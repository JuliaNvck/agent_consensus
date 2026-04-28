# System Design Document: Multi-Agent LLM Consensus

## 1. System Overview
A fault-tolerant, two-module Python 3.11 asynchronous pipeline that filters and aggregates multi-agent LLM outputs based on generation-process signals and semantic clustering.
- **Phase 1 (Generation/Offline):** Agents generate text via a local `vLLM` server. All outputs and per-token log-probabilities are generated once and cached to a local JSON file.
- **Phase 2 (Evaluation/Pipeline):** The filtering and aggregation pipeline runs entirely on the cached JSON data. **Crucial:** No `vLLM` imports or GPU generation code are permitted in Phase 2 to ensure lightweight reproducibility.

## 2. Core Data Structures
The pipeline relies on these explicit, statically typed data structures passing between modules:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AgentGeneration:
    agent_id: str
    output_text: str
    token_logprobs: List[float]
    is_faulty: bool
    fault_type: Optional[str] # None, 'F1_crash', 'F2_byzantine', 'F3_drifter'

@dataclass
class ConsensusResult:
    final_answer: str
    admitted_agents: List[str] # List of agent_ids that passed Module 1
    is_low_confidence: bool    # True if the fallback trigger was activated
```

## 3. Architecture & Interfaces

### 3.1 Orchestrator (`coordination/orchestrator.py`) ✅
- **Role:** Asynchronous manager handling timeouts, routing, and liveness.
- **Input:** `List[AgentGeneration]` retrieved from the data cache.
- **Logic:** 1. Broadcast task (using simulated `asyncio.sleep` delays).
  2. Collect responses with strict timeouts.
  3. Pass generations to Module 1.
  4. **Liveness Fallback:** If `< 2f+1` agents pass Module 1, set `is_low_confidence = True` and admit all available agents.
  5. Pass admitted pool to Module 2.
- **Implementation notes:** `f` is a constructor argument (`Orchestrator(f, agent_timeout)`). F1_crash agents are assigned a sleep delay that exceeds `agent_timeout`, causing `asyncio.wait_for` to cancel them. Empty admitted pool short-circuits before Module 2 with `is_low_confidence=True`.

### 3.2 Module 1: Reliability Filter (`pipeline/filter.py`) ✅
- **Role:** Binary admission control using generation-process signals.
- **Input:** `AgentGeneration` objects (specifically `token_logprobs`).
- **`token_logprobs` layout:** Flat list of top-5 logprobs per token position; length = `5 × T` for a T-token output.
- **Mathematical Objective:** Compute the sliding W=64 window:
  `TopKMass(W) = (1/W) * SUM(top-5 token probs)`
- **Logic:**
  - Calculate the mean and variance of the `TopKMass` trajectory for each agent.
  - Drop agents whose score falls below the threshold τ (calibrated at the 10th percentile of clean-agent scores on a dev slice).
- **Output:** Filtered `List[AgentGeneration]`.
- **Implementation notes:** τ is an explicit caller-supplied parameter to `filter_agents(generations, tau)`. The causal sliding window mean is computed in **O(T)** via a prefix-sum array (`np.cumsum`), avoiding an O(T×W) inner loop. Agents with empty `token_logprobs` (e.g. F1_crash) are dropped unconditionally. Input length not divisible by 5 raises `ValueError`.

### 3.3 Module 2: Robust Semantic Aggregation (`pipeline/aggregation.py`) ✅
- **Role:** Extract the highest-fidelity answer from the admitted pool.
- **Stage 1 (Cluster - O(N)):**
  - Embed the `output_text` of admitted agents using `sentence-transformers/all-mpnet-base-v2`.
  - Calculate the geometric median of the embeddings using the Weiszfeld algorithm via `scipy.optimize`.
  - Identify the candidate text nearest to this median centroid.
- **Stage 2 (Verify - O(1)):**
  - Run bidirectional entailment between the centroid's nearest candidate and the centroid itself.
  - Model: `cross-encoder/nli-deberta-v3-large`.
  - **Constraint:** Must evaluate `(A entails B)` and `(B entails A)` in a single batched tensor operation.
- **Output:** The verified final string.
- **Implementation notes:**
  - **Lazy Loading:** Both models (`SentenceTransformer`, `AutoModelForSequenceClassification`) are module-level singletons initialised on first call via `_get_embed_model()` / `_get_nli_model()`. They are never imported at module load time, so the rest of the pipeline (and the entire test suite) starts instantly without touching disk or GPU.
  - **Geometric Median — Weiszfeld via L-BFGS-B:** `_geometric_median` minimises `f(y) = Σ‖x_i − y‖₂` using `scipy.optimize.minimize(method='L-BFGS-B')` with an analytic gradient `∇f(y) = −Σ (x_i − y) / ‖x_i − y‖`. Distance denominators are clamped to `1e-10` to handle the degenerate case where two embeddings are identical (prevents division-by-zero at the starting point).
  - **Centroid Proxy Logic:** The design doc specifies "bidirectional entailment between the nearest candidate and the centroid itself." Three steps make this concrete:
    1. **Final Selection — Primary Candidate:** The geometric median is a mathematical coordinate in embedding space, not a text string. To determine the final answer, the system computes the Euclidean distance from this coordinate to every admitted agent's embedding and selects the text of the **nearest agent** as the Primary Candidate. This is the string ultimately returned by `aggregate`.
    2. **Verification Logic — Centroid Proxy:** Because an NLI model requires two text strings, the geometric median vector cannot be passed directly to it. The **second-nearest** admitted agent output serves as the Centroid Proxy — an independent agent output that is the closest distinct view of the same semantic cluster. Bidirectional entailment between the Primary Candidate and this proxy confirms that two independent, centrally-located agents agree semantically.
    3. **Fallback on Entailment Failure:** If either direction of the entailment check fails (predicts contradiction or neutral), the system logs a `WARNING` but still returns the Primary Candidate unchanged. Suppressing the answer is not `aggregate`'s responsibility — the orchestrator flags the round with `ConsensusResult.is_low_confidence = True`. Edge case: when `N == 1`, Stage 2 is skipped entirely (single-agent shortcut at the top of `aggregate`).
  - **Tensor Batching Workaround:** `CrossEncoder.predict()` (the sentence-transformers high-level API) iterates over pairs internally and may issue multiple forward passes. To guarantee a single batched tensor operation, Stage 2 uses raw `transformers.AutoTokenizer` + `AutoModelForSequenceClassification` directly: the two pairs `[A, B]` and `[B, A]` are tokenized together and passed through `model(**inputs)` once, yielding logits of shape `(2, 3)` in one GPU kernel call.
  - **Dynamic Label Resolution:** The entailment class index is read from `model.config.id2label` at runtime (`{v.lower(): int(k) for k, v in ...}`) rather than hardcoded. This makes the code robust to label-order differences between NLI model checkpoints.
  - **Entailment Failure Policy:** A failed bidirectional check (one or both directions predict contradiction/neutral) logs a `WARNING` but does not suppress the answer. The return type of `aggregate` is `str`; low-confidence signalling is the orchestrator's responsibility via `ConsensusResult.is_low_confidence`.
  - **Test Isolation Strategy:** `tests/conftest.py` contains an `autouse=True` function-scoped fixture that stubs `_embed` (returns `zeros((N, 2))`) and `_batched_entailment` (returns `(True, True)`) for every test in the suite. Orchestrator integration tests run without triggering model downloads. Aggregation unit/integration tests override these stubs per-test with `monkeypatch.setattr`, which shares the same `monkeypatch` instance and reverts cleanly in LIFO order.

## 4. Fault Models & Injection (`faults/injector.py`) ✅
Applies mutations to the cached data to simulate three fault conditions at varying fractions (β ∈ {0%, 15%, 30%, 45%}):
- **F1 (Crash):** Simulate timeout. Agent produces no output (`output_text=""`, `token_logprobs=[]`). Unconditionally dropped by Module 1 (empty logprob branch).
- **F2 (Byzantine):** Replace output with a pre-specified adversarial wrong answer. Logprobs are spoofed to `−0.02` per entry → `5 × exp(−0.02) ≈ 4.90` mean TopKMass per position → **intentionally passes Module 1** at any realistic τ.
- **F3 (Drifter):** Replace output with syntactically plausible but semantically off-task text, simulating temperature 1.5. Logprobs are spoofed to `−10.0` per entry → `5 × exp(−10) ≈ 2.3×10⁻⁴` mean TopKMass → **intentionally fails Module 1** at any τ > 0.001.
- **Implementation notes:**
  - **Entry point:** `inject_faults(generations, beta, fault_type, seed)` where `fault_type ∈ {F1, F2, F3, mix}`.
  - **Exact count:** Exactly `math.floor(N × β)` agents are mutated — no rounding ambiguity.
  - **Determinism:** Uses `random.Random(seed)` (isolated, never touches global state). `rng.sample(range(N), n_faults)` selects fault indices; for `mix`, `rng.choice(('F1','F2','F3'))` assigns per-agent types.
  - **Immutability:** Original `AgentGeneration` objects are never mutated. Clean agents are returned by reference; faulty agents are new objects.
  - **Logprob length:** Spoofed logprob lists match the source agent's token count (`T = len(original.token_logprobs) // 5`), defaulting to T=20 if the original is empty.

## 5. Evaluation Harness (`eval/runner.py`) ✅
- **Role:** Iterate over the cached, fault-injected data and generate metric reports.
- **Ablation Conditions:**
  - Baseline: No filter, majority voting aggregation.
  - Soft-weighting: Score-weighted geometric median.
  - Hard-filtering: Module 1 admission -> majority voting.
  - Full System: Module 1 admission -> Module 2 aggregation.
- **External Baseline (`eval/decent_baseline.py`) ✅:** Re-implementation of DecentLLMs worker/evaluator scoring (Jo & Park) — isolated from the main pipeline.
- **Output:** Pandas DataFrame exported to CSV tracking Accuracy, Admission Rate, and Fallback Frequency.
- **Implementation notes:**
  - **Cache format:** JSON file with `{"questions": [{"question_id", "ground_truth", "generations": [{agent fields}]}]}`. Minimum 7 agents per question (to support N=7). `load_cache(filepath)` returns `List[Tuple[str, List[AgentGeneration]]]`.
  - **Baseline functions (`eval/baselines.py`):** `majority_voting` uses `collections.Counter.most_common(1)`. `soft_weighted_geometric_median` computes per-agent TopKMass mean as weight, runs a weighted Weiszfeld L-BFGS-B minimisation (`_weighted_geometric_median`), returns text of nearest agent. Falls back to uniform weights when all weights < 1e-15 (all F1 agents).
  - **Embedding reuse in baselines:** `eval/baselines.py` accesses `_embed` via `from pipeline import aggregation as _aggregation; _aggregation._embed(texts)` so that `conftest.py`'s `autouse` monkeypatch on `pipeline.aggregation._embed` applies in tests without extra patching.
  - **Liveness fallback in runner:** `_run_condition` replicates orchestrator logic directly (`await filter_agents(agents, tau)` + `if len(admitted) < 2f+1: fallback`) without the asyncio.sleep broadcast overhead. `f = (N-1)//3` (BFT standard: N=5→f=1, threshold=3; N=7→f=2, threshold=5).
  - **Configurable grid:** `run_experiment_1` accepts `n_values`, `beta_values`, `fault_types` keyword args (defaulting to the full grid) so tests can pass minimal single-element slices for speed.
  - **DataFrame schema:** `condition | n_agents | beta | fault_type | accuracy | admission_rate | fallback_frequency`. 128 rows for the full grid (4 conditions × 2 N × 4 β × 4 fault types). `accuracy` = exact-match fraction; `admission_rate` = mean `n_admitted/N` (always 1.0 for baseline/soft_weighting); `fallback_frequency` = fraction of liveness-fallback rounds.
  - **DecentLLMs external baseline (`eval/decent_baseline.py`):** Entry point `run_decent_baseline(agents, num_evaluators=5) -> str`. For each worker: (1) call `_evaluate_candidate(text, evaluator_id)` for each of `N_e=5` evaluators → `(N_e, 5)` score matrix over C=5 criteria (scores 0–20); (2) compute geometric median of that matrix via uniform-weight Weiszfeld (`_weighted_geometric_median` from `eval/baselines.py`); (3) sum the 5 robust-median components → scalar worker score. Winner = highest scalar score; tie-break = largest SHA-256 hex digest of `output_text`. `_evaluate_candidate` is a deterministic mockable helper (SHA-256 seed → `np.random.default_rng`) — no real LLM calls.

## 6. Live Data Generation (`scripts/generate_cache.py`) ✅
- **Role:** Offline Phase 1 script to run local LLM inference and build the static `cache.json` dataset for Phase 2 evaluation.
- **Input:** Sample subsets (50 questions each) from HuggingFace `gsm8k` (`main` config, `test` split) and `wics/strategy-qa` (`test` split).
- **Generation:** Uses `vllm.LLM` for synchronous batched inference with `temperature=0.7` (ensures text variance among N=7 agents) and `logprobs=5` (required for Module 1 filtering).
- **Output:** Writes to `cache.json` matching the schema: `{"questions": [{"question_id": str, "ground_truth": str, "generations": [AgentGeneration dicts]}]}`.
- **Implementation notes:**
  - **The GPU Exception:** This is the *only* file in the repository permitted to import `vllm` and `torch`.
  - **Explicit Prompts:** Raw questions are wrapped in dataset-specific templates before applying the model's chat template:
    - GSM8K: `"Solve the following math problem step by step, ending with the final answer:\n{question}"`
    - StrategyQA: `"Answer the following question with a clear 'yes' or 'no' and briefly explain why:\n{question}"`
    - Chat template applied via `tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)`.
  - **Dynamic `max_tokens`:** Two separate `llm.generate` calls are made — one per dataset — to avoid over-allocating GPU memory: GSM8K uses `max_tokens=256` (multi-step reasoning), StrategyQA uses `max_tokens=128` (short yes/no + explanation).
  - **Logprob Geometry:** `CompletionOutput.logprobs` is `List[Dict[int, Logprob]]` — one dict per output token mapping `token_id → Logprob`. `_flatten_logprobs` sorts each dict descending by `.logprob`, takes the top-5, and pads to exactly 5 entries with `-100.0`. Result: a flat `List[float]` of length `5 × T`, strictly satisfying Module 1's `ValueError` guard (`len % 5 != 0 → raise`).
  - **Ground Truth Extraction:** GSM8K answers contain reasoning text followed by `#### <number>` — ground truth is `answer.split("####")[-1].strip()`. StrategyQA `answer` is a boolean — converted to `"yes"` / `"no"`.
  - **Agent IDs:** `f"q{question_id}_a{agent_idx}"` where `agent_idx ∈ 0..6`.
  - **Default Agent State:** All `AgentGeneration` dicts initialised with `is_faulty=False`, `fault_type=None`.
  - **CLI:** `python scripts/generate_cache.py [--model ...] [--n-questions 50] [--output cache.json]`.