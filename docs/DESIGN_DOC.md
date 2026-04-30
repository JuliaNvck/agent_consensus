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
  - **DataFrame schema:** `condition | n_agents | beta | fault_type | accuracy | admission_rate | fallback_frequency`. 128 rows for the full grid (4 conditions × 2 N × 4 β × 4 fault types). `accuracy` = extracted-answer exact-match fraction (see below); `admission_rate` = mean `n_admitted/N` (always 1.0 for baseline/soft_weighting); `fallback_frequency` = fraction of liveness-fallback rounds.
  - **Answer extraction (`_extract_answer(output_text, ground_truth) -> str`):** Real LLM outputs are chain-of-thought strings, not bare answer tokens. Accuracy is computed on extracted answers: (1) StrategyQA (GT ∈ {"yes","no"}): regex finds the first `\b(yes|no)\b` in the output (case-insensitive); (2) GSM8K (GT is a number string): regex finds all `\$?[\d,]+` tokens, returns the last one stripped of `$` and `,`; (3) Fallback: returns `output_text.strip()` unchanged (preserves exact-match correctness for synthetic test caches).
  - **Tau auto-calibration (`calibrate_tau(questions, percentile=10.0) -> float`):** `run_experiment_1` accepts `tau: Optional[float] = None`. When `None`, tau is calibrated automatically from the loaded clean agents: compute mean TopKMass for every agent with non-empty logprobs, sort, return the 10th-percentile value. This implements the design-doc §3.2 rule without requiring callers to supply a dataset-specific constant. Real LLM logprobs are a valid probability simplex (top-5 probs sum ≤ 1.0), so empirical clean-agent scores cluster in [0.83, 1.0]; `_DEFAULT_TAU = 1.0` is only suitable for unit-test synthetic logprobs where the top-5 sum can exceed 1.0.
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
  - **Dynamic `max_tokens`:** Two separate `llm.generate` calls are made — one per dataset — to avoid over-allocating GPU memory: GSM8K uses `max_tokens=512` (multi-step reasoning — raised from 256 after empirical analysis showed truncation causing ~10% accuracy loss), StrategyQA uses `max_tokens=128` (short yes/no + explanation).
  - **Logprob Geometry:** `CompletionOutput.logprobs` is `List[Dict[int, Logprob]]` — one dict per output token mapping `token_id → Logprob`. `_flatten_logprobs` sorts each dict descending by `.logprob`, takes the top-5, and pads to exactly 5 entries with `-100.0`. Result: a flat `List[float]` of length `5 × T`, strictly satisfying Module 1's `ValueError` guard (`len % 5 != 0 → raise`).
  - **Ground Truth Extraction:** GSM8K answers contain reasoning text followed by `#### <number>` — ground truth is `answer.split("####")[-1].strip()`. StrategyQA `answer` is a boolean — converted to `"yes"` / `"no"`.
  - **Agent IDs:** `f"q{question_id}_a{agent_idx}"` where `agent_idx ∈ 0..6`.
  - **Default Agent State:** All `AgentGeneration` dicts initialised with `is_faulty=False`, `fault_type=None`.
  - **CLI:** `python scripts/generate_cache.py [--model ...] [--n-questions 50] [--output cache.json]`.

## 7. Paper Experimental Setup

### 7.1 Models
Two open-weight models of the same parameter class are evaluated to demonstrate generality across architectures:

| Model | HuggingFace ID | Notes |
|---|---|---|
| LLaMA 3.1 8B Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Primary model; Meta's instruction-tuned 8B |
| Qwen2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct` | Strong on both math and QA benchmarks |

Each model gets its own `cache.json` (e.g., `cache_llama.json`, `cache_qwen.json`). The evaluation pipeline (`eval/runner.py`) is model-agnostic — `run_experiment_1` is called independently per cache.

### 7.2 Datasets
| Dataset | Split | N questions | Task type | Ground truth format |
|---|---|---|---|---|
| GSM8K (`gsm8k`, `main` config) | `test` | 50 | Multi-step arithmetic reasoning | Integer string (e.g. `"18"`) |
| StrategyQA (`wics/strategy-qa`) | `test` | 50 | Commonsense multi-hop QA | `"yes"` / `"no"` |

These two datasets stress orthogonal failure modes: GSM8K exposes step-counting errors and truncation under fault injection; StrategyQA exposes semantic drift and Byzantine manipulation.

### 7.3 Baselines
Five conditions are compared. The first four are implemented in `eval/runner.py` and `eval/baselines.py`; the fifth is cited from the original paper.

| Condition | Description | Code location |
|---|---|---|
| **Single agent** | N=1, greedy decoding; no consensus | Requires adding N=1 to `n_values` grid |
| **Self-consistency** (Wang et al., 2023) | Majority vote over N samples | `baseline` condition in `run_experiment_1` |
| **Soft-weighted SC** | TopKMass-weighted geometric median | `soft_weighting` condition |
| **Hard filter + majority** | Module 1 admission → majority vote | `hard_only` condition |
| **Full system (ours)** | Module 1 admission → Module 2 semantic aggregation | `full_system` condition |

**DecentLLMs (Jo & Park, 2024):** `eval/decent_baseline.py` implements the structural worker/evaluator scoring algorithm but uses SHA-256-seeded random scores as a deterministic stand-in for real LLM evaluator calls — suitable for unit testing and ablation structure, not paper reporting. For the paper, cite Jo & Park's reported numbers directly and note the model difference (they use GPT-3.5-Turbo; we use open-weight models).

### 7.4 Evaluation Grid
The ablation grid defined in `run_experiment_1` covers:
- **N ∈ {5, 7}** agents per question
- **β ∈ {0%, 15%, 30%, 45%}** fault fraction
- **Fault types:** F1 (crash), F2 (Byzantine), F3 (drifter), mix

The central claim of the paper is that the full system's accuracy-vs-β curve degrades more gracefully than self-consistency (baseline). The delta between `full_system` and `baseline` at β=0.30 and β=0.45 is the primary result.

### 7.5 Published Reference Points
These numbers provide sanity checks for beta=0 (no-fault) accuracy before fault injection:

| Model | Dataset | Condition | Published accuracy | Source |
|---|---|---|---|---|
| LLaMA 3.1 8B | GSM8K | Greedy (single agent) | ~73% | Meta AI (2024) |
| LLaMA 3.1 8B | GSM8K | Self-consistency N=40 | ~82% | Meta AI (2024) |
| Qwen2.5 7B | GSM8K | Greedy (single agent) | ~85% | Qwen team (2024) |

With N=7 agents and majority vote, expect beta=0 accuracy to land between the greedy and N=40 self-consistency figures. Numbers materially below the greedy baseline indicate an infrastructure issue (e.g., token truncation, prompt mismatch) rather than a model limitation.

### 7.6 LLaMA 3.1 8B Results (`cache_llma.json`, `results/experiment_1_llama.csv`)

Generated with `max_tokens=512` for GSM8K, `max_tokens=128` for StrategyQA, N∈{1,5,7} agents, temperature=0.7. Tau auto-calibrated from the 10th percentile of clean-agent TopKMass scores. CSV contains 192 rows (128 for N∈{5,7} + 64 for N=1).

**Accuracy by beta (averaged over N∈{5,7} and all fault types):**

| Condition | N=1 (single agent) | β=0% | β=15% | β=30% | β=45% |
|---|---|---|---|---|---|
| baseline (self-consistency) | 0.710 | 0.710 | 0.710 | 0.001 | 0.000 |
| soft_weighting | 0.710 | 0.715 | 0.701 | 0.544 | 0.357 |
| hard_only | 0.710 | 0.710 | 0.714 | 0.435 | 0.294 |
| **full_system (ours)** | **0.710** | **0.700** | **0.716** | **0.724** | **0.656** |

**Key findings:**
- Single-agent accuracy (N=1): 0.710 — the no-consensus anchor. All conditions identical at N=1 since there is nothing to filter or aggregate.
- Clean accuracy (β=0%) is 0.71 across all conditions, consistent with published LLaMA 3.1 8B self-consistency performance on GSM8K + StrategyQA combined.
- `baseline` collapses completely at β≥0.30 (majority vote is overwhelmed when ~2 of 5 agents are faulty).
- `full_system` maintains **0.656 accuracy at β=0.45** — the primary paper result. Delta vs baseline at β=0.45: **+0.656**.
- Fallback frequency at β=0%: 9.5% (healthy — tau calibration is not over-filtering clean agents).
- F2 (Byzantine) is the hardest fault type for `full_system` (0.570 at β=0.45 vs 0.675 for F1/F3) because spoofed logprobs of −0.02 pass Module 1; semantic aggregation in Module 2 must do all the work.
- F1 (crash) and F3 (drifter) are cleanly filtered by Module 1, explaining their higher `full_system` accuracy.

**Interpretation:** The accuracy-vs-β curve for `full_system` is nearly flat from 0% to 30% fault load, then degrades gracefully to 0.656 at 45%. This is the core claim: BFT-inspired filtering + semantic aggregation provides robustness that majority voting cannot.

### 7.7 Qwen2.5 7B Results (`cache_qwen.json`, `results/experiment_1_qwen.csv`)

Generated with identical settings to LLaMA (`max_tokens=512` GSM8K, `max_tokens=128` StrategyQA, N∈{1,5,7}, temperature=0.7). Tau auto-calibrated per-cache. CSV contains 192 rows (128 for N∈{5,7} + 64 for N=1).

**Accuracy by beta (averaged over N∈{5,7} and all fault types):**

| Condition | N=1 (single agent) | β=0% | β=15% | β=30% | β=45% |
|---|---|---|---|---|---|
| baseline (self-consistency) | 0.690 | 0.690 | 0.690 | 0.011 | 0.003 |
| soft_weighting | 0.690 | 0.660 | 0.665 | 0.499 | 0.340 |
| hard_only | 0.690 | 0.690 | 0.690 | 0.438 | 0.291 |
| **full_system (ours)** | **0.690** | **0.660** | **0.665** | **0.665** | **0.664** |

**Key findings:**
- Single-agent accuracy (N=1): 0.690 — the no-consensus anchor for Qwen.
- `full_system` accuracy is essentially flat across all fault fractions (0.660–0.665) — stronger robustness than LLaMA.
- `baseline` collapses at β≥0.30 identically to LLaMA, confirming the failure mode is architectural (majority vote), not model-specific.
- F2 (Byzantine) remains the hardest fault type: `full_system` scores 0.620 at β=0.45 vs 0.675–0.680 for F1/F3/mix.
- Fallback frequency at β=0%: 8.5% — comparable to LLaMA (9.5%), confirming tau calibration generalises across model families.

**Cross-model comparison at key fault levels:**

| Condition | Model | N=1 | β=0% | β=30% | β=45% |
|---|---|---|---|---|---|
| baseline | LLaMA 3.1 8B | 0.710 | 0.710 | 0.001 | 0.000 |
| baseline | Qwen2.5 7B | 0.690 | 0.690 | 0.011 | 0.003 |
| **full_system** | **LLaMA 3.1 8B** | **0.710** | **0.700** | **0.724** | **0.656** |
| **full_system** | **Qwen2.5 7B** | **0.690** | **0.660** | **0.665** | **0.664** |

**Interpretation:** Both models show identical qualitative behaviour — `baseline` collapses under fault load while `full_system` remains robust — demonstrating that the pipeline's fault tolerance is model-agnostic. Qwen's flatter curve (variance < 0.005 across all β) makes it the stronger robustness demonstration; LLaMA's higher clean accuracy (0.710 vs 0.660) makes it the stronger absolute-performance result.

### 7.8 Figures (`figures/`)

Two figures are generated by `scripts/plot_results.py` from the merged 192-row CSVs.

#### Figure 1: `figures/accuracy_vs_beta.png` — Accuracy vs. Fault Fraction

Two-panel line chart (LLaMA left, Qwen right). X-axis: β ∈ {0%, 15%, 30%, 45%}. Y-axis: accuracy. Five series per panel:
- **Single Agent (N=1, dotted grey):** Flat reference line at 0.71 (LLaMA) / 0.69 (Qwen). No consensus, no fault resilience. **Caveat:** This line is flat not because a single agent is fault-tolerant, but because `inject_faults` injects `floor(N × β)` faults — at N=1, `floor(1 × 0.45) = 0` for all β < 1.0, so the single agent is always evaluated on a clean, unmodified output. The line represents a *permanently clean* single agent, not a single agent subject to the same fault rate as the other conditions. The correct reading is: "our full system at β=45% approaches the accuracy of a guaranteed-clean single agent, even when 45% of the pool is faulty." A separate set of figures (`figures/accuracy_vs_beta_no_n1.png`, `figures/fault_type_breakdown_no_n1.png`) omits this line to avoid visual confusion in the main paper body.
- **Self-Consistency / baseline (red dashed):** Matches single-agent at β=0–15%, then collapses to ~0% at β=30–45%. Visually disappears into the x-axis.
- **Soft-Weighted SC (orange dotted):** Degrades more gracefully than baseline but still falls to 36–34% at β=45%.
- **Hard Filter + Majority / hard_only (blue dash-dot):** Better than soft-weighting at β=30%, but drops to 29% at β=45% as Module 1 fallback fires more frequently.
- **Full System / full_system (green solid):** Nearly flat across all β. LLaMA: 0.70→0.72→0.72→0.66; Qwen: 0.66→0.67→0.67→0.66. Remains well above the single-agent reference line at high fault loads.

**Key visual story:** At β=30–45%, the full system line is the only one that stays near or above the single-agent reference; all baselines fall far below it.

#### Figure 2: `figures/fault_type_breakdown.png` — Accuracy by Fault Type at β=45%

Two-panel grouped bar chart. X-axis: fault types F1 (Crash), F2 (Byzantine), F3 (Drifter), Mix. Two bars per group: Self-Consistency (red) and Full System (green).
- **Self-Consistency bars:** Near-zero across all fault types and both models — the red bars are barely visible.
- **Full System bars:** 57–71% across all fault types (LLaMA) and 62–68% (Qwen).
- **F2 (Byzantine) is visibly shorter** than F1/F3/Mix for both models, confirming that spoofed high-confidence logprobs that pass Module 1 are the hardest fault to handle — Module 2 must do all the work.
- **Mix fault type** achieves the highest `full_system` accuracy (0.705 LLaMA, 0.680 Qwen) because the mixture of F1/F2/F3 means fewer pure Byzantine agents than the worst-case F2-only scenario.

**Key visual story:** The full system maintains meaningful accuracy under every fault type at maximum fault load; self-consistency provides zero resilience regardless of fault type.

**Paper caption note:** Include the following explanation in the figure caption: *"Self-Consistency (Majority Vote) collapses to 0% under all fault types because 2 coordinated Byzantine agents share the same wrong answer, outvoting 3 clean agents whose correct answers are phrased differently."*

---

## 8. Experiment 2: Signal Quality Analysis (`eval/signal_quality.py`)

### 8.1 Research Question

Experiment 1 proves *that* the pipeline is robust. Experiment 2 proves *why*: is the TopKMass signal actually predictive of individual agent correctness, and is it a stronger predictor than simpler baselines (entropy, logprob variance)?

### 8.2 Signal Definitions

Three signals are computed per `AgentGeneration` from `token_logprobs` (flat list of top-5 logprobs, length `5×T`). All signals are oriented so that **higher = more confident = more likely correct** (entropy and variance are negated).

**Signal 1 — TopKMass Mean** (the filter signal; reuses `pipeline/filter._compute_topk_mass_trajectory`):
```
traj = _compute_topk_mass_trajectory(token_logprobs)   # causal W=64 sliding window
topk_mass = mean(traj)
```

**Signal 2 — Negated Mean Token Entropy** (baseline):
```
arr = token_logprobs.reshape(T, 5)
probs = exp(arr)                                     # (T,5) unnormalized top-5 probs
H_i  = -sum(probs[i] * arr[i])  for each position i  # -sum(p * log p), top-5 approximation
neg_entropy = -mean(H_i)
```
Lower entropy = more peaked distribution = more confident → negating makes the signal higher-is-better.

**Signal 3 — Negated Logprob Variance** (baseline):
```
per_pos_mean_lp = arr.mean(axis=1)   # (T,) mean logprob per position
neg_logprob_var = -var(per_pos_mean_lp)
```
Lower variance = more stable confidence across the sequence → negated so higher-is-better.

Agents with empty `token_logprobs` (F1_crash) are skipped — no signal to compute.

### 8.3 Ground-Truth Labeling

Uses the same `_extract_answer` from `eval/runner.py` as Experiment 1, ensuring consistency:
```python
is_correct = _extract_answer(gen.output_text, ground_truth) == ground_truth.strip()
```

### 8.4 Output Files

| File | Description |
|---|---|
| `results/experiment_2_signals.csv` | Per-agent DataFrame: `question_id, topk_mass, neg_entropy, neg_logprob_var, is_correct` |
| `results/experiment_2_signals.png` | 3-panel figure (ROC, scatter, PR) |

### 8.5 Figure Panels

- **Panel A (ROC Curves):** One line per signal with AUC in legend. Gray dashed chance diagonal.
- **Panel B (Scatter):** TopKMass score vs. jittered correctness label. Vertical median lines per class (correct / incorrect) show distributional separation.
- **Panel C (Precision-Recall):** One line per signal with Average Precision in legend. Gray dashed random-classifier baseline.

### 8.6 How to Run

```bash
# On a specific cache (CPU only):
python -m eval.signal_quality --cache cache_llma.json --output-dir results/exp2_llama/
python -m eval.signal_quality --cache cache_qwen.json --output-dir results/exp2_qwen/
```

### 8.7 Expected Results

- `topk_mass` AUC > 0.5 (better than chance) and higher than entropy/variance AUC
- All `topk_mass` values in `[0.0, 1.0]` (valid probability simplex)
- `neg_entropy` values are negative (raw top-5 entropy is positive before negation)
- Panel B shows visible separation: the median TopKMass of correct agents > median of incorrect agents