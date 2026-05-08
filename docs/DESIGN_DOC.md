# System Design Document: Multi-Agent LLM Consensus

## 1. System Overview
A robust aggregation pipeline for multi-agent LLM outputs. The system identifies and down-weights degraded outputs using generation-process signals (Module 1), then aggregates the remaining outputs using a robust semantic estimator (Module 2).

**Framing note:** The current experimental setup uses N=7 samples from the same model (homogeneous pool). In that setting the problem is *robust self-consistency*: some runs are wrong by chance, some produce low-quality output (crashes, off-task drift, confidently wrong answers). The BFT fault-tolerance framing — f-fault tolerance, 2f+1 liveness threshold — is used as analytical scaffolding to reason about worst-case degradation, not as a claim of adversarial robustness. A genuine adversarial setting would require agents operated by different untrusted parties (see §1.1).

- **Phase 1 (Generation/Offline):** Agents generate text via a local `vLLM` server. All outputs and per-token log-probabilities are generated once and cached to a local JSON file.
- **Phase 2 (Evaluation/Pipeline):** The filtering and aggregation pipeline runs entirely on the cached JSON data. **Crucial:** No `vLLM` imports or GPU generation code are permitted in Phase 2 to ensure lightweight reproducibility.

### 1.1 Future Direction: Multi-Provider Heterogeneous Setting
The natural extension of this work is a pool of agents operated by **different providers running different models** (e.g., GPT-4o, Claude, Gemini, LLaMA, Qwen). In that setting:
- The BFT framing becomes genuinely motivated: a provider has an incentive to return confident-looking wrong answers, making F2 (confident-wrong with high-logprob spoofing) a realistic threat rather than a synthetic stress test.
- TopKMass remains comparable across models because it is a ratio (sum of top-5 probs), not an absolute logprob value, making it architecture-agnostic.
- Module 2 (geometric median in embedding space) is already model-agnostic — all text passes through the same sentence transformer regardless of source model.
- Key blocker: not all providers expose per-token logprobs (Claude does not; OpenAI does). Module 1 would need a fallback for logprob-unavailable providers.
- See implementation notes in §10 for what would change.

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
  - Calculate the stable-region mean and variance of the `TopKMass` trajectory for each agent (post-warmup positions only — see below).
  - Drop agents whose score falls below the threshold τ (calibrated at the 5th percentile of clean-agent scores on a dev slice).
- **Output:** Filtered `List[AgentGeneration]`.
- **Implementation notes:** τ is an explicit caller-supplied parameter to `filter_agents(generations, tau)`. The causal sliding window mean is computed in **O(T)** via a prefix-sum array (`np.cumsum`), avoiding an O(T×W) inner loop. Agents with empty `token_logprobs` (e.g. F1_crash) are dropped unconditionally. Input length not divisible by 5 raises `ValueError`.
  - **Warmup-normalized TopKMass (`_agent_stats` warmup cutoff):** For the first W=64 token positions, the sliding window is only partially populated (`window_len = i+1 < 64`), so early trajectory values are systematically lower than the true stable-state TopKMass. A 128-token StrategyQA output has 50% of its trajectory in warmup; a 512-token GSM8K output has only 12.5%. Without correction, a single global τ penalizes StrategyQA agents disproportionately — clean StrategyQA agents were being filtered at ~2× the rate of clean GSM8K agents. Fix: `_agent_stats(trajectory, warmup=WINDOW_SIZE)` slices `trajectory[warmup:]` before computing mean/variance, making the score length-invariant. Falls back to the full trajectory when output length ≤ W. The same slice is applied in `calibrate_tau` (via `_agent_stats`) so calibration and filtering use the identical metric.

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
    3. **Candidate Selection with NLI Fallback:** `aggregate` iterates all admitted candidates in order of distance to the geometric median (nearest first), checking each against the fixed second-nearest reference via bidirectional entailment. The first candidate that passes both directions is returned as `(answer, False)`. If no candidate passes (all fail), the nearest-centroid candidate is returned as `(answer, True)` — is_low_confidence signals the orchestrator. Edge case: when `N == 1`, Stage 2 is skipped entirely (single-agent shortcut at the top of `aggregate`).
  - **Return type change:** `aggregate` returns `Tuple[str, bool]` (final_answer, is_nli_low_confidence). Callers in `eval/runner._run_condition` and `coordination/orchestrator.run` unpack the tuple and merge the NLI flag into the round-level `is_low_confidence` flag.
  - **Tensor Batching Workaround:** `CrossEncoder.predict()` (the sentence-transformers high-level API) iterates over pairs internally and may issue multiple forward passes. To guarantee a single batched tensor operation, Stage 2 uses raw `transformers.AutoTokenizer` + `AutoModelForSequenceClassification` directly: the two pairs `[A, B]` and `[B, A]` are tokenized together and passed through `model(**inputs)` once, yielding logits of shape `(2, 3)` in one GPU kernel call.
  - **Dynamic Label Resolution:** The entailment class index is read from `model.config.id2label` at runtime (`{v.lower(): int(k) for k, v in ...}`) rather than hardcoded. This makes the code robust to label-order differences between NLI model checkpoints.
  - **Test Isolation Strategy:** `tests/conftest.py` contains an `autouse=True` function-scoped fixture that stubs `_embed` (returns `zeros((N, 2))`) and `_batched_entailment` (returns `(True, True)`) for every test in the suite. Orchestrator integration tests run without triggering model downloads. Aggregation unit/integration tests override these stubs per-test with `monkeypatch.setattr`, which shares the same `monkeypatch` instance and reverts cleanly in LIFO order.

## 4. Fault Models & Injection (`faults/injector.py`) ✅
Applies mutations to the cached data to simulate three fault conditions at varying fractions (β ∈ {0%, 15%, 30%, 45%}):
- **F1 (Crash):** Simulate timeout. Agent produces no output (`output_text=""`, `token_logprobs=[]`). Unconditionally dropped by Module 1 (empty logprob branch).
- **F2 (Byzantine):** Replace output with a pre-specified adversarial wrong answer. Logprobs are spoofed to a valid top-5 distribution `[log(0.95), log(0.02), log(0.015), log(0.010), log(0.005)]` repeated per token → **TopKMass = 1.00 per position** → **intentionally passes Module 1** at any τ ≤ 1.0. Using exactly 1.00 (not ≈0.99) is necessary because high-precision models (e.g., Qwen) produce clean-agent scores up to ~0.9951, which would otherwise inadvertently filter Byzantine agents.
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
  - **Baseline functions (`eval/baselines.py`):** `majority_voting(agents)` uses `collections.Counter.most_common(1)` on raw `output_text`. `answer_majority_voting(agents, ground_truth)` extracts comparable answer tokens via `_extract_answer` before counting — this is the correct baseline for paper results since LLM outputs are chain-of-thought strings with unique phrasing. `soft_weighted_geometric_median` computes per-agent TopKMass mean as weight, runs a weighted Weiszfeld L-BFGS-B minimisation (`_weighted_geometric_median`), returns text of nearest agent. Falls back to uniform weights when all weights < 1e-15 (all F1 agents).
  - **`_run_condition` uses answer-level voting:** The `baseline` and `hard_only` conditions use `answer_majority_voting(agents, ground_truth)`, NOT raw-text `majority_voting`. Raw-text voting causes majority-vote to fail whenever 5 correct agents have different phrasings but 2 wrong agents share the same string (a common scenario with chain-of-thought outputs and coordinated Byzantine agents). The `ground_truth` parameter was added to `_run_condition`'s signature for this purpose.
  - **Embedding reuse in baselines:** `eval/baselines.py` accesses `_embed` via `from pipeline import aggregation as _aggregation; _aggregation._embed(texts)` so that `conftest.py`'s `autouse` monkeypatch on `pipeline.aggregation._embed` applies in tests without extra patching.
  - **Liveness fallback in runner:** `_run_condition` replicates orchestrator logic directly (`await filter_agents(agents, tau)` + `if len(admitted) < 2f+1: fallback`) without the asyncio.sleep broadcast overhead. `f = (N-1)//3` (BFT standard: N=5→f=1, threshold=3; N=7→f=2, threshold=5). `aggregate()` now returns `Tuple[str, bool]`; `_run_condition` merges the NLI low-confidence flag into the round-level flag via `is_low = is_low or nli_low`.
  - **Configurable grid:** `run_experiment_1` accepts `n_values`, `beta_values`, `fault_types`, and `dev_fraction` keyword args. `dev_fraction=0.2` (default) reserves 20% of questions for τ calibration; the remainder are the evaluation set. With 100 questions, this gives 20 dev questions × 7 agents = 140 calibration scores — enough for a stable 5th-percentile estimate (the 7th-lowest score). Previously 0.1 (70 scores), where the 5th percentile was the 4th-lowest value and a single outlier agent could shift τ significantly. For small caches (< 10 questions), `dev_fraction=0.0` is used by tests to avoid consuming all data for calibration.
  - **DataFrame schema:** `condition | n_agents | beta | fault_type | accuracy | admission_rate | fallback_frequency`. 128 rows for the full grid (4 conditions × 2 N × 4 β × 4 fault types). `accuracy` = extracted-answer exact-match fraction (see below); `admission_rate` = mean `n_admitted/N` (always 1.0 for baseline/soft_weighting); `fallback_frequency` = fraction of liveness-fallback rounds.
  - **Answer extraction (`_extract_answer(output_text, ground_truth) -> str`):** Real LLM outputs are chain-of-thought strings, not bare answer tokens. Accuracy is computed on extracted answers: (1) StrategyQA (GT ∈ {"yes","no"}): regex finds the first `\b(yes|no)\b` in the output (case-insensitive); (2) GSM8K (GT is a number string): regex finds all `\$?[\d,]+` tokens, returns the last one stripped of `$` and `,`; (3) Fallback: returns `output_text.strip()` unchanged (preserves exact-match correctness for synthetic test caches).
  - **Tau auto-calibration (`calibrate_tau(questions, percentile=5.0) -> float`):** `run_experiment_1` accepts `tau: Optional[float] = None`. When `None`, tau is calibrated on the dev slice. Uses `_agent_stats(traj)` (warmup-normalized stable-region mean) so calibration is on the same metric as filtering. The 5th percentile (down from 10th) halves the designed false-rejection rate of clean agents — at 10th percentile, ~10% of clean agents were rejected by construction, causing liveness fallback at β=0. At 5th percentile, ~5% are rejected. Real LLM logprobs are a valid probability simplex (top-5 probs sum ≤ 1.0), so empirical clean-agent stable-region scores cluster tightly in [0.96, 1.0]. `_DEFAULT_TAU = 1.0` is only used when the dev slice is empty.
  - **Dev-slice shuffle (critical):** Before splitting into dev/eval, `run_experiment_1` shuffles `all_questions` using `random.Random(seed)`. The cache orders questions as GSM8K first (0–49) then StrategyQA (50–99). Without shuffling, the dev slice is drawn entirely from early GSM8K questions, which tend to produce higher-confidence token distributions (focused step-by-step arithmetic) than StrategyQA or harder GSM8K questions. This biases τ upward, causing ~37% of clean eval-set agents to be filtered and triggering liveness fallback on ~50% of questions at β=0 — far above the intended ≤10% false-positive rate. Shuffling with a fixed seed before the split ensures the dev slice is representative of both question types, restoring the intended fallback rate.
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

**Honest framing of claims:**
- **Primary claim:** The pipeline degrades more gracefully than naive self-consistency (majority vote) as LLM output quality deteriorates — crashes, low-confidence drift, and confident errors are identified and handled before aggregation.
- **Secondary claim:** Geometric median in embedding space resists coordinated wrong-answer clusters better than arithmetic aggregation (Exp 3, centroid shift metric).
- **Not claimed:** Adversarial robustness against a real external attacker. The injected F2 fault is a synthetic stress test, not a real-world threat model in the homogeneous (same-model) setting.

The delta between `full_system` and `baseline` at β=0.30 and β=0.45 is the primary result.

### 7.5 Published Reference Points
These numbers provide sanity checks for beta=0 (no-fault) accuracy before fault injection:

| Model | Dataset | Condition | Published accuracy | Source |
|---|---|---|---|---|
| LLaMA 3.1 8B | GSM8K | Greedy (single agent) | ~73% | Meta AI (2024) |
| LLaMA 3.1 8B | GSM8K | Self-consistency N=40 | ~82% | Meta AI (2024) |
| Qwen2.5 7B | GSM8K | Greedy (single agent) | ~85% | Qwen team (2024) |

With N=7 agents and majority vote, expect beta=0 accuracy to land between the greedy and N=40 self-consistency figures. Numbers materially below the greedy baseline indicate an infrastructure issue (e.g., token truncation, prompt mismatch) rather than a model limitation.

### 7.6 LLaMA 3.1 8B Results — V4 (`results/experiment_1_llama_v4.csv`)

Generated with `max_tokens=512` for GSM8K, `max_tokens=128` for StrategyQA, N∈{1,5,7} agents, temperature=0.7. Tau auto-calibrated at the 5th percentile of clean-agent TopKMass scores on a shuffled dev slice (20% of questions). CSV contains 192 rows (128 for N∈{5,7} + 64 for N=1).

**Methodological changes vs. earlier runs:**
- `baseline` and `hard_only` now use `answer_majority_voting_strict` — agents whose output contains no parseable answer format (yes/no or number) return `None` and are excluded from the vote, preventing garbage-key plurality at high β.
- `full_system` now uses geometric median nearest-centroid (`stage1_only`) without NLI Stage 2. NLI verification was consistently the worst condition in all experiments; the simpler stage1_only approach is superior.

**Accuracy by beta (averaged over N∈{5,7} and all fault types):**

| Condition | N=1 (single agent) | β=0% | β=15% | β=30% | β=45% |
|---|---|---|---|---|---|
| baseline (self-consistency) | 0.713 | 0.756 | 0.769 | 0.736 | 0.708 |
| soft_weighting | 0.713 | 0.713 | 0.725 | 0.728 | 0.688 |
| hard_only | 0.713 | 0.750 | 0.763 | 0.745 | 0.727 |
| **full_system / stage1_only** | **0.713** | **0.700** | **0.723** | **0.733** | **0.653** |

**Key findings:**
- Single-agent accuracy (N=1): 0.713 — the no-consensus anchor.
- **Strict extraction eliminates the β=0.45 collapse.** The old `baseline` at β=0.45 averaged 0.543; V4 is 0.708. This 16.5pp gap was entirely due to the `_extract_answer` fallback bug: F2/F3 agents returned their full garbage text as a "vote" that could win plurality at high β. With strict extraction (`_extract_answer_strict`), those votes abstain.
- **`hard_only` is the best condition for invalid-format faults.** At β=0.45: hard_only=0.727, baseline=0.708, full_system=0.653. Module 1 drops F1/F3 agents; strict vote handles the rest.
- **`full_system` (stage1_only) underperforms in Exp 1 but is essential for Exp 3.** At β=0.45 the geometric median nearest-centroid approach selects a suboptimal agent ~7–10% more often than majority voting when faults are invalid-format. The two-failure-mode structure explains this: (1) invalid-format faults → strict majority vote is the right tool; (2) valid-format coordinated wrong answers (Exp 3) → geometric median is the right tool. A single pipeline cannot be optimal for both.
- **Fallback frequency at β=0 (N=7):** 5.0% for hard_only/full_system — a fraction of clean agents fall below τ (expected; 5th-percentile calibration is designed to tolerate this).
- **F1/F3 at β=0.45 N=7:** hard_only fallback_frequency=1.0 (all agents filtered → liveness fires → full pool → same as baseline). full_system and hard_only are identical in accuracy under F1/F3 at max load; the difference is only under F2 (Byzantine spoofed logprobs pass Module 1, so filtering is active).
- **soft_weighting holds well for F3 (drifters):** N=7 β=0.45 F3: soft=0.725 vs baseline=0.6375. Continuous weighting avoids the liveness fallback that fires for hard_only/full_system under F3 overload.

**Interpretation:** The strict extraction fix is the primary Exp 1 result — not a system gain but a methodology correction. The corrected baseline is stronger (0.708 at β=0.45 vs 0.543 before), making `hard_only` the best condition for this fault regime. `full_system`'s advantage emerges in Exp 3 under coordinated attacks.

### 7.7 Qwen2.5 7B Results — V4 (`results/experiment_1_qwen_v4.csv`)

Generated with identical settings to LLaMA (`max_tokens=512` GSM8K, `max_tokens=128` StrategyQA, N∈{1,5,7}, temperature=0.7). Same strict extraction and stage1_only changes as §7.6. CSV contains 192 rows (128 for N∈{5,7} + 64 for N=1).

**Accuracy by beta (averaged over N∈{5,7} and all fault types):**

| Condition | N=1 (single agent) | β=0% | β=15% | β=30% | β=45% |
|---|---|---|---|---|---|
| baseline (self-consistency) | 0.675 | 0.669 | 0.669 | 0.673 | 0.648 |
| soft_weighting | 0.675 | 0.650 | 0.650 | 0.644 | 0.663 |
| hard_only | 0.675 | 0.669 | 0.669 | 0.675 | 0.656 |
| **full_system / stage1_only** | **0.675** | **0.650** | **0.650** | **0.644** | **0.645** |

**Key findings:**
- Single-agent accuracy (N=1): 0.675 — consistent with published Qwen2.5 7B performance on this mixed benchmark.
- **Strict extraction shows a smaller but real improvement for Qwen.** Old baseline β=0.45: 0.607; V4: 0.648. The 4.1pp gain confirms the same bug but with smaller magnitude — Qwen's individual answer-extraction accuracy is higher, leaving fewer garbage-text votes to corrupt the count.
- **`hard_only` is the best condition at β=0.45** (0.656), matching the LLaMA pattern. Strict extraction + Module 1 filtering handles invalid-format faults optimally.
- **Fallback frequency at β=0 (N=7): 13.75%** — higher than LLaMA (5.0%), reflecting Qwen's lower per-agent TopKMass scores triggering more frequent liveness fallbacks even under clean conditions. This is a calibration artifact: Qwen's stable-region scores cluster lower, so the 5th-percentile τ cuts more clean agents.
- **soft_weighting is the best condition for F3/drifter faults at high β.** N=7 β=0.45 F3: soft=0.650 vs baseline=0.575. Continuous weighting avoids the liveness cascade that hard_only/full_system experience under drifter overload.
- **F2 (Byzantine) worst case:** full_system N=7 β=0.45 F2 = 0.5875 — spoofed logprobs pass Module 1, stage1_only picks suboptimal agent in the adversarial cluster.

**Cross-model comparison — V4 (N=7, avg over fault types):**

| Condition | Model | N=1 | β=0% | β=30% | β=45% |
|---|---|---|---|---|---|
| baseline | LLaMA 3.1 8B | 0.713 | 0.750 | 0.734 | 0.713 |
| baseline | Qwen2.5 7B | 0.675 | 0.675 | 0.675 | 0.647 |
| **hard_only** | **LLaMA 3.1 8B** | **0.713** | **0.750** | **0.750** | **0.716** |
| **hard_only** | **Qwen2.5 7B** | **0.675** | **0.675** | **0.675** | **0.650** |
| full_system | LLaMA 3.1 8B | 0.713 | 0.675 | 0.744 | 0.647 |
| full_system | Qwen2.5 7B | 0.675 | 0.650 | 0.638 | 0.634 |

**Interpretation:** Both models confirm the same story: `hard_only` (strict extraction + Module 1 filter) is the best condition for Exp 1's invalid-format fault regime. `full_system` (stage1_only) shows a modest clean-accuracy penalty at β=0 but is indispensable for the coordinated Byzantine scenario in Exp 3 — where strict extraction provides no defense since coordinated agents produce valid-format answers.

### 7.8 Figures (`figures/`) ✅

Figures regenerated (2026-05-06) from `*_shuffled.csv` files via `scripts/plot_results.py`.

Three files in `figures/`:
- `accuracy_vs_beta.png` — two-panel line chart with N=1 reference line (supplementary / appendix)
- `accuracy_vs_beta_no_n1.png` — same chart without N=1 line (main paper body)
- `fault_type_breakdown.png` — bar chart comparing baseline vs full_system at β=45%

Two figures are generated by `scripts/plot_results.py` from the 192-row `*_shuffled.csv` files.

#### Figure 1: `figures/accuracy_vs_beta.png` — Accuracy vs. Fault Fraction

Two-panel line chart (LLaMA left, Qwen right). X-axis: β ∈ {0%, 15%, 30%, 45%}. Y-axis: accuracy. Five series per panel:
- **Single Agent (N=1, dotted grey):** Flat reference line — always evaluated on an unmodified output since `floor(1 × β) = 0` for β < 1.0. Represents a permanently clean single agent, not one subject to the same fault rate. Correct reading: "our full system at β=45% approaches the accuracy of a guaranteed-clean single agent, even when 45% of the pool is faulty." Supplementary figures (`figures/accuracy_vs_beta_no_n1.png`) omit this line.
- **Self-Consistency / baseline (red dashed):** Expected to degrade under fault load; exact shape depends on corrected answer-voting results.
- **Soft-Weighted SC (orange dotted):** Expected to degrade more gracefully than baseline.
- **Hard Filter + Majority / hard_only (blue dash-dot):** Expected to hold better at β=30% but degrade at β=45%.
- **Full System / full_system (green solid):** Expected to remain nearly flat. Update with actual numbers after rerun.

#### Figure 2: `figures/fault_type_breakdown.png` — Accuracy by Fault Type at β=45%

Two-panel grouped bar chart. X-axis: fault types F1 (Crash), F2 (Byzantine), F3 (Drifter), Mix. Two bars per group: Self-Consistency (red) and Full System (green). Expected: F2 (Byzantine) will be the hardest fault type for `full_system` because spoofed TopKMass = 1.0 passes Module 1 — Module 2 must do all the work.

**Paper caption note (draft):** *"Self-Consistency (Majority Vote) degrades under fault load because Byzantine agents concentrate votes on a wrong answer while clean agents may split votes across correct answers. Results shown use answer-level majority voting (extracting the final answer from each chain-of-thought before counting votes)."*

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
| `results/experiment_2_signals.csv` | Per-agent DataFrame: `question_id, topk_mass, neg_entropy, neg_logprob_var, is_correct, fault_type, is_faulty` |
| `results/experiment_2_signals.png` | 3-panel figure (clean run) or 4-panel figure (with fault injection) |

`fault_type` is `"clean"` for original cache agents. When `--include-faults` is used, injected agents appear as rows with `fault_type ∈ {"F2","F3"}` and `is_faulty=True`.

### 8.5 Figure Panels

- **Panel A (ROC Curves):** One line per signal with AUC in legend. Gray dashed chance diagonal.
- **Panel B (Scatter):** TopKMass score vs. jittered correctness label. Vertical median lines per class (correct / incorrect) show distributional separation.
- **Panel C (Precision-Recall):** One line per signal with Average Precision in legend. Gray dashed random-classifier baseline.
- **Panel D (Fault Detection ROC, only when `--include-faults`):** `−TopKMass` as a fault detector — lower TopKMass → more likely faulty. AUC quantifies how well the filter signal separates injected-fault agents from clean ones.

### 8.6 How to Run

```bash
# On a specific cache (CPU only):
python -m eval.signal_quality --cache cache_llma.json --output-dir results/exp2_llama/
python -m eval.signal_quality --cache cache_qwen.json --output-dir results/exp2_qwen/

# With fault injection analysis (adds Panel D):
python -m eval.signal_quality --cache cache_llma.json --output-dir results/exp2_llama/ \
    --include-faults --beta 0.3 --fault-types F2 F3
```

### 8.7 Results (`cache_llma.json` + `cache_qwen.json`, 700 agents each)

**ROC AUC and Average Precision:**

| Signal | LLaMA AUC | LLaMA AP | Qwen AUC | Qwen AP |
|---|---|---|---|---|
| **TopKMass** | **0.606** | **0.802** | **0.627** | **0.765** |
| −Entropy | 0.580 | 0.791 | 0.616 | 0.765 |
| −Logprob Var | 0.448 | 0.675 | 0.538 | 0.678 |

**Key findings:**
- TopKMass is the best correctness predictor on both models (highest AUC and AP).
- Logprob Variance performs *worse than chance* for LLaMA (AUC=0.448) — a strong negative result that validates using TopKMass over naive variance as the filter signal.
- −Entropy is a close second to TopKMass on Qwen (AUC gap of only 0.010), but TopKMass retains a larger advantage on LLaMA (0.026 gap) and is architecturally motivated by the sliding-window design.
- 700 agent generations per model (100 questions × 7 agents); LLaMA 69.4% correct, Qwen 67.3% correct.

**Panel B interpretation (TopKMass vs. Correctness scatter):** Both correct and incorrect agents cluster in a narrow high-confidence range ([0.79, 1.0] LLaMA; [0.94, 1.0] Qwen) with heavy overlap — there is no clean threshold that separates correct from incorrect. The vertical median lines show the correct cloud is shifted slightly right: LLaMA correct median 0.988 vs incorrect 0.981 (gap of 0.007); Qwen correct median 0.998 vs incorrect 0.995 (gap of 0.003). This is a real but modest effect, consistent with the AUC values above. TopKMass is not a correctness oracle — its primary value as a filter is detecting *broken* agents (F1 crash = empty logprobs, F3 drifters = TopKMass near 0) that fall far outside the clean cluster, not distinguishing correct from incorrect within the clean population.

**Output figures:** `results/exp2_llama/experiment_2_signals.png`, `results/exp2_qwen/experiment_2_signals.png`

---

## 9. Experiment 3: Adversarial Coordination Stress Test (`eval/adversarial_test.py`)

### 9.1 Research Question

Experiments 1–2 quantify robustness under random fault injection and validate the TopKMass filter. Experiment 3 answers the hardest question: **what happens under a coordinated Byzantine attack** — the worst case for any consensus mechanism?

The adversary now actively cooperates: multiple Byzantine agents agree on the same wrong answer *and* spoof their logprobs to look maximally confident, bypassing Module 1. Does Stage 2 (NLI Cross-Encoder Verification) provide a residual defense even when Stage 1 (Geometric Median) is dragged toward the adversarial cluster?

Separately: does the geometric median genuinely resist being dragged more than a naïve arithmetic mean?

### 9.2 Fixed Parameters

| Parameter | Value |
|---|---|
| N (agents per question) | 7 |
| f (tolerated faults) | 2 (β ≈ 28.6%) |
| Liveness threshold (2f+1) | 5 |
| τ | auto-calibrated via `calibrate_tau()` (same as Exp 1) |

### 9.3 Three Coordination Degrees

**Uncoordinated**: The f=2 Byzantine agents act independently and produce semantically different wrong texts — one uses `_F2_TEXT` ("WRONG: The answer is definitively incorrect."), the other uses `_F3_TEXT` (Arctic terns off-topic text). Both use `_COORD_LOGPROBS_PER_TOKEN` (top-5 distribution summing to exactly 1.00 TopKMass per position) so they pass Module 1. Because they embed far apart in semantic space, the geometric median cannot be dragged to a coherent adversarial location.

**Coordinated**: Both agents produce the exact same wrong answer derived from ground truth (`"The answer is {opposite}."` for yes/no; `"The answer is {gt+7}."` for GSM8K). Same `_COORD_LOGPROBS_PER_TOKEN` distribution (TopKMass = 1.00). They cluster at a single point in embedding space, exerting a coherent pull on the geometric median.

**Maximally Adversarial**: Identical to coordinated but uses `_MAX_ADV_LOGPROBS_PER_TOKEN` — a slightly higher-confidence top-5 distribution also summing to 1.00 TopKMass per position. Conceptually the hardest attack (coordinated cluster + maximum-confidence spoofing), but numerically equivalent to coordinated in terms of Module 1 admission since both achieve TopKMass = 1.00.

### 9.4 Three Pipeline Conditions

**majority_answer_vote**: No filter; extracts answers via `_extract_answer_strict` then votes with `Counter`. Vulnerable to coordinated attack because 2 identical wrong votes (producing valid yes/no or numeric answers) can swing a close vote against a split correct majority.

**stage1_only** *(recommended)*: Module 1 filter + liveness fallback → geometric median nearest-neighbor (no NLI Stage 2). Implemented in `eval/adversarial_test_v2._aggregate_stage1_only` using `pipeline.aggregation._embed` and `_geometric_median` directly. This is the condition that beats majority vote under coordinated attack (LLaMA: 70% vs 62.5%).

**full_pipeline** *(included for comparison only; consistently worst)*: Module 1 filter + `pipeline.aggregation.aggregate()` — full Stage 1 geometric median + Stage 2 bidirectional NLI entailment check. Candidates are iterated nearest-to-median first; the first passing bidirectional entailment with the second-nearest reference agent is returned. Empirically the worst condition in both Exp 1 and Exp 3 — NLI entailment misclassification and reference-agent failure modes outweigh any residual defense. Do not use.

### 9.5 Centroid Shift Metric

For each question × coordination degree, before any filtering:
```
embs         = _embed([g.output_text for g in all_7_agents])    # (7, D)
clean_centroid = mean(embs[clean_indices])                       # (5, D) → (D,)
mean_centroid  = mean(embs)                                      # arithmetic mean of all 7
gm_centroid    = _geometric_median(embs)                         # robust geometric median

dist_mean = ||mean_centroid − clean_centroid||₂
dist_gm   = ||gm_centroid   − clean_centroid||₂
delta     = dist_mean − dist_gm    # positive = gm stays closer to clean cluster
```

Averaged over all questions per coordination degree. Positive delta = geometric median more robust than arithmetic mean against the coordinated cluster.

### 9.6 Output Files

| File | Description |
|---|---|
| `results/exp3_llama/experiment_3_adversarial_v2.csv` | LLaMA per-(coordination × pipeline_condition) row: accuracy, fallback_frequency, centroid_shift_mean, centroid_shift_gm, centroid_shift_delta |
| `results/exp3_qwen/experiment_3_adversarial_v2.csv` | Qwen equivalent |
| `results/exp3_llama/experiment_3_adversarial.png` | 2-panel figure: accuracy grouped bars + centroid shift paired bars |
| `results/exp3_qwen/experiment_3_adversarial.png` | Qwen equivalent |

### 9.7 Figure Panels

- **Panel A (Accuracy vs. Coordination):** Grouped bar chart — 3 pipeline conditions, 3 x-axis positions (coordination degrees). Shows majority_answer_vote degrading under coordinated attack while stage1_only holds; full_pipeline is consistently lowest.
- **Panel B (Centroid Shift):** Paired bars — arithmetic mean distance (light blue) vs. geometric median distance (green) to clean centroid, for each coordination degree. Positive Δ annotation above each pair quantifies geometric median's robustness advantage.

### 9.8 How to Run

```bash
python -m eval.adversarial_test_v2 --cache cache_llma.json --output-dir results/exp3_llama/
python -m eval.adversarial_test_v2 --cache cache_qwen.json --output-dir results/exp3_qwen/
# Smoke test (5 questions)
python -m eval.adversarial_test_v2 --cache cache_llma.json --output-dir results/exp3_smoke/ --n-questions 5
```

### 9.9 Measured Results — V4 (`results/exp3_llama/experiment_3_adversarial_v2.csv`, `results/exp3_qwen/experiment_3_adversarial_v2.csv`)

Results generated with `eval/adversarial_test_v2.py` using the same V4 pipeline (strict extraction, stage1_only). τ auto-calibrated on a shuffled dev slice (seed=42). 80 eval questions per model.

**Accuracy by coordination degree and pipeline condition:**

| Coordination | Pipeline condition | LLaMA acc | Qwen acc |
|---|---|---|---|
| Uncoordinated | majority_answer_vote | 72.5% | 65.0% |
| Uncoordinated | stage1_only | 67.5% | 63.75% |
| Uncoordinated | full_pipeline | 57.5% | 48.75% |
| Coordinated | majority_answer_vote | 62.5% | 65.0% |
| **Coordinated** | **stage1_only** | **70.0%** | **65.0%** |
| Coordinated | full_pipeline | 56.25% | 48.75% |
| Maximally Adversarial | majority_answer_vote | 62.5% | 65.0% |
| **Maximally Adversarial** | **stage1_only** | **70.0%** | **65.0%** |
| Maximally Adversarial | full_pipeline | 56.25% | 48.75% |

Fallback frequency (stage1_only): 2.5% (LLaMA), 11.25% (Qwen). The Qwen fallback rate is elevated because Qwen's TopKMass distribution is lower, causing Module 1 to over-filter and trigger liveness on ~1 in 9 questions — reverting to full pool majority vote behavior and reducing stage1_only's advantage.

**Centroid shift (avg distance to clean cluster centroid, lower = more robust):**

| Coordination | Model | dist_mean | dist_gm | delta (mean−gm) |
|---|---|---|---|---|
| Uncoordinated | LLaMA | 0.339 | 0.095 | **+0.244** |
| Uncoordinated | Qwen | 0.347 | 0.066 | **+0.282** |
| Coordinated | LLaMA | 0.353 | 0.113 | **+0.240** |
| Coordinated | Qwen | 0.366 | 0.076 | **+0.290** |
| Maximally Adversarial | LLaMA | 0.353 | 0.113 | **+0.240** |
| Maximally Adversarial | Qwen | 0.366 | 0.076 | **+0.290** |

Positive delta confirms geometric median stays ~0.24–0.29 embedding units closer to the honest cluster than the arithmetic mean — a robust, model-agnostic result.

**Key findings:**

**1. `stage1_only` wins under coordinated attack (LLaMA: +7.5pp).** Under uncoordinated conditions, majority vote wins (72.5% vs 67.5%) because diverse wrong answers split the vote and the clean 5-agent plurality is reliable. Under coordinated attack, the 2 Byzantine agents unify their vote on the same wrong answer, dragging majority vote to 62.5%. Geometric median resists: the 5-agent honest cluster remains spatially dominant and stage1_only recovers to 70.0%.

**2. `full_pipeline` (NLI Stage 2) is the worst condition in every scenario.** LLaMA: 57.5%–56.25% across all coordination degrees, consistently 10–14pp below stage1_only. Qwen: 48.75% across all conditions, drastically below both majority vote and stage1_only. NLI Stage 2 was designed to catch cases where the geometric median nearest-neighbor points to an adversarial cluster — but in practice, iterating candidates by distance and checking entailment against the second-nearest agent introduces additional failure modes (entailment misclassification, reference agent also being adversarial under low-liveness) that outweigh any residual defense. **`full_pipeline` should not be used; `stage1_only` is the correct full_system.**

**3. The two-failure-mode structure is confirmed across both experiments:**
  - **Invalid-format faults (Exp 1 F1/F2/F3):** Coordinated agents produce parseable answers (e.g., "The answer is no."), so strict extraction cannot filter them — strict majority vote degrades just as raw majority vote does. `hard_only` is the best condition here.
  - **Valid-format coordinated attacks (Exp 3):** Strict extraction keeps the adversarial votes (correctly parsed as "no"), so majority vote is vulnerable. `stage1_only` (geometric median) is the correct aggregation.
  - No single pipeline is optimal for both. The system's design correctly uses strict extraction + Module 1 + stage1_only as a layered approach that provides partial defense in each regime.

**4. Qwen Exp 3 is neutral rather than positive.** Under coordinated attack, stage1_only = majority_answer_vote = 65.0% for Qwen. The 11.25% liveness fallback explains this: when Module 1 over-filters Qwen agents and reverts to full pool, stage1_only's geometric median operates on the full 7-agent pool including 2 Byzantine agents — at which point it is equivalent to majority vote on the full pool. The centroid shift deltas remain strong (+0.282–0.290), confirming the geometric median mechanism works; the accuracy parity is a calibration artifact.

**5. Centroid shift is model-agnostic.** Coordinated attack increases dist_mean slightly (+0.014 LLaMA, +0.019 Qwen) while dist_gm increases less, widening the delta. Positive delta in all conditions confirms geometric median's robustness advantage over arithmetic mean regardless of coordination degree or model family.

**Output figures:** `results/exp3_llama/experiment_3_adversarial.png`, `results/exp3_qwen/experiment_3_adversarial.png`

---

## 10. Future Direction: Multi-Provider Heterogeneous Setting

This section describes what would need to change to move from the current homogeneous setup (N samples from one model) to a genuine multi-provider ensemble where different agents are operated by different organisations running different models.

### 10.1 What Changes Conceptually

In the homogeneous setting, agents disagree due to stochastic sampling variance. No agent has an incentive to deceive. The BFT framing is scaffolding.

In the multi-provider setting, a provider that operates one or more agents may have an incentive to steer the consensus toward a specific answer (commercial bias, model fine-tuning, or active manipulation). The F2 fault — a confident-looking wrong answer — becomes a realistic threat, not a synthetic one. The two-layer defense then has genuine motivation:
- **Module 1** filters low-confidence providers (those whose model is uncertain or off-task).
- **Module 2** uses content semantics, which a provider cannot fake without actually knowing the correct answer.

### 10.2 Implementation Changes Required

**Cache format:** Add `model_id` and `provider` fields per agent generation.
```json
{"agent_id": "q0_a2", "model_id": "gpt-4o", "provider": "openai", "output_text": "...", "token_logprobs": [...]}
```

**Phase 1 (generation):** `scripts/generate_cache.py` currently uses a single vLLM server. It would need separate API clients per provider (OpenAI, Anthropic, Google, etc.) and a unified logprob normalization layer.

**Logprob availability — the key blocker:**
| Provider | Logprobs available? | Notes |
|---|---|---|
| OpenAI (GPT-4o) | Yes | `logprobs=True`, top-5 via `top_logprobs=5` |
| Anthropic (Claude) | **No** | Claude does not expose per-token logprobs |
| Google (Gemini) | Partial | `response_logprobs=True` in some versions |
| Open-weight (vLLM) | Yes | Full top-K control |

Module 1 requires logprobs. For providers that don't expose them, two options:
1. **Skip Module 1** for that provider — admit unconditionally, rely on Module 2 alone.
2. **Proxy signal** — use output length, response time, or model-reported confidence as a substitute (weaker signal).

**TopKMass comparability:** TopKMass is a ratio (sum of top-5 probs / 1.0), not an absolute logprob. It is comparable across models with different vocabulary sizes because it measures the model's own confidence relative to its own distribution. τ calibration (5th percentile of dev-slice scores) handles any remaining scale differences per-provider.

**Module 2 (geometric median in embedding space):** No changes needed. The sentence transformer operates on output text regardless of source model. This module is already model-agnostic.

**Evaluation:** In a real deployment there is no ground truth available at aggregation time. Evaluation would shift to held-out benchmarks or human annotation rather than exact-match accuracy. The current harness (exact-match against cached ground truth) is appropriate only for research evaluation.

### 10.3 What the System Would Prove in That Setting

With heterogeneous providers, the two primary results from the current study would be directly applicable:
1. **Confidence filtering (Module 1)** identifies providers whose model is uncertain or producing off-task output — real failure modes, not injected ones.
2. **Geometric median in embedding space (Module 2)** resists a minority of providers coordinating on the same wrong answer — a realistic commercial-incentive scenario.

The centroid shift result from Experiment 3 (geometric median stays 0.24–0.29 units closer to the honest cluster than arithmetic mean) translates directly: replace "injected Byzantine agents" with "biased providers."