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
  - **Centroid Proxy Logic:** The design doc specifies "bidirectional entailment between the nearest candidate and the centroid itself." Because the geometric median is a vector in embedding space, not text, it cannot be fed directly to an NLI model. The implementation proxies "the centroid itself" with the **second-nearest** candidate text — the closest admitted agent output that is distinct from the primary answer. This gives a non-trivial, semantically meaningful verification pair. Edge case: when `N == 1`, Stage 2 is skipped entirely (single-agent shortcut at the top of `aggregate`).
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

## 5. Evaluation Harness (`eval/runner.py`)
- **Role:** Iterate over the cached, fault-injected data and generate metric reports.
- **Ablation Conditions:**
  - Baseline: No filter, majority voting aggregation.
  - Soft-weighting: Score-weighted geometric median.
  - Hard-filtering: Module 1 admission -> majority voting.
  - Full System: Module 1 admission -> Module 2 aggregation.
- **External Baseline (`eval/decent_baseline.py`):** Re-implementation of DecentLLMs worker/evaluator scoring (isolate this from the main pipeline to prevent hallucinated dependencies).
- **Output:** Pandas DataFrame exported to CSV tracking Accuracy, Admission Rate, and Fallback Frequency.