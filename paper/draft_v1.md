# Failure-Mode-Specific Consensus for Multi-Agent LLM Ensembles

---

## Abstract

Multi-agent LLM ensembles often aggregate independent completions using self-consistency, a majority vote over extracted answers. This paper studies two failure modes of that procedure and argues that they require different defenses rather than a single universally optimal consensus rule. Invalid-format outputs, such as empty or off-task generations, can contaminate voting when answer extraction falls back to raw text. Coordinated valid-format wrong answers create a separate problem: a small group of syntactically valid wrong answers can form a concentrated semantic cluster that majority vote and confidence filtering do not remove. We evaluate three mechanisms: abstention when no parseable final answer is found (strict extraction), rather than treating malformed output as a vote; a sliding-window token-probability filter (TopKMass); and an embedding-space geometric median selector.

Across experiments on LLaMA 3.1 8B and Qwen2.5 7B using 100 questions from GSM8K and StrategyQA, strict answer extraction corrects a 16.5 percentage point accuracy collapse in LLaMA majority-vote results at high fault fractions, and TopKMass filtering plus strict majority vote has the highest observed average accuracy under invalid-format faults. TopKMass is the strongest of three tested logprob-based correctness signals (LLaMA AUC/AP 0.612/0.810; Qwen 0.635/0.776), although its absolute discriminative power is modest. Under coordinated synthetic wrong-answer injection, geometric median nearest-centroid improves LLaMA accuracy over strict majority vote by 7.5 percentage points (70.0% vs. 62.5%), and a separate geometric analysis confirms that the geometric median remains closer to the honest agents' outputs than the arithmetic mean across all tested attack conditions. Multi-provider experiments delimit the scope of this defense: geometric median helps when wrong answers form a tight coordinated cluster, but not under natural provider diversity or scattered provider errors.

---

## 1. Introduction

Multi-agent LLM ensembles improve output reliability by sampling multiple independent completions and aggregating their answers. The most common aggregation rule is self-consistency (Wang et al., 2023): extract a final answer from each completion and return the plurality answer. This rule is simple and effective when errors are independent and answer extraction is reliable, but it provides little guidance about how to handle malformed outputs, confidence signals, or coordinated wrong answers.

We study these issues through a failure-mode-specific lens. The core observation is that self-consistency can fail for qualitatively different reasons, and defenses that help in one regime can be neutral or harmful in another.

**Failure Mode 1: invalid-format outputs.** Some completions contain no parseable final answer. If extraction falls back to treating the entire output as a vote key, these malformed strings can contaminate the vote pool. Strict answer extraction addresses this by abstaining when no valid answer is found. TopKMass filtering provides a complementary mechanism by identifying crash-like or off-task outputs whose token-probability traces fall far outside the clean-agent distribution.

**Failure Mode 2: coordinated valid-format wrong answers.** A separate problem arises when a minority of agents produce extractable answers that agree on the same wrong value. Strict extraction keeps these answers, and confidence filtering does not remove them when their confidence traces are spoofed or otherwise high. In this regime, the relevant structure is semantic: if honest outputs form the dominant cluster in embedding space, geometric median nearest-centroid aggregation can resist a smaller wrong-answer cluster.

The table below summarizes the taxonomy used throughout the paper.

| Failure mode | Mechanism that helps | Main empirical role |
|---|---|---|
| Invalid or unparseable outputs | Strict answer extraction | Prevents raw malformed text from entering the vote |
| Low-confidence crash or drift outputs | TopKMass filtering | Removes agents outside the clean confidence distribution |
| Coordinated valid wrong answers | Geometric median nearest-centroid | Selects from the semantic center of the admitted pool |
| Scattered provider errors | Majority vote | Remains competitive when wrong answers do not form a cluster |

This paper makes four contributions:

1. A failure-mode taxonomy for multi-agent LLM consensus, separating invalid-format failures from coordinated valid-format wrong answers.
2. An evaluation of TopKMass filtering with the same stable-region statistic used for both calibration and signal-quality analysis.
3. An embedding-space geometric median selector and centroid-shift analysis for coordinated synthetic wrong-answer injection.
4. A set of ablations across two open-weight models, two datasets, multiple fault types, and multi-provider settings that identifies where each mechanism helps and where it does not.

---

## 2. Related Work

**Self-consistency** (Wang et al., 2023) is the standard approach to multi-sample LLM aggregation: sample N chain-of-thought completions, extract the final answer token from each, and return the plurality answer. Our work takes self-consistency as the primary baseline, identifies its failure modes under fault injection and coordinated attack, and proposes complementary mechanisms to address each failure mode. The strict extraction fix (abstaining when no parseable answer is found) is itself a methodology improvement applicable to any self-consistency implementation.

**DecentLLMs** (Jo and Park, 2024) is the closest prior system-level approach. DecentLLMs uses a worker/evaluator structure where each candidate output is scored by multiple evaluators across several criteria, and the geometric median is applied over the resulting score matrix to select a winner. The key difference is the aggregation space: DecentLLMs applies geometric median in criterion score space (evaluator ratings), while our system applies geometric median in sentence embedding space (output text semantics). The two approaches address different failure modes; DecentLLMs does not expose per-token logprobs and therefore cannot perform confidence-based pre-filtering. For the paper we cite DecentLLMs' reported numbers directly, noting that they use GPT-3.5-Turbo while our experiments use open-weight models.

**Byzantine Fault Tolerance** (Lamport et al., 1982; Castro and Liskov, 1999) provides the analytical scaffolding for our liveness threshold and fault model. Classical BFT protocols achieve Byzantine-fault tolerance with N >= 3f+1 nodes through multi-round message passing with O(N^2) message complexity. Our system is not a BFT protocol: it is a one-shot aggregation pipeline with a confidence filter, and the setting is a homogeneous single-model pool where agents have no incentive to collude. The BFT framing motivates the liveness threshold (2f+1 admitted agents required, f = floor((N-1)/3)) and the fault fraction regime (beta=0.45 with N=7 corresponds to f=3, which is super-threshold and causes consistent liveness fallback). We do not claim adversarial robustness against an external attacker in the homogeneous setting.

**Geometric median robustness** (Weiszfeld, 1937; Minsker, 2015) provides the theoretical foundation for the embedding-space selector. The geometric median minimizes sum of L2 distances, whereas the arithmetic mean minimizes sum of squared distances. In Euclidean space, the geometric median has a breakdown point of 1/2, so a minority of outlying points cannot move it arbitrarily far. Our centroid-shift analysis is consistent with this mechanism, but we treat it as mechanistic evidence rather than a sufficient condition for accuracy improvement.

**Minimum Bayes Risk (MBR) decoding** (Eikema and Aziz, 2020) selects the hypothesis that minimizes expected loss across a candidate pool using pairwise similarity. Our geometric median nearest-centroid is conceptually related but operates in embedding space and has O(N) complexity after embedding, compared to O(N^2) pairwise evaluation for MBR.

**Sentence embedding aggregation** (Li et al., 2023) applies embedding clustering for reasoning answer selection. Our approach uses geometric median rather than k-means or nearest-neighbor clustering, providing a different robustness objective for clustered outliers.

**Calibration and per-token confidence** (Guo et al., 2017) is the background for Module 1's signal design. TopKMass is a sliding-window stability measure (the mean sum of top-5 token probabilities over a W=64 window) rather than an instantaneous per-token confidence. This design captures whether a model maintains high-confidence generation across a sequence, which is more predictive of output reliability than any single token's confidence.

---

## 3. System Description

### 3.1 Architecture

The pipeline separates GPU-intensive generation from CPU evaluation. Phase 1 (offline generation) uses `vLLM` with temperature=0.7 and logprobs=5 to produce N=7 completions per question, caching all outputs and per-token top-5 logprobs to a JSON file. Phase 2 (evaluation) runs entirely on cached data with no GPU or `vLLM` dependency.

All data passed between modules conforms to two typed dataclasses:

```python
@dataclass
class AgentGeneration:
    agent_id: str
    output_text: str
    token_logprobs: List[float]   # flat list, length = 5 * T
    is_faulty: bool
    fault_type: Optional[str]     # None, 'F1_crash', 'F2_byzantine', 'F3_drifter'
    model_id: Optional[str] = None
    provider: Optional[str] = None

@dataclass
class ConsensusResult:
    final_answer: str
    admitted_agents: List[str]
    is_low_confidence: bool
```

### 3.2 Module 1: TopKMass Confidence Filter

**Signal computation.** For each agent with T output tokens, the token logprob list has length 5T (top-5 logprobs per position). The TopKMass trajectory is computed using a causal sliding window of width W=64:

```
TopKMass(t) = (1 / min(t+1, W)) * sum_{i=max(0, t-W+1)}^{t} sum_{k=1}^{5} exp(logprob_{i,k})
```

This is implemented in O(T) via prefix sums (`numpy.cumsum`), avoiding an O(T x W) inner loop.

**Warmup normalization.** For outputs shorter than 2W tokens (common in StrategyQA with max_tokens=128), a large fraction of the trajectory is in the warmup region where the window is partially filled and scores are systematically lower than the stable-state value. Without correction, a single global threshold tau would penalize short outputs relative to long ones. The stable-region mean is computed on the trajectory slice at positions [W:], falling back to the full trajectory when output length is at most W. This is applied identically in both filtering and tau calibration, ensuring the same metric is used throughout.

**Threshold calibration.** Tau is calibrated at the 5th percentile of stable-region TopKMass scores on a 20% shuffled dev slice. Shuffling is critical: the cache orders GSM8K questions first (indices 0-49), and without shuffling the dev slice draws entirely from higher-confidence GSM8K questions, biasing tau upward and causing approximately 50% liveness fallback at beta=0. With shuffling (fixed seed=42), the 5th-percentile tau is calibrated on a representative mix and produces approximately 5% clean-agent false rejection.

**Liveness fallback.** The tolerated fault count is f = floor((N-1)/3) (N=7 gives f=2; N=5 gives f=1). If fewer than 2f+1 agents pass filtering, all available agents are admitted and `is_low_confidence=True` is set. This ensures the pipeline always returns an answer, at the cost of reduced confidence.

The table below summarizes the expected behavior by injected fault type.

| Fault | TopKMass Score | Module 1 Outcome |
|---|---|---|
| F1 (crash) | 0.0 (empty logprobs) | Dropped unconditionally |
| F3 (drifter) | ~2.3 x 10^-4 (logprob spoof: -10.0 per entry) | Dropped (below any tau > 0.001) |
| F2 (Byzantine) | 1.00 (logprob spoof: sum = 1.00 exactly) | **Passes** at any tau <= 1.0 |
| Clean agent | [0.96, 1.00] (typical range) | Passes unless rare outlier |

F2 Byzantine logprobs are spoofed to TopKMass=1.00 (not 0.9999) because high-precision models (notably Qwen2.5) produce clean-agent scores up to ~0.9951; the exact value 1.00 is required to avoid inadvertent filtering of Byzantine agents.

### 3.3 Module 2: Geometric Median Nearest-Centroid

Admitted agents' output texts are embedded using `sentence-transformers/all-mpnet-base-v2` (768-dimensional). The geometric median of the embedding matrix is computed by minimizing:

```
f(y) = sum_{i=1}^{N} ||x_i - y||_2
```

via L-BFGS-B with analytic gradient `grad f(y) = -sum (x_i - y) / ||x_i - y||_2` and denominator clamping at 1e-10. The final answer is the `output_text` of the agent with minimum Euclidean distance to this geometric median (nearest-centroid selection). The reported homogeneous experiments use this nearest-centroid selector for the Experiment 1 `full_system` condition and the Experiment 3 `stage1_only` condition. Implementation provenance is provided in Appendix A.4.

Earlier aggregation variants included bidirectional NLI verification and distance-weighted answer voting. These are retained only as ablations and are not the recommended geometric-median condition.

### 3.4 Ablation Conditions

| Condition | Module 1 | Aggregation |
|---|---|---|
| `baseline` | None | Strict majority vote with abstention on unparseable outputs |
| `soft_weighting` | None (scores as weights) | TopKMass-weighted geometric median |
| `hard_only` | TopKMass filter + liveness | Strict majority vote |
| `full_system` / `stage1_only` | TopKMass filter + liveness | Geometric median nearest-centroid |

### 3.5 Fault Injection

The fault injector (`faults/injector.py`) applies deterministic mutations to cached data. Exactly floor(N x beta) agents are mutated per question, selected via `random.Random(seed).sample`. F1 agents receive empty `output_text` and `token_logprobs`. F2 agents receive a pre-specified wrong answer string and logprobs spoofed to TopKMass=1.00. F3 agents receive syntactically plausible off-task text and logprobs spoofed to -10.0 per entry. The `mix` condition assigns each mutated agent an independently drawn fault type. Original AgentGeneration objects are never mutated; clean agents are returned by reference.

---

## 4. Experiment 1: Fault Injection Ablation

### 4.1 Setup

We evaluate four pipeline conditions across N in {5, 7} agents, beta in {0, 0.15, 0.30, 0.45} fault fraction, and fault types {F1, F2, F3, mix} on 100 questions (50 GSM8K multi-step arithmetic, 50 StrategyQA commonsense yes/no). LLaMA 3.1 8B Instruct and Qwen2.5 7B Instruct each receive their own cached generation sets (N=7 agents, temperature=0.7, max_tokens=512 for GSM8K and 128 for StrategyQA). Tau is auto-calibrated per run.

Strict answer extraction is applied to `baseline` and `hard_only`: agents whose output contains no parseable answer (no `yes`/`no` match for StrategyQA; no numeric token for GSM8K) return `None` and are excluded from the vote. This is the V4 methodology; earlier runs without strict extraction showed a 16.5pp LLaMA collapse at beta=0.45.

### 4.2 Results

**Table 1.** LLaMA 3.1 8B accuracy by condition and fault fraction (averaged over N in {5,7} and all fault types).

| Condition | N=1 (ref) | beta=0% | beta=15% | beta=30% | beta=45% |
|---|---|---|---|---|---|
| baseline (self-consistency) | 0.713 | 0.756 | 0.769 | 0.736 | 0.708 |
| soft_weighting | 0.713 | 0.713 | 0.725 | 0.728 | 0.688 |
| hard_only | 0.713 | 0.750 | 0.763 | 0.745 | **0.727** |
| full_system (stage1_only) | 0.713 | 0.700 | 0.723 | 0.733 | 0.653 |

**Table 2.** Qwen2.5 7B accuracy by condition and fault fraction (averaged over N in {5,7} and all fault types).

| Condition | N=1 (ref) | beta=0% | beta=15% | beta=30% | beta=45% |
|---|---|---|---|---|---|
| baseline (self-consistency) | 0.675 | 0.669 | 0.669 | 0.673 | 0.648 |
| soft_weighting | 0.675 | 0.650 | 0.650 | 0.644 | 0.663 |
| hard_only | 0.675 | 0.669 | 0.669 | 0.675 | **0.656** |
| full_system (stage1_only) | 0.675 | 0.650 | 0.650 | 0.644 | 0.645 |

The N=1 single-agent reference is immune to fault injection (floor(1 x beta) = 0 for any beta < 1.0) and serves as the no-consensus anchor. LLaMA N=1 accuracy (0.713) is consistent with published greedy-decoding results (~73%, Meta AI 2024); the mixed-benchmark setting (50% StrategyQA) accounts for the slight downward shift from GSM8K-only benchmarks.

### 4.3 Key Findings

**Finding 1: Strict extraction is the primary Exp 1 result.** Without strict extraction, the LLaMA baseline at beta=0.45 was 0.543; with strict extraction it is 0.708, a 16.5pp improvement. The gap arose because F2 and F3 agents with unscorable outputs contributed their full text as vote keys, which could win plurality when parseable-answer agents did not form a single dominant extracted answer. Strict extraction removes these spurious votes. The 4.1pp Qwen improvement (0.607 to 0.648) is smaller because Qwen's individual answer-extraction accuracy is higher, leaving fewer invalid votes. This isolates answer extraction as an important confound in self-consistency evaluation.

**Finding 2: hard_only has the highest observed average accuracy for invalid-format faults.** At beta=0.45, hard_only outperforms baseline by 1.9pp for LLaMA (0.727 vs. 0.708) and by 0.8pp for Qwen (0.656 vs. 0.648). These small differences should be interpreted cautiously, but they are consistent with the intended mechanism: TopKMass filtering drops F1 and F3 agents before strict majority voting.

**Finding 3: soft_weighting is strongest in the high-drifter cell.** At N=7, beta=0.45, F3-only, LLaMA soft_weighting achieves 0.725 vs. 0.638 for baseline. At high drifter fractions, Module 1 hard-filters all F3 agents, triggering liveness fallback at 100% of questions and reverting to the full pool. Soft-weighting avoids this by continuously down-weighting low-confidence agents rather than hard-filtering them.

**Finding 4: full_system underperforms for invalid-format faults.** At beta=0.45, full_system is 5.4pp below hard_only for LLaMA (0.653 vs. 0.727). When liveness fires and the full pool is admitted (including drifter or crash agents), geometric median in embedding space can be drawn toward outlier embeddings even while strict majority vote correctly ignores agents with no parseable answer. This is not a failure of the geometric median mechanism; it is a consequence of the liveness fallback providing no protection against invalid-format embeddings. The advantage of full_system emerges in Experiment 3, where the threat structure is different.

**Finding 5: Qwen beta=0 fallback rate is elevated.** At beta=0, hard_only and full_system trigger liveness fallback on 13.75% of Qwen questions (vs. 5.0% for LLaMA). Qwen's stable-region TopKMass scores cluster lower than LLaMA's, so the 5th-percentile tau cuts more clean Qwen agents. This is a calibration artifact that causes Module 1 to do less useful work under clean conditions for Qwen; accuracy recovers via the liveness fallback (reverting to full pool), but the elevated fallback rate reduces the filter's net contribution.

See Figure 1 for accuracy vs. beta line charts and Figure 2 for per-fault-type breakdown at beta=0.45.

---

## 5. Experiment 2: TopKMass Signal Quality Validation

### 5.1 Setup

Experiment 2 evaluates TopKMass as a filter signal by measuring its predictive value for individual agent correctness, compared to two simpler logprob-based alternatives. All 700 agents per model (100 questions x 7 agents) are labeled `is_correct` via shared answer extraction against ground truth. Three signals are computed from `token_logprobs`:

- **TopKMass (stable-region mean):** the post-warmup mean of the causal W=64 TopKMass trajectory, falling back to the full trajectory for outputs of length at most W. This is the same statistic used for filtering and tau calibration.
- **Negated mean token entropy:** `-mean(-sum(p * log p))` over token positions, using top-5 unnormalized probabilities.
- **Negated logprob variance:** `-var(mean logprob per position)`, negated so higher is better (lower variance = more stable confidence).

All signals are oriented so that higher values indicate greater confidence (more likely correct). ROC AUC and Average Precision (AP) are computed for each signal as a binary correctness classifier.

### 5.2 Results

**Table 3.** Signal quality for correctness prediction (700 agents per model).

| Signal | LLaMA AUC | LLaMA AP | Qwen AUC | Qwen AP |
|---|---|---|---|---|
| **TopKMass** | **0.612** | **0.810** | **0.635** | **0.776** |
| Negated Entropy | 0.580 | 0.791 | 0.616 | 0.765 |
| Negated Logprob Variance | 0.448 | 0.675 | 0.538 | 0.678 |

### 5.3 Key Findings

**TopKMass is the strongest of the tested predictors on both models,** achieving the highest AUC and AP on LLaMA (0.612, 0.810) and Qwen (0.635, 0.776). The absolute AUC values are modest, so this result should be interpreted as comparative signal validation rather than strong correctness prediction. Reporting TopKMass with the stable-region mean makes this validation consistent with the filter design.

**Logprob variance performs below chance for LLaMA (AUC=0.448).** High logprob variance reflects natural variation in token confidence across a multi-step reasoning chain (for example, arithmetic steps alternating with reasoning steps at different confidence levels), not output unreliability. Using variance as a filter signal would disadvantage models on precisely the tasks (multi-step reasoning) where they perform strongly. This negative result supports the sliding-window design of TopKMass, which measures confidence stability across the sequence rather than raw variance.

**No signal is a correctness oracle.** Both correct and incorrect agents cluster in a narrow high-confidence range ([0.96, 1.00] for LLaMA, [0.94, 1.00] for Qwen) with heavy overlap. The correct-agent TopKMass median is shifted slightly right relative to incorrect agents (0.988 vs. 0.981 for LLaMA, a gap of 0.007), but no threshold cleanly separates the two classes. TopKMass's primary value as a filter is detecting agents far outside this clean cluster: F1 agents score exactly 0.0 (empty logprobs), F3 drifter agents score ~2.3 x 10^-4 (logprob spoof: -10.0 per entry). These are detectable at any tau > 0.001, which explains why Module 1 can remove crash and drifter agents while retaining most clean agents in this evaluation.

See Figure 4 for ROC curves, TopKMass scatter, and Precision-Recall curves.

---

## 6. Experiment 3: Coordinated Adversarial Attack

### 6.1 Setup

Experiment 3 evaluates coordinated synthetic wrong-answer injection: a minority of agents produce valid extractable wrong answers, agree on the same wrong value, and spoof logprobs to pass Module 1. The setting is N=7, f=2 (2 injected agents, 5 clean agents). Three coordination degrees are evaluated:

- **Uncoordinated:** The 2 Byzantine agents produce semantically different wrong texts. Both spoof TopKMass=1.00, so they pass Module 1, but they embed far apart in sentence space and cannot form a coherent adversarial cluster.
- **Coordinated:** Both agents produce the same wrong answer derived from the ground truth ("The answer is {opposite}." for StrategyQA; "The answer is {gt+7}." for GSM8K). They cluster at a single point in embedding space.
- **Maximally adversarial:** Identical to coordinated in answer content, with a higher-confidence logprob spoof. Because both coordinated settings achieve TopKMass=1.00, this condition is numerically equivalent to coordinated for Module 1 admission and is interpreted as a duplicate stress setting rather than a stronger attack.

Three pipeline conditions are compared: strict answer majority vote, geometric median nearest-centroid, and a distance-weighted voting ablation. The distance-weighted rows are not NLI results.

### 6.2 Centroid Shift Metric

For each question, we compute the distance from the arithmetic mean and geometric median of all 7 agents' embeddings to the honest-agent centroid (mean embedding of the 5 clean agents):

```
delta = dist(arithmetic_mean, honest_centroid) - dist(geometric_median, honest_centroid)
```

Positive delta indicates the geometric median stays closer to the honest cluster than the arithmetic mean. This metric is computed before any filtering and averaged across all evaluation questions. It measures embedding-space robustness relative to the arithmetic mean, not accuracy relative to majority vote.

### 6.3 Results

**Table 4.** Accuracy by coordination degree and pipeline condition (80 evaluation questions per model).

| Coordination | Condition | LLaMA Acc. | Qwen Acc. |
|---|---|---|---|
| Uncoordinated | majority_answer_vote | 72.5% | 65.0% |
| Uncoordinated | stage1_only | 67.5% | 63.75% |
| Uncoordinated | full_pipeline | 57.5% | 48.75% |
| Coordinated | majority_answer_vote | 62.5% | 65.0% |
| **Coordinated** | **stage1_only** | **70.0%** | **65.0%** |
| Coordinated | full_pipeline | 56.25% | 48.75% |
| Max adversarial | majority_answer_vote | 62.5% | 65.0% |
| **Max adversarial** | **stage1_only** | **70.0%** | **65.0%** |
| Max adversarial | full_pipeline | 56.25% | 48.75% |

**Table 5.** Centroid shift metric (distance to honest cluster centroid; lower = more robust).

| Coordination | Model | dist_mean | dist_gm | delta (mean - gm) |
|---|---|---|---|---|
| Uncoordinated | LLaMA | 0.339 | 0.095 | **+0.244** |
| Uncoordinated | Qwen | 0.347 | 0.066 | **+0.282** |
| Coordinated | LLaMA | 0.353 | 0.113 | **+0.240** |
| Coordinated | Qwen | 0.366 | 0.076 | **+0.290** |
| Max adversarial | LLaMA | 0.353 | 0.113 | **+0.240** |
| Max adversarial | Qwen | 0.366 | 0.076 | **+0.290** |

### 6.4 Key Findings

**Finding 1: geometric median nearest-centroid improves LLaMA under coordinated injection.** Under uncoordinated conditions, majority vote outperforms geometric median nearest-centroid (72.5% vs. 67.5%) because the two diverse wrong-answer embeddings do not form a cluster. Under coordinated injection, the two injected agents unify on a single wrong answer, reducing majority vote accuracy to 62.5%. Geometric median nearest-centroid reaches 70.0%, a 7.5pp improvement over majority vote in this setting.

**Finding 2: the weighted-vote ablation is not retained.** LLaMA weighted-vote accuracy ranges from 56.25% to 57.5% across coordination degrees, and Qwen weighted-vote accuracy is 48.75% across conditions. Its failure mode is vote fragmentation: diverse correct answers can split weighted vote mass, while a coordinated wrong-answer cluster concentrates weight on one extractable answer. Geometric median nearest-centroid is therefore the retained geometric-median condition for this experiment.

**Finding 3: Qwen Experiment 3 is neutral.** Under coordinated injection, geometric median nearest-centroid and majority vote both achieve 65.0% accuracy for Qwen. The liveness fallback rate for geometric median nearest-centroid is 11.25% (vs. 2.5% for LLaMA): Qwen's lower TopKMass distribution causes Module 1 to over-filter clean agents, reverting to the full 7-agent pool on roughly 1 in 9 questions. The centroid shift deltas remain positive (+0.282 to +0.290), but this mechanistic signal is not sufficient to produce an accuracy gain for Qwen.

**Finding 4: Centroid shift supports the mechanism but does not imply accuracy gains.** Positive delta is observed in all 6 conditions. Coordination increases dist_mean slightly (0.339 to 0.353 for LLaMA, 0.347 to 0.366 for Qwen) while dist_gm increases less. This supports the embedding-space mechanism, but Qwen's neutral accuracy result shows that centroid shift alone is not sufficient for improved final-answer accuracy.

See Figure 3 for accuracy vs. coordination degree and centroid shift comparisons.

---

## 7. Multi-Provider Experiments

### 7.1 Experiment A: Provider Diversity

To test whether multi-provider pools improve over the strongest single provider in this evaluation, we constructed a mixed cache combining LLaMA 3.1 8B, Qwen2.5 7B, Mistral 7B Instruct v0.3, and Phi-3 mini 4k in a 2-2-2-1 agent ratio (N=7 total) with no fault injection. Per-provider tau calibration was applied.

**Table 6.** Experiment A: Multi-provider diversity results.

| Condition | Accuracy | Low-Confidence Freq. | Mean Agents Admitted |
|---|---|---|---|
| single_llama | 0.6875 | 0.025 | 6.21 |
| multi_provider | 0.6750 | 0.038 | 6.05 |
| single_phi3 | 0.6625 | 0.013 | 6.21 |
| single_qwen | 0.6625 | 0.088 | 6.56 |
| single_mistral | 0.5625 | 0.075 | 6.18 |

The multi-provider pool does not exceed the strongest single provider in this evaluation: multi_provider achieves 67.5% vs. single_llama at 68.75% (-1.25pp). This small difference should be interpreted cautiously, but it indicates that provider diversity alone, without coordinated failures among wrong agents, does not provide an observed accuracy advantage in this setup.

### 7.2 Experiment B: Biased Provider Injection

Experiment B tests whether geometric median can absorb a systematically biased provider. Two designated biased providers (phi3, mistral) are evaluated under natural bias (as-is generation) and injected bias (F1/F2 fault injection), compared to an oracle condition (biased provider removed entirely).

**Table 7.** Experiment B: Biased provider results (phi3 and mistral as biased providers).

| Biased Provider | Condition | Bias Type | Accuracy |
|---|---|---|---|
| phi3 | no_phi3 (oracle) | natural | 0.675 |
| phi3 | majority_vote_all | natural | **0.700** |
| phi3 | geometric_median_all | natural | 0.675 |
| phi3 | no_phi3 (oracle) | injected | 0.675 |
| phi3 | majority_vote_all | injected | **0.725** |
| phi3 | geometric_median_all | injected | 0.675 |
| mistral | no_mistral (oracle) | natural | 0.700 |
| mistral | majority_vote_all | natural | **0.700** |
| mistral | geometric_median_all | natural | 0.675 |
| mistral | no_mistral (oracle) | injected | 0.700 |
| mistral | majority_vote_all | injected | **0.7125** |
| mistral | geometric_median_all | injected | 0.650 |

Majority vote has higher observed accuracy than geometric median in every condition. For Phi-3 injected bias, majority_vote_all reaches 0.725 while geometric_median_all remains at 0.675. For Mistral injected bias, geometric_median_all degrades to 0.650 while majority_vote_all reaches 0.7125.

### 7.3 Mechanistic Interpretation

The contrast between Experiment 3 and Experiment B is mechanistically informative. In Experiment 3, 2 injected agents coordinate on the same wrong answer ("The answer is no."), forming a tight cluster in embedding space. The 5 clean agents form a larger cluster. Geometric median nearest-centroid improves LLaMA accuracy in this setting.

In Experiment B, a biased provider generates different wrong answers on different questions (natural variation in model output). One or two diverse wrong votes out of seven are usually handled by the honest majority. There is no tight adversarial cluster for the geometric median to resist. Both majority vote and geometric median are drawn toward the honest majority, and majority vote is simpler and at least as accurate in this evaluation.

The relevant mechanism is cluster cohesion among wrong agents, not only the size of the wrong minority. The centroid shift metric (Section 6.2) quantifies this embedding-space effect: coordinated wrong answers form a tight spatial cluster that the arithmetic mean is drawn toward more strongly than the geometric median. Against scattered wrong answers, the observed advantage disappears.

---

## 8. Discussion

### 8.1 The Two-Failure-Mode Thesis

The experimental results support a unified thesis: the two structural failure modes of self-consistency require distinct mechanisms to address, and no single pipeline condition is optimal for both.

**Table 8.** Failure mode taxonomy and recommended conditions.

| Failure Mode | Highest observed condition | Mechanism |
|---|---|---|
| F1 crash / F3 drifter (invalid-format) | hard_only, with soft_weighting strongest in high-F3 cells | Module 1 drops low-confidence agents; liveness fallback matters when overloaded |
| F2 Byzantine (spoofed logprob, scattered) | hard_only | Spoofed agents pass Module 1; strict extraction + majority vote handles scattered wrong answers |
| Coordinated valid-format attack (Exp 3) | stage1_only | Strict extraction keeps adversarial votes; geometric median resists tight cluster |
| Diverse provider bias (Exp A/B) | majority vote | No tight cluster; geometric median offers no observed advantage |

Module 1 is effective when the broken agents fall far outside the clean TopKMass cluster (F1 score=0, F3 score~2.3 x 10^-4). It is deliberately ineffective against F2 Byzantine agents (TopKMass=1.00 by design), because real logprob spoofing is a realistic threat in deployed systems and a filter that is defeatable by logprob manipulation provides false assurance.

Module 2 (geometric median) is most useful when wrong agents form a spatially coherent cluster in embedding space. Its robustness comes from the L1-type (sum of distances) objective, which has a theoretical breakdown point of 1/2, compared to the L2-type (sum of squared distances) objective of the arithmetic mean. The centroid shift results (+0.240 to +0.290 embedding units, positive across all conditions and both models) support this mechanism, but they are not sufficient for accuracy gains.

### 8.2 Deprecated Aggregation Variants

Two aggregation variants were explored before the final stage1-only framing.

The first is bidirectional NLI entailment verification. It was designed to check whether the nearest-centroid candidate entails and is entailed by the second-nearest agent's output. This path is retained in the repository for provenance and helper reuse, but it is not the reported `full_pipeline` condition in Experiment 3.

The second is distance-weighted answer voting. This is the `full_pipeline` condition in the reported Experiment 3 results. It underperforms stage1_only in every reported Exp 3 condition. Its main failure mode is vote fragmentation: semantically diverse correct answers can split weighted vote mass, while a coordinated wrong-answer cluster concentrates weight on one extractable answer. For the reported results, stage1_only is the recommended configuration.

### 8.3 BFT Framing: Scope and Limitations

The liveness threshold (2f+1 with f = floor((N-1)/3)) and fault fraction vocabulary (beta) are drawn from Byzantine Fault Tolerance theory. This framing is useful for reasoning about worst-case degradation scenarios and for specifying the pipeline's behavior when the fault load exceeds design thresholds. It should not be interpreted as a claim that the system provides Byzantine-fault-tolerant consensus in the classical sense.

The current evaluation uses a homogeneous pool (N samples from one model). In this setting, no agent has an incentive to deceive; F2 Byzantine injection is a synthetic stress test, not a real-world threat. The BFT framing becomes more motivated in a multi-provider heterogeneous setting, where a provider operating one or more agents may have incentives to steer the consensus answer. The coordinated-attack defense is a hypothesis for coordinated provider failures, not a result established by the natural provider-bias experiments here. Such a setting also requires per-provider logprob availability and is left to future work.

### 8.4 Limitations

**Small evaluation set and aggregate CSVs.** Each main evaluation uses 80 questions per model after the 20% dev split, so absolute accuracy values carry non-trivial uncertainty. The canonical Exp 1 and Exp 3 CSVs store aggregate accuracies rather than per-question correctness vectors; reporting paired confidence intervals would require re-extracting question-level outcomes from the generation caches. We therefore interpret small differences cautiously.

**Qwen calibration artifact.** Qwen's 13.75% liveness fallback rate at beta=0 and 11.25% in Experiment 3 indicate that the 5th-percentile tau over-filters Qwen clean agents. This reduces Module 1's effective contribution for Qwen and explains the neutral Exp 3 result. Adaptive per-model calibration with larger dev slices would likely improve Qwen's results.

**Homogeneous pool.** All primary experiments use N=7 samples from a single model at temperature=0.7. Diversity within the pool is stochastic, not structural. The multi-provider experiments (Section 7) extend to four models but find neutral or negative results for multi-provider consensus under non-coordinated conditions.

**Synthetic fault injection.** F2 Byzantine agents use a fixed wrong-answer template and perfectly spoofed logprobs. Real-world failure modes may have different logprob signatures or different patterns of answer coordination. The centroid shift metric provides mechanistic evidence about embedding-space behavior, but the accuracy results may differ under different adversarial strategies.

---

## 9. Conclusion

We have presented a failure-mode-specific study of multi-agent LLM consensus under two structurally distinct failure modes of self-consistency. Strict answer extraction prevents invalid outputs from entering the vote pool as raw-text keys. TopKMass filtering identifies low-confidence invalid-format outputs using a stable-region sliding-window mean of top-5 token probabilities. Geometric median nearest-centroid aggregation provides a separate mechanism for coordinated valid-format wrong answers by minimizing sum of embedding distances rather than sum of squared distances.

The primary Experiment 1 result is a methodology finding: strict answer extraction corrects a 16.5pp collapse in one LLaMA high-fault baseline, and TopKMass filtering plus strict majority vote has the highest observed average accuracy for invalid-format faults. Experiment 2 validates TopKMass as the strongest of the tested logprob-based correctness predictors (AUC=0.612 for LLaMA, vs. 0.580 for negated entropy and 0.448 for negated logprob variance), while confirming that its primary value is detecting broken agents far outside the clean cluster rather than discriminating correct from incorrect among healthy agents. Experiment 3 shows that geometric median nearest-centroid improves LLaMA accuracy by 7.5pp over majority vote under coordinated synthetic wrong-answer injection. Multi-provider experiments delineate the scope of this defense: it requires coordinated wrong answers forming a tight cluster and does not extend to natural provider diversity or scattered provider bias in this evaluation.

No single pipeline condition is uniformly optimal. Strict extraction and TopKMass filtering are most useful for invalid-format failures; geometric median nearest-centroid is most useful when wrong answers form a coordinated semantic cluster; majority vote remains simpler and at least as effective under several uncoordinated regimes. Understanding the specific conditions under which each mechanism helps, supported by mechanistic metrics such as TopKMass AUC and centroid shift, is the primary contribution of this work.

Future directions include: (1) multi-provider heterogeneous pools with mixed logprob availability, including providers that do not expose per-token logprobs; (2) adaptive tau calibration per provider or per output length using larger dev slices; (3) larger evaluation sets with per-question confidence intervals; (4) arithmetic-mean nearest-centroid and other additional aggregation ablations; and (5) streaming consensus settings where agent count varies dynamically.

---

## References

Castro, M. and Liskov, B. (1999). Practical Byzantine fault tolerance. In *Proceedings of the 3rd Symposium on Operating Systems Design and Implementation (OSDI)*, pages 173-186.

Eikema, B. and Aziz, W. (2020). Is MAP decoding all you need? The inadequacy of the mode in neural machine translation. In *Proceedings of COLING 2020*.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of ICML 2017*.

Jo, D. and Park, S. (2024). DecentLLMs: Decentralized consensus among heterogeneous LLMs. *arXiv preprint*.

Lamport, L., Shostak, R., and Pease, M. (1982). The Byzantine generals problem. *ACM Transactions on Programming Languages and Systems*, 4(3):382-401.

Li, Y., Lin, Z., Zhang, S., Fu, Q., Chen, B., Lou, J.-G., and Chen, W. (2023). Making language models better reasoners with step-aware verifier. In *Proceedings of ACL 2023*.

Meta AI. (2024). Llama 3.1 model card. Meta AI technical report.

Minsker, S. (2015). Geometric median and robust estimation in Banach spaces. *Bernoulli*, 21(4):2308-2335.

Qwen Team. (2024). Qwen2.5 technical report. Alibaba Cloud technical report.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language models. In *Proceedings of ICLR 2023*.

Weiszfeld, E. (1937). Sur le point pour lequel la somme des distances de n points donnes est minimum. *Tohoku Mathematical Journal*, 43:355-386.

---

## Appendix

### A.1 Figure: Accuracy vs. Fault Fraction (with N=1 Reference)

Figure A1 (`figures/accuracy_vs_beta.png`) reproduces the main accuracy-vs-beta line charts with an additional gray dotted reference line for the N=1 single-agent condition. The N=1 reference line is flat by construction: floor(1 x beta) = 0 for any beta in {0, 0.15, 0.30, 0.45}, so the single agent's generation is never perturbed by fault injection. This line should be read as an unperturbed single-agent accuracy floor, not as a condition exposed to the same fault rate as the ensemble conditions.

### A.2 Figure: Qwen Signal Quality

Figure A2 (`results/exp2_qwen/experiment_2_signals.png`) presents the three-panel signal quality analysis for Qwen2.5 7B, parallel to the LLaMA Figure 4 in the main body. Qwen TopKMass AUC (0.635) exceeds LLaMA (0.612); negated entropy is a closer second for Qwen (AUC gap 0.019 vs. 0.032 for LLaMA). Logprob variance remains below chance for Qwen (AUC=0.538, though above the LLaMA value of 0.448). The narrow TopKMass score range for Qwen ([0.94, 1.00]) vs. LLaMA ([0.79, 1.00]) reflects Qwen's generally higher per-token confidence and contributes to the calibration artifact observed in Experiments 1 and 3.

### A.3 Table: Liveness Fallback Frequency

**Table A1.** Liveness fallback frequency by condition, N, beta, and model (averaged over fault types).

| Model | Condition | N | beta=0% | beta=15% | beta=30% | beta=45% |
|---|---|---|---|---|---|---|
| LLaMA 3.1 8B | hard_only | 7 | 5.0% | 7.5% | 18.8% | 100.0% (F1/F3) |
| LLaMA 3.1 8B | full_system | 7 | 5.0% | 7.5% | 18.8% | 100.0% (F1/F3) |
| Qwen2.5 7B | hard_only | 7 | 13.75% | 16.25% | 26.25% | 100.0% (F1/F3) |
| Qwen2.5 7B | full_system | 7 | 13.75% | 16.25% | 26.25% | 100.0% (F1/F3) |

At beta=0.45 with N=7, floor(7 x 0.45) = 3 faults. The BFT threshold is 2f+1 = 5 agents. With 3 F1 or F3 agents filtered by Module 1, only 4 agents remain, which is below the threshold, and liveness fires at 100% of questions. Hard_only and full_system are therefore equivalent to the unfiltered baseline under F1/F3 at beta=0.45 N=7. The difference between conditions at beta=0.45 in Table 1 and Table 2 is driven by F2 (Byzantine) questions, where spoofed logprobs pass Module 1 and filtering remains active.

### A.4 Reproducibility: Code Paths and Result Artifacts

The canonical TopKMass filter is implemented in `pipeline/filter.py`. The final geometric median nearest-centroid helper is implemented in `pipeline/stage1.py`, reusing embedding and geometric-median helpers from `pipeline/aggregation.py`. The older NLI verification path remains in `pipeline/aggregation.py` for provenance and helper reuse. The distance-weighted voting ablation used by the Experiment 3 `full_pipeline` rows is implemented in `pipeline_v2/aggregation.py`.

The Experiment 1 runner is `eval/runner_v2.py`; the Experiment 2 signal-quality runner is `eval/signal_quality.py`; the Experiment 3 coordinated-injection runner is `eval/adversarial_test_v2.py`. Multi-provider experiments use `eval/runner_multi.py`, `eval/biased_provider_test.py`, and `pipeline_multi/`. Shared cache loading and answer extraction helpers are in `eval/io.py` and `eval/answer_extraction.py`.

The main result artifacts are `results/experiment_1_llama_v4.csv`, `results/experiment_1_qwen_v4.csv`, the Experiment 2 signal-quality directories under `results/exp2_*`, the Experiment 3 `_v2.csv` files, and the multi-provider CSVs reported in Section 7. No confidence intervals are reported because the canonical Experiment 1 and Experiment 3 CSVs contain aggregate accuracies rather than per-question correctness vectors.
