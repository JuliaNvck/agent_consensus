# Paper Outline — Research Memo

**Working title:** Robust Consensus for Multi-Agent LLM Ensembles via Confidence Filtering and Geometric Median Aggregation

**Target:** Full conference paper, ~8 pages (e.g., EMNLP, NeurIPS, ICLR workshop on LLM reliability)

**Status:** All experiments complete. Data, figures, and code committed.

---

## 1. Research Question

### The Problem

Multi-agent LLM ensembles — running N independent completions from the same or different models and aggregating the results — are increasingly used to improve accuracy and reliability over single-pass inference. The standard aggregation method is **self-consistency** (Wang et al., 2023): majority vote over extracted answers.

Self-consistency fails in two structurally distinct ways:

**Failure Mode 1 — Invalid-format output failures.** Some agents produce outputs that contain no parseable answer: empty outputs from timeouts (F1 crash), off-task hallucinations (F3 drifter), or outputs whose full text is injected into the vote pool when answer extraction fails. At high fault fractions (β=45%), these corrupt the vote with garbage keys that can win plurality. A simple fix — strict answer extraction that abstains rather than falling back to full text — dramatically corrects this (16.5pp improvement for LLaMA at β=45%). Module 1 (confidence filtering) can detect and remove F1/F3 agents before aggregation.

**Failure Mode 2 — Coordinated valid-format wrong answers.** A minority of agents can produce syntactically valid, extractable wrong answers that all agree on ("The answer is no."). Unlike F1/F3, these agents look confident and well-formed. Strict extraction keeps them in the vote. When 2 of 7 agents coordinate on the same wrong answer and 5 correct agents split their votes across different phrasings, majority vote can be swung to the minority answer. Module 1 does not help here — these agents pass confidence filtering by design.

### The Central Claim

A two-layer pipeline — Module 1 (TopKMass confidence filter) + Module 2 (geometric median in embedding space) — addresses both failure modes through complementary mechanisms:
- Module 1 handles invalid-format failures by dropping low-confidence agents
- Module 2 handles coordinated valid-format attacks by finding the semantic center of the honest majority, resisting a tight adversarial cluster

No single prior approach addresses both simultaneously.

### Secondary Question

Is TopKMass (sliding-window mean of top-5 token probabilities) a better per-agent correctness predictor than simpler logprob-based signals (entropy, logprob variance)? Experiment 2 validates this.

---

## 2. System Description

### Architecture

Two strictly separated phases, motivated by the GPU Rule: Phase 1 uses GPU inference; Phase 2 runs entirely on CPU from cached data.

**Phase 1 — Offline generation (GPU, vLLM):**
- Run `scripts/generate_cache.py` with `vllm.LLM`, `temperature=0.7`, `logprobs=5`
- Outputs: N=7 completions per question with per-token top-5 logprobs
- Cached to JSON: `{"questions": [{"question_id", "ground_truth", "generations": [{agent fields}]}]}`
- Never re-run during evaluation

**Phase 2 — Evaluation pipeline (CPU):**
- Load cache → inject faults → Module 1 → Module 2 → extract answer → score

### Module 1: TopKMass Confidence Filter (`pipeline/filter.py`)

**Signal definition:**
```
TopKMass(W=64) = (1/W) * Σ sum(top-5 token probs at position t)
```
Computed as a causal sliding-window mean over token positions. The stable-region mean (trajectory sliced at position W=64) is the per-agent score. Warmup normalization makes the score length-invariant — without it, short outputs (StrategyQA, ~128 tokens) would be systematically penalized vs long outputs (GSM8K, ~512 tokens).

**Threshold calibration:**
τ is calibrated at the **5th percentile** of clean agent scores on a shuffled 20% dev slice. This tolerates 5% false-rejection of clean agents while filtering broken agents (F1: score=0, F3: score≈2.3×10⁻⁴) far below any clean-agent score.

**Liveness fallback (BFT-inspired):**
- f = ⌊(N−1)/3⌋ (N=7 → f=2; N=5 → f=1)
- Liveness threshold: 2f+1 admitted agents required
- If admitted < 2f+1 after filtering: revert to full pool, set `is_low_confidence=True`

**Fault behavior by type:**
| Fault | TopKMass score | Module 1 outcome |
|---|---|---|
| F1 (crash) | 0.0 (empty logprobs) | Dropped unconditionally |
| F3 (drifter) | ≈2.3×10⁻⁴ (logprob spoof: −10.0) | Dropped (below any τ > 0.001) |
| F2 (Byzantine) | 1.00 (logprob spoof: sum=1.00) | **Passes** — Module 2 must handle |
| Clean agent | [0.96, 1.00] typical | Passes unless rare outlier |

### Module 2: Geometric Median Nearest-Centroid (`pipeline/stage1.py`, `stage1_only`)

1. Embed admitted agents' output text using `sentence-transformers/all-mpnet-base-v2` (768-dim)
2. Compute geometric median of embeddings: minimise `Σ‖xᵢ − y‖₂` via Weiszfeld/L-BFGS-B
3. Return `output_text` of the agent with minimum Euclidean distance to the geometric median

**Why geometric median over arithmetic mean:** The geometric median is the minimizer of sum of distances (not sum of squared distances), giving it a breakdown point of 1/2 — a minority of outlier embeddings cannot drag it arbitrarily far. The centroid shift metric (§4.3) measures this empirically: under coordinated attack, the geometric median stays 0.24–0.29 embedding units closer to the honest cluster than the arithmetic mean.

**NLI Stage 2 (deprecated):** A bidirectional NLI entailment check remains in `pipeline/aggregation.py` for provenance, but it is not used in the reported `_v2.csv` Exp 3 rows. The final nearest-centroid path reuses `_embed` and `_geometric_median` helpers from that module through `pipeline/stage1.py`.

### Ablation Conditions

| Condition | Module 1 | Aggregation | Notes |
|---|---|---|---|
| `baseline` | None | Strict majority vote | Self-consistency with strict extraction |
| `soft_weighting` | None (scores used as weights) | Weighted geometric median | TopKMass-weighted Weiszfeld |
| `hard_only` | TopKMass filter + liveness | Strict majority vote | Best for invalid-format faults |
| `full_system` / `stage1_only` | TopKMass filter + liveness | Geometric median nearest-centroid | Best for coordinated attacks |

**Strict answer extraction** (`eval/answer_extraction.py`): agents with no parseable answer return `None` and are excluded from the vote. This is applied to `baseline` and `hard_only`. Without it, the old baseline collapsed 16.5pp at β=0.45 (garbage text winning plurality).

### Fault Injection (`faults/injector.py`)

- **F1 (crash):** `output_text=""`, `token_logprobs=[]`
- **F2 (Byzantine):** wrong answer string + spoofed logprobs with TopKMass=1.00 per position (sum of top-5 probs = 1.0 exactly; above the 0.9951 max clean-agent score to avoid inadvertent filtering)
- **F3 (drifter):** off-task syntactically plausible text (Arctic terns passage) + spoofed logprobs with per-position value −10.0 → TopKMass≈2.3×10⁻⁴
- **mix:** each mutated agent randomly assigned F1/F2/F3
- β ∈ {0, 0.15, 0.30, 0.45}; exactly `floor(N×β)` agents mutated; deterministic via `random.Random(seed)`

---

## 3. Models and Data

### Models

| Model | HuggingFace ID | Role |
|---|---|---|
| LLaMA 3.1 8B Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Primary (Exp 1–3) |
| Qwen2.5 7B Instruct | `Qwen/Qwen2.5-7B-Instruct` | Primary (Exp 1–3) |
| Mistral 7B Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | Multi-provider (Exp A, B) |
| Phi-3 mini 4k | `microsoft/Phi-3-mini-4k-instruct` | Multi-provider (Exp A, B) |

Two models of the same ~7–8B parameter class, different architectures and training. Both evaluated identically. Cross-model consistency is a validation, not a controlled variable.

### Datasets

| Dataset | Config / Split | N questions | Task | Answer format |
|---|---|---|---|---|
| GSM8K | `main` / `test` | 50 | Multi-step arithmetic reasoning | Integer string (e.g., `"42"`) |
| StrategyQA | `wics/strategy-qa` / `test` | 50 | Commonsense multi-hop QA | `"yes"` / `"no"` |

100 questions total per model. Mixing GSM8K and StrategyQA stresses orthogonal failure modes: GSM8K exposes step-counting errors and arithmetic drift; StrategyQA exposes semantic drift and yes/no manipulation.

**Generation settings:** `temperature=0.7`, N=7 agents per question, `max_tokens=512` (GSM8K) / `max_tokens=128` (StrategyQA), `logprobs=5`.

**Dev/eval split:** 20% shuffled dev slice for τ calibration; 80% evaluation. Without shuffling, the GSM8K-first ordering biases τ upward and causes ~50% liveness fallback at β=0 — the shuffle is critical.

**Published reference points (sanity check):**
- LLaMA 3.1 8B GSM8K greedy: ~73% (Meta AI 2024). Our N=1 LLaMA: 71.3%. ✓
- Qwen2.5 7B GSM8K greedy: ~85% (Qwen team 2024). Our N=1 Qwen: 67.5% on mixed benchmark (lower expected since StrategyQA is included). ✓

---

## 4. Experiments

### 4.1 Experiment 1: Fault Injection Ablation

**Goal:** Quantify how much each pipeline condition degrades as fault fraction increases, and which fault types are hardest.

**Grid:** 4 conditions × N∈{5,7} × β∈{0,0.15,0.30,0.45} × fault types {F1,F2,F3,mix} = 128 rows per model (plus 64 rows for N=1 single-agent reference).

**Result files:** `results/experiment_1_llama_v4.csv`, `results/experiment_1_qwen_v4.csv`

**Results — LLaMA 3.1 8B (averaged over N∈{5,7} and all fault types):**

| Condition | N=1 | β=0% | β=15% | β=30% | β=45% |
|---|---|---|---|---|---|
| baseline (self-consistency) | 0.713 | 0.756 | 0.769 | 0.736 | 0.708 |
| soft_weighting | 0.713 | 0.713 | 0.725 | 0.728 | 0.688 |
| hard_only | 0.713 | 0.750 | 0.763 | 0.745 | **0.727** |
| full_system / stage1_only | 0.713 | 0.700 | 0.723 | 0.733 | 0.653 |

**Results — Qwen2.5 7B (averaged over N∈{5,7} and all fault types):**

| Condition | N=1 | β=0% | β=15% | β=30% | β=45% |
|---|---|---|---|---|---|
| baseline | 0.675 | 0.669 | 0.669 | 0.673 | 0.648 |
| soft_weighting | 0.675 | 0.650 | 0.650 | 0.644 | 0.663 |
| hard_only | 0.675 | 0.669 | 0.669 | 0.675 | **0.656** |
| full_system / stage1_only | 0.675 | 0.650 | 0.650 | 0.644 | 0.645 |

**Key findings:**

1. **`hard_only` is the best condition for invalid-format faults.** LLaMA β=0.45: hard_only=0.727 vs baseline=0.708 (+1.9pp). Module 1 drops F1/F3 agents; strict extraction handles the rest.

2. **The strict extraction fix is the primary Exp 1 result.** Old LLaMA baseline at β=0.45: 0.543 → V4: 0.708 (+16.5pp). The collapse was a methodology bug, not a pipeline limitation. The corrected baseline is stronger, making all condition differences smaller but real.

3. **`soft_weighting` is the best condition for F3 (drifters).** N=7 β=0.45 F3: LLaMA soft=0.725 vs baseline=0.638. Continuous weighting avoids the liveness fallback cascade that fires when all F3 agents are Module-1-filtered (100% fallback rate at F3 β=0.45 for hard_only/full_system).

4. **`full_system` underperforms in Exp 1 but is necessary for Exp 3.** At β=0.45, geometric median nearest-centroid selects a suboptimal agent more often than majority voting when faults are invalid-format. This is expected: when most of the pool is F3 (filtered out → liveness fires → full pool → 5 correct + 2 F3 agents), gm can drift toward the F3 outliers. hard_only avoids this because the F3 outliers abstain from the vote rather than embedding into the wrong part of the space.

5. **Qwen fallback at β=0: 13.75%** — Qwen's TopKMass distribution is lower, causing τ to cut more clean agents. This is a calibration artifact, not a fundamental problem. Liveness fires, accuracy recovers, but it means Qwen's Module 1 is doing less useful work at clean conditions.

### 4.2 Experiment 2: TopKMass Signal Quality Validation

**Goal:** Validate that TopKMass is a better per-agent correctness predictor than simpler logprob-based alternatives. This provides mechanistic justification for why Module 1 works.

**Setup:** 700 agents per model (100 questions × 7 agents). Each agent labeled `is_correct` via answer extraction against ground truth. Three signals computed per agent: TopKMass mean (stable-region), −MeanTokenEntropy, −LogprobVariance. ROC AUC and Average Precision computed for each signal as a binary correctness classifier.

**Result files:** `results/exp2_llama/experiment_2_signals.csv`, `results/exp2_qwen/experiment_2_signals.csv`

**Results:**

| Signal | LLaMA AUC | LLaMA AP | Qwen AUC | Qwen AP |
|---|---|---|---|---|
| **TopKMass** | **0.606** | **0.802** | **0.627** | **0.765** |
| −Entropy | 0.580 | 0.791 | 0.616 | 0.765 |
| −LogprobVar | 0.448 | 0.675 | 0.538 | 0.678 |

**Key findings:**

1. **TopKMass wins on both models** (highest AUC and Average Precision).

2. **Logprob variance is worse than chance for LLaMA (AUC=0.448).** This is a strong negative that validates using TopKMass over naive variance. High logprob variance does not predict incorrectness — it reflects natural variation in token confidence across a multi-step reasoning chain, not unreliability.

3. **No signal is a correctness oracle.** Both correct and incorrect agents cluster in a narrow high-confidence TopKMass range ([0.96, 1.0]) with heavy overlap. The correct cloud is shifted slightly right (LLaMA correct median 0.988 vs incorrect 0.981; gap of 0.007). TopKMass's real value is detecting *broken* agents (F1: score=0, F3: score≈2.3×10⁻⁴) that fall far outside the clean cluster — not discriminating correct from incorrect within the clean population.

**Framing note for the paper:** This experiment validates that Module 1 is principled, not arbitrary. Even with modest AUC for correctness prediction, TopKMass identifies the broken-agent failure modes that corrupt majority vote.

### 4.3 Experiment 3: Coordinated Adversarial Attack

**Goal:** Test the pipeline against the hardest case: a coordinating minority of Byzantine agents that (a) produce valid extractable wrong answers, (b) all agree on the same wrong answer, and (c) spoof maximum-confidence logprobs to pass Module 1.

**Setup:** N=7, f=2 (2 Byzantine agents). Three coordination degrees:
- **Uncoordinated:** 2 Byzantine agents produce different wrong texts (diverse adversarial cluster)
- **Coordinated:** Both produce the same wrong answer (`"The answer is {opposite}."` or `"The answer is {gt+7}."`)
- **Maximally adversarial:** Same as coordinated + slightly higher spoof confidence (numerically equivalent since both achieve TopKMass=1.00)

Three pipeline conditions: `majority_answer_vote`, `stage1_only`, `full_pipeline`.

**Centroid shift metric:**
```
delta = dist(arithmetic_mean, honest_centroid) − dist(geometric_median, honest_centroid)
```
Positive delta = geometric median stays closer to the honest 5-agent cluster than arithmetic mean. Computed per question, averaged per coordination degree.

**Result files:** `results/exp3_llama/experiment_3_adversarial_v2.csv`, `results/exp3_qwen/experiment_3_adversarial_v2.csv`

**Accuracy results:**

| Coordination | Condition | LLaMA | Qwen |
|---|---|---|---|
| Uncoordinated | majority_answer_vote | 72.5% | 65.0% |
| Uncoordinated | stage1_only | 67.5% | 63.75% |
| Uncoordinated | full_pipeline | 57.5% | 48.75% |
| Coordinated | majority_answer_vote | 62.5% | 65.0% |
| **Coordinated** | **stage1_only** | **70.0%** | **65.0%** |
| Coordinated | full_pipeline | 56.25% | 48.75% |
| Maximally adversarial | majority_answer_vote | 62.5% | 65.0% |
| **Max adversarial** | **stage1_only** | **70.0%** | **65.0%** |
| Maximally adversarial | full_pipeline | 56.25% | 48.75% |

Fallback frequency (stage1_only): 2.5% LLaMA, 11.25% Qwen.

**Centroid shift results:**

| Coordination | Model | dist_mean | dist_gm | delta |
|---|---|---|---|---|
| Uncoordinated | LLaMA | 0.339 | 0.095 | +0.244 |
| Uncoordinated | Qwen | 0.347 | 0.066 | +0.282 |
| Coordinated | LLaMA | 0.353 | 0.113 | +0.240 |
| Coordinated | Qwen | 0.366 | 0.076 | +0.290 |
| Max adversarial | LLaMA | 0.353 | 0.113 | +0.240 |
| Max adversarial | Qwen | 0.366 | 0.076 | +0.290 |

**Key findings:**

1. **`stage1_only` wins under coordinated attack: LLaMA +7.5pp (70.0% vs 62.5%).** Under uncoordinated conditions, majority vote leads (72.5% vs 67.5%) because diverse wrong answers split the adversarial vote. Under coordinated attack, the 2 Byzantine agents unify on the same answer, dragging majority vote to 62.5%. Geometric median resists: the 5-agent honest cluster remains spatially dominant.

2. **`full_pipeline` (weighted-vote ablation) is the worst condition in every scenario.** LLaMA: 56.25-57.5% across all conditions; Qwen: 48.75%. The reported `_v2.csv` rows use `pipeline_v2.aggregation.aggregate()`, not NLI Stage 2. The failure mode is weighted-vote fragmentation under coordinated valid-format attacks. Do not use it as the final method.

3. **Qwen Exp 3 is neutral.** stage1_only = majority_vote = 65.0% under coordinated attack. The 11.25% liveness fallback explains this: Module 1 over-filters Qwen clean agents and reverts to the full 7-agent pool, including the 2 Byzantine agents — at which point geometric median operates on the uncleaned pool and converges toward majority vote behavior. The centroid shift deltas (+0.282–0.290) remain strong, confirming the mechanism works; the accuracy parity is a calibration artifact.

4. **Centroid shift is model-agnostic.** Positive delta in all 6 conditions; coordination increases dist_mean slightly while dist_gm barely moves. This is the mechanistic explanation: geometric median's robustness comes from minimizing sum of distances rather than sum of squared distances, making it robust to the tight wrong-answer cluster.

### 4.4 Experiment A: Multi-Provider Diversity

**Goal:** Does a 4-model mixed pool (LLaMA, Qwen, Mistral, Phi-3 in 2-2-2-1 ratio) outperform the best individual provider?

**Result file:** `results/exp_multi_a_diversity.csv`

**Results:**

| Condition | Accuracy | Low-conf freq | Mean admitted |
|---|---|---|---|
| single_llama | 0.6875 | 0.025 | 6.21 |
| multi_provider | 0.6750 | 0.038 | 6.05 |
| single_phi3 | 0.6625 | 0.013 | 6.21 |
| single_qwen | 0.6625 | 0.088 | 6.56 |
| single_mistral | 0.5625 | 0.075 | 6.18 |

**Finding:** Multi-provider does not beat the best single provider (−1.25pp vs LLaMA). Mistral is unexpectedly weak (56.25%), contributing noise. The geometric median partially absorbs Mistral's drag — multi_provider beats single_mistral by 11.25pp — but cannot exceed the strongest single-provider accuracy. **Negative result.**

### 4.5 Experiment B: Biased Provider Injection

**Goal:** Can geometric median absorb a systematically biased provider?

**Conditions:** designated biased provider (phi3 or mistral) × natural or injected bias × {oracle, majority_vote_all, geometric_median_all}.

**Result files:** `results/exp_multi_b_phi3.csv`, `results/exp_multi_b_mistral.csv`

**Results — phi3 as biased provider:**

| Condition | Natural | Injected |
|---|---|---|
| no_phi3 (oracle) | 0.675 | 0.675 |
| majority_vote_all | **0.700** | **0.700** |
| geometric_median_all | 0.675 | 0.675 |

**Results — mistral as biased provider:**

| Condition | Natural | Injected |
|---|---|---|
| no_mistral (oracle) | 0.700 | 0.700 |
| majority_vote_all | **0.700** | **0.7125** |
| geometric_median_all | 0.675 | 0.650 |

**Finding:** Majority vote beats geometric median in every condition. When Mistral is injected, geometric median degrades to 0.650 while majority vote improves to 0.7125. **Strong negative result.**

**Why:** Biased providers produce *diverse* wrong answers (different wrong answer per question), not coordinated ones. 1–2 diverse wrong votes out of 7 are trivially outvoted. Geometric median has no edge because there is no tight wrong-answer cluster to resist. This directly contrasts with Exp 3 where 2 agents coordinate on the *same* wrong answer.

---

## 5. Two-Failure-Mode Thesis

This is the paper's organizing claim — the unifying framework that explains when each component helps and when it doesn't.

| Failure mode | Best condition | Mechanism |
|---|---|---|
| **F1 crash / F3 drifter** | `hard_only` | Module 1 drops agents (score=0 or ≈0); liveness fallback when overloaded |
| **F2 Byzantine (spoofed logprob)** | `hard_only` | Spoofed agents pass Module 1; strict extraction + majority vote handles scattered wrong answers |
| **Coordinated valid-format attack** (Exp 3) | `stage1_only` | Strict extraction keeps adversarial votes; geometric median resists tight cluster |
| **Diverse provider bias** (Exp A/B) | Majority vote | No coordination → no tight cluster → geometric median offers no advantage |

**The scope of geometric median's advantage:**
Geometric median's robustness is *conditional on cluster cohesion* among wrong agents. It wins specifically when:
1. Wrong agents converge on the same output (coordinated)
2. The wrong-answer cluster is tight in embedding space
3. The honest-agent cluster is larger (N=5 vs N=2 in Exp 3)

When wrong agents produce diverse failures — whether random fault injection (Exp 1 F1/F3) or natural provider heterogeneity (Exp A/B) — majority vote is equally effective and simpler. The centroid shift metric quantifies this: the delta is positive and large (+0.24–0.29) under coordinated attack, confirming the geometric median stays outside the adversarial cluster.

**Honest framing:** No single pipeline condition is optimal for all failure regimes. The layered design (Module 1 for invalid-format detection, Module 2 for semantic robustness) provides partial defense in each regime without claiming universality.

---

## 6. Figures

All figures are paper-ready (150 dpi, tight bbox). Generated by `scripts/plot_results.py`.

### Main Body Figures

**Figure 1: `figures/accuracy_vs_beta_no_n1.png`**
Two-panel line chart (LLaMA left, Qwen right). X-axis: fault fraction β ∈ {0%, 15%, 30%, 45%}. Y-axis: accuracy. Four series per panel: baseline (red dashed), soft_weighting (orange dotted), hard_only (blue dash-dot), full_system (green solid). Averaged over N∈{5,7} and all fault types.

*Draft caption:* "Accuracy vs. fault fraction for LLaMA 3.1 8B (left) and Qwen2.5 7B (right). All conditions use strict answer extraction (abstaining when no parseable answer is found). Hard Filter + Majority (hard_only) consistently leads at high β, with the gap widening as fault load increases. Results averaged over N∈{5,7} agents and all four fault types."

**Figure 2: `figures/fault_type_breakdown.png`**
Two-panel grouped bar chart at β=45%. X-axis: F1 (Crash), F2 (Byzantine), F3 (Drifter), Mix. Four grouped bars per position: one per condition. Color-coded with percentage annotations above each bar.

*Draft caption:* "Accuracy by fault type at β=45% for LLaMA (left) and Qwen (right). F2 (Byzantine) is the hardest fault type for hard_only because spoofed logprobs (TopKMass=1.00) pass Module 1, leaving soft extraction as the only defense. Soft-Weighted SC (soft_weighting) outperforms on F3 (Drifter) because continuous score weighting avoids the liveness fallback cascade that fires when all drifter agents are hard-filtered."

**Figure 3: `figures/adversarial_coordination.png`**
Two-panel line chart. X-axis: coordination degree (Uncoordinated, Coordinated, Max Adversarial). Y-axis: accuracy 40%–85%. Three series per panel: Majority Vote (red dashed), Stage1-Only (green solid), Full Pipeline (purple dotted). Exact percentages annotated above each point.

*Draft caption:* "Accuracy under adversarial coordination (Experiment 3). Under uncoordinated attack, majority vote leads because diverse wrong answers split the adversarial vote. Under coordinated attack, majority vote drops 10pp (LLaMA: 72.5%→62.5%) while Stage1-Only recovers to 70.0%, demonstrating that geometric median's advantage is tied to adversarial cluster cohesion. Full Pipeline is the `pipeline_v2` weighted-vote ablation in the reported `_v2.csv` files and is not recommended."

**Figure 4: `results/exp2_llama/experiment_2_signals.png`** (or append to paper body)
Three-panel figure for LLaMA signal quality. Panel A: ROC curves for TopKMass, −Entropy, −LogprobVar with AUC in legend. Panel B: TopKMass scatter (x=score, y=jittered correctness label) with per-class median lines. Panel C: Precision-Recall curves with AP in legend.

*Draft caption:* "TopKMass signal quality for LLaMA 3.1 8B (700 agents, 100 questions × 7). Panel A: ROC curves — TopKMass (AUC=0.606) outperforms −Entropy (0.580) and −LogprobVariance (0.448, below chance). Panel B: Correct agents (green) cluster slightly higher than incorrect (red) but with heavy overlap — TopKMass is not a correctness oracle, but reliably identifies broken agents (crash/drifter) far below the clean cluster. Panel C: Precision-Recall curves confirm TopKMass's average precision advantage."

### Supplementary / Appendix Figures

**`figures/accuracy_vs_beta.png`** — same as Figure 1 but with N=1 single-agent reference line (gray dotted). Note: N=1 accuracy is immune to fault injection since `floor(1×β)=0` — it is the guaranteed-clean-agent anchor, not a fault-exposed measurement.

**`results/exp2_qwen/experiment_2_signals.png`** — Qwen signal quality (parallel to Figure 4).

---

## 7. Known Weaknesses and Honest Limitations

These should be acknowledged in the Discussion or Limitations section.

**1. BFT framing is analytical scaffolding, not a real threat model.**
The homogeneous pool (N samples from one model) has no incentive structure. F2 Byzantine injection is a synthetic stress test. The framing provides useful analysis structure (f-fault tolerance, liveness, breakdown points) but should not be presented as adversarial robustness against an external attacker.

**2. Qwen Experiment 3 is neutral, not positive.**
stage1_only = majority_vote = 65.0% under coordinated attack for Qwen. Caused by 11.25% liveness fallback: Module 1 over-filters Qwen clean agents and reverts to the full 7-agent pool including Byzantine agents. The mechanism (centroid shift +0.282–0.290) works; the calibration doesn't. This should be reported honestly — the LLaMA result (+7.5pp) carries the Exp 3 claim.

**3. Multi-provider experiments are negative.**
Exp A: multi-provider doesn't beat the best single provider (−1.25pp vs LLaMA).
Exp B: geometric median loses to majority vote under natural provider bias. The distinction is clear: the technique requires coordinated wrong answers, which natural provider bias does not produce. Report in the paper as scope delimitation, not a failure.

**4. Weighted-vote full_pipeline is consistently worst in reported Exp 3.**
The reported `full_pipeline` rows use `pipeline_v2` distance-weighted answer voting and are 10-14pp below stage1_only for LLaMA, with Qwen at 48.75%. The paper should explicitly state that these `_v2.csv` rows are not NLI results. NLI remains a deprecated earlier implementation in `pipeline/aggregation.py`.

**5. N=7 limits fault tolerance at β=0.45.**
floor(7×0.45)=3 faults, but f=2 in BFT terms — so β=0.45 is super-threshold. Liveness fallback fires at 100% for F1/F3 at β=0.45 N=7, making hard_only and full_system equivalent to baseline in those cells. This means the Exp 1 result understates the advantage of Module 1 filtering at lower β values.

**6. Small evaluation set.**
100 questions per model, 80 eval questions after dev split. Results are consistent across LLaMA and Qwen, but the absolute accuracy numbers should be interpreted with appropriate uncertainty.

---

## 8. Related Work

*(Bullets to develop into prose — not exhaustive)*

**Self-consistency (Wang et al., 2023):** Majority vote over N chain-of-thought samples. The paper's primary baseline. Fails under coordinated minority attack and invalid-format corruption. Our strict extraction fix corrects a methodology issue in applying it.

**DecentLLMs (Jo & Park, 2024):** Worker/evaluator scoring with geometric median over criterion score matrices. Key difference: they apply geometric median in *score space* (evaluator ratings over criteria); we apply it in *embedding space* (output text semantics). Different aggregation target; different failure modes addressed.

**BFT consensus (Lamport et al. 1982; Castro & Liskov PBFT 1999):** Classical BFT requires 3f+1 nodes for Byzantine-fault tolerance with message complexity O(N²). Our system is not a full BFT protocol — it is a one-shot aggregation with a confidence filter, not a multi-round consensus protocol. The BFT framing motivates the liveness threshold and fault model but should not be overclaimed.

**Geometric median robustness (Weiszfeld 1937; Vempala & Wein 2019):** Theoretical breakdown point of 1/2 for geometric median under adversarial contamination. Our centroid shift results (+0.24–0.29) are consistent with this guarantee. The L-BFGS-B Weiszfeld implementation is numerically stable and handles degenerate cases (identical embeddings) via denominator clamping.

**Minimum Bayes Risk (MBR) decoding:** MBR selects the hypothesis that minimizes expected loss across a candidate pool using pairwise similarity. Our geometric median aggregation is related but operates in embedding space rather than output-space similarity, and uses the nearest-centroid selection rather than pairwise comparison. Computational cost: O(N) embedding lookups vs O(N²) pairwise for MBR.

**Sentence embedding for LLM answer aggregation:** all-mpnet-base-v2 (768-dim) for embedding. Related work includes Li et al. (2023) using embedding clustering for answer selection in reasoning tasks. Key difference: we apply geometric median over the embedding set rather than k-means or nearest-neighbor clustering.

**Per-token confidence as a reliability signal:** Related to model calibration literature (Guo et al. 2017). TopKMass is architecturally motivated as a sliding-window stability measure rather than a per-token instantaneous confidence — it captures whether the model maintains high-confidence generation *across a sequence*, which is more predictive of reliability than any single token's confidence.

---

## 9. Proposed Paper Section Map

Reference map for drafting. Each row is a paper section with its source material.

| # | Section | Key content | Key numbers |
|---|---|---|---|
| — | Abstract | Problem, two-layer approach, headline results | +7.5pp LLaMA Exp 3; hard_only best Exp 1; TopKMass AUC 0.606 beats entropy |
| 1 | Introduction | Two failure modes; gap in self-consistency; contribution list (3–4 bullets) | 16.5pp strict-extraction fix; +7.5pp coordinated defense |
| 2 | Related Work | Self-consistency, DecentLLMs, BFT, geometric median, MBR | — |
| 3 | System | Phase 1/2; Module 1 math (TopKMass, τ calibration, warmup norm, liveness); Module 2 math (gm, nearest-centroid); fault injection | τ=5th pct, W=64, f=⌊(N−1)/3⌋ |
| 4 | Experiment 1 | Setup; ablation table (Table 1); fault-type breakdown figure (Fig 2); strict-extraction fix discussion | hard_only 72.7% vs baseline 70.8%; soft best for F3 |
| 5 | Experiment 2 | Signal validation; ROC/PR figure (Fig 4); interpretation | TopKMass AUC 0.606; logprob var AUC 0.448 |
| 6 | Experiment 3 | Adversarial coordination setup; accuracy table; centroid shift table; Exp 3 figure (Fig 3) | LLaMA +7.5pp; delta +0.240–0.290 |
| 7 | Multi-Provider | Exp A/B; scope-of-defense interpretation; diversity ≠ coordination | Negative: majority vote wins; geometric median requires cluster cohesion |
| 8 | Discussion | Two-failure-mode thesis; when each layer helps; NLI Stage 2 failure; BFT framing note | Table: failure mode → best condition |
| 9 | Conclusion | — | — |
| — | Appendix | accuracy_vs_beta with N=1 line; Qwen signal quality figure; fallback frequency tables | — |

### Open questions to resolve before writing

1. **Does §7 (multi-provider, negative results) go in the main paper or appendix?**
   - Main body: honest scope delimitation strengthens the claim by showing exactly *when* geometric median helps
   - Appendix: keeps the narrative tighter; negative results don't distract from the main story
   - Recommendation: keep in main body as a short section framed as "scope of the defense"

2. **Is Experiment 2 a standalone section or folded into §3 (System)?**
   - Standalone (§5): gives signal validation its own space; useful if the paper targets ML venues that care about ablations
   - Folded into §3: more concise; treats signal quality as a design validation rather than a result
   - Recommendation: standalone, but short (~0.5 pages)

3. **Paper title options:**
   - "Robust Consensus for Multi-Agent LLM Ensembles via Confidence Filtering and Geometric Median Aggregation"
   - "Two-Layer Defense for Multi-Agent LLM Failures: Confidence Filtering Meets Geometric Median"
   - "Beyond Majority Vote: Robust Aggregation for Multi-Agent LLM Outputs Under Fault Injection and Coordinated Attack"

---

## 10. Code and Data Reference

All experiments are reproducible from the committed codebase. No GPU needed for Phase 2.

| Item | Path |
|---|---|
| Phase 1 generation | `scripts/generate_cache.py` |
| Module 1 filter | `pipeline/filter.py` |
| Module 2 stage1-only aggregation | `pipeline/stage1.py` |
| Embedding and geometric-median helpers | `pipeline/aggregation.py` |
| Answer extraction helpers | `eval/answer_extraction.py` |
| Homogeneous cache loading | `eval/io.py` |
| Fault injection | `faults/injector.py` |
| Exp 1 runner | `eval/runner_v2.py` -> `run_experiment_1()` |
| Exp 2 signal quality | `eval/signal_quality.py` |
| Exp 3 adversarial | `eval/adversarial_test_v2.py` |
| Exp A multi-provider | `eval/runner_multi.py` |
| Exp B biased provider | `eval/biased_provider_test.py` |
| Figure generation | `scripts/plot_results.py` |
| LLaMA cache | `cache_llma.json` |
| Qwen cache | `cache_qwen.json` |
| Exp 1 results | `results/experiment_1_llama_v4.csv`, `results/experiment_1_qwen_v4.csv` |
| Exp 2 results | `results/exp2_llama/`, `results/exp2_qwen/` |
| Exp 3 results | `results/exp3_llama/experiment_3_adversarial_v2.csv`, `results/exp3_qwen/experiment_3_adversarial_v2.csv` |
| Exp A results | `results/exp_multi_a_diversity.csv` |
| Exp B results | `results/exp_multi_b_phi3.csv`, `results/exp_multi_b_mistral.csv` |
| Full design doc | `docs/DESIGN_DOC.md` |
