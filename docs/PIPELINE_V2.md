# Pipeline V2: Distance-Weighted Majority Vote

## Overview

Pipeline V2 replaces Module 2's NLI-based candidate selection with a
distance-weighted majority vote. Module 1 (the TopKMass reliability filter)
is unchanged. All other infrastructure — cache format, τ calibration, BFT
liveness fallback, fault injection, evaluation harness — is identical to V1,
enabling direct CSV comparison.

---

## What Changed from V1

### Module 2: Removed NLI, Added Weighted Voting

**V1 Module 2 (pipeline/aggregation.py)**

1. Embed all admitted agent outputs.
2. Compute geometric median of embeddings.
3. Sort agents by distance to median.
4. Iterate candidates nearest-first; run bidirectional NLI entailment against
   the second-nearest "reference" agent.
5. Return the first candidate that passes NLI.

**Problem observed:** NLI never fired (low_confidence_frequency ≈ 0 across all
experiments). The system effectively always returned `texts[sorted_idx[0]]`
— the single output text nearest to the geometric median. This is
**nearest-centroid selection**, not a robust aggregator.

Consequence: at β=0 (all agents clean), clean agents produce diverse but
correct outputs. A compact minority wrong-answer cluster can pull the geometric
median toward it, making the nearest-centroid output wrong even when the
majority is correct. This caused a consistent 5–8% accuracy penalty at β=0
relative to majority-vote baseline.

**V2 Module 2 (pipeline_v2/aggregation.py)**

1. Embed all admitted agent outputs.
2. Compute geometric median of embeddings.
3. Compute each agent's proximity weight: `w_i = exp(-dist_i / mean_dist)`.
4. Infer answer type from output content (numeric if >50% of outputs contain
   3+ number tokens; yes/no otherwise).
5. Extract answer key from each output; accumulate weighted votes per key.
6. Return the answer with the highest total weight.

**Why this fixes β=0:** At β=0, all agents are clean and their embeddings
cluster similarly around the geometric median. Distances are approximately
uniform, so all weights are nearly equal, and weighted voting reduces to plain
majority vote. No accuracy penalty.

**Why this preserves β>0 gains:** Faulty agents diverge from the clean cluster:
- F1 (crash): empty output → embedding far from the clean median → low weight.
- F2 (Byzantine): wrong content → embedding in a different semantic region →
  the geometric median sits in the correct cluster (majority); wrong agents are
  farther from it → low weight.
- F3 (drifter): low-confidence, uncertain output → embedding drifts from the
  confident correct cluster → low weight.

In all cases the correct-answer group accumulates higher total weight.

### Removed: NLI Model

`pipeline/aggregation.py` loaded `cross-encoder/nli-deberta-v3-large` (~1.4 GB)
at runtime. V2 removes this dependency entirely. The sentence-transformer model
(`all-mpnet-base-v2`, ~420 MB) is still used for embedding.

### Renamed: `nli_fallback_frequency` → `low_confidence_frequency`

Experiment 3 V2 CSV uses `low_confidence_frequency` to track questions where
the winning answer received <50% of total vote weight. This replaces the V1
column `nli_fallback_frequency` which was always 0 (NLI never rejected).

---

## Algorithm Detail

### Proximity Weight Formula

```
mean_d = mean(||embedding_i - median||)    # self-normalizing
w_i    = exp(-dist_i / mean_d)             # w_i ∈ (0, 1]
```

**At β=0 (clean pool):** all distances are similar (clustered embeddings) →
`dist_i ≈ mean_d` for all i → all weights ≈ `exp(-1) ≈ 0.37` (uniform) →
weighted vote = majority vote.

**At β=0.45, F1:** crash agents have `dist ≈ 3–10 × mean_d` → weights near
zero. Four clean agents at dist≈0 have weights near 1.0. Correct answer
dominates.

**At β=0.45, F2:** even though F2 agents pass Module 1 (spoofed TopKMass=1.0),
their wrong-content embeddings cluster away from the correct-answer region.
Geometric median sits in the clean cluster; F2 agents are far → downweighted.

### Answer Type Inference

Without access to ground_truth at aggregation time, answer type is inferred:

```python
is_numeric = (count of outputs with ≥ 3 number tokens) > 0.5 × N
```

GSM8K outputs always contain a multi-step computation trace with many numbers.
StrategyQA outputs may reference 0–2 numbers incidentally but rarely 3+.
The threshold of 3 separates these distributions cleanly.

### Low-Confidence Signal

```python
is_low_confidence = (winning_weight / total_weight) < 0.5
```

True when no single answer accumulated a majority of the total vote weight.
This indicates genuine agent disagreement and propagates upward as the
`fallback_frequency` flag in Experiment 1 results.

---

## File Structure

```
pipeline_v2/
  __init__.py
  aggregation.py          # New Module 2: weighted voting, no NLI

eval/
  runner_v2.py            # Experiment 1 runner using pipeline_v2
  adversarial_test_v2.py  # Experiment 3 runner using pipeline_v2

tests/
  test_aggregation_v2.py  # 28 tests covering new aggregate logic

docs/
  PIPELINE_V2.md          # This document
```

**Unchanged files (V1 system):**
```
pipeline/aggregation.py   # Original NLI-based Module 2
pipeline/filter.py        # Module 1 — unchanged, shared by both systems
eval/runner.py            # Original Experiment 1 runner
eval/adversarial_test.py  # Original Experiment 3 runner
results/*_v2.csv          # V1 experiment results
```

---

## Running Experiments

All commands run from the project root on EC2. V2 results go to `*_v3.csv`
files so V1 and V2 results coexist for comparison.

### Experiment 1 — Ablation Grid

```bash
# LLaMA — run in tmux window 1
python -m eval.runner_v2 \
    --cache cache_llma.json \
    --output results/experiment_1_llama_v3.csv \
    --include-n1

# Qwen — run in tmux window 2
python -m eval.runner_v2 \
    --cache cache_qwen.json \
    --output results/experiment_1_qwen_v3.csv \
    --include-n1
```

### Experiment 3 — Adversarial Coordination

```bash
# LLaMA — run in tmux window 3
python -m eval.adversarial_test_v2 \
    --cache cache_llma.json \
    --output-dir results/exp3_llama_v3/

# Qwen — run in tmux window 4
python -m eval.adversarial_test_v2 \
    --cache cache_qwen.json \
    --output-dir results/exp3_qwen_v3/
```

All four can run simultaneously on a g6.4xlarge (L4 24 GB) without GPU
contention — the sentence-transformer uses ~420 MB and inference is fast.

### Smoke Test (5 questions, ~2 min)

```bash
python -m eval.runner_v2 \
    --cache cache_llma.json \
    --output /tmp/smoke_v3.csv \
    --n-questions 5
```

**Expected sanity checks on v3 results:**
- `full_system` at β=0 should be within ±2% of `baseline` (gap closes)
- `full_system` at β=0.45 should still beat `baseline` by ≥10% for LLaMA F1/F3
- `fallback_frequency` at β=0, N=7 should remain ≤5% for LLaMA
- `low_confidence_frequency` in Exp 3 may now be non-zero (meaningful signal)

---

## Key Differences: V1 vs V2 Results

| Metric | V1 | V2 (expected) |
|---|---|---|
| β=0 gap (full_system vs baseline) | -5% to -8% LLaMA | ~0% (closes) |
| β=0.45 F1 gain vs baseline | +20% LLaMA | ≥+15% (preserved) |
| β=0.45 F2 gain vs baseline | +15% LLaMA | ≥+10% (preserved) |
| NLI model loaded | Yes (1.4 GB) | No |
| low_confidence ever True | Rarely | Yes (meaningful signal) |
| Exp 3: stage1_only vs full_pipeline | Nearly identical (NLI never fired) | Measurable difference |