# Multi-Provider Consensus Pipeline — Design Document

## Goal

Build a multi-provider consensus system using 4 open-weight models as independent
"providers." The core claim: geometric median aggregation over heterogeneous
provider outputs is more robust to per-provider systematic bias than (a) any
single provider alone, and (b) naive plurality voting.

This extends the existing single-model self-consistency pipeline with a structurally
more realistic adversarial scenario: one model family that systematically fails on
a subset of questions.

---

## Motivation

The existing pipeline's "adversarial agent" framing injected faults synthetically
(β-fraction of agents get F1/F2/F3 faults). A **biased provider** is structurally
identical but more realistic: a smaller or differently-trained model family
systematically produces wrong answers on a subset of questions. Geometric median
over a mixed provider pool should absorb this without knowing which provider is biased.

---

## Providers

| Provider tag | Model | VRAM | Cache file | Status |
|---|---|---|---|---|
| `llama` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | ~16 GB | `cache_llma.json` | Existing |
| `qwen` | `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | `cache_qwen.json` | Existing |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | ~14 GB | `cache_mistral.json` | **Needs generation** |
| `phi3` | `microsoft/Phi-3-mini-4k-instruct` | ~7.6 GB | `cache_phi3.json` | **Needs generation** |

Run one model at a time (single L4 24GB GPU). Each cache = ~30 min on EC2.

---

## Cache Generation (EC2 commands)

```bash
# Step 1 — Generate new provider caches (run one at a time)
python -m scripts.generate_cache_multi \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --provider-name mistral \
    --n-questions 50 \
    --output cache_mistral.json

python -m scripts.generate_cache_multi \
    --model microsoft/Phi-3-mini-4k-instruct \
    --provider-name phi3 \
    --n-questions 50 \
    --output cache_phi3.json

# Step 2 — Mix into a single 7-agent pool (2+2+2+1 split)
python -m scripts.mix_caches \
    --inputs cache_llma.json:llama cache_qwen.json:qwen \
             cache_mistral.json:mistral cache_phi3.json:phi3 \
    --agents-per-provider 2 2 2 1 \
    --output cache_mixed.json \
    --seed 42
```

The mixed cache has N=7 agents per question from 4 providers.
Provider split rationale: equal representation for the 3 large models (7-8B);
Phi-3 mini gets 1 agent because it is smaller and will serve as the "biased provider"
in Experiment B.

---

## Module 1: Per-Provider TopKMass Filter (`pipeline_multi/filter.py`)

Different model families have different TopKMass score distributions — a global
threshold τ would be miscalibrated for Phi-3 mini vs LLaMA.

**Calibration**: for each provider, set τ_provider = 25th-percentile of TopKMass
scores across that provider's dev-set agents.  This admits 75% of each provider's
agents even in the worst case.

```python
def calibrate_tau(agents: List[AgentGeneration]) -> Dict[str, float]:
    # Groups agents by provider, returns {provider: percentile_25_topkmass}
```

Agents without a provider tag (legacy caches), or whose provider is missing from
the calibrated map, fall back to `GLOBAL_TAU_FALLBACK = 0.5`.

---

## Module 2: Geometric Median Nearest-Centroid (`pipeline_multi/aggregation.py`)

Algorithm:
1. Filter admitted agents using per-provider τ (Module 1).
2. If < 2 admitted: fall back to majority vote on unfiltered pool (liveness).
3. Embed admitted outputs with `sentence-transformers/all-mpnet-base-v2`.
4. Compute geometric median (L-BFGS-B, reused from `pipeline.aggregation`).
5. Return the admitted agent text nearest to the geometric median.

**Why nearest-centroid, not weighted vote?**  
V3 experiments showed distance-weighted majority vote is the *worst* condition under
adversarial inputs (56.25% vs 70% for nearest-centroid). Diverse correct answers
split weighted votes while a coordinated wrong-answer cluster concentrates them.
Nearest-centroid is immune to this fragmentation.

---

## Experiments

### Experiment A: Provider Diversity (`eval/runner_multi.py`)

**Question**: Does a multi-provider ensemble beat the best single provider?

| Condition | Agents used | Aggregation |
|---|---|---|
| `single_llama` | llama only (2 agents) | geometric median nearest-centroid |
| `single_qwen` | qwen only (2 agents) | geometric median nearest-centroid |
| `single_mistral` | mistral only (2 agents) | geometric median nearest-centroid |
| `single_phi3` | phi3 only (1 agent) | geometric median nearest-centroid |
| `multi_provider` | all 7 mixed agents | geometric median nearest-centroid |

**Metric**: accuracy gap = multi_provider - best_single.

**Expected result**: multi_provider ≥ best_single, gap ~1-3%. If negative, it means
Phi-3 mini noise outweighs diversity benefit.

```bash
python -m eval.runner_multi --cache cache_mixed.json --output results/exp_multi_a_diversity.csv
```

**Output CSV columns**: `condition, accuracy, low_confidence_freq, n_admitted_mean`

---

### Experiment B: Biased Provider (`eval/biased_provider_test.py`)

**Question**: Can geometric median absorb a systematically wrong provider?

| Condition | Description |
|---|---|
| `no_phi3` | Oracle: remove phi3 agents before aggregation |
| `majority_vote_all` | Naive plurality vote on raw output_text, all 7 agents |
| `geometric_median_all` | Our system: Module 1 filter + geometric median, all 7 agents |

Evaluated under two bias types:
- **natural**: use phi3 answers as-is (measures real systematic error from smaller model)
- **injected**: replace all phi3 agents with F1 crash faults (empty output, no logprobs)

**Expected result**: geometric_median_all ≥ majority_vote_all under injected bias
(F1 crash agents have empty logprobs → filtered by Module 1 → effectively same as no_phi3).
Under natural bias: result depends on how systematically wrong phi3 is.

```bash
python -m eval.biased_provider_test --cache cache_mixed.json --output results/exp_multi_b_biased_provider.csv
```

**Output CSV columns**: `condition, bias_type, accuracy, fallback_freq`

---

## File Map

| File | Purpose |
|---|---|
| `pipeline_multi/__init__.py` | Package init |
| `pipeline_multi/DESIGN_MULTI.md` | This document |
| `pipeline_multi/filter.py` | Module 1: per-provider τ calibration |
| `pipeline_multi/aggregation.py` | Module 2: geometric median nearest-centroid |
| `scripts/generate_cache_multi.py` | Phase 1: provider-tagged vLLM cache generation |
| `scripts/mix_caches.py` | Phase 1: merge provider caches into mixed pool |
| `eval/runner_multi.py` | Experiment A runner |
| `eval/biased_provider_test.py` | Experiment B runner |
| `tests/test_aggregation_multi.py` | Unit tests for filter + aggregation |
| `tests/test_mixer.py` | Unit tests for mix_caches |

---

## Known Limitations

1. **Mistral ≈ LLaMA**: both 7-8B instruction-tuned models with similar pretraining.
   Their answers may be near-identical, reducing diversity benefit.

2. **Phi-3 mini is small**: at 3.8B, it is consistently weaker on GSM8K math
   reasoning. This makes it a natural "biased provider" but also means Experiment A
   may show a small accuracy drop from including it.

3. **2-2-2-1 split is fixed**: the unequal contribution (phi3 gets 1 agent vs 2 for
   others) was chosen to reflect realistic proportional deployment, but alternative
   splits could be explored.

4. **No logprob spoofing**: unlike the original adversarial framing, the biased
   provider here sends *real* logprobs (natural) or *empty* logprobs (injected).
   This is the realistic threat model.

---

## Progress Tracker

- [x] Extend `models.py` with `model_id` and `provider` fields
- [x] `scripts/generate_cache_multi.py`
- [x] `scripts/mix_caches.py`
- [x] `pipeline_multi/filter.py`
- [x] `pipeline_multi/aggregation.py`
- [x] `eval/runner_multi.py`
- [x] `eval/biased_provider_test.py`
- [x] `tests/test_aggregation_multi.py` — 33 tests, all passing
- [x] `tests/test_mixer.py`
- [ ] Generate `cache_mistral.json` on EC2
- [ ] Generate `cache_phi3.json` on EC2
- [ ] Run `scripts/mix_caches.py` to create `cache_mixed.json`
- [ ] Run Experiment A → `results/exp_multi_a_diversity.csv`
- [ ] Run Experiment B → `results/exp_multi_b_biased_provider.csv`
