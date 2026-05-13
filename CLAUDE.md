# Project: Multi-Agent LLM Consensus Pipeline

**Core Directive:** You are a Senior Distributed Systems Engineer. Before making architectural decisions, referencing data structures, or building new modules, you MUST read `docs/DESIGN_DOC.md`. Do not hallucinate implementations outside of those requirements.

## Architectural Boundaries & Rules

1. **Strict Decoupling (The GPU Rule):** - **Phase 1 (Generation):** `vLLM` is ONLY allowed in offline generation scripts to build the cache.
   - **Phase 2 (Evaluation):** The pipeline (`coordination/`, `pipeline/`, `eval/`) runs ENTIRELY on the cached JSON data. **Never import `vllm` in Phase 2 scripts.**
2. **Data Contracts:** All data passed between modules must strictly adhere to the `AgentGeneration` and `ConsensusResult` dataclasses defined in `docs/DESIGN_DOC.md`.
3. **Concurrency:** All orchestration must use Python 3.11 `asyncio` patterns.
4. **Type Safety:** Enforce strict Python static type hints (`->`, `Optional`, `List`, `Dict`, etc.) across all functions and classes.
5. **Math Accuracy:** For mathematical implementations (TopKMass, Geometric Median), refer exactly to the formulas provided in the design documents or ask for them if missing.

## Paper Reference Map

**Primary homogeneous experiments:**
- `pipeline/filter.py`: canonical Module 1 TopKMass filter.
- `pipeline/stage1.py`: shared geometric median nearest-centroid (`stage1_only`) wrapper used by final homogeneous runners.
- `eval/io.py`: shared homogeneous cache loader copied from the final runner behavior.
- `eval/answer_extraction.py`: shared answer extraction and strict answer-voting helpers.
- `eval/runner_v2.py`: canonical Exp 1 V4 runner. Uses shared strict answer extraction for `baseline` and `hard_only`; implements `full_system` via `pipeline.stage1.nearest_centroid_text`.
- `eval/adversarial_test_v2.py`: canonical Exp 3 runner for `experiment_3_adversarial_v2.csv`. Its `stage1_only` condition uses the shared geometric median nearest-centroid method. Its `full_pipeline` condition is the legacy weighted-vote ablation from `pipeline_v2`, not NLI.
- `eval/signal_quality.py`: canonical Exp 2 runner. It uses shared cache loading and answer extraction, but currently reports TopKMass using the full trajectory mean (`traj.mean()`), not the post-warmup stable-region mean used for filtering/calibration.
- `eval/baselines.py`: keeps raw majority voting and soft-weighted geometric median baselines; answer-voting helpers are re-exported from `eval.answer_extraction`.
- Results: `results/experiment_1_{llama,qwen}_v4.csv`, `results/exp3_{llama,qwen}/experiment_3_adversarial_v2.csv`, `results/exp2_{llama,qwen}/`.

**Shared helpers and older ablations:**
- `pipeline/aggregation.py`: contains reusable `_embed` and `_geometric_median` helpers used by the final stage1-only path, plus the older public `aggregate()` implementation with NLI verification. Do not describe reported Exp 3 `_v2.csv` `full_pipeline` rows as NLI.
- `pipeline_v2/aggregation.py`: distance-weighted vote ablation. It regressed under coordinated adversarial conditions, but it is still referenced by Exp 3 `full_pipeline` rows in `eval/adversarial_test_v2.py`.
- `eval/runner.py` and `eval/adversarial_test.py`: older runners. Useful for provenance only; do not cite as final Exp 1 or Exp 3 results.

**Multi-provider experiments:**
- `pipeline_multi/`: relevant for Exp A/B only. It is not the primary homogeneous-system implementation, but it is part of the reported multi-provider results.
- `eval/runner_multi.py`: Exp A, multi-provider diversity.
- `eval/biased_provider_test.py`: Exp B, biased-provider stress test.
- Results: `results/exp_multi_a_diversity.csv`, `results/exp_multi_b_{phi3,mistral}.csv`.

**Do not cite as final paper results:**
- Any Exp 1 CSV without `_v4`.
- Root-level `results/experiment_2_signals.csv`.
- `results/exp3_smoke/` and `smoke_*.csv`.
- `pipeline_v2` as the final method; cite it only as the failed weighted-vote ablation when explaining Exp 3 `full_pipeline`.

## Workflow Commands
- Run tests: `pytest tests/`
- Run type checker: `mypy .`
- Format code: `black . && isort .`

## Claude Code Protocols
- **Formatting Constraints:** Do NOT use em dashes or double hyphen separators in your conversational text, code comments, documents/writing, or documentation updates. Use commas, parentheses, or separate sentences instead.
- **Plan Mode:** For any task touching 3 or more files, or involving complex math/async logic, use Plan Mode first to draft a strategy.
- **Write Tests First:** Before implementing complex logic (e.g., sliding windows, tensor batching), write the `pytest` file first to verify the math.
- **Stay Concise:** When responding, do not over-explain. Acknowledge instructions briefly and write the code.
- **Living Documentation:** Treat `docs/DESIGN_DOC.md` as the ultimate source of truth. When you complete a major module, or right before a `/compact` command is run, automatically update the `DESIGN_DOC.md` to reflect the current state of the architecture, newly discovered edge cases, and cross off completed steps.
- **Confidence Check:** Do not make any changes, until you have 95% confidence that you know what to build ask me follow up questions until you have that confidence.
