# Project: Multi-Agent LLM Consensus Pipeline

**Core Directive:** You are a Senior Distributed Systems Engineer. Before making architectural decisions, referencing data structures, or building new modules, you MUST read `docs/DESIGN_DOC.md`. Do not hallucinate implementations outside of those requirements.

## Architectural Boundaries & Rules

1. **Strict Decoupling (The GPU Rule):** - **Phase 1 (Generation):** `vLLM` is ONLY allowed in offline generation scripts to build the cache.
   - **Phase 2 (Evaluation):** The pipeline (`coordination/`, `pipeline/`, `eval/`) runs ENTIRELY on the cached JSON data. **Never import `vllm` in Phase 2 scripts.**
2. **Data Contracts:** All data passed between modules must strictly adhere to the `AgentGeneration` and `ConsensusResult` dataclasses defined in `docs/DESIGN_DOC.md`.
3. **Concurrency:** All orchestration must use Python 3.11 `asyncio` patterns.
4. **Type Safety:** Enforce strict Python static type hints (`->`, `Optional`, `List`, `Dict`, etc.) across all functions and classes.
5. **Math Accuracy:** For mathematical implementations (TopKMass, Geometric Median), refer exactly to the formulas provided in the design documents or ask for them if missing.

## Canonical vs. Legacy Files (Paper Reference Guide)

**Canonical (cite these):**
- `pipeline/filter.py`, `pipeline/aggregation.py` — Module 1 (TopKMass) and Module 2 (geometric median nearest-centroid)
- `eval/runner_v2.py` — Exp 1 (fault injection ablation, final strict extraction + stage1_only)
- `eval/signal_quality.py` — Exp 2 (TopKMass vs entropy vs logprob variance)
- `eval/adversarial_test_v2.py` — Exp 3 (adversarial coordination, produces `_v2.csv`)
- `eval/runner_multi.py` — Exp A (multi-provider diversity)
- `eval/biased_provider_test.py` — Exp B (biased provider stress test)
- Results: `experiment_1_{llama,qwen}_v4.csv`, `exp3_{llama,qwen}/experiment_3_adversarial_v2.csv`, `exp2_{llama,qwen}/`, `exp_multi_a_diversity.csv`, `exp_multi_b_{phi3,mistral}.csv`

**Legacy (never cite, never reference in paper):**
- `pipeline_v2/` — distance-weighted majority vote; regressed under adversarial conditions, replaced by stage1_only
- `pipeline_multi/` — per-provider filter variant; superseded, not used in final experiments
- `eval/runner.py` — original Exp 1 runner; no strict extraction, broken at high beta
- `eval/adversarial_test.py` — produces `experiment_3_adversarial.csv` (no `_v2`); superseded by `adversarial_test_v2.py`
- `eval/baselines.py`, `eval/decent_baseline.py` — early helpers, logic folded into runner_v2.py
- Results: any Exp 1 CSV without `_v4` suffix, `exp3_smoke/`, `smoke_*.csv`, root-level `experiment_2_signals.csv`

## Workflow Commands
- Run tests: `pytest tests/`
- Run type checker: `mypy .`
- Format code: `black . && isort .`

## Claude Code Protocols
- **Formatting Constraints:** Do NOT use em dashes (`—` or `--`) in your conversational text, code comments, documents/writing, or documentation updates. Use commas, parentheses, or separate sentences instead.
- **Plan Mode:** For any task touching 3 or more files, or involving complex math/async logic, use Plan Mode first to draft a strategy.
- **Write Tests First:** Before implementing complex logic (e.g., sliding windows, tensor batching), write the `pytest` file first to verify the math.
- **Stay Concise:** When responding, do not over-explain. Acknowledge instructions briefly and write the code.
- **Living Documentation:** Treat `docs/DESIGN_DOC.md` as the ultimate source of truth. When you complete a major module, or right before a `/compact` command is run, automatically update the `DESIGN_DOC.md` to reflect the current state of the architecture, newly discovered edge cases, and cross off completed steps.
- **Confidence Check:** Do not make any changes, until you have 95% confidence that you know what to build ask me follow up questions until you have that confidence.