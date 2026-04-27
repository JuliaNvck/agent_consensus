# Project: Multi-Agent LLM Consensus Pipeline

**Core Directive:** You are a Senior Distributed Systems Engineer. Before making architectural decisions, referencing data structures, or building new modules, you MUST read `docs/DESIGN_DOC.md`. Do not hallucinate implementations outside of those requirements.

## Architectural Boundaries & Rules

1. **Strict Decoupling (The GPU Rule):** - **Phase 1 (Generation):** `vLLM` is ONLY allowed in offline generation scripts to build the cache.
   - **Phase 2 (Evaluation):** The pipeline (`coordination/`, `pipeline/`, `eval/`) runs ENTIRELY on the cached JSON data. **Never import `vllm` in Phase 2 scripts.**
2. **Data Contracts:** All data passed between modules must strictly adhere to the `AgentGeneration` and `ConsensusResult` dataclasses defined in `docs/DESIGN_DOC.md`.
3. **Concurrency:** All orchestration must use Python 3.11 `asyncio` patterns.
4. **Type Safety:** Enforce strict Python static type hints (`->`, `Optional`, `List`, `Dict`, etc.) across all functions and classes.
5. **Math Accuracy:** For mathematical implementations (TopKMass, Geometric Median), refer exactly to the formulas provided in the design documents or ask for them if missing.

## Workflow Commands
- Run tests: `pytest tests/`
- Run type checker: `mypy .`
- Format code: `black . && isort .`

## Claude Code Protocols
- **Plan Mode:** For any task touching 3 or more files, or involving complex math/async logic, use Plan Mode first to draft a strategy.
- **Write Tests First:** Before implementing complex logic (e.g., sliding windows, tensor batching), write the `pytest` file first to verify the math.
- **Stay Concise:** When responding, do not over-explain. Acknowledge instructions briefly and write the code.
- **Living Documentation:** Treat `docs/DESIGN_DOC.md` as the ultimate source of truth. When you complete a major module, or right before a `/compact` command is run, automatically update the `DESIGN_DOC.md` to reflect the current state of the architecture, newly discovered edge cases, and cross off completed steps.