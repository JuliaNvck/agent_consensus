#!/usr/bin/env python3
"""Phase 1: Offline vLLM inference → cache.json

GPU Exception: this is the only file in the repo permitted to import vllm and torch.
See docs/DESIGN_DOC.md §6.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import torch  # noqa: F401  (GPU exception — Phase 1 only)
from datasets import load_dataset
from vllm import LLM, SamplingParams

_PAD_LOGPROB: float = -100.0

_GSM8K_PROMPT = "Solve the following math problem step by step, ending with the final answer:\n{question}"
_STRATQA_PROMPT = "Answer the following question with a clear 'yes' or 'no' and briefly explain why:\n{question}"


def _flatten_logprobs(logprobs_per_token: list[dict[int, Any]]) -> list[float]:
    """Flatten vLLM's nested logprob structure to a 1D list of length 5×T.

    vLLM returns List[Dict[token_id, Logprob]] — one dict per output token.
    Module 1 requires a flat list of exactly 5 logprob floats per token position.
    Tokens with fewer than 5 entries are right-padded with _PAD_LOGPROB (-100.0).
    """
    flat: list[float] = []
    for token_dict in logprobs_per_token:
        top5 = sorted(token_dict.values(), key=lambda lp: lp.logprob, reverse=True)[:5]
        values = [lp.logprob for lp in top5]
        while len(values) < 5:
            values.append(_PAD_LOGPROB)
        flat.extend(values)
    return flat


def _load_questions(n: int) -> tuple[list[dict], list[dict]]:
    """Load up to n questions from each dataset, returning (gsm_questions, sqa_questions)."""
    gsm_questions: list[dict] = []
    gsm = load_dataset("gsm8k", "main", split="test")
    for i, row in enumerate(gsm.select(range(min(n, len(gsm))))):
        gsm_questions.append({
            "question_id": f"gsm8k_{i}",
            "prompt": _GSM8K_PROMPT.format(question=row["question"]),
            "ground_truth": row["answer"].split("####")[-1].strip(),
        })

    sqa_questions: list[dict] = []
    sqa = load_dataset("wics/strategy-qa", split="test")
    for i, row in enumerate(sqa.select(range(min(n, len(sqa))))):
        sqa_questions.append({
            "question_id": f"stratqa_{i}",
            "prompt": _STRATQA_PROMPT.format(question=row["question"]),
            "ground_truth": "yes" if row["answer"] else "no",
        })

    return gsm_questions, sqa_questions


def _apply_chat_template(tokenizer: Any, raw_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_records(questions: list[dict], results: list[Any]) -> list[dict]:
    """Convert vLLM RequestOutput objects to the cache.json schema."""
    records = []
    for q, req_out in zip(questions, results):
        generations = [
            {
                "agent_id": f"q{q['question_id']}_a{agent_idx}",
                "output_text": completion.text,
                "token_logprobs": _flatten_logprobs(completion.logprobs or []),
                "is_faulty": False,
                "fault_type": None,
            }
            for agent_idx, completion in enumerate(req_out.outputs)
        ]
        records.append({
            "question_id": q["question_id"],
            "ground_truth": q["ground_truth"],
            "generations": generations,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cache.json via vLLM batched inference (Phase 1)."
    )
    parser.add_argument(
        "--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model ID to load via vLLM.",
    )
    parser.add_argument(
        "--n-questions", type=int, default=50,
        help="Number of questions to sample from each dataset (default: 50).",
    )
    parser.add_argument(
        "--output", default="cache.json",
        help="Destination path for the generated cache file (default: cache.json).",
    )
    args = parser.parse_args()

    gsm_questions, sqa_questions = _load_questions(args.n_questions)

    llm = LLM(model=args.model, max_model_len=4096)
    tokenizer = llm.get_tokenizer()

    gsm_prompts = [_apply_chat_template(tokenizer, q["prompt"]) for q in gsm_questions]
    sqa_prompts = [_apply_chat_template(tokenizer, q["prompt"]) for q in sqa_questions]

    # Separate SamplingParams per dataset to optimise GPU time
    gsm_params = SamplingParams(n=7, temperature=0.7, logprobs=5, max_tokens=256)
    sqa_params = SamplingParams(n=7, temperature=0.7, logprobs=5, max_tokens=128)

    print(f"Generating {len(gsm_prompts)} GSM8K × 7 completions (max_tokens=256)...", flush=True)
    gsm_results = llm.generate(gsm_prompts, gsm_params) if gsm_prompts else []

    print(f"Generating {len(sqa_prompts)} StrategyQA × 7 completions (max_tokens=128)...", flush=True)
    sqa_results = llm.generate(sqa_prompts, sqa_params) if sqa_prompts else []

    all_records = _build_records(gsm_questions, gsm_results) + _build_records(sqa_questions, sqa_results)

    with open(args.output, "w") as fh:
        json.dump({"questions": all_records}, fh, indent=2)

    print(f"Wrote {len(all_records)} questions → {args.output}")


if __name__ == "__main__":
    main()
