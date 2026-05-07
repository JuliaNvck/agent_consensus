#!/usr/bin/env python3
"""Phase 1: Merge provider-specific cache files into a single mixed-pool cache.

Usage:
    python -m scripts.mix_caches \\
        --inputs cache_llma.json:llama cache_qwen.json:qwen \\
                 cache_mistral.json:mistral cache_phi3.json:phi3 \\
        --agents-per-provider 2 2 2 1 \\
        --output cache_mixed.json \\
        --seed 42

The --agents-per-provider list must have the same length as --inputs, in the same
order.  For each question, `k` agents are sampled uniformly at random (without
replacement) from the corresponding provider cache.

Existing caches without model_id/provider fields are retrofitted using the tag
supplied in the --inputs argument (e.g. "cache_llma.json:llama" injects
provider="llama").
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Dict, List, Optional, Tuple


def _load_cache(path: str) -> Dict[str, dict]:
    """Load a cache file and return a mapping from question_id → record."""
    with open(path) as fh:
        data = json.load(fh)
    return {r["question_id"]: r for r in data["questions"]}


def _retrofit_provider(generations: List[dict], provider: str, model_id: Optional[str]) -> List[dict]:
    """Inject provider/model_id into generation dicts that are missing them."""
    for gen in generations:
        if not gen.get("provider"):
            gen["provider"] = provider
        if model_id and not gen.get("model_id"):
            gen["model_id"] = model_id
    return generations


def _parse_input_spec(spec: str) -> Tuple[str, str]:
    """Parse 'path/to/cache.json:provider_tag' → (path, tag)."""
    parts = spec.rsplit(":", 1)
    if len(parts) != 2 or not parts[1]:
        print(f"ERROR: --inputs entry must be 'path:provider_tag', got: {spec!r}", file=sys.stderr)
        sys.exit(1)
    return parts[0], parts[1]


def mix_question(
    records_by_provider: Dict[str, dict],
    agents_per_provider: Dict[str, int],
    provider_order: List[str],
    rng: random.Random,
) -> List[dict]:
    """Sample agents from each provider for a single question."""
    agents: List[dict] = []
    for provider in provider_order:
        n = agents_per_provider[provider]
        record = records_by_provider[provider]
        available = record["generations"]
        if len(available) < n:
            # Sample with replacement if fewer agents than requested
            sampled = rng.choices(available, k=n)
        else:
            sampled = rng.sample(available, k=n)
        agents.extend(sampled)
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-provider cache files into a single mixed-pool cache."
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True, metavar="PATH:PROVIDER",
        help="Space-separated list of 'cache_path:provider_tag' entries.",
    )
    parser.add_argument(
        "--agents-per-provider", nargs="+", type=int, required=True,
        metavar="N",
        help="Number of agents to sample per provider per question (same order as --inputs).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Destination path for the mixed cache JSON.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    args = parser.parse_args()

    if len(args.inputs) != len(args.agents_per_provider):
        print(
            f"ERROR: --inputs ({len(args.inputs)} entries) and "
            f"--agents-per-provider ({len(args.agents_per_provider)} entries) must have the same length.",
            file=sys.stderr,
        )
        sys.exit(1)

    provider_order: List[str] = []
    caches: Dict[str, Dict[str, dict]] = {}  # provider → {question_id → record}

    for spec, n_agents in zip(args.inputs, args.agents_per_provider):
        path, provider = _parse_input_spec(spec)
        print(f"Loading {path} as provider={provider!r} ({n_agents} agents/q)...")
        cache = _load_cache(path)
        # Retrofit provider tags into generation records
        for record in cache.values():
            _retrofit_provider(record["generations"], provider, model_id=None)
        caches[provider] = cache
        provider_order.append(provider)

    agents_per_provider: Dict[str, int] = {
        p: n for p, n in zip(provider_order, args.agents_per_provider)
    }

    # Find the intersection of question_ids across all providers
    all_ids = [set(caches[p].keys()) for p in provider_order]
    common_ids: List[str] = sorted(set.intersection(*all_ids))
    missing_per_provider = {
        p: sorted(set(caches[p].keys()) - set(common_ids)) for p in provider_order
    }
    for provider, missing in missing_per_provider.items():
        if missing:
            print(
                f"WARNING: provider={provider!r} is missing {len(missing)} question(s) "
                f"from the intersection. Skipping: {missing[:5]}{'...' if len(missing) > 5 else ''}",
                file=sys.stderr,
            )

    print(f"Mixing {len(common_ids)} questions × {sum(args.agents_per_provider)} agents each...")

    rng = random.Random(args.seed)
    mixed_questions: List[dict] = []

    for qid in common_ids:
        records_by_provider = {p: caches[p][qid] for p in provider_order}
        ground_truth = records_by_provider[provider_order[0]]["ground_truth"]
        agents = mix_question(records_by_provider, agents_per_provider, provider_order, rng)
        mixed_questions.append({
            "question_id": qid,
            "ground_truth": ground_truth,
            "generations": agents,
        })

    with open(args.output, "w") as fh:
        json.dump({"questions": mixed_questions}, fh, indent=2)

    total_agents = len(mixed_questions) * sum(args.agents_per_provider)
    print(f"Wrote {len(mixed_questions)} questions, {total_agents} total agent records → {args.output}")


if __name__ == "__main__":
    main()
