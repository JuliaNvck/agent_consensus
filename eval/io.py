from __future__ import annotations

import json
from typing import List, Tuple

from models import AgentGeneration


def load_cache(filepath: str) -> List[Tuple[str, List[AgentGeneration]]]:
    """Load a JSON generation cache.

    Returns a list of (ground_truth, List[AgentGeneration]) pairs, one per question.
    This is the shared Phase 2 cache reader used by the final homogeneous runners.
    """
    with open(filepath) as fh:
        data = json.load(fh)

    result: List[Tuple[str, List[AgentGeneration]]] = []
    for q in data["questions"]:
        ground_truth: str = q["ground_truth"]
        generations = [
            AgentGeneration(
                agent_id=g["agent_id"],
                output_text=g["output_text"],
                token_logprobs=g["token_logprobs"],
                is_faulty=g["is_faulty"],
                fault_type=g.get("fault_type"),
            )
            for g in q["generations"]
        ]
        result.append((ground_truth, generations))
    return result
