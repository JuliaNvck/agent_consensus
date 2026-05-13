from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from models import AgentGeneration


def extract_answer(output_text: str, ground_truth: str) -> str:
    """Extract a comparable answer token from a chain-of-thought output.

    StrategyQA (GT in {"yes", "no"}): returns first yes/no found.
    GSM8K (GT is numeric): returns the last number-like token, stripping $ and commas.
    Fallback: returns output_text.strip() unchanged.
    """
    gt = ground_truth.strip().lower()
    if gt in {"yes", "no"}:
        m = re.search(r"\b(yes|no)\b", output_text.lower())
        return m.group(1) if m else output_text.strip()
    numbers = re.findall(r"\$?[\d,]+", output_text)
    if numbers:
        return numbers[-1].replace("$", "").replace(",", "")
    return output_text.strip()


def extract_answer_strict(output_text: str, ground_truth: str) -> Optional[str]:
    """Strict extraction: return None when no valid answer format is found.

    Unlike extract_answer, this never falls back to returning the full output_text.
    Returning None signals that the agent's output should be skipped in a vote.
    """
    gt = ground_truth.strip().lower()
    if gt in {"yes", "no"}:
        m = re.search(r"\b(yes|no)\b", output_text.lower())
        return m.group(1) if m else None
    numbers = re.findall(r"\$?[\d,]+", output_text)
    if numbers:
        return numbers[-1].replace("$", "").replace(",", "")
    return None


def answer_majority_voting(
    agents: List[AgentGeneration],
    ground_truth: str,
) -> str:
    """Vote over extracted answers, not raw output_text."""
    if not agents:
        return ""
    answers = [extract_answer(g.output_text, ground_truth) for g in agents]
    return Counter(answers).most_common(1)[0][0]


def answer_majority_voting_strict(
    agents: List[AgentGeneration],
    ground_truth: str,
) -> str:
    """Strict majority vote: skip agents whose output has no parseable answer.

    Falls back to non-strict answer voting if no agent produces a parseable answer,
    preserving the previous liveness behavior.
    """
    if not agents:
        return ""
    extracted = [extract_answer_strict(g.output_text, ground_truth) for g in agents]
    valid = [a for a in extracted if a is not None]
    if not valid:
        return answer_majority_voting(agents, ground_truth)
    return Counter(valid).most_common(1)[0][0]


_extract_answer = extract_answer
_extract_answer_strict = extract_answer_strict
