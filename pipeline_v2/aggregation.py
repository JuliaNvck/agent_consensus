from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from models import AgentGeneration

_EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

_embed_model: Optional[object] = None


def _get_embed_model() -> object:
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer as _ST
        _embed_model = _ST(_EMBED_MODEL_NAME)
    return _embed_model


def _embed(texts: List[str]) -> np.ndarray:
    return _get_embed_model().encode(texts, convert_to_numpy=True)  # type: ignore[union-attr]


def _geometric_median(embeddings: np.ndarray) -> np.ndarray:
    """L1 geometric median via L-BFGS-B minimisation of Σ‖x_i − y‖₂.

    Uses an analytic gradient with a small epsilon floor on distances to handle
    degenerate cases (duplicate embeddings) without division by zero.
    """
    def _obj(y: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(embeddings - y, axis=1)))

    def _grad(y: np.ndarray) -> np.ndarray:
        diffs = embeddings - y                                    # (N, D)
        dists = np.linalg.norm(diffs, axis=1, keepdims=True)     # (N, 1)
        dists = np.maximum(dists, 1e-10)
        return -np.sum(diffs / dists, axis=0)                     # (D,)

    x0 = embeddings.mean(axis=0)
    result = minimize(_obj, x0, jac=_grad, method="L-BFGS-B")
    return result.x


def _infer_is_numeric(texts: List[str]) -> bool:
    """Infer whether outputs are GSM8K-style (numeric) or StrategyQA-style (yes/no).

    Heuristic: if more than half of outputs contain 3 or more distinct number
    tokens, classify as numeric. GSM8K agents always produce a multi-step
    computation with many numbers; StrategyQA agents may reference 0-2 numbers
    incidentally in their reasoning but rarely 3+.
    """
    if not texts:
        return False
    numeric_count = sum(
        1 for t in texts if len(re.findall(r"\$?[\d,]+", t)) >= 3
    )
    return numeric_count > len(texts) * 0.5


def _extract_answer_key(text: str, is_numeric: bool) -> str:
    """Extract a comparable answer key from a single output text.

    is_numeric=True  → last number token (strips $ and commas), e.g. '42'.
    is_numeric=False → first yes/no word (case-insensitive), e.g. 'yes'.
    Returns '' if no match found (e.g. empty crash-agent output).
    """
    if is_numeric:
        nums = re.findall(r"\$?[\d,]+", text)
        return nums[-1].replace("$", "").replace(",", "") if nums else ""
    m = re.search(r"\b(yes|no)\b", text.lower())
    return m.group(1) if m else ""


async def aggregate(admitted: List[AgentGeneration]) -> Tuple[str, bool]:
    """Module 2 v2: distance-weighted majority vote.

    Replaces the NLI-based candidate selection from v1 with a principled
    soft-voting scheme. Each admitted agent casts a vote for its answer,
    weighted by exp(-distance_to_median / mean_distance). Agents close to
    the geometric median (the consensus centre) vote with full weight;
    outliers — crash agents (empty embeddings), Byzantine agents (wrong
    content), and drifters (low-confidence outputs) — are automatically
    downweighted because their embeddings diverge from the clean cluster.

    At β=0 (all agents clean), distances are roughly uniform, so weights are
    nearly equal and the result approximates plain majority vote.
    At β>0 (faulty agents present), the geometric median sits in the clean
    cluster; faulty agents are far from it and receive low weight.

    Args:
        admitted: Agents admitted by Module 1 filter, or all agents on a
                  BFT liveness fallback.

    Returns:
        (final_answer, is_low_confidence)
        final_answer:       Full output text of one agent whose extracted answer
                            key won the weighted vote.
        is_low_confidence:  True when the winning answer received < 50% of
                            total vote weight, indicating genuine disagreement.
    """
    if not admitted:
        return "", False
    if len(admitted) == 1:
        return admitted[0].output_text, False

    texts = [gen.output_text for gen in admitted]

    # Stage 1: embed → geometric median → proximity weights
    embeddings = _embed(texts)                                    # (N, D)
    median = _geometric_median(embeddings)                        # (D,)
    dists = np.linalg.norm(embeddings - median, axis=1)          # (N,)

    mean_d = float(np.mean(dists))
    if mean_d < 1e-10:
        mean_d = 1.0  # degenerate case: all embeddings identical
    weights = np.exp(-dists / mean_d)                            # w_i ∈ (0, 1]

    # Stage 2: infer answer type, extract keys, accumulate weighted votes
    is_numeric = _infer_is_numeric(texts)

    answer_weights: Dict[str, float] = {}
    answer_rep: Dict[str, str] = {}  # representative full output per key

    for gen, w in zip(admitted, weights):
        key = _extract_answer_key(gen.output_text, is_numeric)
        answer_weights[key] = answer_weights.get(key, 0.0) + w
        if key not in answer_rep:
            answer_rep[key] = gen.output_text

    if not answer_weights:
        # Degenerate: no answers extractable (all agents crashed)
        nearest_idx = int(np.argmin(dists))
        return texts[nearest_idx], True

    best_key = max(answer_weights, key=lambda k: answer_weights[k])
    total_weight = float(sum(answer_weights.values()))
    is_low_confidence = bool(answer_weights[best_key] / total_weight < 0.5)

    return answer_rep[best_key], is_low_confidence
