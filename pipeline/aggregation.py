from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from models import AgentGeneration

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

_EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-large"

_embed_model: Optional[SentenceTransformer] = None
_nli_tokenizer: Optional[AutoTokenizer] = None
_nli_model: Optional[AutoModelForSequenceClassification] = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer as _ST
        _embed_model = _ST(_EMBED_MODEL_NAME)
    return _embed_model


def _get_nli_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    global _nli_tokenizer, _nli_model
    if _nli_model is None:
        from transformers import (
            AutoModelForSequenceClassification as _Model,
            AutoTokenizer as _Tok,
        )
        _nli_tokenizer = _Tok.from_pretrained(_NLI_MODEL_NAME)
        _nli_model = _Model.from_pretrained(_NLI_MODEL_NAME)
        _nli_model.eval()
    return _nli_tokenizer, _nli_model  # type: ignore[return-value]


def _embed(texts: List[str]) -> np.ndarray:
    return _get_embed_model().encode(texts, convert_to_numpy=True)  # type: ignore[return-value]


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


def _batched_entailment(text_a: str, text_b: str) -> Tuple[bool, bool]:
    """Evaluate (A entails B) and (B entails A) in a single batched forward pass.

    Returns (a_entails_b, b_entails_a).
    """
    import torch

    tokenizer, model = _get_nli_model()

    pairs = [[text_a, text_b], [text_b, text_a]]
    inputs = tokenizer(
        pairs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        logits = model(**inputs).logits  # (2, 3) — single forward pass

    # Resolve entailment label index dynamically from the model's own config
    label_to_idx = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    ent_idx = label_to_idx["entailment"]

    predicted = logits.argmax(dim=-1)  # (2,)
    a_entails_b = bool(predicted[0].item() == ent_idx)
    b_entails_a = bool(predicted[1].item() == ent_idx)
    return a_entails_b, b_entails_a


async def aggregate(admitted: List[AgentGeneration]) -> str:
    """Module 2: extract the highest-fidelity answer from the admitted pool.

    Stage 1 — Cluster (O(N)): embed outputs, compute geometric median, find
    the nearest candidate text.
    Stage 2 — Verify (O(1)): bidirectional NLI entailment between the nearest
    candidate and the second-nearest (the closest independent cluster member).
    Entailment failure logs a warning but does not suppress the answer; the
    return type is str and confidence signalling is the orchestrator's job.
    """
    if not admitted:
        return ""
    if len(admitted) == 1:
        return admitted[0].output_text

    texts = [gen.output_text for gen in admitted]

    # Stage 1: geometric median → nearest candidate
    embeddings = _embed(texts)                                     # (N, D)
    median = _geometric_median(embeddings)                         # (D,)
    dists = np.linalg.norm(embeddings - median, axis=1)           # (N,)
    sorted_idx = np.argsort(dists)
    nearest_idx = int(sorted_idx[0])
    reference_idx = int(sorted_idx[1])

    candidate = texts[nearest_idx]
    reference = texts[reference_idx]

    # Stage 2: single-batch bidirectional entailment
    a_ok, b_ok = _batched_entailment(candidate, reference)
    if not (a_ok and b_ok):
        logger.warning(
            "Weak bidirectional entailment for nearest-centroid candidate "
            "(a→b=%s, b→a=%s). Returning candidate anyway.",
            a_ok,
            b_ok,
        )

    return candidate
