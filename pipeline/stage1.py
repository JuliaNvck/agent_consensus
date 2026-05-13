from __future__ import annotations

from typing import List

import numpy as np

from models import AgentGeneration
from pipeline import aggregation as _aggregation


def nearest_centroid_text(admitted: List[AgentGeneration]) -> str:
    """Geometric median nearest-centroid selection used by stage1_only."""
    if not admitted:
        return ""
    if len(admitted) == 1:
        return admitted[0].output_text

    texts = [gen.output_text for gen in admitted]
    embeddings = _aggregation._embed(texts)
    median = _aggregation._geometric_median(embeddings)
    dists = np.linalg.norm(embeddings - median, axis=1)
    return texts[int(np.argmin(dists))]


stage1_only = nearest_centroid_text
