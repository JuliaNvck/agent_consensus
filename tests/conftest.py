"""
Session-wide autouse fixture that stubs out the two heavy model calls in
pipeline/aggregation.py so that tests which exercise the orchestrator or
other non-aggregation code never trigger a model download or GPU call.

Individual tests in test_aggregation.py override these stubs with their
own monkeypatch.setattr calls to inject controlled behavior.
"""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _stub_aggregation_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace _embed and _batched_entailment with lightweight CPU stubs."""
    monkeypatch.setattr(
        "pipeline.aggregation._embed",
        lambda texts: np.zeros((len(texts), 2), dtype=np.float64),
    )
    monkeypatch.setattr(
        "pipeline.aggregation._batched_entailment",
        lambda a, b: (True, True),
    )
