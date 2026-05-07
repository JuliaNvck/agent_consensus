"""Tests for scripts/mix_caches.py — cache mixing logic."""
from __future__ import annotations

import json
import os
import random
import tempfile
from typing import Dict, List

import pytest

# Import the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.mix_caches import (
    _load_cache,
    _parse_input_spec,
    _retrofit_provider,
    mix_question,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_generation(agent_id: str, provider: str = "", model_id: str = "") -> dict:
    gen: dict = {
        "agent_id": agent_id,
        "output_text": f"answer from {agent_id}",
        "token_logprobs": [0.0] * 10,
        "is_faulty": False,
        "fault_type": None,
    }
    if provider:
        gen["provider"] = provider
    if model_id:
        gen["model_id"] = model_id
    return gen


def _make_cache_file(questions: List[dict]) -> str:
    """Write questions to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump({"questions": questions}, fh)
    return path


# ---------------------------------------------------------------------------
# _parse_input_spec
# ---------------------------------------------------------------------------

class TestParseInputSpec:
    def test_valid_spec(self):
        path, tag = _parse_input_spec("cache_llma.json:llama")
        assert path == "cache_llma.json"
        assert tag == "llama"

    def test_path_with_colon_in_directory(self):
        path, tag = _parse_input_spec("/some/path/cache.json:phi3")
        assert path == "/some/path/cache.json"
        assert tag == "phi3"

    def test_missing_tag_raises(self):
        with pytest.raises(SystemExit):
            _parse_input_spec("cache_llma.json")

    def test_empty_tag_raises(self):
        with pytest.raises(SystemExit):
            _parse_input_spec("cache_llma.json:")


# ---------------------------------------------------------------------------
# _retrofit_provider
# ---------------------------------------------------------------------------

class TestRetrofitProvider:
    def test_injects_provider_when_missing(self):
        gens = [_make_generation("a0"), _make_generation("a1")]
        result = _retrofit_provider(gens, "llama", None)
        assert all(g["provider"] == "llama" for g in result)

    def test_does_not_overwrite_existing_provider(self):
        gens = [_make_generation("a0", provider="qwen")]
        result = _retrofit_provider(gens, "llama", None)
        assert result[0]["provider"] == "qwen"  # preserved

    def test_injects_model_id_when_missing(self):
        gens = [_make_generation("a0")]
        result = _retrofit_provider(gens, "llama", "meta-llama/Llama-3.1-8B")
        assert result[0]["model_id"] == "meta-llama/Llama-3.1-8B"

    def test_does_not_overwrite_existing_model_id(self):
        gens = [_make_generation("a0", model_id="original-model")]
        result = _retrofit_provider(gens, "llama", "new-model")
        assert result[0]["model_id"] == "original-model"

    def test_no_model_id_arg_leaves_field_absent(self):
        gens = [_make_generation("a0")]
        result = _retrofit_provider(gens, "llama", None)
        assert "model_id" not in result[0] or result[0].get("model_id") is None


# ---------------------------------------------------------------------------
# _load_cache
# ---------------------------------------------------------------------------

class TestLoadCache:
    def test_loads_all_questions(self):
        questions = [
            {"question_id": "gsm8k_0", "ground_truth": "5", "generations": []},
            {"question_id": "stratqa_0", "ground_truth": "yes", "generations": []},
        ]
        path = _make_cache_file(questions)
        try:
            cache = _load_cache(path)
            assert set(cache.keys()) == {"gsm8k_0", "stratqa_0"}
        finally:
            os.unlink(path)

    def test_keyed_by_question_id(self):
        questions = [{"question_id": "gsm8k_7", "ground_truth": "42", "generations": []}]
        path = _make_cache_file(questions)
        try:
            cache = _load_cache(path)
            assert "gsm8k_7" in cache
            assert cache["gsm8k_7"]["ground_truth"] == "42"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# mix_question
# ---------------------------------------------------------------------------

class TestMixQuestion:
    def _make_record(self, provider: str, n_agents: int) -> dict:
        return {
            "generations": [
                _make_generation(f"{provider}_a{i}", provider=provider)
                for i in range(n_agents)
            ]
        }

    def test_correct_total_agent_count(self):
        rng = random.Random(0)
        records = {
            "llama": self._make_record("llama", 7),
            "qwen": self._make_record("qwen", 7),
            "mistral": self._make_record("mistral", 7),
            "phi3": self._make_record("phi3", 7),
        }
        agents_per = {"llama": 2, "qwen": 2, "mistral": 2, "phi3": 1}
        order = ["llama", "qwen", "mistral", "phi3"]
        agents = mix_question(records, agents_per, order, rng)
        assert len(agents) == 7

    def test_provider_composition(self):
        rng = random.Random(0)
        records = {
            "llama": self._make_record("llama", 7),
            "phi3": self._make_record("phi3", 7),
        }
        agents_per = {"llama": 3, "phi3": 1}
        order = ["llama", "phi3"]
        agents = mix_question(records, agents_per, order, rng)
        llama_count = sum(1 for a in agents if a["provider"] == "llama")
        phi3_count = sum(1 for a in agents if a["provider"] == "phi3")
        assert llama_count == 3
        assert phi3_count == 1

    def test_no_duplicates_when_sampling_without_replacement(self):
        rng = random.Random(0)
        records = {"llama": self._make_record("llama", 7)}
        agents_per = {"llama": 4}
        order = ["llama"]
        agents = mix_question(records, agents_per, order, rng)
        ids = [a["agent_id"] for a in agents]
        assert len(ids) == len(set(ids))

    def test_sampling_with_replacement_when_insufficient_agents(self):
        rng = random.Random(0)
        records = {"phi3": self._make_record("phi3", 2)}  # only 2 available
        agents_per = {"phi3": 5}  # request 5
        order = ["phi3"]
        agents = mix_question(records, agents_per, order, rng)
        assert len(agents) == 5  # sampling with replacement fills the gap

    def test_reproducible_with_same_seed(self):
        records = {
            "llama": self._make_record("llama", 7),
            "qwen": self._make_record("qwen", 7),
        }
        agents_per = {"llama": 2, "qwen": 2}
        order = ["llama", "qwen"]
        run1 = mix_question(records, agents_per, order, random.Random(99))
        run2 = mix_question(records, agents_per, order, random.Random(99))
        assert [a["agent_id"] for a in run1] == [a["agent_id"] for a in run2]

    def test_different_seeds_give_different_results(self):
        records = {"llama": self._make_record("llama", 7)}
        agents_per = {"llama": 3}
        order = ["llama"]
        results: list[list[dict]] = []
        for seed in range(20):
            agents = mix_question(records, agents_per, order, random.Random(seed))
            results.append([a["agent_id"] for a in agents])
        # Not all 20 runs should be identical
        assert len(set(map(tuple, results))) > 1
