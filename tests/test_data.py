"""Streaming cache I/O and dataset loader tests for pdd/data.py."""
import json
import os

import pytest

from pdd.config import DataConfig
from pdd.data import DatasetLoader, PreferenceExample


def _sample(n=500):
    return [PreferenceExample(i, f"prompt {i} ✓", f"chosen {i}", f"rejected {i}") for i in range(n)]


def test_save_streamed_format_matches_canonical_json(tmp_path):
    exs = _sample(200)
    p = str(tmp_path / "ex.json")
    DatasetLoader.save_json_cache(exs, p)
    canonical = json.dumps([e.to_dict() for e in exs], separators=(",", ":"))
    assert open(p, encoding="utf-8").read() == canonical


def test_load_roundtrip_orjson_and_stdlib(tmp_path, monkeypatch):
    exs = _sample(300)
    p = str(tmp_path / "ex.json")
    DatasetLoader.save_json_cache(exs, p)
    assert DatasetLoader.load_json_cache(p) == exs

    # Force the stdlib fallback path by blocking the orjson import.
    import builtins
    real = builtins.__import__

    def no_orjson(name, *a, **k):
        if name == "orjson":
            raise ImportError("simulated")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_orjson)
    assert DatasetLoader.load_json_cache(p) == exs


def test_load_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        DatasetLoader.load_json_cache("/nonexistent/path/ex.json")


def test_preference_example_dict_cycle():
    ex = PreferenceExample(3, "p", "c", "r")
    assert PreferenceExample.from_dict(ex.to_dict()) == ex
