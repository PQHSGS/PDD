"""LazyExampleStore: offset-indexed random access over examples.json sidecars."""
import os

import numpy as np
import pytest

from pdd.data import DatasetLoader, LazyExampleStore, PreferenceExample


def _sample(n=2000):
    return [PreferenceExample(i, f"prompt {i} — ünïcode ✓ {i*7}", f"chosen {i}", f"rejected {i}")
            for i in range(n)]


def test_build_and_random_access(tmp_path):
    exs = _sample()
    p = str(tmp_path / "examples.json")
    DatasetLoader.save_json_cache(exs, p)

    store = LazyExampleStore.build_if_missing(p)
    assert len(store) == 2000
    assert os.path.exists(str(tmp_path / "examples.ndjson"))
    assert os.path.exists(str(tmp_path / "examples_offsets.npy"))

    import random
    rng = random.Random(1)
    for i in rng.sample(range(2000), 50) + [0, 1999]:
        assert store[i] == exs[i]
    assert store[-1] == exs[-1]
    assert store[5:10] == exs[5:10]


def test_reopen_uses_existing_sidecars(tmp_path):
    exs = _sample(300)
    p = str(tmp_path / "examples.json")
    DatasetLoader.save_json_cache(exs, p)
    first = LazyExampleStore.build_if_missing(p)
    nd_mtime = os.path.getmtime(str(tmp_path / "examples.ndjson"))

    second = LazyExampleStore.build_if_missing(p)      # must NOT rebuild
    assert second[299] == exs[299]
    assert os.path.getmtime(str(tmp_path / "examples.ndjson")) == nd_mtime
    first.close(); second.close()


def test_corrupt_index_is_rebuilt(tmp_path):
    exs = _sample(500)
    p = str(tmp_path / "examples.json")
    DatasetLoader.save_json_cache(exs, p)
    LazyExampleStore.build_if_missing(p).close()

    np.save(str(tmp_path / "examples_offsets.npy"), np.array([0, 5], dtype=np.int64))
    store = LazyExampleStore.build_if_missing(p)
    assert len(store) == 500 and store[42] == exs[42]


def test_out_of_range_and_missing_source():
    with pytest.raises(IndexError):
        LazyExampleStore.build_if_missing("/nonexistent/examples.json")  # returns None
        raise IndexError  # unreachable guard
    store = LazyExampleStore.build_if_missing("/nonexistent/examples.json")
    assert store is None


def test_slice_and_negative_index(tmp_path):
    exs = _sample(50)
    p = str(tmp_path / "examples.json")
    DatasetLoader.save_json_cache(exs, p)
    store = LazyExampleStore.build_if_missing(p)
    assert store[-3:] == exs[-3:]
    assert store[10:13] == [exs[10], exs[11], exs[12]]
