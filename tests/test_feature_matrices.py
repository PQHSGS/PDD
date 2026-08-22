"""Round-trip and state-fingerprint tests for pdd/feature_matrices.py."""
import numpy as np
import pytest
import scipy.sparse as sp

from pdd.feature_matrices import (
    FeatureMatrices,
    _example_hash,
    matrices_state,
    mmap_dir_complete,
    state_valid,
    write_matrices_state,
)


def _random_fm(n: int = 40, d: int = 30, seed: int = 0) -> FeatureMatrices:
    mats = {}
    for i, name in enumerate(["P_max", "P_freq", "C_max", "C_freq", "R_max", "R_freq"]):
        m, _ = make_sparse(n, d, 0.15, seed + i)
        mats[name] = m
    return FeatureMatrices(example_ids=np.arange(n, dtype=np.int64), **mats)


def make_sparse(n, d, density, seed):
    return sp.random(n, d, density=density, format="csr", dtype=np.float32, random_state=seed), None


def _assert_same(a: sp.csr_matrix, b: sp.csr_matrix):
    assert a.shape == b.shape
    assert (a != b).nnz == 0


def test_npz_roundtrip(small_matrices):
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.npz")
        small_matrices.save_npz(p)
        loaded = FeatureMatrices.load_npz(p)
        for name in ["P_max", "P_freq", "C_max", "C_freq", "R_max", "R_freq"]:
            _assert_same(getattr(loaded, name), getattr(small_matrices, name))
        assert np.array_equal(np.asarray(loaded.example_ids), np.arange(6))


def test_mmap_dir_roundtrip_lazy(small_matrices):
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "mm")
        small_matrices.save_mmap_dir(d)
        assert mmap_dir_complete(d)
        lazy = FeatureMatrices.load_mmap_dir(d)
        # Nothing materialized yet: all backing attrs are None until first property access.
        assert lazy._C_max is None
        for name in ["C_max", "R_freq"]:
            _assert_same(getattr(lazy, name), getattr(small_matrices, name))
        # Missing shape marker => incomplete (merge-crash guard).
        os.remove(os.path.join(d, "R_freq_shape.npy"))
        assert not mmap_dir_complete(d)


def test_missing_matrix_raises_attribute_error():
    fm = FeatureMatrices(example_ids=np.zeros(3, dtype=np.int64))  # no arrays, no mmap dir
    with pytest.raises(AttributeError):
        _ = fm.C_max


def test_union_p1_matches_direct_computation():
    n, d = 50, 20
    c, _ = make_sparse(n, d, 0.3, 1)
    r, _ = make_sparse(n, d, 0.3, 2)
    c.data[:] = 1.0
    r.data[:] = 1.0
    fm = FeatureMatrices(example_ids=np.arange(n, dtype=np.int64), C_freq=c, R_freq=r)
    p1 = fm.union_p1(d)
    direct = np.asarray(((c > 0) + (r > 0)).astype(np.float32).sum(axis=0)).ravel() / n
    assert np.allclose(p1, direct, atol=1e-6)


def test_example_hash_changes_with_ids():
    a = np.arange(10)
    b = np.arange(10)[::-1].copy()
    assert _example_hash(a) != _example_hash(b)
    assert _example_hash(a) == _example_hash(a.copy())


def test_state_write_and_valid(tmp_path):
    fm = _random_fm()
    write_matrices_state(str(tmp_path), fm)
    assert state_valid(fm, str(tmp_path))
    # The fingerprint is {N, d_sae, ex_hash(example_ids)} — content-independent by
    # design — so a *different shape or id vector* is what must invalidate it.
    other = _random_fm(n=39, seed=99)
    assert not state_valid(other, str(tmp_path))


def test_state_valid_legacy_fallback_warns_and_returns_true(tmp_path, caplog):
    fm = _random_fm(n=5, d=4, seed=3)
    with caplog.at_level("WARNING"):
        ok = state_valid(fm, str(tmp_path))
    assert ok is True
    assert any("extraction-state" in r.message for r in caplog.records)
    # And it persisted the state for next time.
    assert state_valid(fm, str(tmp_path))


def test_matrices_state_fields():
    fm = _random_fm(n=7, d=5, seed=5)
    st = matrices_state(fm)
    assert st["N"] == 7 and st["d_sae"] == 5 and len(st["ex_hash"]) == 16
