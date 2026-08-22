"""Shared synthetic fixtures for the PDD test suite (no GPU, no model downloads)."""
import numpy as np
import pytest
import scipy.sparse as sp

from pdd.feature_clusters import FeatureClusterMap
from pdd.feature_matrices import FeatureMatrices


def make_sparse_rng(n_rows: int, d_cols: int, density: float, seed: int):
    """Deterministic sparse CSR generator."""
    rng = np.random.default_rng(seed)
    return sp.random(n_rows, d_cols, density=density, format="csr", dtype=np.float32, random_state=seed), rng


@pytest.fixture
def small_matrices() -> FeatureMatrices:
    """6x4 example/feature fixture with known non-zero structure per row."""
    # Row i activates feature i%4 strongly on the chosen side, even rows also fire f3 on rejected.
    c = np.zeros((6, 4), dtype=np.float32)
    r = np.zeros((6, 4), dtype=np.float32)
    for i in range(6):
        c[i, i % 4] = 0.5 + 0.1 * i
        if i % 2 == 0:
            r[i, 3] = 0.2
    return FeatureMatrices(
        example_ids=np.arange(6, dtype=np.int64),
        P_max=sp.csr_matrix(np.abs(c) / 2),
        P_freq=sp.csr_matrix((c > 0).astype(np.float32)),
        C_max=sp.csr_matrix(c),
        C_freq=sp.csr_matrix((c > 0).astype(np.float32)),
        R_max=sp.csr_matrix(r),
        R_freq=sp.csr_matrix((r > 0).astype(np.float32)),
    )


@pytest.fixture
def two_cluster_map() -> FeatureClusterMap:
    """Two feature communities over 4 features: T_1={0,1}, T_2={2}; f3 unassigned."""
    clusters = {1: [0, 1], 2: [2]}
    f2c = {0: 1, 1: 1, 2: 2, 3: 0}
    return FeatureClusterMap(clusters=clusters, feature_to_cluster=f2c)
