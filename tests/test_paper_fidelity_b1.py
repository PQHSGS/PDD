"""Paper-fidelity tests: Appendix B.1 primitive & statistic formulas on hand-checkable data.

The pipeline's k-means assignment step is stochastic in label identity, so these tests
recompute every statistic from the RETURNED matrices/assignments using the paper
formulas and assert the emitted HypothesisPair fields match exactly.
"""
import numpy as np

from pdd.config import FeatureConditionedConfig
from pdd.feature_conditioned import FeatureConditionedPipeline
from pdd.feature_matrices import FeatureMatrices

import scipy.sparse as sp


def _tiny_fm() -> FeatureMatrices:
    """N=12 examples x d=3 features. C_freq/R_freq are BINARY presence matrices.

    Group A rows 0..5 : CHOSEN fires f0            -> u(T_1) = +1 (amplify)
    Group B rows 6..11: REJECTED fires f0, CHOSEN fires f2
                          -> u(T_1) = -1 (suppress), u(T_2) = +1
    The s-matrix directions ([1,0] vs [1,1]) separate cleanly under spherical k-means.
    """
    c = np.zeros((12, 3), dtype=np.float32)
    r = np.zeros((12, 3), dtype=np.float32)
    for i in range(12):
        if i < 6:
            c[i, 0] = 0.8 + 0.01 * i          # magnitude is irrelevant after binarization
        else:
            r[i, 0] = 0.9 - 0.01 * i
            c[i, 2] = 0.5
    return FeatureMatrices(
        example_ids=np.arange(12, dtype=np.int64),
        P_max=sp.csr_matrix(np.zeros((12, 3), dtype=np.float32)),
        P_freq=sp.csr_matrix(np.zeros((12, 3), dtype=np.float32)),
        C_max=sp.csr_matrix(c),
        C_freq=sp.csr_matrix((c > 0).astype(np.float32)),
        R_max=sp.csr_matrix(r),
        R_freq=sp.csr_matrix((r > 0).astype(np.float32)),
    )


CFG = FeatureConditionedConfig(
    tau=0.01,
    silent_pct=0.0,
    n_data_clusters=2,
    min_feat_cluster_size=1,
    min_data_cluster_size=2,
)


def _run():
    fm = _tiny_fm()
    from pdd.feature_clusters import FeatureClusterMap
    clusters = {1: [0], 2: [2]}
    f2c = {0: 1, 2: 2}
    cmap = FeatureClusterMap(clusters=clusters, feature_to_cluster=f2c)
    return FeatureConditionedPipeline(CFG).run(fm, cmap, seed=0, checkpoint_dir=None, use_checkpoint=False)


def test_b1_primitives_s_v_u_exact():
    res = _run()
    fm = _tiny_fm()
    c = np.asarray(fm.C_freq.todense())
    r = np.asarray(fm.R_freq.todense())
    A = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])       # f0->T_1, f2->T_2 indicator

    assert np.allclose(res.s_matrix, (c + r) @ A, atol=1e-6)
    assert np.allclose(res.v_matrix, (c - r) @ A, atol=1e-6)

    u_expected = ((c > CFG.tau).astype(np.float32) @ A - (r > CFG.tau).astype(np.float32) @ A)
    u_expected /= np.array([1.0, 1.0])                        # |T_1| = |T_2| = 1
    assert np.allclose(res.u_matrix, u_expected, atol=1e-6)

    # Disparity direction sanity from the one-sided firing pattern.
    assert np.all(res.u_matrix[:6, 0] > 0)     # group A amplifies T_1
    assert np.all(res.u_matrix[6:, 0] < 0)     # group B suppresses T_1
    assert np.all(res.u_matrix[6:, 1] > 0)     # group B amplifies T_2


def test_b1_welch_statistics_match_paper_formulas():
    res = _run()
    u = res.u_matrix
    assign = res.cluster_assignments
    active = ~res.silent_mask
    n_pool = int(active.sum())

    for h in res.hypotheses:
        col = {1: 0, 2: 1}[h.m]
        in_mask = (assign == h.k) & active
        out_mask = active & (assign != h.k)
        n_k = int(in_mask.sum()); n_out = int(out_mask.sum())

        u_in = u[in_mask].mean(axis=0)
        u_out = u[out_mask].mean(axis=0)
        delta = u_in[col] - u_out[col]
        var_in = u[in_mask].var(axis=0, ddof=1)[col]
        var_out = u[out_mask].var(axis=0, ddof=1)[col]
        z = delta / np.sqrt(var_in / n_k + var_out / n_out + 1e-12)
        pooled = ((n_k - 1) * var_in + (n_out - 1) * var_out) / max(1, n_pool - 2)
        d = delta / np.sqrt(pooled + 1e-12)

        assert h.n_k == n_k and abs(h.delta - delta) < 1e-5
        assert abs(h.z_score - z) < 1e-4
        assert abs(h.cohens_d - d) < 1e-4
        assert h.is_chosen_leaning == bool(delta > 0)

        # Split-half parity masks (paper: row-index even vs odd)
        idx = np.arange(len(assign))
        a_m = in_mask & (idx % 2 == 0); b_m = in_mask & (idx % 2 == 1)
        oa_m = out_mask & (idx % 2 == 0); ob_m = out_mask & (idx % 2 == 1)
        d_a = u[a_m].mean(axis=0)[col] - u[oa_m].mean(axis=0)[col]
        d_b = u[b_m].mean(axis=0)[col] - u[ob_m].mean(axis=0)[col]

        assert abs(h.delta_A - d_a) < 1e-5 and abs(h.delta_B - d_b) < 1e-5
        sc = (abs(d_a) > CFG.split_half_eps and abs(d_b) > CFG.split_half_eps
              and np.sign(d_a) == np.sign(d_b))
        assert h.sign_consistent == bool(sc)
        assert abs(h.delta_min - min(abs(d_a), abs(d_b))) < 1e-5

        # Paper filters
        assert h.t_m >= CFG.min_feat_cluster_size and h.n_k >= CFG.min_data_cluster_size
