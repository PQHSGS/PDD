"""Pure-algorithm tests for pdd/inspection.py (Tab 3/4 explorer layer)."""
import numpy as np
import pytest
from types import SimpleNamespace

from pdd.inspection import (
    cluster_signals,
    hypothesis_evidence,
    pair_concepts,
    parse_conditions,
    predicted_behavior_shifts,
    project_clusters,
    rank_cluster_samples,
    rank_compound_samples,
    sae_feature_item,
    top_examples,
    top_sae_features,
)


# ---------------------------------------------------------------------------
# parse_conditions
# ---------------------------------------------------------------------------

def test_parse_conditions_full_form():
    assert parse_conditions("7:amplify:0.03,4:suppress") == [(7, "amplify", 0.03), (4, "suppress", 0.01)]


def test_parse_conditions_skips_malformed_and_bad_threshold():
    out = parse_conditions("abc:amplify,9:amplify:notafloat,,5:backflip")
    assert (9, "amplify", 0.01) in out
    # invalid m dropped; bad direction coerced to amplify; bad threshold -> default
    assert all(m != "abc" for m, _, _ in [(str(m), d, t) for m, d, t in out])


def test_parse_conditions_custom_default_threshold():
    assert parse_conditions("1:suppress", default_thresh=0.5) == [(1, "suppress", 0.5)]


# ---------------------------------------------------------------------------
# cluster_signals / hypothesis_evidence
# ---------------------------------------------------------------------------

def test_cluster_signals_sum_and_mean():
    act = np.array([1.0, 2.0, 3.0, 4.0])
    clusters = {10: [0, 1], 20: [2], 30: [99]}          # 99 out of bounds
    sig = cluster_signals(act, clusters, mode="sum")
    assert sig[10] == 3.0 and sig[20] == 3.0 and sig[30] == 0.0
    mean = cluster_signals(act, clusters, mode="mean")
    assert mean[10] == 1.5


def test_cluster_signals_empty_activations():
    assert cluster_signals(None, {1: [0]}, mode="sum") == {}
    assert cluster_signals(np.array([]), {1: [0]}, mode="sum") == {}


def test_hypothesis_evidence_weights_delta_by_signal():
    hypos = [{"m": 1, "delta": 0.5}, {"m": 2, "delta": -0.4}, {"m": None, "delta": 9.0}]
    ev = hypothesis_evidence(hypos, {1: 2.0})
    by_m = {h["m"]: v for h, v in ev}
    assert by_m[1] == pytest.approx(1.0)
    assert by_m[2] == 0.0            # no live signal for m=2
    assert by_m[None] == 0.0


# ---------------------------------------------------------------------------
# rank_cluster_samples / rank_compound_samples
# ---------------------------------------------------------------------------

def _stub_fc(n: int, u_cols: dict):
    cols = list(u_cols.values())
    return SimpleNamespace(
        u_matrix=np.stack(cols, axis=1).astype(np.float64),
        s_matrix=np.zeros((n, len(cols))),
        cluster_assignments=np.arange(n) % 3,
    )


def test_rank_cluster_samples_directional_filter_and_order():
    n = 6
    fc = _stub_fc(n, {5: np.array([+2, +1, 0, -1, -3, +0.5])})
    examples = [SimpleNamespace(prompt=f"p{i}", chosen=f"c{i}", rejected=f"r{i}") for i in range(n)]
    view = lambda ex: {"prompt": ex.prompt, "chosen": ex.chosen, "rejected": ex.rejected}

    amp = rank_cluster_samples(5, "amplify", 10, fc, examples, mats=None,
                               feature_clusters={5: [1, 2]}, cluster_ids=[5],
                               example_view_fn=view, neuronpedia_url_fn=lambda f: None)
    assert amp["total_matching"] == 3                      # u > 0 rows only
    assert [s["u"] for s in amp["samples"]] == [2.0, 1.0, 0.5]   # |u| descending

    sup = rank_cluster_samples(5, "suppress", 10, fc, examples, mats=None,
                               feature_clusters={5: [1, 2]}, cluster_ids=[5],
                               example_view_fn=view, neuronpedia_url_fn=lambda f: None)
    assert sup["total_matching"] == 2
    assert [s["u"] for s in sup["samples"]] == [-3.0, -1.0]      # most negative first


def test_rank_cluster_samples_shape_mismatch_returns_empty():
    fc = _stub_fc(3, {5: np.zeros(3)})
    out = rank_cluster_samples(5, "amplify", 5, fc, [], mats=None,
                               feature_clusters={5: [1]}, cluster_ids=[5, 6, 7, 8],
                               example_view_fn=lambda e: {}, neuronpedia_url_fn=lambda f: None)
    assert out["total_matching"] == 0 and out["samples"] == []


def test_rank_compound_samples_and_mask_and_score():
    n = 5
    fc = _stub_fc(n, {
        1: np.array([+1, +2, -1, +3, 0]),
        2: np.array([+4, -1, -2, +1, 0]),
    })
    examples = [SimpleNamespace(prompt="", chosen="", rejected="") for _ in range(n)]
    res = rank_compound_samples([(1, "amplify", 0.01), (2, "amplify", 0.01)], 10, fc, examples,
                                mats=None, feature_clusters={}, cluster_ids=[1, 2],
                                example_view_fn=lambda e: {}, neuronpedia_url_fn=lambda f: None)
    # AND of (u1>0, u2>0) keeps rows 0 and 3; scores |u1|+|u2| => row0 (5) beats row3 (4)
    assert res["total_matching"] == 2
    assert [s["index"] for s in res["samples"]] == [0, 3]
    assert res["samples"][0]["score"] == pytest.approx(5.0)
    assert res["samples"][1]["score"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# predicted_behavior_shifts / pair_concepts / top helpers
# ---------------------------------------------------------------------------

def _labels():
    feat_labels = {4: {"title": "Step-by-Step", "description": "d"}}
    data_labels = [{"cluster_id": 7, "title": "Math Prompts"}]
    return feat_labels, data_labels


def test_predicted_behavior_shifts_titles_and_sort():
    scored = [{
        "cluster_id": 7,
        "hypos": [
            {"k": 7, "m": 4, "delta": 0.10, "z_score": 2.0, "cohens_d": 0.5},
            {"k": 7, "m": 5, "delta": 0.05, "z_score": 1.0, "cohens_d": 0.2},
        ],
    }]
    shifts = predicted_behavior_shifts(scored, {4: 3.0}, *_labels(), limit=10)
    assert shifts[0]["feature_cluster_title"] == "Step-by-Step"
    assert shifts[0]["data_cluster_title"] == "Math Prompts"
    assert shifts[0]["effect_direction"].startswith("Amplified")
    assert abs(shifts[0]["live_activity"] - 3.0) < 1e-9


def test_pair_concepts_promoted_vs_suppressed_split():
    feat_labels, data_labels = _labels()
    u_sig = {4: 0.20, 5: -0.30}
    best_by_m = {4: {"k": 7, "delta": 0.12, "z_score": 3.0}, 5: {"k": 8, "delta": -0.2, "z_score": 2.0}}
    promoted, suppressed = pair_concepts(u_sig, best_by_m, {4: [1], 5: [1]},
                                         min_feat_cluster_size=1,
                                         feature_cluster_labels=feat_labels,
                                         data_cluster_labels=data_labels)
    assert [p["feature_cluster_m"] for p in promoted] == [4]
    assert [s["feature_cluster_m"] for s in suppressed] == [5]
    assert suppressed[0]["signal_strength"] == "Strong"


def test_top_examples_bounds_and_positive_only():
    exs = [SimpleNamespace(prompt=str(i), chosen="", rejected="") for i in range(5)]
    out = top_examples(np.array([0.0, 3.0, -1.0, 2.0]), exs, lambda e: {"prompt": e.prompt}, top_n=10)
    assert [o["prompt"] for o in out] == ["1", "3"]


def test_top_sae_features_tags_cluster_and_caps():
    act = np.zeros(8); act[[3, 1, 6]] = [0.9, 0.5, 0.1]
    items = top_sae_features(act, {3: 1, 1: 1, 6: 2}, {1: [1, 3], 2: [6]},
                             min_partition_size=1, top_n=2, neuronpedia_url_fn=lambda i: f"u{i}")
    assert [it["feature_index"] for it in items] == [3, 1]
    assert items[0]["neuronpedia_url"] == "u3"
    assert sae_feature_item(3, 0.9, 1, None)["activation"] == 0.9


def test_project_clusters_strips_internal_arrays():
    scored = [{"a": 1, "hypos": [1, 2]}]
    assert project_clusters(scored)[0] == {"a": 1}
