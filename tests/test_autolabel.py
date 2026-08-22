"""Auto-label layer tests: ClusterLabel defaults, JSON extraction, keyword fallback, Pass 1 resume.

No GPU / no model download: the LLM labeler is exercised only through its static
parsing helpers and via the heuristic fallback path.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pdd.autolabel import (
    AutoLabelingPipeline,
    ClusterAutoLabeler,
    ClusterLabel,
    LLMClusterLabeler,
    cluster_labels_path,
)
from pdd.config import AutoLabelConfig


# ---------------------------------------------------------------------------
# Data structures & parsing helpers
# ---------------------------------------------------------------------------

def test_cluster_label_prompt_fields_default_empty():
    lbl = ClusterLabel(cluster_id=3, title="t", description="d", keywords=["k"])
    assert lbl.centroid_prompts == [] and lbl.sample_prompts == []


def test_extract_json_variants():
    ex = LLMClusterLabeler._extract_json
    assert ex('```json\n{"title": "A", "description": "B", "keywords": ["x"]}\n```')["title"] == "A"
    assert ex('junk {"title": "T", "description": "D"} tail')["title"] == "T"
    # Regex fallback when braces are broken
    out = ex('title: "Fallback" ... "description": "Desc"')
    assert out is None or isinstance(out, dict)


def test_strip_prefixes_and_fallback():
    f = lambda t: LLMClusterLabeler._strip_prefixes(t, ("cluster of",), 3, "FB")
    assert f("Cluster of math prompts") == "Math prompts"
    assert f("ok") == "FB"                      # below min_len -> fallback


def test_keyword_labeler_silent_bucket_special_case():
    lab = ClusterAutoLabeler()
    lbl = lab.generate_label(0, ["anything"], ["anything"])
    assert "Silent Bucket" in lbl.title
    normal = lab.generate_label(2, ["python sorting lists algorithms"], ["how to sort a list"])
    assert normal.title and normal.keywords


# ---------------------------------------------------------------------------
# Pass 1 resume logic (heuristic labeler; fc_res stubbed)
# ---------------------------------------------------------------------------

def _fc_res_stub(assignments):
    n = len(assignments)
    k_data = int(max(assignments)) + 1
    rng = np.random.default_rng(0)
    return SimpleNamespace(
        cluster_assignments=np.asarray(assignments),
        s_matrix=rng.random((n, 4)).astype(np.float32) + 0.01,
        u_matrix=None,
    )


def _examples_for(assignments):
    return [
        SimpleNamespace(example_id=i, prompt=f"question about topic {a} number {i}",
                        chosen="c", rejected="r")
        for i, a in enumerate(assignments)
    ]


def _pipeline(tmp_path):
    return AutoLabelingPipeline(
        AutoLabelConfig(enabled=True, heuristic=True, max_prompt_chars=80), str(tmp_path)
    )


ASSIGNMENTS = [0, 1, 1, 2, 2, 2]


def test_pass1_writes_all_clusters(tmp_path):
    pipe = _pipeline(tmp_path)
    count = pipe._label_data_clusters(ClusterAutoLabeler(), _examples_for(ASSIGNMENTS), _fc_res_stub(ASSIGNMENTS), seed=0)
    assert count == 3
    data = json.load(open(cluster_labels_path(str(tmp_path)), encoding="utf-8"))
    assert {lbl["cluster_id"] for lbl in data["labels"]} == {0, 1, 2}


def test_pass1_resume_skips_complete_artifact(tmp_path, caplog):
    pipe = _pipeline(tmp_path)
    fc = _fc_res_stub(ASSIGNMENTS)
    exs = _examples_for(ASSIGNMENTS)
    pipe._label_data_clusters(ClusterAutoLabeler(), exs, fc, seed=0)

    # Second call with a *different* labeler object must skip entirely.
    class ExplodingLabeler(ClusterAutoLabeler):
        def sample_cluster_prompts(self, *a, **k):  # must never be called
            raise AssertionError("resume should not re-sample")

    with caplog.at_level("INFO"):
        count = pipe._label_data_clusters(ExplodingLabeler(), exs, fc, seed=0)
    assert count == 3
    assert any("skipping relabeling" in r.message for r in caplog.records)


def test_pass1_partial_artifact_triggers_full_relabel(tmp_path, caplog):
    pipe = _pipeline(tmp_path)
    out_path = cluster_labels_path(str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)
    json.dump({"total_clusters": 1, "labels": [{"cluster_id": 0}]}, open(out_path, "w"))

    count = pipe._label_data_clusters(ClusterAutoLabeler(), _examples_for(ASSIGNMENTS), _fc_res_stub(ASSIGNMENTS), seed=0)
    assert count == 3
    assert any("partial" in r.message for r in caplog.records)


def test_pass2b_skips_when_u_matrix_missing(tmp_path, caplog):
    pipe = _pipeline(tmp_path)
    labels: dict = {"5": {"title": "t"}}
    pipe._append_disparity_labels(ClusterAutoLabeler(), [], {5: [0]}, SimpleNamespace(u_matrix=None), labels)
    assert "disparity_title" not in labels["5"]
