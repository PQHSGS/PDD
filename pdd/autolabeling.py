"""Auto-interpretation pipeline stage (Appendix B.1.7 + viewer interpretation).

Final pipeline stage producing every derived artifact the viewer needs for
cluster interpretation, from the artifacts already computed in-memory by the
pipeline — no recomputation of the B.1 / B.2 hypotheses:

  Pass 1  Data clusters B_k        -> <run>/cluster_labels.json
           (LLM labels from real centroid/random sampled prompts via s_matrix)
  Pass 2  SAE feature clusters T_m -> <run>/feature_cluster_labels.json
           (whole-cluster LLM labels from real response examples firing each T_m)
  Pass 3  Prompt clusters A_k /
           response-delta clusters R_m -> <run>/prompt_conditioned_cluster_examples.json
           (real-example indices expressing each cluster via c_matrix / |u_matrix|,
           plus the score-weighted top content tokens of those examples)

When called standalone (fc_res/pc_res = None) the two pipelines are re-run from
the cached matrices; the main pipeline always passes them in.

The artifact-path helpers below are the single source of truth shared with the
viewer (pdd/viewer_server.py) so writer and reader cannot drift.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from tqdm import tqdm

from .config import AutoLabelConfig, FeatureConditionedConfig, PromptConditionedConfig
from .logger import get_logger

logger = get_logger("PDD.AutoLabel")


def cluster_labels_path(run_dir: Union[str, "os.PathLike[str]"]) -> str:
    """LLM labels for data clusters B_k (title/description/keywords)."""
    return os.path.join(run_dir, "cluster_labels.json")


def feature_cluster_labels_path(run_dir: Union[str, "os.PathLike[str]"]) -> str:
    """Whole-cluster LLM labels for SAE feature clusters T_m."""
    return os.path.join(run_dir, "feature_cluster_labels.json")


def pc_cluster_examples_path(run_dir: Union[str, "os.PathLike[str]"]) -> str:
    """Real example indices expressing prompt clusters A_k / response-delta clusters R_m."""
    return os.path.join(run_dir, "prompt_conditioned_cluster_examples.json")


def _save_json(path: str, data: Dict[str, Any]) -> None:
    """Atomic JSON write (tmp + replace) so a crash never corrupts an artifact."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)
    logger.info(f"Saved '{path}'.")


def _load_examples(checkpoint_dir: str) -> List[Any]:
    """Load the REAL cached preference examples (the pipeline's in-memory copies are text-stubs when checkpoints are reused)."""
    from .data import PreferenceExample

    ex_path = os.path.join(checkpoint_dir, "examples.json")
    if not os.path.exists(ex_path):
        raise FileNotFoundError(f"No cached examples.json in checkpoint '{checkpoint_dir}' for auto-labeling.")
    with open(ex_path, "r", encoding="utf-8") as f:
        return [PreferenceExample.from_dict(d) for d in json.load(f)]


_STOPWORDS = frozenset({
    "about", "after", "again", "also", "been", "being", "before", "could",
    "does", "from", "have", "into", "more", "most", "only", "other", "over",
    "please", "should", "some", "such", "than", "that", "their", "there",
    "these", "they", "this", "those", "through", "using", "want", "were",
    "what", "when", "where", "which", "while", "with", "would", "your",
})


def _top_tokens(texts: Sequence[str], weights: Sequence[float], n_tokens: int = 8) -> List[str]:
    """Score-weighted top content tokens across a cluster's strongest examples.

    Higher-weight examples (stronger c_matrix / |u_matrix|) contribute more to a
    token's total, so the returned tokens are the ones that most express the
    cluster in the real data. Pure offline text statistics — no model calls.
    """
    counts: Dict[str, float] = {}
    for text, w in zip(texts, weights):
        if not text:
            continue
        for tok in re.split(r"[^\w']+", text.lower()):
            tok = tok.strip("'")
            if len(tok) > 4 and tok.isalpha() and tok not in _STOPWORDS:
                counts[tok] = counts.get(tok, 0.0) + w
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[:n_tokens]]


class AutoLabelingPipeline:
    """Runs the three auto-interpretation passes and writes them under the run directory."""

    def __init__(self, cfg: AutoLabelConfig, run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir

    def _labeler(self):
        """Build the LLM or keyword labeler (lazy import: this module must stay light for the viewer)."""
        from .autolabel import ClusterAutoLabeler, LLMClusterLabeler

        if self.cfg.heuristic:
            logger.info("Using keyword-heuristic labels (auto_label.heuristic).")
            return ClusterAutoLabeler(max_prompt_chars=self.cfg.max_prompt_chars)
        logger.info(f"Using local LLM labels ({self.cfg.label_model}).")
        return LLMClusterLabeler(
            model_path=self.cfg.label_model,
            max_prompt_chars=self.cfg.max_prompt_chars,
            max_examples=self.cfg.max_examples,
        )

    def _label_data_clusters(
        self,
        labeler: Any,
        examples: List[Any],
        fc_res: Any,
        seed: int,
    ) -> int:
        """Pass 1: LLM (or keyword) labels for data clusters B_k."""
        unique_clusters = sorted(set(fc_res.cluster_assignments.tolist()))
        if self.cfg.num_clusters > 0:
            unique_clusters = unique_clusters[: self.cfg.num_clusters]

        out_path = cluster_labels_path(self.run_dir)
        labels: List[Dict[str, Any]] = []
        for idx, k in enumerate(tqdm(unique_clusters, desc="Pass 1: labeling data clusters", unit="cluster")):
            centroid_p, sample_p = labeler.sample_cluster_prompts(
                examples=examples,
                cluster_assignments=fc_res.cluster_assignments,
                s_matrix=fc_res.s_matrix,
                cluster_id=k,
                seed=seed,
            )
            labels.append(asdict(labeler.generate_label(k, centroid_p, sample_p)))
            if (idx + 1) % 10 == 0:
                _save_json(out_path, {"total_clusters": len(labels), "labels": labels})
                logger.info(f"  ...{idx + 1}/{len(unique_clusters)} labeled (checkpoint saved).")

        _save_json(out_path, {"total_clusters": len(labels), "labels": labels})
        return len(labels)

    def _label_feature_clusters(
        self,
        labeler: Any,
        examples: List[Any],
        matrices: Any,
        clusters: Dict[int, Sequence[int]],
    ) -> int:
        """Pass 2: whole-cluster labels for SAE feature clusters T_m.

        Each cluster's meaning is derived from the real dataset examples that fire
        it most (rows of C_max + R_max over the member features), fed to the LLM as
        response texts — independent of any Neuronpedia dashboard. Additionally,
        when an LLM labeler is active, the top ``label_top_features`` member
        features per cluster get their own local LLM labels (feature_labels) as a
        per-feature fallback for SAE models without a Neuronpedia dashboard.
        """
        from .autolabel import LLMClusterLabeler

        d_sae = matrices.C_max.shape[1]
        out_path = feature_cluster_labels_path(self.run_dir)
        labels: Dict[str, Dict[str, Any]] = {}
        feature_labels: Dict[str, Dict[str, Any]] = {}
        for m in tqdm(sorted(clusters.keys()), desc="Pass 2: labeling feature clusters", unit="cluster"):
            feats = [f for f in clusters[m] if 0 <= f < d_sae]
            if not feats:
                continue
            firing = matrices.C_max[:, feats] + matrices.R_max[:, feats]
            scores = np.asarray(firing.sum(axis=1)).ravel()
            idxs = [int(i) for i in np.argsort(scores)[-10:][::-1] if scores[int(i)] > 0]
            texts = [(examples[i].chosen or examples[i].rejected or examples[i].prompt or "").strip() for i in idxs]

            if isinstance(labeler, LLMClusterLabeler):
                parsed = labeler._label_dict(texts, kind="response")
                if parsed is None:
                    parsed = {"title": f"Feature cluster T_{m}", "description": "Cluster of SAE features (no LLM label)", "keywords": []}
                title, desc, kws = parsed.get("title", ""), parsed.get("description", ""), parsed.get("keywords", [])
            else:
                fallback = labeler.generate_label(m, texts, [])
                title, desc, kws = fallback.title, fallback.description, fallback.keywords

            if isinstance(kws, str):
                kws = [kws]
            labels[str(m)] = {
                "title": str(title)[:120],
                "description": str(desc)[:600],
                "keywords": [str(k) for k in kws][:5],
            }
            logger.info(f"T_{m}: {labels[str(m)]['title']}")

            if isinstance(labeler, LLMClusterLabeler) and self.cfg.label_top_features > 0:
                colsum = np.asarray(firing.sum(axis=0)).ravel()
                top_loc = np.argsort(colsum)[-self.cfg.label_top_features:][::-1]
                for j in top_loc:
                    f_idx = int(feats[j])
                    col = (
                        np.asarray(matrices.C_max[:, f_idx].toarray()).ravel()
                        + np.asarray(matrices.R_max[:, f_idx].toarray()).ravel()
                    )
                    f_idxs = [int(i) for i in np.argsort(col)[-8:][::-1] if col[int(i)] > 0]
                    f_texts = [
                        (examples[i].chosen or examples[i].rejected or examples[i].prompt or "").strip()
                        for i in f_idxs
                    ]
                    if not f_texts:
                        continue
                    f_parsed = labeler._label_dict(f_texts, kind="response")
                    if not f_parsed:
                        continue
                    f_kws = f_parsed.get("keywords", [])
                    if isinstance(f_kws, str):
                        f_kws = [f_kws]
                    feature_labels[str(f_idx)] = {
                        "title": str(f_parsed.get("title", ""))[:120],
                        "description": str(f_parsed.get("description", ""))[:600],
                        "keywords": [str(k) for k in f_kws][:5],
                    }

        payload: Dict[str, Any] = {"feature_clusters": labels}
        if feature_labels:
            payload["feature_labels"] = feature_labels
            logger.info(f"Labeled {len(feature_labels)} individual features (Neuronpedia fallback).")
        _save_json(out_path, payload)
        return len(labels)

    def _pc_cluster_examples(self, pc_res: Any, examples: List[Any], n_top: int) -> int:
        """Pass 3: real-example indices + top tokens for prompt clusters A_k and response-delta clusters R_m.

        Meaning for A_k = the examples with the strongest c_matrix (prompt-side)
        values; for R_m = the examples with the strongest |u_matrix| (response-delta).
        ``examples`` are the real cached preference examples used to extract the
        score-weighted top content tokens per cluster.
        """
        prompt_ex: Dict[str, List[int]] = {}
        prompt_tokens: Dict[str, List[str]] = {}
        for col, k in enumerate(sorted(pc_res.prompt_clusters.keys())):
            scores = np.asarray(pc_res.c_matrix[:, col]).ravel()
            idxs = [int(i) for i in np.argsort(scores)[-n_top:][::-1]]
            prompt_ex[str(k)] = idxs
            prompt_tokens[str(k)] = _top_tokens(
                [examples[i].prompt or "" for i in idxs], [float(scores[i]) for i in idxs]
            )

        resp_ex: Dict[str, List[int]] = {}
        resp_tokens: Dict[str, List[str]] = {}
        for col, m in enumerate(sorted(pc_res.resp_clusters.keys())):
            scores = np.abs(np.asarray(pc_res.u_matrix[:, col]).ravel())
            idxs = [int(i) for i in np.argsort(scores)[-n_top:][::-1]]
            resp_ex[str(m)] = idxs
            resp_tokens[str(m)] = _top_tokens(
                [examples[i].chosen or "" for i in idxs], [float(scores[i]) for i in idxs]
            )

        _save_json(pc_cluster_examples_path(self.run_dir), {
            "n_top": n_top,
            "prompt_cluster_examples": prompt_ex,
            "response_cluster_examples": resp_ex,
            "prompt_cluster_tokens": prompt_tokens,
            "response_cluster_tokens": resp_tokens,
        })
        return len(prompt_ex) + len(resp_ex)

    def run(
        self,
        matrices: Any,
        cluster_map: Any,
        fc_res: Optional[Any] = None,
        pc_res: Optional[Any] = None,
        seed: int = 0,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, int]:
        """Execute all three passes. ``fc_res`` / ``pc_res`` short-circuit Pass 1/3.

        Returns a small {pass_name: count} report.
        """
        from .feature_conditioned import FeatureConditionedPipeline
        from .prompt_conditioned import PromptConditionedPipeline

        if checkpoint_dir is None:
            raise ValueError("AutoLabelingPipeline.run requires checkpoint_dir to load the real cached examples.")
        examples = _load_examples(checkpoint_dir)
        labeler = self._labeler()

        counts: Dict[str, int] = {}

        if fc_res is None:
            logger.info("Re-running B.1 feature-conditioned pipeline for auto-labeling (no precomputed result passed).")
            fc_res = FeatureConditionedPipeline(FeatureConditionedConfig()).run(matrices, cluster_map, seed=seed)
        counts["data_clusters"] = self._label_data_clusters(labeler, examples, fc_res, seed=seed)

        if not self.cfg.skip_feature_clusters:
            counts["feature_clusters"] = self._label_feature_clusters(
                labeler, examples, matrices, cluster_map.clusters
            )

        if not self.cfg.skip_pc_examples:
            if pc_res is None:
                logger.info("Re-running B.2 prompt-conditioned pipeline for auto-labeling (no precomputed result passed).")
                pc_cfg = PromptConditionedConfig()
                pc_res = PromptConditionedPipeline(pc_cfg).run(matrices, seed=seed, checkpoint_dir=checkpoint_dir)
            counts["pc_clusters"] = self._pc_cluster_examples(pc_res, examples, self.cfg.pc_n_top)

        return counts