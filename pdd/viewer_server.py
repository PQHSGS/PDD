"""FastAPI server for the Predictive Data Debugging (PDD) Interactive Viewer.

Serves run metadata, feature-conditioned & prompt-conditioned hypotheses,
cluster statistics, and live prompt/preference pair inspection endpoints.
Points directly to a target run directory and its linked checkpoint artifacts.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .autolabel import (
    cluster_labels_path,
    feature_cluster_labels_path,
    pc_cluster_examples_path,
)

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    raise ImportError("FastAPI and Uvicorn are required for the viewer. Install via `pip install fastapi uvicorn`.")

logger = logging.getLogger("PDD.Viewer")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

app = FastAPI(
    title="Predictive Data Debugging (PDD) Viewer",
    description="Interactive Explorer for Predictive Data Debugging (arXiv:2606.12360)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

VIEWER_DIR = Path(__file__).parent.parent / "viewer"


class PromptInspectionRequest(BaseModel):
    prompt: str
    top_k: int = 5


class PreferencePairInspectionRequest(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    top_k: int = 5


class ViewerState:
    """Manages the target run directory, linked checkpoints, and hypothesis indices."""

    def __init__(self, run_dir: Optional[str] = None):
        if run_dir is None:
            run_dir = os.environ.get("PDD_RUN_DIR", "runs/qwen3_1.7b_dolci")
        self.run_dir = Path(run_dir)
        self.summary: Dict[str, Any] = {}
        self.checkpoint_dir: Optional[Path] = None
        self.feature_clusters: Dict[int, List[int]] = {}
        self.cluster_labels: List[Dict[str, Any]] = []
        self.fc_hypos: List[Dict[str, Any]] = []
        self.pc_hypos: List[Dict[str, Any]] = []
        self.k_to_fc: Dict[int, List[Dict[str, Any]]] = {}
        self.k_to_pc: Dict[int, List[Dict[str, Any]]] = {}
        self.inspector = None
        self._feat_to_cluster: Optional[Dict[int, int]] = None
        self._feature_matrices = None
        self._examples = None
        self._feat_delta: Optional[np.ndarray] = None
        self._pc_cluster_examples = None
        self._feature_cluster_labels: Optional[Dict[int, Dict[str, Any]]] = None
        self._np_set: Optional[Tuple[str, str]] = None
        self._cluster_info_cache: Dict[str, Any] = {}
        self._feature_totals: Optional[np.ndarray] = None

        self.load()

    def load(self) -> None:
        """Load target run summary, cluster definitions from checkpoints, and pre-index hypotheses."""
        if not self.run_dir.exists():
            alt = Path("runs") / self.run_dir
            if alt.exists():
                self.run_dir = alt
            else:
                logger.warning(f"Run directory '{self.run_dir}' not found on disk.")
                return

        # 1. Load PDD Summary
        sum_path = self.run_dir / "pdd_summary.json"
        if sum_path.exists():
            try:
                with open(sum_path, "r", encoding="utf-8") as f:
                    self.summary = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading summary: {e}")

        # 2. Resolve Checkpoint Subfolder for Cluster Maps
        ckpt_path_str = self.summary.get("checkpoint_subfolder")
        if ckpt_path_str and Path(ckpt_path_str).exists():
            self.checkpoint_dir = Path(ckpt_path_str)
            clusters_file = self.checkpoint_dir / "clusters.json"
            if clusters_file.exists():
                try:
                    with open(clusters_file, "r", encoding="utf-8") as f:
                        raw_clusters = json.load(f).get("clusters", {})
                        self.feature_clusters = {int(k): v for k, v in raw_clusters.items()}
                except Exception as e:
                    logger.warning(f"Error loading clusters.json: {e}")

        # 3. Load Auto-Labels
        lbl_file = Path(cluster_labels_path(str(self.run_dir)))
        if lbl_file.exists():
            try:
                with open(lbl_file, "r", encoding="utf-8") as f:
                    self.cluster_labels = json.load(f).get("labels", [])
            except Exception:
                pass

        # 4. Instant Seed from Summary
        # Seed with summary top hypotheses initially
        self.fc_hypos = self.summary.get("top_feature_conditioned_hypotheses", [])
        self.pc_hypos = self.summary.get("top_prompt_conditioned_hypotheses", [])

        logger.info(
            f"ViewerState initialized for '{self.run_dir.name}': "
            f"{len(self.feature_clusters)} feature clusters, ready for instant requests."
        )

    @property
    def prompt_hypotheses_map(self) -> Dict[int, List[Dict[str, Any]]]:
        """Lazy load and cache prompt hypotheses map on demand."""
        if not self.k_to_pc:
            pc_file = self.run_dir / "prompt_conditioned_hypotheses.json"
            if pc_file.exists():
                try:
                    with open(pc_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.pc_hypos = data.get("hypotheses", data) if isinstance(data, dict) else data
                        k_to_pc = {}
                        for h in self.pc_hypos:
                            k = h.get("k")
                            if k is not None:
                                k_to_pc.setdefault(k, []).append(h)
                        self.k_to_pc = k_to_pc
                except Exception as e:
                    logger.warning(f"Error reading pc hypotheses: {e}")
        return self.k_to_pc

    @property
    def feature_hypotheses_map(self) -> Dict[int, List[Dict[str, Any]]]:
        """Lazy load and cache feature hypotheses map on demand."""
        if not self.k_to_fc:
            fc_file = self.run_dir / "feature_conditioned_hypotheses.json"
            if fc_file.exists():
                try:
                    with open(fc_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.fc_hypos = data.get("hypotheses", data) if isinstance(data, dict) else data
                        k_to_fc = {}
                        for h in self.fc_hypos:
                            k = h.get("k")
                            if k is not None:
                                k_to_fc.setdefault(k, []).append(h)
                        self.k_to_fc = k_to_fc
                except Exception as e:
                    logger.warning(f"Error reading fc hypotheses: {e}")
        return self.k_to_fc

    def _cluster_signals(self, activations: np.ndarray, mode: str = "sum") -> Dict[int, float]:
        """Map a (d_sae,) live activation vector onto per-feature-cluster signals.

        ``sum`` mirrors the paper's s = C + R cluster aggregate; ``mean`` mirrors
        the per-cluster disparity u = (C>tau - R>tau)/|T_m|. Keys are feature
        cluster ids (T_m) from the checkpoint clusters.json map.
        """
        signals: Dict[int, float] = {}
        if activations is None or len(activations) == 0:
            return signals
        for m, feats in self.feature_clusters.items():
            idx = np.asarray(feats, dtype=np.int64)
            idx = idx[idx < len(activations)]
            if len(idx) == 0:
                signals[m] = 0.0
                continue
            vals = activations[idx]
            signals[m] = float(vals.mean()) if mode == "mean" else float(vals.sum())
        return signals

    @staticmethod
    def _hypothesis_evidence(hypos: List[Dict[str, Any]], signals: Dict[int, float]) -> List[Tuple[Dict[str, Any], float]]:
        """Score each hypothesis by |delta| x live per-feature-cluster signal."""
        ev: List[Tuple[Dict[str, Any], float]] = []
        for h in hypos:
            m = h.get("m")
            sig = signals.get(m, 0.0) if m is not None else 0.0
            ev.append((h, abs(float(h.get("delta", 0.0))) * abs(sig)))
        return ev

    def _cluster_keywords(self, activations: np.ndarray, feature_ms, top_n: int = 3) -> List[str]:
        """Top individual SAE features by live activation within the given feature clusters."""
        if activations is None or len(activations) == 0:
            return []
        feats: List[int] = []
        for m in feature_ms:
            feats.extend(self.feature_clusters.get(m, []))
        if not feats:
            return []
        idx = np.asarray(feats, dtype=np.int64)
        idx = idx[idx < len(activations)]
        if len(idx) == 0:
            return []
        vals = activations[idx]
        order = np.argsort(vals)[-top_n:][::-1]
        return [f"SAE-Feat_{int(idx[i])} (act={vals[i]:.2f})" for i in order]

    def _score_data_clusters(self, signal: Dict[int, float], feat_for_keywords: np.ndarray) -> List[Dict[str, Any]]:
        """Shared scorer for both inspect modes: rank data clusters B_k by live-evidence-weighted hypotheses.

        ``signal`` is the live per-feature-cluster activity (Mode A: sum over prompt
        features; Mode B: mean of per-feature u). Each scored cluster carries its
        hypotheses + best hypothesis for the downstream shift extraction.
        """
        label_map = {cl.get("cluster_id"): cl for cl in self.cluster_labels}
        scored = []
        for k, hypos in self.feature_hypotheses_map.items():
            ev = self._hypothesis_evidence(hypos, signal)
            if not ev:
                continue
            best_h, best_ev = max(ev, key=lambda t: t[1])
            if best_ev <= 0:
                continue
            cl_info = label_map.get(k, {})
            title = cl_info.get("title", f"Data Topic B_{k}")
            desc = cl_info.get("description", f"Active dataset topic cluster with {len(hypos)} verified hypotheses.")
            feature_ms = {h.get("m") for h in hypos if h.get("m") is not None}
            scored.append({
                "cluster_id": k,
                "title": title,
                "description": desc,
                "matched_keywords": self._cluster_keywords(feat_for_keywords, feature_ms),
                "relevance_score": float(best_ev),
                "best_hypothesis": best_h,
                "hypos": hypos,
            })
        return sorted(scored, key=lambda x: x["relevance_score"], reverse=True)

    @staticmethod
    def _project_clusters(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop the internal hypos/best_hypothesis payload before sending to the client."""
        return [{
            "cluster_id": c["cluster_id"],
            "title": c["title"],
            "description": c["description"],
            "matched_keywords": c["matched_keywords"],
            "relevance_score": c["relevance_score"],
        } for c in scored]

    @property
    def feature_to_cluster_map(self) -> Dict[int, int]:
        """Inverse map feature index -> feature cluster T_m (lazy, cached)."""
        if self._feat_to_cluster is None:
            m: Dict[int, int] = {}
            for cl, feats in self.feature_clusters.items():
                for f in feats:
                    m[int(f)] = cl
            self._feat_to_cluster = m
        return self._feat_to_cluster

    @staticmethod
    def _neuronpedia_sae_set(sae_cfg: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Map the run's SAE to a (model_id, sae_set) Neuronpedia pair, or None."""
        repo = sae_cfg.get("repo", "")
        layer = sae_cfg.get("layer")
        d_sae = int(sae_cfg.get("d_sae") or 0)
        width = {16384: "16k", 32768: "32k", 65536: "65k", 131072: "131k",
                 262144: "262k", 524288: "524k", 1048576: "1m"}.get(d_sae, "16k")
        if repo == "gemma-scope-2b-pt-res-canonical":
            return ("gemma-2-2b", f"{layer}-gemmascope-res-{width}")
        if "adamkarvonen" in repo.lower():
            return ("qwen3-1.7b", f"{layer}-resid-batchtopk-65k__l0-80")
        return None

    @staticmethod
    @functools.lru_cache(maxsize=16)
    def _neuronpedia_verified(model_id: str, sae_set: str) -> bool:
        """One-time, cached check that the Neuronpedia slug actually resolves."""
        try:
            import urllib.request
            url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_set}/100"
            req = urllib.request.Request(
                url, method="GET",
                headers={"User-Agent": "PDD-Viewer/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status == 200
        except Exception as e:
            logger.warning(f"Neuronpedia slug not verified ({url}): {e}")
            return False

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def _neuronpedia_feature(model_id: str, sae_set: str, f: int) -> Optional[Dict[str, Any]]:
        """Cached per-feature Neuronpedia metadata (auto-interpretation payload). None on any failure."""
        import urllib.request

        url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_set}/{f}"
        try:
            req = urllib.request.Request(
                url, method="GET",
                headers={"User-Agent": "PDD-Viewer/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                d = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            logger.warning(f"Neuronpedia feature fetch failed ({url}): {e}")
            return None
        pos = list(zip(d.get("pos_str") or [], d.get("pos_values") or []))[:10]
        neg = list(zip(d.get("neg_str") or [], d.get("neg_values") or []))[:8]
        return {
            "name": d.get("name"),
            "description": d.get("description"),
            "max_act_approx": d.get("maxActApprox"),
            "pos_tokens": [{"token": t, "value": v} for t, v in pos],
            "neg_tokens": [{"token": t, "value": v} for t, v in neg],
            "correlated_features": (d.get("correlated_features_indices") or [])[:10],
            "aligned_neurons": (d.get("neuron_alignment_indices") or [])[:10],
        }

    def _neuronpedia_set(self) -> Optional[Tuple[str, str]]:
        """Cached (model_id, sae_set) pair only if the Neuronpedia slug was runtime-verified."""
        if self._np_set is None:
            slug = self._neuronpedia_sae_set(self.summary.get("config", {}).get("sae", {}))
            if slug and self._neuronpedia_verified(*slug):
                self._np_set = slug
        return self._np_set

    def _neuronpedia_url(self, feature_index: int) -> Optional[str]:
        np_set = self._neuronpedia_set()
        if np_set is None:
            return None
        return f"https://www.neuronpedia.org/{np_set[0]}/{np_set[1]}/{feature_index}"

    def _top_sae_features(self, activations: np.ndarray, top_n: int = 8) -> List[Dict[str, Any]]:
        """Top individual SAE features by activation, with cluster membership + Neuronpedia link.

        Only emits a Neuronpedia URL when the slug was runtime-verified, so we never
        surface 404 links. When the SAE has no dashboard, cluster-membership + activation
        remain for the example-based interpretation fallback.
        """
        if activations is None or len(activations) == 0:
            return []
        order = np.argsort(activations)[-top_n:][::-1]
        ftoc = self.feature_to_cluster_map
        feat_delta = self._feature_delta()
        out: List[Dict[str, Any]] = []
        for i in order:
            i = int(i)
            val = float(activations[i])
            if val == 0.0:
                continue
            item: Dict[str, Any] = {"feature_index": i, "activation": val, "cluster_m": ftoc.get(i)}
            url = self._neuronpedia_url(i)
            if url:
                item["neuronpedia_url"] = url
            if feat_delta is not None and i < feat_delta.shape[0]:
                d = float(feat_delta[i])
                item["dp_delta"] = d
                item["dp_direction"] = "amplified" if d > 1e-4 else ("suppressed" if d < -1e-4 else "neutral")
            out.append(item)
        return out

    def _feature_delta(self) -> Optional[np.ndarray]:
        """Per-feature DPO-push signal: u_f = P(fires in chosen) - P(fires in rejected).

        Computed lazily (one column-mean pass over the cached C_max/R_max sparse
        matrices, ~d_sae floats) and cached. Positive => DPO amplifies the feature,
        negative => DPO suppresses it. The paper's per-feature primitive `u` (B.1).
        """
        if self._feat_delta is None:
            mats = self._load_feature_matrices()
            if mats is None:
                return None
            try:
                c_rate = (mats.C_max > 0.01).mean(axis=0)
                r_rate = (mats.R_max > 0.01).mean(axis=0)
                self._feat_delta = (
                    np.asarray(c_rate).ravel() - np.asarray(r_rate).ravel()
                ).astype(np.float32)
            except Exception as e:
                logger.warning(f"Error computing per-feature delta: {e}")
                return None
        return self._feat_delta

    def _load_feature_matrices(self):
        """Lazily mmap the cached per-example feature matrices for example-based cluster interpretation."""
        if self._feature_matrices is None and self.checkpoint_dir is not None:
            from pdd.feature_matrices import FeatureMatrices
            mmap_dir = self.checkpoint_dir / "matrices_mmap"
            npz = self.checkpoint_dir / "matrices.npz"
            try:
                if mmap_dir.is_dir():
                    self._feature_matrices = FeatureMatrices.load_mmap_dir(str(mmap_dir))
                elif npz.exists():
                    self._feature_matrices = FeatureMatrices.load_npz(str(npz))
            except Exception as e:
                logger.warning(f"Error loading feature matrices for interpretation: {e}")
        return self._feature_matrices

    @staticmethod
    def _ex_get(ex: Any, key: str) -> str:
        if isinstance(ex, dict):
            return ex.get(key, "") or ""
        return getattr(ex, key, "") or ""

    def _load_examples(self):
        """Lazily load the cached dataset examples for example-based cluster interpretation."""
        if self._examples is None and self.checkpoint_dir is not None:
            ex_path = self.checkpoint_dir / "examples.json"
            if ex_path.exists():
                try:
                    import orjson
                    with open(ex_path, "rb") as f:
                        self._examples = orjson.loads(f.read())
                except Exception:
                    try:
                        with open(ex_path, "r", encoding="utf-8") as f:
                            self._examples = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading examples for interpretation: {e}")
        return self._examples

    def _load_pc_cluster_examples(self):
        """Lazily load per-cluster top example indices for prompt clusters A_k / response-delta clusters R_m.

        Written by the auto-labeling pipeline stage (prompt_conditioned_cluster_examples.json);
        gives each A_k/R_m a concrete, readable meaning via the real examples that
        express it (no SAE feature breakdown).
        """
        if self._pc_cluster_examples is None:
            pc_path = Path(pc_cluster_examples_path(str(self.run_dir)))
            if pc_path.exists():
                try:
                    with open(pc_path, "r", encoding="utf-8") as f:
                        self._pc_cluster_examples = json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading prompt-conditioned cluster examples: {e}")
        return self._pc_cluster_examples

    def _pc_cluster_tokens(self, cluster_type: str, cid: int) -> List[str]:
        """Score-weighted top content tokens expressing prompt cluster A_k / response-delta cluster R_m."""
        pc = self._load_pc_cluster_examples()
        if pc is None:
            return []
        key = f"{cluster_type}_cluster_tokens"
        return pc.get(key, {}).get(str(cid), [])

    def _pc_cluster_top_examples(self, cluster_type: str, cid: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Real examples expressing prompt cluster A_k or response-delta cluster R_m."""
        pc = self._load_pc_cluster_examples()
        examples = self._load_examples()
        if pc is None or examples is None:
            return []
        key = f"{cluster_type}_cluster_examples"
        idxs = pc.get(key, {}).get(str(cid), [])
        if not idxs:
            return []
        out = []
        for i in idxs[:top_n]:
            if int(i) >= len(examples):
                continue
            ex = examples[int(i)]
            if cluster_type == "response":
                desc = f"chosen-vs-rejected SAE delta signal (cluster R_{cid})"
            else:
                desc = f"prompt expressing A_{cid}"
            out.append({
                "index": int(i),
                "prompt": self._ex_get(ex, "prompt"),
                "chosen": self._ex_get(ex, "chosen"),
                "rejected": self._ex_get(ex, "rejected"),
                "note": desc,
            })
        return out

    def _load_feature_cluster_labels(self) -> Dict[int, Dict[str, Any]]:
        """Lazily load whole-cluster LLM labels for SAE feature clusters T_m."""
        if self._feature_cluster_labels is None:
            self._feature_cluster_labels = {}
            lbl_file = Path(feature_cluster_labels_path(str(self.run_dir)))
            if lbl_file.exists():
                try:
                    with open(lbl_file, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                        self._feature_cluster_labels = {int(k): v for k, v in raw.get("feature_clusters", {}).items()}
                except Exception as e:
                    logger.warning(f"Error loading feature cluster labels: {e}")
        return self._feature_cluster_labels

    def _feature_firings(self) -> Optional[np.ndarray]:
        """Total firings for all SAE features across C_max and R_max in a single fast 10ms pass."""
        if self._feature_totals is None:
            mats = self._load_feature_matrices()
            if mats is None:
                return None
            try:
                c_data = mats.C_max.data
                c_idx = mats.C_max.indices
                r_data = mats.R_max.data
                r_idx = mats.R_max.indices
                d_sae = mats.C_max.shape[1]
                c_sum = np.bincount(c_idx, weights=c_data, minlength=d_sae)
                r_sum = np.bincount(r_idx, weights=r_data, minlength=d_sae)
                self._feature_totals = (c_sum + r_sum).astype(np.float32)
            except Exception as e:
                logger.warning(f"Error computing fast feature totals: {e}")
                return None
        return self._feature_totals

    def _top_cluster_features(self, m: int, top_n: int = 8) -> List[Dict[str, Any]]:
        """Top SAE features inside feature cluster T_m, ranked by firing in < 1ms."""
        m_int = int(m)
        feats = self.feature_clusters.get(m_int, [])
        if not feats:
            return []
        tot = self._feature_firings()
        if tot is not None and len(tot) > 0:
            d_sae = len(tot)
            feats = [f for f in feats if 0 <= f < d_sae]
            if feats:
                firings = tot[feats]
                order = np.argsort(firings)[-top_n:][::-1]
                return [{
                    "feature_index": int(feats[j]),
                    "firing": float(firings[j]),
                    "neuronpedia_url": self._neuronpedia_url(int(feats[j])),
                } for j in order]
        return [{"feature_index": int(f), "firing": 0.0, "neuronpedia_url": self._neuronpedia_url(int(f))} for f in feats[:top_n]]

    def _feature_cluster_info(self, m: int, top_n_examples: int = 5) -> Dict[str, Any]:
        """One payload for the T_m dropdown: whole-cluster label + top member features + real examples."""
        m_int = int(m)
        cache_key = f"T_{m_int}_{top_n_examples}"
        if cache_key in self._cluster_info_cache:
            return self._cluster_info_cache[cache_key]

        feats = self.feature_clusters.get(m_int, [])
        label = self._load_feature_cluster_labels().get(m_int)
        res = {
            "cluster_m": m_int,
            "label": label or {"title": f"Feature cluster T_{m}", "description": "", "keywords": []},
            "n_features": len(feats),
            "top_features": self._top_cluster_features(m_int, top_n=8),
            "examples": self._cluster_top_examples(m_int, top_n=top_n_examples),
        }
        self._cluster_info_cache[cache_key] = res
        return res

    def _data_cluster_info(self, k: int, top_n_examples: int = 5) -> Dict[str, Any]:
        """Interpretation for data cluster B_k: title, description, keywords, and sampled centroid/random prompts."""
        k_int = int(k)
        cache_key = f"B_{k_int}_{top_n_examples}"
        if cache_key in self._cluster_info_cache:
            return self._cluster_info_cache[cache_key]

        label_obj = None
        for lab in (self.cluster_labels or []):
            if lab.get("cluster_id") == k_int:
                label_obj = lab
                break
        if label_obj is None:
            label_obj = {
                "cluster_id": k_int,
                "title": f"Data Cluster B_{k}",
                "description": "",
                "keywords": [],
                "centroid_prompts": [],
                "sample_prompts": [],
            }
        res = {
            "cluster_id": k_int,
            "title": label_obj.get("title", f"Data Cluster B_{k}"),
            "description": label_obj.get("description", ""),
            "keywords": label_obj.get("keywords", []),
            "centroid_prompts": label_obj.get("centroid_prompts", [])[:top_n_examples],
            "sample_prompts": label_obj.get("sample_prompts", [])[:top_n_examples],
        }
        self._cluster_info_cache[cache_key] = res
        return res

    def _cluster_top_examples(self, m: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Top dataset examples firing feature cluster T_m in < 10ms via searchsorted on CSR indices."""
        mats = self._load_feature_matrices()
        examples = self._load_examples()
        m_int = int(m)
        feats = self.feature_clusters.get(m_int, [])
        if mats is None or examples is None or not feats:
            return []

        top_mems = self._top_cluster_features(m_int, top_n=5)
        top_f_indices = np.array([f["feature_index"] for f in top_mems], dtype=np.int64)
        if len(top_f_indices) == 0:
            return []

        row_scores: Dict[int, float] = {}
        try:
            for mat in (mats.C_max, mats.R_max):
                mask = np.isin(mat.indices, top_f_indices)
                if not np.any(mask):
                    continue
                pos = np.nonzero(mask)[0]
                rows = np.searchsorted(mat.indptr, pos, side="right") - 1
                weights = mat.data[pos]
                for r, w in zip(rows, weights):
                    r_int = int(r)
                    row_scores[r_int] = row_scores.get(r_int, 0.0) + float(w)
        except Exception as e:
            logger.warning(f"Fast example lookup fallback: {e}")

        order = sorted(row_scores.keys(), key=lambda x: row_scores[x], reverse=True)[:top_n]
        out: List[Dict[str, Any]] = []
        for i in order:
            if int(i) >= len(examples):
                continue
            ex = examples[int(i)]
            out.append({
                "index": int(i),
                "score": float(row_scores[i]),
                "prompt": self._ex_get(ex, "prompt")[-600:],
                "chosen": self._ex_get(ex, "chosen")[-400:],
                "rejected": self._ex_get(ex, "rejected")[-400:],
            })
        return out

    def _feature_detail(self, f: int, top_n: int = 5) -> Dict[str, Any]:
        """Per-feature interpretation: run firing stats, top firing examples, Neuronpedia metadata."""
        f = int(f)
        cache_key = f"feat_{f}_{top_n}"
        if cache_key in self._cluster_info_cache:
            return self._cluster_info_cache[cache_key]

        mats = self._load_feature_matrices()
        examples = self._load_examples()
        out: Dict[str, Any] = {"feature_index": f}
        if mats is None:
            out["error"] = "feature matrices not cached for this run"
            return out
        d_sae = mats.C_max.shape[1]
        if not (0 <= f < d_sae):
            out["error"] = f"feature index out of range (d_sae={d_sae})"
            return out

        def _col(a: Any, i: int) -> np.ndarray:
            if hasattr(a, "toarray"):
                return np.asarray(a[:, i].toarray()).ravel()
            return np.asarray(a[:, i]).ravel()

        act = _col(mats.C_max, f) + _col(mats.R_max, f)
        pos_mask = act > 0
        n_firing = int(pos_mask.sum())
        pos_vals = act[pos_mask]
        firing_stats: Dict[str, float] = {
            "n_examples": n_firing,
            "n_total": int(len(act)),
            "max": float(pos_vals.max()) if n_firing else 0.0,
            "mean": float(pos_vals.mean()) if n_firing else 0.0,
        }
        out["firing"] = firing_stats

        examples_out: List[Dict[str, Any]] = []
        if examples is not None:
            order = np.argsort(act)[::-1]
            for i in order:
                if act[i] <= 0:
                    break
                if int(i) >= len(examples):
                    continue
                ex = examples[int(i)]
                examples_out.append({
                    "index": int(i),
                    "score": float(act[i]),
                    "prompt": self._ex_get(ex, "prompt")[-600:],
                    "chosen": self._ex_get(ex, "chosen")[-400:],
                    "rejected": self._ex_get(ex, "rejected")[-400:],
                })
                if len(examples_out) >= top_n:
                    break
        out["examples"] = examples_out

        url = self._neuronpedia_url(f)
        if url:
            out["neuronpedia_url"] = url
            np_set = self._neuronpedia_set()
            if np_set:
                np_data = self._neuronpedia_feature(np_set[0], np_set[1], f)
                if np_data:
                    out["neuronpedia"] = np_data
        self._cluster_info_cache[cache_key] = out
        return out

    def get_inspector(self):
        """Get or initialize the NeuralInspector configured for this target run."""
        if self.inspector is None:
            from pdd.neural_inspector import get_neural_inspector
            model_cfg = self.summary.get("config", {}).get("model", {})
            sae_cfg = self.summary.get("config", {}).get("sae", {})

            model_path = model_cfg.get("path", "google/gemma-2-2b")
            sae_repo = sae_cfg.get("repo", "gemma-scope-2b-pt-res-canonical")
            sae_id = sae_cfg.get("sae_id")
            layer = sae_cfg.get("layer", 12)
            d_in = sae_cfg.get("d_in")
            d_sae = sae_cfg.get("d_sae")
            k = sae_cfg.get("k")

            self.inspector = get_neural_inspector(
                model_path=model_path,
                sae_repo=sae_repo,
                sae_id=sae_id,
                layer=layer,
                d_in=d_in,
                d_sae=d_sae,
                k=k,
            )
        return self.inspector


_STATE: Optional[ViewerState] = None


def get_state() -> ViewerState:
    """Lazy global state accessor for the single targeted run (set via --run_dir)."""
    global _STATE
    if _STATE is None:
        _STATE = ViewerState()
    return _STATE


@app.get("/api/runs")
def list_runs() -> Dict[str, Any]:
    """Return the active targeted run only (single-run mode)."""
    state = get_state()
    metrics = state.summary.get("metrics", {})
    return {
        "runs": [{
            "name": state.run_dir.name,
            "path": str(state.run_dir),
            "timestamp": state.summary.get("timestamp", "N/A"),
            "config_name": state.summary.get("config", {}).get("name", state.run_dir.name),
            "model": state.summary.get("config", {}).get("model", {}).get("path", "N/A"),
            "sae": state.summary.get("config", {}).get("sae", {}).get("repo", "N/A"),
            "num_examples": metrics.get("num_examples", 0),
            "num_clusters": metrics.get("num_sae_feature_clusters", 0),
        }]
    }


@app.get("/api/run_data")
def get_run_data() -> Dict[str, Any]:
    """Retrieve summary, validation metrics, cluster labels, and top hypotheses for the targeted run."""
    state = get_state()
    val_file = state.run_dir / "p4_validation" / "p4_r2_metrics.json"
    validation_metrics = {}
    if val_file.exists():
        try:
            with open(val_file, "r", encoding="utf-8") as f:
                validation_metrics = json.load(f)
        except Exception:
            pass

    # Reload the full hypothesis lists (cached per run) so tables show top-100
    # ranked by the pipeline's saved ordering instead of the summary top-10.
    # The properties self-log any load failure and leave the summary seed intact.
    state.feature_hypotheses_map
    state.prompt_hypotheses_map

    return {
        "summary": state.summary,
        "validation_metrics": validation_metrics,
        "cluster_labels": state.cluster_labels,
        "feature_cluster_labels": state._load_feature_cluster_labels(),
        "top_feature_conditioned_hypotheses": state.fc_hypos[:100],
        "top_prompt_conditioned_hypotheses": state.pc_hypos[:100],
    }


@app.get("/api/feature_cluster_info")
def get_feature_cluster_info(m: int = Query(..., description="Feature cluster id T_m"),
                             top_n: int = Query(5)) -> Dict[str, Any]:
    """Whole-cluster interpretation for SAE feature cluster T_m."""
    state = get_state()
    return state._feature_cluster_info(m, top_n_examples=top_n)


@app.get("/api/pc_cluster_examples")
def get_pc_cluster_examples(cluster_type: str = Query(..., description="'prompt' for A_k or 'response' for R_m"),
                            cid: int = Query(..., description="Cluster id"),
                            top_n: int = Query(5)) -> Dict[str, Any]:
    """Meaning for prompt cluster A_k / response-delta cluster R_m: the real examples that express them."""
    state = get_state()
    if cluster_type not in ("prompt", "response"):
        raise HTTPException(400, "cluster_type must be 'prompt' or 'response'.")
    examples = state._pc_cluster_top_examples(cluster_type, cid, top_n=top_n)
    tokens = state._pc_cluster_tokens(cluster_type, cid)
    return {
        "cluster_type": cluster_type,
        "cluster_id": cid,
        "tokens": tokens,
        "examples": examples,
    }


@app.get("/api/cluster_detail")
def get_cluster_detail(type: str = Query(..., description="'data' (B_k), 'feature' (T_m), 'prompt' (A_k), or 'response' (R_m)"),
                       id: int = Query(..., description="Cluster ID integer"),
                       top_n: int = Query(5)) -> Dict[str, Any]:
    """Unified endpoint to fetch full interpretation for any of the 4 cluster types."""
    state = get_state()
    t = type.lower().strip()
    if t in ("data", "b", "bk"):
        return {"cluster_family": "B", "cluster_type": "data", "id": id, **state._data_cluster_info(id, top_n_examples=top_n)}
    elif t in ("feature", "t", "tm"):
        return {"cluster_family": "T", "cluster_type": "feature", "id": id, **state._feature_cluster_info(id, top_n_examples=top_n)}
    elif t in ("prompt", "a", "ak"):
        return {
            "cluster_family": "A",
            "cluster_type": "prompt",
            "id": id,
            "tokens": state._pc_cluster_tokens("prompt", id),
            "examples": state._pc_cluster_top_examples("prompt", id, top_n=top_n),
        }
    elif t in ("response", "r", "rm"):
        return {
            "cluster_family": "R",
            "cluster_type": "response",
            "id": id,
            "tokens": state._pc_cluster_tokens("response", id),
            "examples": state._pc_cluster_top_examples("response", id, top_n=top_n),
        }
    raise HTTPException(400, f"Unsupported cluster type: '{type}'. Allowed: data (B), feature (T), prompt (A), response (R).")


@app.get("/api/feature_detail")
def get_feature_detail(f: int = Query(..., ge=0, description="SAE feature index"),
                       top_n: int = Query(5)) -> Dict[str, Any]:
    """Per-feature interpretation: run firing stats + top examples + cached Neuronpedia metadata."""
    state = get_state()
    return state._feature_detail(f, top_n=top_n)


@app.post("/api/inspect_prompt")
def inspect_prompt(req: PromptInspectionRequest) -> Dict[str, Any]:
    """Mode A: Inspect prompt through live GPU Model + SAE forward pass and predict downstream behavioral shifts.

    Cluster matching is driven by the *live* prompt activation: each hypothesis
    (k, m) is scored by |delta_{k,m}| x (prompt's per-feature-cluster T_m signal),
    so different prompts genuinely match different data clusters / concepts.
    """
    state = get_state()
    prompt_text = req.prompt.strip()
    if not prompt_text:
        return {"prompt": "", "matched_clusters": [], "predicted_behavior_shifts": []}

    # 1. Real GPU Forward Pass -> SAE Features P(x)
    inspector = state.get_inspector()
    p_feat = inspector.extract_prompt_features(prompt_text)

    # 2. Per-feature-cluster activity of the live prompt (T_m <- sum of P(x) members)
    act = state._cluster_signals(p_feat, mode="sum")

    scored_clusters = state._score_data_clusters(act, p_feat)[:req.top_k]
    matched_clusters = state._project_clusters(scored_clusters)

    # 3. Extract Predicted Shifts from live-evidence-ranked hypotheses
    predicted_shifts = []
    for c in scored_clusters:
        ordered = sorted(c["hypos"], key=lambda h: abs(float(h.get("delta", 0.0))) * abs(act.get(h.get("m"), 0.0)), reverse=True)
        for h in ordered[:2]:
            k = h.get("k")
            m = h.get("m")
            delta = float(h.get("delta", 0.0))
            z = float(h.get("z_score", 0.0))
            d = float(h.get("cohens_d", 0.0))
            evidence = act.get(m, 0.0)
            is_amplified = delta > 0

            direction_word = "AMPLIFIED (Boosted)" if is_amplified else "SUPPRESSED (Inhibited)"
            interpretation = (
                f"This prompt fires SAE feature cluster T_{m} with live activity {evidence:.3f}. "
                f"In the training data, examples of type B_{k} are chosen-leaning on this cluster "
                f"(Δ = {delta:+.5f}, Welch z = {z:.2f}), so post-training will likely {direction_word} "
                f"this response behavior for similar prompts."
            )

            predicted_shifts.append({
                "prompt_cluster_k": k,
                "response_cluster_m": m,
                "delta": delta,
                "effect_direction": "Amplified after DPO" if is_amplified else "Suppressed after DPO",
                "z_score": z,
                "cohens_d": d,
                "live_activity": evidence,
                "interpretation": interpretation,
            })

    predicted_shifts.sort(key=lambda s: abs(float(s["delta"])) * abs(float(s.get("live_activity", 0.0))), reverse=True)
    predicted_shifts = predicted_shifts[:10]

    return {
        "prompt": prompt_text,
        "matched_clusters": matched_clusters,
        "predicted_behavior_shifts": predicted_shifts,
        "top_sae_features": state._top_sae_features(p_feat),
    }


@app.post("/api/inspect_preference_pair")
def inspect_preference_pair(req: PreferencePairInspectionRequest) -> Dict[str, Any]:
    """Mode B: Batched GPU forward pass on preference pair to measure exact SAE disparity u = 1(C>0.01) - 1(R>0.01).

    Promoted/suppressed concepts are selected by the *live* per-feature-cluster
    disparity aggregated from u, not by precomputed hypothesis deltas alone.
    """
    state = get_state()
    prompt_text = req.prompt.strip()
    chosen_text = req.chosen.strip()
    rejected_text = req.rejected.strip()

    if not prompt_text or not chosen_text or not rejected_text:
        return {"matched_clusters": [], "promoted_concepts": [], "suppressed_concepts": []}

    # 1. Batched GPU Forward Pass -> Chosen (C), Rejected (R), and Disparity (u)
    inspector = state.get_inspector()
    c_p, r_p, u = inspector.extract_pair_features(prompt_text, chosen_text, rejected_text)

    # 2. Per-feature-cluster live disparity: u_m = mean over T_m members of u
    u_sig = state._cluster_signals(u, mode="mean")
    pair_act = c_p + r_p

    # 3. Score Data Clusters B_k by live-evidence-weighted hypotheses
    scored_clusters = state._score_data_clusters(u_sig, pair_act)[:req.top_k]
    matched_clusters = state._project_clusters(scored_clusters)

    # 4. Extract Promoted vs. Suppressed Concepts from the LIVE disparity
    # Map feature cluster m -> strongest hypothesis referencing it (for context).
    best_by_m: Dict[int, Dict[str, Any]] = {}
    for k, hypos in state.feature_hypotheses_map.items():
        for h in hypos:
            m = h.get("m")
            if m is None:
                continue
            d = abs(float(h.get("delta", 0.0)))
            if d > best_by_m.get(m, {}).get("_abs_delta", 0.0):
                best_by_m[m] = dict(h, _abs_delta=d)

    promoted_concepts = []
    suppressed_concepts = []
    for m, uval in sorted(u_sig.items(), key=lambda t: abs(t[1]), reverse=True):
        if uval == 0.0:
            continue
        h = best_by_m.get(m, {})
        k = h.get("k")
        z = float(h.get("z_score", 0.0))
        is_chosen = uval > 0
        item = {
            "feature_cluster_m": m,
            "data_cluster_k": k,
            "delta": float(uval),
            "hypothesis_delta": float(h.get("delta", 0.0)),
            "z_score": z,
            "signal_strength": "Strong" if abs(uval) > 0.15 else "Moderate",
            "explanation": (
                f"Live SAE disparity: the chosen response fires feature cluster T_{m} "
                f"more than the rejected (net u = {uval:+.3f})."
                + (f" Consistent with training hypothesis B_{k} (Δ = {h.get('delta', 0.0):+.4f}, Welch z = {z:.2f})." if k is not None else "")
            ),
        }
        if is_chosen:
            promoted_concepts.append(item)
        else:
            suppressed_concepts.append(item)

    promoted_concepts.sort(key=lambda x: x["delta"], reverse=True)
    suppressed_concepts.sort(key=lambda x: x["delta"])
    promoted_concepts = promoted_concepts[:5]
    suppressed_concepts = suppressed_concepts[:5]

    return {
        "prompt": prompt_text,
        "chosen_length": len(chosen_text),
        "rejected_length": len(rejected_text),
        "promoted_sae_features_count": int((u > 0).sum()),
        "suppressed_sae_features_count": int((u < 0).sum()),
        "matched_clusters": matched_clusters,
        "promoted_concepts": promoted_concepts,
        "suppressed_concepts": suppressed_concepts,
        "top_sae_features": state._top_sae_features(np.abs(u)),
    }


# Mount Frontend Static Assets
if VIEWER_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(VIEWER_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = VIEWER_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"status": "PDD Viewer Backend Running", "frontend_path": str(index_file)})


def main():
    parser = argparse.ArgumentParser(description="Launch PDD Interactive Web Viewer")
    parser.add_argument("--run_dir", type=str, default="runs/qwen3_1.7b_dolci", help="Path to target PDD run directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7000, help="Port to bind server")
    args = parser.parse_args()

    os.environ["PDD_RUN_DIR"] = args.run_dir
    global _STATE
    _STATE = ViewerState(run_dir=args.run_dir)

    # Pre-warm cached examples in a background daemon thread to eliminate first-click disk latency
    import threading
    threading.Thread(target=_STATE._load_examples, daemon=True).start()

    logger.info(f"Starting PDD Viewer for '{args.run_dir}' at http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
