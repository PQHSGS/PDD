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
import threading
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


def _read_json(path: Path) -> Optional[Any]:
    """Read a JSON file; return None (with a warning log) instead of raising on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error reading {path.name}: {e}")
        return None


def _mtime_of(path: Path) -> float:
    """mtime of ``path`` (0.0 when missing) so lazy loaders can detect pipeline updates."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0

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
        self._cluster_labels_raw: Optional[Any] = None
        self.fc_hypos: List[Dict[str, Any]] = []
        self.pc_hypos: List[Dict[str, Any]] = []
        self.k_to_fc: Dict[int, List[Dict[str, Any]]] = {}
        self.k_to_pc: Dict[int, List[Dict[str, Any]]] = {}
        self.inspector = None
        self._examples_lock = threading.Lock()
        self._scores_lock = threading.Lock()
        self._member_cache_lock = threading.Lock()
        self._feat_to_cluster: Optional[Dict[int, int]] = None
        self._feature_matrices = None
        self._examples = None
        self._feat_delta: Optional[np.ndarray] = None
        self._pc_cluster_examples: Optional[Any] = None
        self._feature_cluster_labels: Optional[Any] = None
        self._feature_cluster_labels_mtime: float = 0.0
        self._pc_cluster_examples_mtime: float = 0.0
        self._cluster_labels_mtime: float = 0.0
        self._np_set: Optional[Tuple[str, str]] = None
        self._np_verifying: bool = False
        self._np_verify_lock = threading.Lock()
        self._best_hypo_by_m: Optional[Dict[int, Dict[str, Any]]] = None
        self._cluster_info_cache: Dict[str, Any] = {}
        self._feature_totals: Optional[np.ndarray] = None
        self._all_member_cols: Optional[np.ndarray] = None
        self._member_positions_cache: Dict[str, np.ndarray] = {}
        self._member_matrices: Dict[str, np.ndarray] = {}
        self._example_u: Optional[np.ndarray] = None
        self._example_s: Optional[np.ndarray] = None
        self._example_cluster_ids: Optional[np.ndarray] = None

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
        sum_data = _read_json(sum_path)
        if sum_data is not None:
            self.summary = sum_data

        # 2. Resolve Checkpoint Subfolder for Cluster Maps
        ckpt_path_str = self.summary.get("checkpoint_subfolder")
        if ckpt_path_str and Path(ckpt_path_str).exists():
            self.checkpoint_dir = Path(ckpt_path_str)
            clusters_file = self.checkpoint_dir / "clusters.json"
            raw_clusters = _read_json(clusters_file)
            if raw_clusters is not None:
                self.feature_clusters = {int(k): v for k, v in raw_clusters.get("clusters", {}).items()}

        # 3. Auto-Labels (B_k, T_m, A_k/R_m) are read lazily and refresh automatically
        #    when the pipeline rewrites them (see _load_data_cluster_labels, _load_feature_cluster_labels,
        #    _load_pc_cluster_examples) — no manual rebuild step.

        # 4. Instant Seed from Summary
        # Seed with summary top hypotheses initially
        self.fc_hypos = self.summary.get("top_feature_conditioned_hypotheses", [])
        self.pc_hypos = self.summary.get("top_prompt_conditioned_hypotheses", [])

        logger.info(
            f"ViewerState initialized for '{self.run_dir.name}': "
            f"{len(self.feature_clusters)} feature clusters, ready for instant requests."
        )

    def _load_data_cluster_labels(self) -> List[Dict[str, Any]]:
        """B_k labels, re-read when cluster_labels.json changes (auto-label pipeline update)."""
        raw = self._reload_if_changed(
            Path(cluster_labels_path(str(self.run_dir))), "_cluster_labels_raw", "_cluster_labels_mtime")
        return raw.get("labels", []) if raw is not None else []

    def _reload_if_changed(self, path: Path, cache_attr: str, mtime_attr: str) -> Any:
        """Re-read a JSON artifact when its mtime changes; memoize the raw value otherwise."""
        mtime = _mtime_of(path)
        cached = getattr(self, cache_attr)
        if cached is None or mtime != getattr(self, mtime_attr):
            setattr(self, cache_attr, _read_json(path))
            setattr(self, mtime_attr, mtime)
        return getattr(self, cache_attr)

    @staticmethod
    def _parse_hypotheses(data: Any) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
        """Split a hypotheses artifact into the flat list and the k-indexed map."""
        hypos = data.get("hypotheses", data) if isinstance(data, dict) else data
        k_to: Dict[int, List[Dict[str, Any]]] = {}
        for h in hypos:
            k = h.get("k")
            if k is not None:
                k_to.setdefault(k, []).append(h)
        return hypos, k_to

    @staticmethod
    def _cluster_covered_hypos(hypos: List[Dict[str, Any]], n_per: int) -> List[Dict[str, Any]]:
        """Top-n hypotheses per response-feature cluster (h['m']), preserving pipeline ranking.

        Guarantees every cluster that produced hypotheses appears in the table, even when
        its effect sizes rank far down the global list (the old [:100] slice exposed only
        ~2/3 of the run's T_m clusters).
        """
        out: List[Dict[str, Any]] = []
        per: Dict[int, int] = {}
        for h in hypos:
            m = h.get("m")
            if m is None:
                continue
            if per.get(m, 0) < n_per:
                per[m] = per.get(m, 0) + 1
                out.append(h)
        return out

    @property
    def prompt_hypotheses_map(self) -> Dict[int, List[Dict[str, Any]]]:
        """Lazy load and cache prompt hypotheses map on demand."""
        if not self.k_to_pc:
            data = _read_json(self.run_dir / "prompt_conditioned_hypotheses.json")
            if data is not None:
                self.pc_hypos, self.k_to_pc = self._parse_hypotheses(data)
        return self.k_to_pc

    @property
    def feature_hypotheses_map(self) -> Dict[int, List[Dict[str, Any]]]:
        """Lazy load and cache feature hypotheses map on demand."""
        if not self.k_to_fc:
            data = _read_json(self.run_dir / "feature_conditioned_hypotheses.json")
            if data is not None:
                self.fc_hypos, self.k_to_fc = self._parse_hypotheses(data)
            elif not self.fc_hypos and self.summary:
                self.fc_hypos, self.k_to_fc = self._parse_hypotheses(
                    {"hypotheses": self.summary.get("top_feature_conditioned_hypotheses", [])}
                )
        return self.k_to_fc

    @property
    def _fc_cfg(self) -> Dict[str, Any]:
        """The run's feature-conditioned config block (all thresholds are config-driven)."""
        return self.summary.get("config", {}).get("feature_conditioned", {})

    @property
    def min_feat_cluster_size(self) -> int:
        """Configured minimum feature cluster size for hypothesis emission and predictions (from config JSON)."""
        return int(self._fc_cfg.get("min_feat_cluster_size", 10))

    @property
    def min_data_cluster_size(self) -> int:
        """Configured minimum data cluster size n_k for hypothesis emission (from config JSON)."""
        return int(self._fc_cfg.get("min_data_cluster_size", 25))

    @property
    def min_partition_cluster_size(self) -> int:
        """Configured minimum graph community size for the coordinate partition (from config JSON)."""
        fcl_cfg = self.summary.get("config", {}).get("feature_clustering", {})
        return int(fcl_cfg.get("min_cluster_size", 4))

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
        label_map = {cl.get("cluster_id"): cl for cl in self._load_data_cluster_labels()}
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

    def _score_prompt_conditioned_hypotheses(self, prompt_text: str, p_feat: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Extract top local Prompt-Conditioned hypotheses (A_k x R_m) matching the prompt."""
        import re
        pc_ex = self._load_pc_cluster_examples()
        prompt_hypos_map = self.prompt_hypotheses_map
        if not prompt_hypos_map:
            return []

        prompt_tokens_map = pc_ex.get("prompt_cluster_tokens", {}) if pc_ex else {}
        prompt_words = set(re.findall(r"\w+", prompt_text.lower()))

        scored_ak: List[Tuple[int, float]] = []
        for k_str, tokens in prompt_tokens_map.items():
            try:
                k = int(k_str)
            except (ValueError, TypeError):
                continue
            if not tokens:
                continue
            match_count = sum(1 for t in tokens if t.lower() in prompt_words or any(t.lower() in w for w in prompt_words))
            if match_count > 0:
                scored_ak.append((k, float(match_count) / max(1, len(tokens))))

        if not scored_ak:
            scored_ak = [(int(k), 1.0) for k in list(prompt_hypos_map.keys())[:10]]

        scored_ak.sort(key=lambda x: x[1], reverse=True)

        pc_shifts: List[Dict[str, Any]] = []
        for k, score in scored_ak[:top_k]:
            hypos = prompt_hypos_map.get(k, [])
            if not hypos:
                continue
            sorted_h = sorted(hypos, key=lambda h: abs(float(h.get("cohens_d", 0.0))), reverse=True)
            for h in sorted_h[:2]:
                m = h.get("m")
                delta = float(h.get("delta", 0.0))
                z = float(h.get("z_score", 0.0))
                d = float(h.get("cohens_d", 0.0))
                is_amplified = delta > 0

                p_tokens = self._pc_cluster_tokens("prompt", k)
                r_tokens = self._pc_cluster_tokens("response", m)

                direction_word = "AMPLIFIED (Boosted)" if is_amplified else "SUPPRESSED (Inhibited)"
                interpretation = (
                    f"Prompt matches prompt-feature cluster A_{k} (expressed by: {', '.join(p_tokens[:4]) if p_tokens else 'local prompt subspace'}). "
                    f"In local preference data, this condition shifts response-delta cluster R_{m} "
                    f"(expressed by: {', '.join(r_tokens[:4]) if r_tokens else 'response disparity features'}) "
                    f"with local effect size Cohen's d = {d:.2f} (Δ = {delta:+.5f}, Welch z = {z:.2f}), "
                    f"predicting post-training will {direction_word} this response pattern."
                )

                pc_shifts.append({
                    "prompt_cluster_k": k,
                    "response_cluster_m": m,
                    "pipeline_type": "prompt_conditioned",
                    "delta": delta,
                    "effect_direction": "Amplified after DPO" if is_amplified else "Suppressed after DPO",
                    "z_score": z,
                    "cohens_d": d,
                    "relevance_score": float(score),
                    "prompt_tokens": p_tokens[:8],
                    "response_tokens": r_tokens[:8],
                    "prompt_examples": self._pc_cluster_top_examples("prompt", k, top_n=2),
                    "response_examples": self._pc_cluster_top_examples("response", m, top_n=2),
                    "interpretation": interpretation,
                })

        pc_shifts.sort(key=lambda s: abs(float(s["cohens_d"])) * float(s.get("relevance_score", 1.0)), reverse=True)
        return pc_shifts[:10]

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

    def _best_hypothesis_by_m(self) -> Dict[int, Dict[str, Any]]:
        """Strongest hypothesis per feature cluster T_m (by |delta|), computed once per run."""
        if self._best_hypo_by_m is None:
            best: Dict[int, Dict[str, Any]] = {}
            for hypos in self.feature_hypotheses_map.values():
                for h in hypos:
                    m = h.get("m")
                    if m is None:
                        continue
                    d = abs(float(h.get("delta", 0.0)))
                    if d > best.get(m, {}).get("_abs_delta", 0.0):
                        best[m] = dict(h, _abs_delta=d)
            self._best_hypo_by_m = best
        return self._best_hypo_by_m

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
        import urllib.request
        url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_set}/100"
        try:
            req = urllib.request.Request(
                url, method="GET",
                headers={"User-Agent": "PDD-Viewer/1.0", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status == 200
        except Exception as e:
            logger.warning(f"Neuronpedia slug not verified ({url}): {e}")
            return False

    def _neuronpedia_cache_dir(self) -> Path:
        """Persistent disk directory for cached Neuronpedia responses."""
        p = self.run_dir / "viewer_cache" / "neuronpedia"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _get_neuronpedia_feature(self, f: int) -> Optional[Dict[str, Any]]:
        """Get Neuronpedia metadata using memory cache -> persistent disk cache -> HTTP fetch."""
        np_set = self._neuronpedia_set()
        if not np_set:
            return None

        # 1. Check persistent disk cache
        cache_file = self._neuronpedia_cache_dir() / f"{f}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception as e:
                logger.warning(f"Corrupt Neuronpedia cache file {cache_file.name}; refetching: {e}")

        # 2. Fetch over HTTP
        data = self._neuronpedia_feature(np_set[0], np_set[1], f)
        if data is not None:
            try:
                with open(cache_file, "w", encoding="utf-8") as fp:
                    json.dump(data, fp)
            except Exception as e:
                logger.debug(f"Failed to persist Neuronpedia cache for feature {f}: {e}")
        return data

    def _prewarm_neuronpedia_features(self, feature_indices: List[int]) -> None:
        """Asynchronously pre-warm Neuronpedia cache in the background for predicted features."""
        np_set = self._neuronpedia_set()
        if not np_set or not feature_indices:
            return

        def _worker() -> None:
            for f_idx in feature_indices[:16]:
                try:
                    self._get_neuronpedia_feature(int(f_idx))
                except Exception as e:
                    logger.warning(f"Neuronpedia pre-warm failed for feature {f_idx}: {e}")

        threading.Thread(target=_worker, daemon=True, name="PDD-NeuronpediaPrewarmer").start()

    @staticmethod
    @functools.lru_cache(maxsize=1024)
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
        ex = d.get("explanations") or []
        auto = next((e for e in ex if e.get("typeName") == "auto" and e.get("description")), None)
        expl = auto or next((e for e in ex if e.get("description")), None)
        pos = list(zip(d.get("pos_str") or [], d.get("pos_values") or []))[:10]
        neg = list(zip(d.get("neg_str") or [], d.get("neg_values") or []))[:8]
        return {
            "name": d.get("name"),
            "description": (expl or {}).get("description") or d.get("description") or d.get("name"),
            "explanation_model": (expl or {}).get("explanationModelName"),
            "max_act_approx": d.get("maxActApprox"),
            "pos_tokens": [{"token": t, "value": v} for t, v in pos],
            "neg_tokens": [{"token": t, "value": v} for t, v in neg],
            "correlated_features": (d.get("correlated_features_indices") or [])[:10],
            "aligned_neurons": (d.get("neuron_alignment_indices") or [])[:10],
        }

    def _neuronpedia_set(self) -> Optional[Tuple[str, str]]:
        """Cached (model_id, sae_set) pair only if the Neuronpedia slug was runtime-verified.

        Verification runs once in a background thread so no request ever blocks on the
        slug HTTP check; until it finishes this returns None, so Neuronpedia
        links/explainers simply appear from the first request after verification completes.
        """
        if self._np_set is None:
            with self._np_verify_lock:
                if self._np_set is None and not self._np_verifying:
                    self._np_verifying = True
                    threading.Thread(
                        target=self._verify_neuronpedia_slug,
                        daemon=True,
                        name="PDD-NeuronpediaVerifier",
                    ).start()
        return self._np_set

    def _verify_neuronpedia_slug(self) -> None:
        try:
            slug = self._neuronpedia_sae_set(self.summary.get("config", {}).get("sae", {}))
            if slug and self._neuronpedia_verified(*slug):
                self._np_set = slug
        except Exception as e:
            logger.warning(f"Neuronpedia slug verification failed: {e}")

    def _neuronpedia_url(self, feature_index: int) -> Optional[str]:
        np_set = self._neuronpedia_set()
        if np_set is None:
            return None
        return f"https://www.neuronpedia.org/{np_set[0]}/{np_set[1]}/{feature_index}"

    def _sae_feature_item(self, i: int, val: float, m: Optional[int]) -> Dict[str, Any]:
        """One top-feature entry shared by the robust-cluster and raw fallback paths."""
        item: Dict[str, Any] = {"feature_index": i, "activation": val, "cluster_m": m}
        url = self._neuronpedia_url(i)
        if url:
            item["neuronpedia_url"] = url
        return item

    def _top_sae_features(self, activations: np.ndarray, top_n: int = 8, min_cluster_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Top individual SAE features by activation belonging to robust feature clusters (|T_m| >= min_cluster_size)."""
        if activations is None or len(activations) == 0:
            return []

        if min_cluster_size is None:
            min_cluster_size = self.min_feat_cluster_size
        
        ftoc = self.feature_to_cluster_map
        feat_delta = self._feature_delta()

        # Find non-zero firing features
        nonzero_indices = np.flatnonzero(activations > 0)
        if len(nonzero_indices) == 0:
            return []

        # Sort non-zero features by activation descending
        sorted_indices = nonzero_indices[np.argsort(activations[nonzero_indices])[::-1]]

        out: List[Dict[str, Any]] = []
        for i in sorted_indices:
            i = int(i)
            val = float(activations[i])
            m = ftoc.get(i)
            # Require feature to belong to a robust cluster (|T_m| >= min_cluster_size from config)
            if m is not None and len(self.feature_clusters.get(m, [])) >= min_cluster_size:
                item = self._sae_feature_item(i, val, m)
                if feat_delta is not None and i < feat_delta.shape[0]:
                    d = float(feat_delta[i])
                    item["dp_delta"] = d
                    item["dp_direction"] = "amplified" if d > 1e-4 else ("suppressed" if d < -1e-4 else "neutral")
                out.append(item)
                if len(out) >= top_n:
                    break

        # If no robust-cluster features fired, fallback to top raw features
        if len(out) == 0:
            for i in sorted_indices[:top_n]:
                out.append(self._sae_feature_item(int(i), float(activations[i]), ftoc.get(int(i))))

        return out

    @staticmethod
    def _col_firing_rate(mat, threshold: float) -> np.ndarray:
        """Per-column fraction of rows with a value > threshold, scanning CSR in row blocks.

        Avoids materializing a boolean copy of the whole sparse matrix: on the 1B-nnz
        65k matrices that copy costs ~5GB per matrix. Peak memory stays ~O(rows_per_block).
        """
        d = int(mat.shape[1])
        n = int(mat.shape[0])
        indptr, data, indices = mat.indptr, mat.data, mat.indices
        counts = np.zeros(d, dtype=np.float64)
        rows_per_block = 20_000
        for start in range(0, n, rows_per_block):
            end = min(start + rows_per_block, n)
            lo, hi = int(indptr[start]), int(indptr[end])
            counts += np.bincount(indices[lo:hi][data[lo:hi] > threshold], minlength=d)
        return counts / n

    def _feature_delta(self) -> Optional[np.ndarray]:
        """Per-feature DPO-push signal: u_f = P(fires in chosen) - P(fires in rejected).

        Computed lazily (one chunked column-mean pass over the cached C_max/R_max sparse
        matrices, ~d_sae floats) and cached. Positive => DPO amplifies the feature,
        negative => DPO suppresses it. The paper's per-feature primitive `u` (B.1).
        """
        if self._feat_delta is None:
            mats = self._load_feature_matrices()
            if mats is None:
                return None
            try:
                logger.info("Computing per-feature delta (chunked column scan over C_max/R_max)...")
                tau = self._feature_delta_tau()
                c_rate = self._col_firing_rate(mats.C_max, tau)
                r_rate = self._col_firing_rate(mats.R_max, tau)
                self._feat_delta = (c_rate - r_rate).astype(np.float32)
                self._persist_feature_delta()
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
    def _example_field(ex: Any, key: str) -> str:
        if isinstance(ex, dict):
            return ex.get(key, "") or ""
        return getattr(ex, key, "") or ""

    def _example_view(self, ex: Any) -> Dict[str, str]:
        """Truncated prompt/chosen/rejected text fields for one example (shared by all sample lists)."""
        return {
            "prompt": self._example_field(ex, "prompt")[-600:],
            "chosen": self._example_field(ex, "chosen")[-400:],
            "rejected": self._example_field(ex, "rejected")[-400:],
        }

    def _top_examples(self, scores: np.ndarray, examples, top_n: int) -> List[Dict[str, Any]]:
        """Top examples by per-example score: index, score, and truncated text fields."""
        out: List[Dict[str, Any]] = []
        for i in np.argsort(scores)[::-1]:
            if scores[i] <= 0:
                break
            if int(i) >= len(examples):
                continue
            ex = examples[int(i)]
            out.append({
                "index": int(i),
                "score": float(scores[i]),
                **self._example_view(ex),
            })
            if len(out) >= top_n:
                break
        return out

    def _load_examples(self):
        """Lazily load the cached dataset examples for example-based cluster interpretation.

        Guarded by a lock so the background prewarm thread and a concurrent request
        never both parse the large examples file at once.
        """
        if self._examples is None:
            with self._examples_lock:
                if self._examples is None and self.checkpoint_dir is not None:
                    ex_path = self.checkpoint_dir / "examples.json"
                    try:
                        import orjson
                        with open(ex_path, "rb") as f:
                            self._examples = orjson.loads(f.read())
                    except Exception as e:
                        logger.debug(f"orjson examples load failed ({e}); falling back to stdlib json.")
                        self._examples = _read_json(ex_path)
        return self._examples

    def _load_pc_cluster_examples(self):
        """Lazily load per-cluster top example indices for prompt clusters A_k / response-delta clusters R_m.

        Written by the auto-labeling pipeline stage (prompt_conditioned_cluster_examples.json);
        gives each A_k/R_m a concrete, readable meaning via the real examples that
        express it (no SAE feature breakdown). Re-reads when the file changes.
        """
        return self._reload_if_changed(
            Path(pc_cluster_examples_path(str(self.run_dir))), "_pc_cluster_examples", "_pc_cluster_examples_mtime")

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
                **self._example_view(ex),
                "note": desc,
            })
        return out

    def _load_feature_cluster_labels(self) -> Dict[int, Dict[str, Any]]:
        """Lazily load whole-cluster LLM labels for SAE feature clusters T_m.

        Re-reads when the file changes so a re-run of the auto-label pipeline is picked
        up without restarting the viewer.
        """
        raw = self._reload_if_changed(
            Path(feature_cluster_labels_path(str(self.run_dir))),
            "_feature_cluster_labels", "_feature_cluster_labels_mtime")
        if raw is None:
            return {}
        return {int(k): v for k, v in raw.get("feature_clusters", {}).items()}

    def _expected_member_cols(self) -> np.ndarray:
        """Sorted unique member columns across all feature clusters."""
        non_empty = [np.asarray(f, dtype=np.int64) for f in self.feature_clusters.values() if len(f) > 0]
        if not non_empty:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(non_empty))

    def _member_positions(self, mats, attr: str) -> np.ndarray:
        """Flat positions in the CSR arrays of entries belonging to any member column.

        Cached after the first ``isin`` pass over the full indices array, so subsequent
        cluster lookups only touch the (small) member entries instead of re-reading the
        whole indices file.
        """
        pos = self._member_positions_cache.get(attr)
        if pos is None:
            if self._all_member_cols is None:
                self._all_member_cols = self._expected_member_cols()
            mat = getattr(mats, attr)
            pos = np.nonzero(np.isin(mat.indices, self._all_member_cols))[0]
            self._member_positions_cache[attr] = pos
        return pos

    def _member_matrix(self, mats, attr: str) -> Optional[np.ndarray]:
        """Dense (N, n_members) matrix of member-column values, built once from the CSR.

        The member columns across all clusters are small (hundreds), so this compact
        in-memory copy (≈1MB per member column) serves every cluster lookup from RAM
        instead of re-reading the billion-nonzero matrix files on disk.
        """
        M = self._member_matrices.get(attr)
        if M is None:
            pos = self._member_positions(mats, attr)
            if len(pos) == 0:
                return None
            mat = getattr(mats, attr)
            cols = mat.indices[pos]
            rows = np.searchsorted(mat.indptr, pos, side="right") - 1
            vals = mat.data[pos]
            slots = np.searchsorted(self._all_member_cols, cols)
            M = np.zeros((mat.shape[0], len(self._all_member_cols)), dtype=np.float32)
            M[rows, slots] = vals
            self._member_matrices[attr] = M
            logger.info(f"Built dense member matrix {attr} {M.shape} ({M.nbytes / 1e6:.0f} MB).")
        return M

    def _feature_firings(self) -> Optional[np.ndarray]:
        """Total firings for all SAE features across C_max and R_max.

        Summed over the compact dense member matrices (no per-cluster disk reads, no
        billion-nonzero float64 casts).
        """
        if self._feature_totals is None:
            mats = self._load_feature_matrices()
            if mats is None:
                return None
            d_sae = mats.C_max.shape[1]
            tot = np.zeros(d_sae, dtype=np.float32)
            for attr in ("C_max", "R_max"):
                M = self._member_matrix(mats, attr)
                if M is not None:
                    tot[self._all_member_cols] += M.sum(axis=0)
            self._feature_totals = tot
        return self._feature_totals

    # ------------------------------------------------------------------
    # Persisted member cache (built once, mmap-loaded on every boot)
    # ------------------------------------------------------------------

    @staticmethod
    def _matrix_source_fingerprint(mat) -> str:
        """Fingerprint of a matrix's on-disk source: nnz + size + mtime of its data file."""
        nnz = len(mat.data)
        fname = getattr(mat.data, "filename", None)
        if fname:
            try:
                st = Path(fname).stat()
                return f"{nnz}:{st.st_size}:{int(st.st_mtime)}"
            except OSError:
                pass
        return f"{nnz}:"

    def _member_cache_meta(self, mats) -> Dict[str, Any]:
        """Identity of the cached build; any change invalidates it and triggers a rebuild."""
        return {
            "run_dir": self.run_dir.name,
            "checkpoint": self.checkpoint_dir.name if self.checkpoint_dir else None,
            "d_sae": int(mats.C_max.shape[1]),
            "n": int(mats.C_max.shape[0]),
            "n_members": len(self._expected_member_cols()),
            "clusters_fp": repr(sorted((int(k), sorted(map(int, v))) for k, v in self.feature_clusters.items())),
            "matrices": {
                "C_max": self._matrix_source_fingerprint(mats.C_max),
                "R_max": self._matrix_source_fingerprint(mats.R_max),
            },
        }

    def _member_cache_valid(self, mats, meta: Optional[Dict[str, Any]]) -> bool:
        if meta is None:
            return False
        try:
            return all(meta[k] == v for k, v in self._member_cache_meta(mats).items())
        except Exception as e:
            logger.debug(f"Member cache meta malformed ({e}); rebuilding.")
            return False

    @staticmethod
    def _save_npy(path: Path, arr: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile
        tmp_file = tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}_{os.getpid()}_{threading.get_ident()}_", suffix=".tmp.npy", delete=False
        )
        try:
            with open(tmp_file.name, "wb") as f:
                np.save(f, arr)
            os.replace(tmp_file.name, path)
        except Exception:
            try:
                if os.path.exists(tmp_file.name):
                    os.remove(tmp_file.name)
            except Exception:
                pass
            raise

    def _feature_delta_tau(self) -> float:
        """The B.1 tau threshold the cached per-feature delta was computed with."""
        return float(self._fc_cfg.get("tau", 0.01))

    def _persist_feature_delta(self) -> None:
        cache_dir = self.run_dir / "viewer_cache"
        if cache_dir.is_dir() and self._feat_delta is not None:
            self._save_npy(cache_dir / "feature_delta.npy", self._feat_delta)
            meta_tmp = cache_dir / "feature_delta_meta.json.tmp"
            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump({"tau": self._feature_delta_tau()}, f)
            os.replace(meta_tmp, cache_dir / "feature_delta_meta.json")

    def _persist_member_cache(self, mats) -> None:
        """Atomically write the built member cache under <run_dir>/viewer_cache/."""
        cache_dir = self.run_dir / "viewer_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "member_cols.npy": self._all_member_cols,
            "positions_C_max.npy": self._member_positions_cache["C_max"],
            "positions_R_max.npy": self._member_positions_cache["R_max"],
            "member_matrix_C_max.npy": self._member_matrices["C_max"],
            "member_matrix_R_max.npy": self._member_matrices["R_max"],
            "feature_totals.npy": self._feature_totals,
        }
        for name, arr in payload.items():
            if arr is not None:
                self._save_npy(cache_dir / name, arr)
        meta_tmp = cache_dir / "meta.json.tmp"
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(self._member_cache_meta(mats), f, indent=2)
        os.replace(meta_tmp, cache_dir / "meta.json")
        logger.info(f"Persisted member cache to {cache_dir}.")

    def _load_member_cache(self, mats) -> None:
        """Load the persisted member cache if valid, else build it once and persist it.

        Dense matrices are memory-mapped so boot stays fast and only the few columns a
        cluster lookup touches are read from disk. Validated against the source matrices
        + feature clusters on every start; any source change (re-clustering, new
        matrices) rebuilds automatically — no manual rebuild script.
        """
        with self._member_cache_lock:
            if not self.feature_clusters:
                logger.info("No feature clusters for this run; skipping member cache.")
                return
            cache_dir = self.run_dir / "viewer_cache"
            cache_files = [
                "member_cols.npy", "positions_C_max.npy", "positions_R_max.npy",
                "member_matrix_C_max.npy", "member_matrix_R_max.npy", "feature_totals.npy",
            ]
            meta = _read_json(cache_dir / "meta.json")
            if all((cache_dir / f).exists() for f in cache_files) and self._member_cache_valid(mats, meta):
                try:
                    self._all_member_cols = np.load(cache_dir / "member_cols.npy")
                    self._member_positions_cache = {
                        "C_max": np.load(cache_dir / "positions_C_max.npy"),
                        "R_max": np.load(cache_dir / "positions_R_max.npy"),
                    }
                    self._member_matrices = {
                        "C_max": np.load(cache_dir / "member_matrix_C_max.npy", mmap_mode="r"),
                        "R_max": np.load(cache_dir / "member_matrix_R_max.npy", mmap_mode="r"),
                    }
                    self._feature_totals = np.load(cache_dir / "feature_totals.npy")
                    fd_path = cache_dir / "feature_delta.npy"
                    if fd_path.exists():
                        fd_meta = _read_json(cache_dir / "feature_delta_meta.json")
                        if fd_meta is None:
                            logger.debug(f"No feature_delta meta in {cache_dir}; assuming default tau.")
                            fd_meta = {}
                        if float(fd_meta.get("tau", 0.01)) == self._feature_delta_tau():
                            self._feat_delta = np.load(fd_path)
                    logger.info(f"Loaded member cache from {cache_dir} (mmap).")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load member cache ({e}); rebuilding.")
            self._feature_firings()
            self._feature_delta()
            self._persist_member_cache(mats)

    def prewarm(self) -> None:
        """Load (or build + persist) the SAE cluster member cache at startup."""
        mats = self._load_feature_matrices()
        if mats is None:
            logger.info("No feature matrices for this run; skipping member-cache prewarm.")
            return
        self._load_member_cache(mats)
        if self._feat_delta is None:
            self._feature_delta()

    def _top_cluster_features(self, m: int, top_n: int = 8) -> List[Dict[str, Any]]:
        """Top SAE features inside feature cluster T_m, ranked by firing in < 1ms."""
        m_int = int(m)
        feats = self.feature_clusters.get(m_int, [])
        if not feats:
            return []
        tot = self._feature_firings()
        if tot is None or len(tot) == 0:
            logger.warning(f"Feature firing totals unavailable; returning T_{m} members unranked.")
            return [{
                "feature_index": int(f), "firing": 0.0,
                "neuronpedia_url": self._neuronpedia_url(int(f)),
            } for f in feats[:top_n]]
        feats = [f for f in feats if 0 <= f < len(tot)]
        if not feats:
            return []
        firings = tot[feats]
        order = np.argsort(firings)[-top_n:][::-1]
        return [{
            "feature_index": int(feats[j]),
            "firing": float(firings[j]),
            "neuronpedia_url": self._neuronpedia_url(int(feats[j])),
        } for j in order]

    def _cached_info(self, key: str, build) -> Dict[str, Any]:
        """Memoize a per-cluster payload in the shared info cache."""
        cached = self._cluster_info_cache.get(key)
        if cached is not None:
            return cached
        res = build()
        self._cluster_info_cache[key] = res
        return res

    def _feature_cluster_info(self, m: int, top_n_examples: int = 5) -> Dict[str, Any]:
        """One payload for the T_m dropdown: whole-cluster label + top member features + real examples."""
        m_int = int(m)
        cache_key = f"T_{m_int}_{top_n_examples}"

        def build() -> Dict[str, Any]:
            feats = self.feature_clusters.get(m_int, [])
            label = self._load_feature_cluster_labels().get(m_int)
            return {
                "cluster_m": m_int,
                "label": label or {"title": f"Feature cluster T_{m}", "description": "", "keywords": []},
                "n_features": len(feats),
                "top_features": self._top_cluster_features(m_int, top_n=8),
                "examples": self._cluster_top_examples(m_int, top_n=top_n_examples),
            }

        res = self._cached_info(cache_key, build)
        # Pre-warm Neuronpedia metadata in the background for top member features
        self._prewarm_neuronpedia_features([tf["feature_index"] for tf in res["top_features"]])
        return res

    def _data_cluster_info(self, k: int, top_n_examples: int = 5) -> Dict[str, Any]:
        """Interpretation for data cluster B_k: title, description, keywords, and sampled centroid/random prompts."""
        k_int = int(k)
        cache_key = f"B_{k_int}_{top_n_examples}"

        def build() -> Dict[str, Any]:
            label_obj = next((lab for lab in self._load_data_cluster_labels() if lab.get("cluster_id") == k_int), None)
            if label_obj is None:
                label_obj = {
                    "cluster_id": k_int,
                    "title": f"Data Cluster B_{k}",
                    "description": "",
                    "keywords": [],
                    "centroid_prompts": [],
                    "sample_prompts": [],
                }
            return {
                "cluster_id": k_int,
                "title": label_obj.get("title", f"Data Cluster B_{k}"),
                "description": label_obj.get("description", ""),
                "keywords": label_obj.get("keywords", []),
                "centroid_prompts": label_obj.get("centroid_prompts", [])[:top_n_examples],
                "sample_prompts": label_obj.get("sample_prompts", [])[:top_n_examples],
            }

        return self._cached_info(cache_key, build)

    def _cluster_top_examples(self, m: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Top dataset examples firing feature cluster T_m, ranked by vectorized row scores.

        Sums the cluster's top member columns from the cached dense member matrix in
        RAM — no per-cluster disk reads.
        """
        mats = self._load_feature_matrices()
        examples = self._load_examples()
        m_int = int(m)
        feats = self.feature_clusters.get(m_int, [])
        if mats is None or examples is None or not feats:
            return []

        top_mems = self._top_cluster_features(m_int, top_n=5)
        top_f = np.array([f["feature_index"] for f in top_mems], dtype=np.int64)
        if len(top_f) == 0:
            return []

        slots = np.searchsorted(self._all_member_cols, top_f)
        scores = np.zeros(mats.C_max.shape[0], dtype=np.float64)
        for attr in ("C_max", "R_max"):
            M = self._member_matrix(mats, attr)
            if M is not None:
                scores += M[:, slots].sum(axis=1)
        return self._top_examples(scores, examples, top_n)

    def _ensure_example_scores(self, mats) -> None:
        """Per-example amplify/suppress scores (u, s) for every feature cluster, computed once.

        u_{i,m} = (1/|T_m|) sum_{g in T_m}[1{C_max_{i,g} > tau} - 1{R_max_{i,g} > tau}],
        s_{i,m} = number of T_m members firing anywhere in the pair (chosen + rejected).
        A single blocked matmul over the dense C_max/R_max member matrices (reused from the
        B.1 dropdown) builds both (N, K) tables, persisted under <run>/viewer_cache/ with the
        same fingerprint as the member cache so re-clustering/rebuilds refresh them. Tab-4
        queries then only sort a 260k column in-RAM — milliseconds, not a per-query scan.
        """
        with self._scores_lock:
            if self._example_u is not None and self._example_s is not None:
                return
            cluster_ids = sorted(int(k) for k in self.feature_clusters.keys())
            if not cluster_ids:
                return
            cache_dir = self.run_dir / "viewer_cache"
            meta = _read_json(cache_dir / "example_scores_meta.json")
            if (
                all((cache_dir / f).exists() for f in ("example_u.npy", "example_s.npy", "example_cluster_ids.npy"))
                and meta is not None
                and meta.get("matrices") == self._member_cache_meta(mats)
                and float(meta.get("tau", -1.0)) == self._feature_delta_tau()
                and meta.get("cluster_ids") == cluster_ids
            ):
                try:
                    self._example_u = np.load(cache_dir / "example_u.npy", mmap_mode="r")
                    self._example_s = np.load(cache_dir / "example_s.npy", mmap_mode="r")
                    self._example_cluster_ids = np.load(cache_dir / "example_cluster_ids.npy")
                    logger.info(f"Loaded per-example scores from {cache_dir} (mmap).")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load per-example scores ({e}); recomputing.")
            if self._all_member_cols is None:
                self._all_member_cols = self._expected_member_cols()
            n_members = int(len(self._all_member_cols))
            K = len(cluster_ids)
            A = np.zeros((n_members, K), dtype=np.float32)
            for c_idx, cid in enumerate(cluster_ids):
                feats = np.asarray(self.feature_clusters[cid], dtype=np.int64)
                if len(feats) == 0:
                    continue
                slots = np.searchsorted(self._all_member_cols, feats)
                A[slots, c_idx] = 1.0
            cluster_sizes = np.maximum(A.sum(axis=0), 1.0)
            tau = self._feature_delta_tau()

            def fire_counts(M: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if M is None:
                    return None
                N = int(M.shape[0])
                out = np.zeros((N, K), dtype=np.float32)
                rows_per_block = 20_000
                for start in range(0, N, rows_per_block):
                    end = min(start + rows_per_block, N)
                    out[start:end] = (M[start:end] > tau).astype(np.float32) @ A
                return out

            M_c = self._member_matrix(mats, "C_max")
            M_r = self._member_matrix(mats, "R_max")
            c_cnt = fire_counts(M_c)
            r_cnt = fire_counts(M_r)
            if c_cnt is None or r_cnt is None:
                logger.warning("Member matrices unavailable; cannot build per-example scores.")
                return
            u = (c_cnt - r_cnt) / cluster_sizes[None, :]
            s = c_cnt + r_cnt
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._save_npy(cache_dir / "example_u.npy", u)
            self._save_npy(cache_dir / "example_s.npy", s)
            self._save_npy(cache_dir / "example_cluster_ids.npy", np.asarray(cluster_ids, dtype=np.int64))
            meta_tmp = cache_dir / f".example_scores_meta_{os.getpid()}_{threading.get_ident()}.json.tmp"
            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump({"tau": tau, "cluster_ids": cluster_ids, "matrices": self._member_cache_meta(mats)}, f)
            os.replace(meta_tmp, cache_dir / "example_scores_meta.json")
            self._example_u = u
            self._example_s = s
            self._example_cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
            logger.info(f"Built per-example scores u/s ({u.shape}) and persisted under {cache_dir}.")

    def _inspect_feature_samples(self, m: int, k: int, side: str = "amplify") -> Dict[str, Any]:
        """Top preference examples whose labels amplify or suppress feature cluster T_m.

        The inverse of inspect_prompt: instead of asking what a prompt predicts about a
        behavior, pick a behavior (T_m) and get the training pairs whose labels drive it.
        Amplify  (u > 0): the chosen response fires the cluster more than the rejected one —
                       post-training rewards this behavior.
        Suppress (u < 0): the rejected response fires it more — post-training penalizes it.
        u/s come from the precomputed per-example tables (see _ensure_example_scores), so a
        query is just a filtered sort of one 260k column — milliseconds.
        """
        m_int = int(m)
        k = max(1, min(int(k), 200))
        side = side if side in ("amplify", "suppress") else "amplify"
        cache_key = f"inspectSamples_{m_int}_{side}_k{k}"

        def build() -> Dict[str, Any]:
            feats = np.asarray(self.feature_clusters.get(m_int, []), dtype=np.int64)
            label = self._load_feature_cluster_labels().get(m_int)
            base = {
                "cluster_m": m_int,
                "label": label or {"title": f"Feature cluster T_{m}", "description": "", "keywords": []},
                "n_features": int(len(feats)),
                "side": side,
                "total_matching": 0,
                "samples": [],
            }
            if len(feats) == 0:
                return base
            mats = self._load_feature_matrices()
            examples = self._load_examples()
            if mats is None or examples is None:
                return base
            self._ensure_example_scores(mats)
            if self._example_u is None or self._example_cluster_ids is None:
                return base
            pos = np.searchsorted(self._example_cluster_ids, m_int)
            if pos >= len(self._example_cluster_ids) or int(self._example_cluster_ids[pos]) != m_int:
                return base
            u = np.asarray(self._example_u[:, pos], dtype=np.float32)
            s = np.asarray(self._example_s[:, pos], dtype=np.float32)
            present = np.flatnonzero(s > 0)
            if len(present) == 0:
                return base
            order = present[np.argsort(-u[present], kind="stable")]
            if side == "suppress":
                order = order[::-1]
            samples = []
            for i in order[:k]:
                i = int(i)
                if i >= len(examples):
                    continue
                ex = examples[i]
                u_i = float(u[i])
                samples.append({
                    "index": i,
                    "u": u_i,
                    "s": float(s[i]),
                    "effect_direction": "Amplified after DPO" if u_i > 0 else ("Suppressed after DPO" if u_i < 0 else "Neutral"),
                    **self._example_view(ex),
                })
            return {**base, "total_matching": int(len(present)), "samples": samples}

        return self._cached_info(cache_key, build)

    def _inspect_compound_samples(self, conditions: List[Tuple[int, str, float]], k: int) -> Dict[str, Any]:
        """Top samples satisfying EVERY condition: (m, direction, tau).

        amplify  => u_m >  tau (chosen carries the cluster more than rejected);
        suppress => u_m < -tau (rejected carries it more). Because example_u is a
        260k x K table, a compound query is just per-condition column masks ANDed
        together, then ranked by total excess score = sum |u_m| / tau over conditions
        (a sample wins by clearing every condition with a healthy margin). ~ms.
        """
        k = max(1, min(int(k), 200))
        conds: List[Tuple[int, str, float]] = []
        for m, direction, tau in conditions:
            direction = direction if direction in ("amplify", "suppress") else "amplify"
            tau = float(tau) if tau is not None and float(tau) > 0 else 0.1
            conds.append((int(m), direction, tau))
        if not conds:
            return {"compound": True, "k": k, "total_matching": 0, "conditions": [], "samples": []}
        cache_key = "compound_" + "_".join(f"{m}:{d}:{t:.3g}" for m, d, t in conds) + f"_k{k}"

        def build() -> Dict[str, Any]:
            base = {"compound": True, "k": k, "total_matching": 0, "conditions": [], "samples": []}
            mats = self._load_feature_matrices()
            examples = self._load_examples()
            if mats is None or examples is None:
                return base
            self._ensure_example_scores(mats)
            if self._example_u is None or self._example_s is None or self._example_cluster_ids is None:
                return base
            col_u, col_s = [], []
            mask = None
            for m, direction, tau in conds:
                pos = int(np.searchsorted(self._example_cluster_ids, m))
                if pos >= len(self._example_cluster_ids) or int(self._example_cluster_ids[pos]) != m:
                    return base
                u = np.asarray(self._example_u[:, pos], dtype=np.float32)
                s = np.asarray(self._example_s[:, pos], dtype=np.float32)
                cond = (u > tau) if direction == "amplify" else (u < -tau)
                mask = cond if mask is None else (mask & cond)
                col_u.append(u)
                col_s.append(s)
            if mask is None:
                return base
            idxs = np.flatnonzero(mask)
            if len(idxs) == 0:
                return base
            score = np.zeros(len(idxs), dtype=np.float32)
            for (m, direction, tau), u in zip(conds, col_u):
                score += np.abs(u[idxs]) / tau
            order = idxs[np.argsort(-score, kind="stable")]
            labels = self._load_feature_cluster_labels()
            cond_list = []
            for m, direction, tau in conds:
                label = labels.get(int(m))
                cond_list.append({
                    "m": int(m),
                    "direction": direction,
                    "tau": tau,
                    "label": label or {"title": f"Feature cluster T_{m}", "description": "", "keywords": []},
                })
            samples = []
            for i in order[:k]:
                i = int(i)
                if i >= len(examples):
                    continue
                ex = examples[i]
                u_map = {str(m): float(u[i]) for (m, _, _), u in zip(conds, col_u)}
                samples.append({
                    "index": i,
                    "u": u_map,
                    "s": {str(m): float(s[i]) for (m, _, _), s in zip(conds, col_s)},
                    "score": float(sum(abs(u_map[str(m)]) / t for (m, _, t) in conds)),
                    "effect_directions": {
                        str(m): "Amplified after DPO" if u[i] > 0 else "Suppressed after DPO"
                        for (m, _, _), u in zip(conds, col_u)
                    },
                    **self._example_view(ex),
                })
            return {**base, "total_matching": int(len(idxs)), "conditions": cond_list, "samples": samples}

        return self._cached_info(cache_key, build)

    @staticmethod
    def _csr_col(a: Any, i: int) -> np.ndarray:
        """Dense column ``i`` of a sparse matrix as a 1-D array."""
        if hasattr(a, "toarray"):
            return np.asarray(a[:, i].toarray()).ravel()
        return np.asarray(a[:, i]).ravel()

    def _feature_act(self, mats, f: int) -> np.ndarray:
        """Per-example C_max + R_max activation vector for feature ``f``.

        Uses the dense member matrix (instant) when ``f`` is a cluster member and it is
        already built; otherwise falls back to a scalar CSR column extraction.
        """
        if self._all_member_cols is not None:
            slot = int(np.searchsorted(self._all_member_cols, f))
            if slot < len(self._all_member_cols) and int(self._all_member_cols[slot]) == f:
                act = np.zeros(mats.C_max.shape[0], dtype=np.float32)
                for attr in ("C_max", "R_max"):
                    M = self._member_matrices.get(attr)
                    if M is not None:
                        act += M[:, slot]
                return act
        return self._csr_col(mats.C_max, f) + self._csr_col(mats.R_max, f)

    def _feature_detail(self, f: int, top_n: int = 5) -> Dict[str, Any]:
        """Per-feature interpretation: run firing stats, top firing examples, Neuronpedia metadata."""
        f = int(f)
        cache_key = f"feat_{f}_{top_n}"

        def build() -> Dict[str, Any]:
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

            act = self._feature_act(mats, f)
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
            out["examples"] = self._top_examples(act, examples, top_n) if examples is not None else []

            url = self._neuronpedia_url(f)
            if url:
                out["neuronpedia_url"] = url
                np_data = self._get_neuronpedia_feature(f)
                if np_data:
                    out["neuronpedia"] = np_data
            return out

        return self._cached_info(cache_key, build)

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
    val_data = _read_json(val_file)
    validation_metrics = val_data if val_data is not None else {}

    # Reload the full hypothesis lists (cached per run) so tables show top-100
    # ranked by the pipeline's saved ordering instead of the summary top-10.
    # The properties self-log any load failure and leave the summary seed intact.
    state.feature_hypotheses_map
    state.prompt_hypotheses_map

    return {
        "summary": state.summary,
        "validation_metrics": validation_metrics,
        "cluster_labels": state._load_data_cluster_labels(),
        "feature_cluster_labels": state._load_feature_cluster_labels(),
        "top_feature_conditioned_hypotheses": state._cluster_covered_hypos(state.fc_hypos, 30),
        "top_prompt_conditioned_hypotheses": state._cluster_covered_hypos(state.pc_hypos, 5),
    }


@app.get("/api/feature_cluster_info")
def get_feature_cluster_info(m: int = Query(..., description="Feature cluster id T_m"),
                             top_n: int = Query(5)) -> Dict[str, Any]:
    """Whole-cluster interpretation for SAE feature cluster T_m."""
    state = get_state()
    return state._feature_cluster_info(m, top_n_examples=top_n)


@app.get("/api/inspect_feature_samples")
def get_inspect_feature_samples(m: Optional[int] = Query(None, description="Feature cluster id T_m (single-cluster query)"),
                                k: int = Query(50, ge=1, le=200, description="Number of top samples"),
                                side: str = Query("amplify", description="'amplify' (chosen carries concept) or 'suppress' (rejected carries concept)"),
                                conditions: Optional[str] = Query(None, description="Compound query: comma-separated 'm:amplify|suppress[:tau]' e.g. '1:amplify:0.2,3:suppress'")) -> Dict[str, Any]:
    """Inverse of /api/inspect_prompt: top training examples whose labels amplify/suppress feature cluster T_m.

    With `conditions`, runs a compound query: top examples satisfying EVERY condition
    (amplify = u_m > tau, suppress = u_m < -tau), ranked by total excess score.
    """
    state = get_state()
    if conditions:
        parsed = []
        for part in conditions.split(","):
            fields = part.strip().split(":")
            if len(fields) < 2:
                continue
            try:
                cm = int(fields[0])
            except ValueError:
                continue
            direction = fields[1].strip() if fields[1].strip() in ("amplify", "suppress") else "amplify"
            try:
                tau = float(fields[2]) if len(fields) > 2 and fields[2].strip() else 0.1
            except ValueError:
                tau = 0.1
            parsed.append((cm, direction, tau))
        if parsed:
            return state._inspect_compound_samples(parsed, k)
        raise HTTPException(400, "No valid conditions parsed; expected 'm:amplify|suppress[:tau],...'")
    if m is None:
        raise HTTPException(400, "Provide either m (single-cluster) or conditions (compound query).")
    return state._inspect_feature_samples(m, k, side)


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
    elif t in ("prompt", "a", "ak", "response", "r", "rm"):
        is_prompt = t in ("prompt", "a", "ak")
        cluster_type = "prompt" if is_prompt else "response"
        return {
            "cluster_family": "A" if is_prompt else "R",
            "cluster_type": cluster_type,
            "id": id,
            "tokens": state._pc_cluster_tokens(cluster_type, id),
            "examples": state._pc_cluster_top_examples(cluster_type, id, top_n=top_n),
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

    # 3. Extract Feature-Conditioned Predicted Shifts (B_k x T_m) from live-evidence-ranked hypotheses
    predicted_shifts = []
    labels_t = state._load_feature_cluster_labels()
    labels_b = state._load_data_cluster_labels()
    b_title_map = {int(lab.get("cluster_id")): lab.get("title", f"Data Cluster B_{lab.get('cluster_id')}") for lab in labels_b if "cluster_id" in lab}

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

            t_info = labels_t.get(int(m), {}) if m is not None else {}
            t_title = t_info.get("title", f"Feature cluster T_{m}")
            t_desc = t_info.get("description", "")
            b_title = b_title_map.get(int(k), f"Topic B_{k}") if k is not None else "N/A"

            direction_word = "AMPLIFIED (Boosted)" if is_amplified else "SUPPRESSED (Inhibited)"
            interpretation = (
                f"This prompt fires SAE feature cluster T_{m} ({t_title}) with live activity {evidence:.3f}. "
                f"In the training data, examples of type B_{k} ({b_title}) are chosen-leaning on this cluster "
                f"(Δ = {delta:+.5f}, Welch z = {z:.2f}), so post-training will likely {direction_word} "
                f"this response behavior for similar prompts."
            )

            predicted_shifts.append({
                "prompt_cluster_k": k,
                "response_cluster_m": m,
                "feature_cluster_title": t_title,
                "feature_cluster_description": t_desc,
                "data_cluster_title": b_title,
                "pipeline_type": "feature_conditioned",
                "delta": delta,
                "effect_direction": "Amplified after DPO" if is_amplified else "Suppressed after DPO",
                "z_score": z,
                "cohens_d": d,
                "live_activity": evidence,
                "interpretation": interpretation,
            })

    predicted_shifts.sort(key=lambda s: abs(float(s["delta"])) * abs(float(s.get("live_activity", 0.0))), reverse=True)
    predicted_shifts = predicted_shifts[:10]

    # 4. Extract Prompt-Conditioned Predicted Shifts (A_k x R_m)
    pc_shifts = state._score_prompt_conditioned_hypotheses(prompt_text, p_feat, top_k=req.top_k)

    top_feats = state._top_sae_features(p_feat)
    state._prewarm_neuronpedia_features([f["feature_index"] for f in top_feats])

    return {
        "prompt": prompt_text,
        "matched_clusters": matched_clusters,
        "predicted_behavior_shifts": predicted_shifts,
        "prompt_conditioned_shifts": pc_shifts,
        "top_sae_features": top_feats,
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
    best_by_m = state._best_hypothesis_by_m()
    labels_t = state._load_feature_cluster_labels()
    labels_b = state._load_data_cluster_labels()
    b_title_map = {int(lab.get("cluster_id")): lab.get("title", f"Data Cluster B_{lab.get('cluster_id')}") for lab in labels_b if "cluster_id" in lab}

    promoted_concepts = []
    suppressed_concepts = []
    for m, uval in sorted(u_sig.items(), key=lambda t: abs(t[1]), reverse=True):
        if uval == 0.0:
            continue
        if len(state.feature_clusters.get(m, [])) < state.min_feat_cluster_size or m not in best_by_m:
            continue
        h = best_by_m.get(m, {})
        k = h.get("k")
        z = float(h.get("z_score", 0.0))
        is_chosen = uval > 0
        t_info = labels_t.get(int(m), {}) if m is not None else {}
        t_title = t_info.get("title", f"Feature cluster T_{m}")
        b_title = b_title_map.get(int(k), f"Topic B_{k}") if k is not None else "N/A"

        item = {
            "feature_cluster_m": m,
            "feature_cluster_title": t_title,
            "data_cluster_k": k,
            "data_cluster_title": b_title,
            "delta": float(uval),
            "hypothesis_delta": float(h.get("delta", 0.0)),
            "z_score": z,
            "signal_strength": "Strong" if abs(uval) > 0.15 else ("Moderate" if abs(uval) > 0.05 else "Weak"),
            "explanation": (
                f"Live SAE disparity: the chosen response fires feature cluster T_{m} ({t_title}) "
                f"more than the rejected (net u = {uval:+.3f})."
                + (f" Consistent with training hypothesis B_{k} ({b_title}) (Δ = {h.get('delta', 0.0):+.4f}, Welch z = {z:.2f})." if k is not None else "")
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

    top_pair_feats = state._top_sae_features(np.abs(u))
    state._prewarm_neuronpedia_features([f["feature_index"] for f in top_pair_feats])

    return {
        "prompt": prompt_text,
        "chosen_length": len(chosen_text),
        "rejected_length": len(rejected_text),
        "promoted_sae_features_count": int((u > 0).sum()),
        "suppressed_sae_features_count": int((u < 0).sum()),
        "matched_clusters": matched_clusters,
        "promoted_concepts": promoted_concepts,
        "suppressed_concepts": suppressed_concepts,
        "top_sae_features": top_pair_feats,
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
    parser.add_argument("--no-prewarm", action="store_true",
                        help="Skip loading/building the SAE member cache at startup")
    args = parser.parse_args()

    os.environ["PDD_RUN_DIR"] = args.run_dir
    global _STATE
    _STATE = ViewerState(run_dir=args.run_dir)

    # Load (or build + persist once) the SAE cluster member cache before serving,
    # so the first cluster click is instant even on the billion-nonzero runs.
    if not args.no_prewarm:
        _STATE.prewarm()

    # Pre-warm cached examples in a background daemon thread to eliminate first-click disk latency
    threading.Thread(target=_STATE._load_examples, daemon=True).start()

    logger.info(f"Starting PDD Viewer for '{args.run_dir}' at http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
