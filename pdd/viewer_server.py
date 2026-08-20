"""FastAPI server for the Predictive Data Debugging (PDD) Interactive Viewer.

Serves run metadata, feature-conditioned (B.1) & prompt-conditioned (B.2) hypotheses,
cluster statistics, and live prompt/preference pair neural inspection endpoints.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    raise ImportError("FastAPI and Uvicorn are required for the viewer. Install via `pip install fastapi uvicorn`.")

logger = logging.getLogger("PDD.Viewer")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")


# ==============================================================================
# Global Utility Functions & Atomic File I/O
# ==============================================================================

def _read_json(path: Path) -> Optional[Any]:
    """Read a JSON file; return None (with a warning log) instead of raising on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error reading {path.name}: {e}")
        return None


def _save_json(path: Path, data: Any, indent: Optional[int] = None) -> None:
    """Atomically write a JSON file using a thread-safe process/thread-unique temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}_{os.getpid()}_{threading.get_ident()}.tmp.json"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise


def _mtime_of(path: Path) -> float:
    """Return modification time of ``path`` (0.0 when missing) for lazy pipeline cache invalidation."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# ==============================================================================
# FastAPI App Setup & Request Models
# ==============================================================================

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
    """Mode A inspection request: live prompt forward pass to predict behavioral shifts."""
    prompt: str = Field(..., description="Prompt text to analyze")
    top_k: int = Field(5, description="Number of top matching clusters/hypotheses to return")


class PreferencePairInspectionRequest(BaseModel):
    """Mode B inspection request: live preference pair forward pass to audit SAE disparity."""
    prompt: str = Field(..., description="Prompt text")
    chosen: str = Field(..., description="Chosen (promoted) response text")
    rejected: str = Field(..., description="Rejected (suppressed) response text")
    top_k: int = Field(5, description="Number of top promoted/suppressed concepts to return")


# ==============================================================================
# ViewerState: Central State Manager for Runs & Caches
# ==============================================================================

class ViewerState:
    """Central state manager for the active run directory, checkpoints, and lazy caches.

    Responsibilities:
    - Resolves and reads run-level metadata (`pdd_summary.json`) and checkpoint artifacts.
    - Maintains thread-safe, memory-mapped caches for high-throughput UI exploration.
    - Manages live NeuralInspector instances for real GPU prompt & preference pair analysis.
    - Lazily parses and reloads auto-generated LLM cluster labels (B_k, T_m, A_k, R_m).
    """

    def __init__(self, run_dir: Optional[str] = None):
        if run_dir is None:
            run_dir = os.environ.get("PDD_RUN_DIR", "runs/qwen3_1.7b_dolci")
        
        # ----------------------------------------------------------------------
        # 1. Target Run Directory & Summary Metadata
        # ----------------------------------------------------------------------
        self.run_dir: Path = Path(run_dir)
        """Path to the active run directory (e.g. `runs/qwen3_1.7b_batchtopk_65k`)."""

        self.summary: Dict[str, Any] = {}
        """Parsed content of `pdd_summary.json` containing run configuration and top metrics."""

        self.checkpoint_dir: Optional[Path] = None
        """Path to the underlying checkpoint folder containing feature matrices and clusters."""

        # ----------------------------------------------------------------------
        # 2. Graph Communities & Partition Maps
        # ----------------------------------------------------------------------
        self.feature_clusters: Dict[int, List[int]] = {}
        """Mapping from feature cluster community ID T_m -> list of constituent SAE feature indices."""

        self._feat_to_cluster: Optional[Dict[int, int]] = None
        """Inverse mapping: individual SAE feature index -> feature cluster community ID T_m."""

        # ----------------------------------------------------------------------
        # 3. Hypothesis Sets & k-Indexed Lookup Maps
        # ----------------------------------------------------------------------
        self.fc_hypos: List[Dict[str, Any]] = []
        """Feature-Conditioned hypotheses list (Appendix B.1: B_k x T_m)."""

        self.pc_hypos: List[Dict[str, Any]] = []
        """Prompt-Conditioned hypotheses list (Appendix B.2: A_k x R_m)."""

        self.k_to_fc: Dict[int, List[Dict[str, Any]]] = {}
        """Mapping from data cluster index k -> list of Feature-Conditioned hypotheses."""

        self.k_to_pc: Dict[int, List[Dict[str, Any]]] = {}
        """Mapping from prompt cluster index k -> list of Prompt-Conditioned hypotheses."""

        self._best_hypo_by_m: Optional[Dict[int, Dict[str, Any]]] = None
        """Mapping from feature cluster index m -> strongest validated hypothesis referencing it."""

        # ----------------------------------------------------------------------
        # 4. Large Dataset & Matrix Artifacts (Lazy Disk/Memory-Mapped)
        # ----------------------------------------------------------------------
        self._feature_matrices = None
        """Lazy-loaded FeatureMatrices object (C_max, R_max, P_max sparse/dense matrices)."""

        self._examples = None
        """Lazy-loaded list of DatasetExample objects cached from `examples.json`."""

        self._feat_delta: Optional[np.ndarray] = None
        """Per-feature empirical delta vector across all dataset pairs: 1(C>tau) - 1(R>tau)."""

        self._example_u: Optional[np.ndarray] = None
        """Memory-mapped (N, K) per-example disparity score matrix for all feature clusters."""

        self._example_s: Optional[np.ndarray] = None
        """Memory-mapped (N, K) per-example total firing score matrix for all feature clusters."""

        self._example_cluster_ids: Optional[np.ndarray] = None
        """Sorted 1-D array of feature cluster community IDs corresponding to columns of example_u/s."""

        # ----------------------------------------------------------------------
        # 5. Dynamic Auto-Label Caches & mtime Tracking
        # ----------------------------------------------------------------------
        self._cluster_labels_raw: Optional[Any] = None
        """Raw cache of `cluster_labels.json` (B_k data cluster titles and descriptions)."""

        self._cluster_labels_mtime: float = 0.0
        """File modification time of `cluster_labels.json` for hot reload on pipeline updates."""

        self._feature_cluster_labels: Optional[Any] = None
        """Raw cache of `feature_cluster_labels.json` (T_m feature cluster titles and descriptions)."""

        self._feature_cluster_labels_mtime: float = 0.0
        """File modification time of `feature_cluster_labels.json` for hot reload."""

        self._pc_cluster_examples: Optional[Any] = None
        """Raw cache of `prompt_conditioned_cluster_examples.json` (A_k / R_m representative tokens)."""

        self._pc_cluster_examples_mtime: float = 0.0
        """File modification time of `prompt_conditioned_cluster_examples.json` for hot reload."""

        # ----------------------------------------------------------------------
        # 6. Dense Cluster Member Lookup Caches
        # ----------------------------------------------------------------------
        self._all_member_cols: Optional[np.ndarray] = None
        """Sorted array of unique SAE feature indices belonging to ANY community >= min_size."""

        self._member_positions_cache: Dict[str, np.ndarray] = {}
        """Row indices of firing examples in C_max and R_max for dense member matrix fast lookups."""

        self._member_matrices: Dict[str, np.ndarray] = {}
        """Memory-mapped dense member matrices for C_max and R_max (N, len(all_member_cols))."""

        self._feature_totals: Optional[np.ndarray] = None
        """Pre-aggregated firing counts across all examples for each member SAE feature."""

        self._cluster_info_cache: Dict[str, Any] = {}
        """In-memory memoization cache for formatted cluster info payloads."""

        # ----------------------------------------------------------------------
        # 7. Thread Synchronization Locks & Neural Inspector
        # ----------------------------------------------------------------------
        self._examples_lock: threading.Lock = threading.Lock()
        """Mutex protecting lazy dataset example parsing and offset indexing."""

        self._scores_lock: threading.Lock = threading.Lock()
        """Mutex serializing background generation of example_u and example_s score tables."""

        self._member_cache_lock: threading.Lock = threading.Lock()
        """Mutex protecting compilation and memory-mapping of dense member matrices."""

        self._np_verify_lock: threading.Lock = threading.Lock()
        """Mutex protecting Neuronpedia dataset slug verification and HTTP probes."""

        self.inspector = None
        """Lazy-loaded NeuralInspector instance for live GPU model/SAE forward passes."""

        self._np_set: Optional[Tuple[str, str]] = None
        """Cached tuple of (model_slug, sae_slug) verified against Neuronpedia."""

        self._np_verifying: bool = False
        """Status flag indicating whether background Neuronpedia slug verification is running."""

        # Execute initial run directory scan and summary loading
        self.load()

    # ==========================================================================
    # SECTION 1: Initialization, Lifecycle & Config Properties
    # ==========================================================================

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

        # 3. Seed top hypotheses initially from summary
        self.fc_hypos = self.summary.get("top_feature_conditioned_hypotheses", [])
        self.pc_hypos = self.summary.get("top_prompt_conditioned_hypotheses", [])

        logger.info(
            f"ViewerState initialized for '{self.run_dir.name}': "
            f"{len(self.feature_clusters)} feature clusters, ready for instant requests."
        )

    def prewarm(self) -> None:
        """Prewarm dense member cache and feature deltas at server startup for zero-latency lookups."""
        mats = self._load_feature_matrices()
        if mats is None:
            logger.info("No feature matrices for this run; skipping member-cache prewarm.")
            return
        self._load_member_cache(mats)
        if self._feat_delta is None:
            self._feature_delta()

    @property
    def _fc_cfg(self) -> Dict[str, Any]:
        """Feature-conditioned pipeline configuration block from run metadata."""
        return self.summary.get("config", {}).get("feature_conditioned", {})

    @property
    def min_feat_cluster_size(self) -> int:
        """Minimum feature cluster community size |T_m| required for hypothesis emission (default: 10)."""
        return int(self._fc_cfg.get("min_feat_cluster_size", 10))

    @property
    def min_data_cluster_size(self) -> int:
        """Minimum prompt cluster size n_k required for hypothesis emission (default: 25)."""
        return int(self._fc_cfg.get("min_data_cluster_size", 25))

    @property
    def min_partition_cluster_size(self) -> int:
        """Minimum community size for the coordinate graph partition (default: 4)."""
        fcl_cfg = self.summary.get("config", {}).get("feature_clustering", {})
        return int(fcl_cfg.get("min_cluster_size", 4))

    # ==========================================================================
    # SECTION 2: Hypothesis Maps & Dynamic Artifact Loaders
    # ==========================================================================

    def _reload_if_changed(self, path: Path, cache_attr: str, mtime_attr: str) -> Any:
        """Re-read a JSON artifact when its file mtime changes; memoize the parsed data otherwise."""
        mtime = _mtime_of(path)
        cached = getattr(self, cache_attr)
        if cached is None or mtime != getattr(self, mtime_attr):
            setattr(self, cache_attr, _read_json(path))
            setattr(self, mtime_attr, mtime)
        return getattr(self, cache_attr)

    def _load_data_cluster_labels(self) -> List[Dict[str, Any]]:
        """Data cluster (B_k) labels, re-read when `cluster_labels.json` updates on disk."""
        raw = self._reload_if_changed(
            Path(cluster_labels_path(str(self.run_dir))), "_cluster_labels_raw", "_cluster_labels_mtime"
        )
        return raw.get("labels", []) if raw is not None else []

    def _load_feature_cluster_labels(self) -> Dict[int, Dict[str, Any]]:
        """Feature cluster (T_m) labels, re-read when `feature_cluster_labels.json` updates on disk."""
        raw = self._reload_if_changed(
            Path(feature_cluster_labels_path(str(self.run_dir))),
            "_feature_cluster_labels",
            "_feature_cluster_labels_mtime",
        )
        if raw is None:
            return {}
        clusters_data = raw.get("feature_clusters", {})
        return {int(k): v for k, v in clusters_data.items()}

    @staticmethod
    def _parse_hypotheses(data: Any) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
        """Split a hypothesis JSON artifact into a flat list and a k-indexed lookup dictionary."""
        hypos = data.get("hypotheses", data) if isinstance(data, dict) else data
        k_to: Dict[int, List[Dict[str, Any]]] = {}
        for h in hypos:
            k = h.get("k")
            if k is not None:
                k_to.setdefault(k, []).append(h)
        return hypos, k_to

    @staticmethod
    def _cluster_covered_hypos(hypos: List[Dict[str, Any]], n_per: int) -> List[Dict[str, Any]]:
        """Select top-n hypotheses per response cluster (m), ensuring full community coverage in tables."""
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
        """Lazy-load and index Prompt-Conditioned hypotheses (A_k x R_m) on demand."""
        if not self.k_to_pc:
            data = _read_json(self.run_dir / "prompt_conditioned_hypotheses.json")
            if data is not None:
                self.pc_hypos, self.k_to_pc = self._parse_hypotheses(data)
        return self.k_to_pc

    @property
    def feature_hypotheses_map(self) -> Dict[int, List[Dict[str, Any]]]:
        """Lazy-load and index Feature-Conditioned hypotheses (B_k x T_m) on demand."""
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
    def feature_to_cluster_map(self) -> Dict[int, int]:
        """Map individual SAE feature index f -> feature cluster community ID T_m."""
        if self._feat_to_cluster is None:
            self._feat_to_cluster = {}
            for m, feats in self.feature_clusters.items():
                for f in feats:
                    self._feat_to_cluster[int(f)] = int(m)
        return self._feat_to_cluster

    def _best_hypothesis_by_m(self) -> Dict[int, Dict[str, Any]]:
        """Find the strongest validated hypothesis referencing each feature cluster m."""
        if self._best_hypo_by_m is not None:
            return self._best_hypo_by_m
        best: Dict[int, Dict[str, Any]] = {}
        for hypos in self.feature_hypotheses_map.values():
            for h in hypos:
                m = h.get("m")
                if m is None:
                    continue
                m_int = int(m)
                abs_d = abs(float(h.get("delta", 0.0)))
                prev = best.get(m_int)
                if prev is None or abs_d > abs(float(prev.get("delta", 0.0))):
                    best[m_int] = h
        self._best_hypo_by_m = best
        return best

    # ==========================================================================
    # SECTION 3: Dataset Examples & Token Interpretation Subsystem
    # ==========================================================================

    def _load_examples(self) -> Optional[List[Any]]:
        """Lazy-load raw dataset examples from cached `examples.json` in the checkpoint directory."""
        with self._examples_lock:
            if self._examples is not None:
                return self._examples
            if not self.checkpoint_dir:
                return None
            ex_file = self.checkpoint_dir / "examples.json"
            if not ex_file.exists():
                return None
            try:
                from .dataset import DatasetLoader
                logger.info(f"Loading cached examples from {ex_file}...")
                self._examples = DatasetLoader.load_cached_examples(str(ex_file))
                logger.info(f"Loaded {len(self._examples)} examples into viewer memory.")
                return self._examples
            except Exception as e:
                logger.warning(f"Failed to load cached examples from {ex_file}: {e}")
                return None

    @staticmethod
    def _example_field(ex: Any, key: str) -> str:
        """Extract a string field from a DatasetExample object or dictionary safely."""
        return str(getattr(ex, key, "") or (ex.get(key, "") if isinstance(ex, dict) else "")).strip()

    def _example_view(self, ex: Any) -> Dict[str, str]:
        """Format a dataset example into clean prompt, chosen, and rejected strings."""
        return {
            "prompt": self._example_field(ex, "prompt"),
            "chosen": self._example_field(ex, "chosen"),
            "rejected": self._example_field(ex, "rejected"),
        }

    def _top_examples(self, scores: np.ndarray, examples: Sequence[Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """Return the top-n scoring dataset examples sorted by activation score in descending order."""
        if examples is None or len(examples) == 0:
            return []
        scores_arr = np.asarray(scores).ravel()
        limit = min(len(scores_arr), len(examples))
        if limit == 0:
            return []
        sub_scores = scores_arr[:limit]
        order = np.argsort(sub_scores)[-top_n:][::-1]
        out = []
        for idx in order:
            i = int(idx)
            val = float(sub_scores[i])
            if val <= 0:
                continue
            out.append({
                "index": i,
                "score": val,
                **self._example_view(examples[i]),
            })
        return out

    def _load_pc_cluster_examples(self) -> Optional[Dict[str, Any]]:
        """Load pre-extracted representative tokens and examples for prompt clusters A_k / R_m."""
        return self._reload_if_changed(
            Path(pc_cluster_examples_path(str(self.run_dir))),
            "_pc_cluster_examples",
            "_pc_cluster_examples_mtime",
        )

    def _pc_cluster_tokens(self, cluster_type: str, cid: int) -> List[str]:
        """Fetch top representative tokens describing prompt cluster A_k or response-delta cluster R_m."""
        pc_ex = self._load_pc_cluster_examples()
        if not pc_ex:
            return []
        key = "prompt_cluster_tokens" if cluster_type == "prompt" else "response_cluster_tokens"
        return pc_ex.get(key, {}).get(str(cid), [])

    def _pc_cluster_top_examples(self, cluster_type: str, cid: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Fetch real dataset examples expressing prompt cluster A_k or response-delta cluster R_m."""
        pc_ex = self._load_pc_cluster_examples()
        examples = self._load_examples()
        if not pc_ex or not examples:
            return []
        key = "prompt_cluster_examples" if cluster_type == "prompt" else "response_cluster_examples"
        idxs = pc_ex.get(key, {}).get(str(cid), [])
        out = []
        for i in idxs[:top_n]:
            i_int = int(i)
            if i_int < len(examples):
                out.append({
                    "index": i_int,
                    **self._example_view(examples[i_int]),
                })
        return out

    # ==========================================================================
    # SECTION 4: High-Performance Dense Member Cache & Per-Example Scores (u, s)
    # ==========================================================================

    def _load_feature_matrices(self):
        """Lazy-load the FeatureMatrices container from `matrices_mmap` or `matrices.npz`."""
        if self._feature_matrices is not None:
            return self._feature_matrices
        if not self.checkpoint_dir:
            return None
        try:
            from .feature_matrices import FeatureMatrixExtractor
            logger.info(f"Loading feature matrices from {self.checkpoint_dir}...")
            extractor = FeatureMatrixExtractor(None, None, None, 0, "cpu")
            self._feature_matrices = extractor.extract(
                None, str(self.checkpoint_dir / "matrices.npz"), use_checkpoint=True
            )
            return self._feature_matrices
        except Exception as e:
            logger.warning(f"Failed to load feature matrices from {self.checkpoint_dir}: {e}")
            return None

    def _expected_member_cols(self) -> np.ndarray:
        """Return sorted 1-D array of unique SAE feature indices in communities >= min_partition_cluster_size."""
        min_sz = self.min_partition_cluster_size
        members = set()
        for feats in self.feature_clusters.values():
            if len(feats) >= min_sz:
                members.update(int(f) for f in feats)
        return np.array(sorted(members), dtype=np.int64)

    def _member_positions(self, mats, attr: str) -> np.ndarray:
        """Precompute binary column firing positions for member features across matrix rows."""
        M = getattr(mats, attr)
        cols = self._all_member_cols
        if M is None or cols is None or len(cols) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        tau = self._feature_delta_tau()
        if hasattr(M, "tocsc"):
            M_csc = M.tocsc()
            row_list, col_list = [], []
            for slot, f in enumerate(cols):
                col = M_csc.getcol(int(f))
                active_rows = col.indices[col.data > tau]
                if len(active_rows) > 0:
                    row_list.append(active_rows)
                    col_list.append(np.full(len(active_rows), slot, dtype=np.int32))
            if row_list:
                rows = np.concatenate(row_list)
                slots = np.concatenate(col_list)
                return np.stack([rows, slots], axis=1).astype(np.int64)
            return np.zeros((0, 2), dtype=np.int64)
        M_sub = np.asarray(M[:, cols])
        rows, slots = np.where(M_sub > tau)
        return np.stack([rows, slots], axis=1).astype(np.int64)

    def _member_matrix(self, mats, attr: str) -> Optional[np.ndarray]:
        """Return dense memory-mapped member matrix (N, len(all_member_cols)) for C_max or R_max."""
        cached = self._member_matrices.get(attr)
        if cached is not None:
            return cached
        M = getattr(mats, attr)
        cols = self._all_member_cols
        if M is None or cols is None or len(cols) == 0:
            return None
        if hasattr(M, "toarray"):
            dense = np.asarray(M[:, cols].toarray(), dtype=np.float32)
        else:
            dense = np.asarray(M[:, cols], dtype=np.float32)
        self._member_matrices[attr] = dense
        return dense

    @staticmethod
    def _col_firing_rate(mat, threshold: float) -> np.ndarray:
        """Compute column firing counts across rows exceeding threshold."""
        if hasattr(mat, "tocsr"):
            mat_csr = mat.tocsr()
            mask = mat_csr.data > threshold
            if not np.any(mask):
                return np.zeros(mat.shape[1], dtype=np.float64)
            data_bin = mask.astype(np.float64)
            bin_csr = mat_csr.__class__((data_bin, mat_csr.indices, mat_csr.indptr), shape=mat.shape)
            return np.asarray(bin_csr.sum(axis=0)).ravel()
        return np.asarray((mat > threshold).sum(axis=0)).ravel().astype(np.float64)

    def _feature_delta_tau(self) -> float:
        """The B.1 tau threshold used to extract per-feature deltas (from config JSON)."""
        return float(self._fc_cfg.get("tau", 0.01))

    def _feature_delta(self) -> Optional[np.ndarray]:
        """Compute and cache per-feature preference disparity vector: 1(C>tau) - 1(R>tau)."""
        if self._feat_delta is not None:
            return self._feat_delta
        mats = self._load_feature_matrices()
        if mats is None or mats.C_max is None or mats.R_max is None:
            return None
        tau = self._feature_delta_tau()
        c_rate = self._col_firing_rate(mats.C_max, tau)
        r_rate = self._col_firing_rate(mats.R_max, tau)
        self._feat_delta = (c_rate - r_rate) / float(mats.C_max.shape[0])
        self._persist_feature_delta()
        return self._feat_delta

    def _feature_firings(self) -> Optional[np.ndarray]:
        """Compute and cache total firing counts across chosen and rejected responses."""
        if self._feature_totals is not None:
            return self._feature_totals
        mats = self._load_feature_matrices()
        if mats is None:
            return None
        tot = np.zeros(mats.C_max.shape[1], dtype=np.float32)
        for attr in ("C_max", "R_max"):
            M = getattr(mats, attr)
            if M is not None:
                if hasattr(M, "sum"):
                    tot += np.asarray(M.sum(axis=0)).ravel().astype(np.float32)
                else:
                    tot += np.asarray(M).sum(axis=0).astype(np.float32)
        self._feature_totals = tot
        return self._feature_totals

    def _member_cache_meta(self, mats) -> Dict[str, Any]:
        """Generate fingerprint metadata to validate cache integrity against source matrices."""
        c_shape = list(mats.C_max.shape) if mats and mats.C_max is not None else []
        return {
            "c_shape": c_shape,
            "min_cluster_size": self.min_partition_cluster_size,
            "n_clusters": len(self.feature_clusters),
            "n_members": int(len(self._all_member_cols)) if self._all_member_cols is not None else 0,
        }

    def _member_cache_valid(self, mats, meta: Optional[Dict[str, Any]]) -> bool:
        """Check whether the persisted member cache on disk matches the current run fingerprint."""
        if meta is None:
            return False
        try:
            return all(meta[k] == v for k, v in self._member_cache_meta(mats).items())
        except Exception as e:
            logger.debug(f"Member cache meta malformed ({e}); rebuilding.")
            return False

    @staticmethod
    def _save_npy(path: Path, arr: np.ndarray) -> None:
        """Atomically persist a NumPy array using a process/thread-unique temporary file."""
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

    def _persist_feature_delta(self) -> None:
        """Atomically save computed feature deltas to `viewer_cache/feature_delta.npy`."""
        cache_dir = self.run_dir / "viewer_cache"
        if cache_dir.is_dir() and self._feat_delta is not None:
            self._save_npy(cache_dir / "feature_delta.npy", self._feat_delta)
            _save_json(cache_dir / "feature_delta_meta.json", {"tau": self._feature_delta_tau()})

    def _persist_member_cache(self, mats) -> None:
        """Atomically persist member matrices and positions under `<run>/viewer_cache/`."""
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
        _save_json(cache_dir / "meta.json", self._member_cache_meta(mats), indent=2)
        logger.info(f"Persisted member cache to {cache_dir}.")

    def _load_member_cache(self, mats) -> None:
        """Load dense memory-mapped member cache if valid, otherwise compile and persist it."""
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
                        fd_meta = _read_json(cache_dir / "feature_delta_meta.json") or {}
                        if float(fd_meta.get("tau", 0.01)) == self._feature_delta_tau():
                            self._feat_delta = np.load(fd_path)
                    logger.info(f"Loaded member cache from {cache_dir} (mmap).")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load member cache ({e}); rebuilding.")
            
            # Rebuild member cache
            self._all_member_cols = self._expected_member_cols()
            self._member_positions_cache = {
                "C_max": self._member_positions(mats, "C_max"),
                "R_max": self._member_positions(mats, "R_max"),
            }
            self._member_matrices = {
                "C_max": self._member_matrix(mats, "C_max"),
                "R_max": self._member_matrix(mats, "R_max"),
            }
            self._feature_firings()
            self._feature_delta()
            self._persist_member_cache(mats)

    def _load_cached_example_scores(self, cache_dir: Path, cluster_ids: List[int], mats) -> bool:
        """Attempt zero-copy memory-map load of per-example scores (u, s) if fingerprint matches."""
        meta = _read_json(cache_dir / "example_scores_meta.json")
        if not meta or meta.get("matrices") != self._member_cache_meta(mats):
            return False
        if float(meta.get("tau", -1.0)) != self._feature_delta_tau() or meta.get("cluster_ids") != cluster_ids:
            return False
        if not all((cache_dir / f).exists() for f in ("example_u.npy", "example_s.npy", "example_cluster_ids.npy")):
            return False

        try:
            self._example_u = np.load(cache_dir / "example_u.npy", mmap_mode="r")
            self._example_s = np.load(cache_dir / "example_s.npy", mmap_mode="r")
            self._example_cluster_ids = np.load(cache_dir / "example_cluster_ids.npy")
            logger.info(f"Loaded per-example scores from {cache_dir} (mmap).")
            return True
        except Exception as e:
            logger.warning(f"Failed to load per-example scores ({e}); rebuilding.")
            return False

    def _ensure_example_scores(self, mats) -> None:
        """Ensure per-example disparity and firing score tables (u, s) are built and memory-mapped."""
        with self._scores_lock:
            if self._example_u is not None and self._example_s is not None:
                return

            cluster_ids = sorted(int(k) for k in self.feature_clusters.keys())
            if not cluster_ids:
                return

            cache_dir = self.run_dir / "viewer_cache"
            if self._load_cached_example_scores(cache_dir, cluster_ids, mats):
                return

            if self._all_member_cols is None:
                self._all_member_cols = self._expected_member_cols()

            n_members = len(self._all_member_cols)
            K = len(cluster_ids)
            A = np.zeros((n_members, K), dtype=np.float32)
            for c_idx, cid in enumerate(cluster_ids):
                feats = np.asarray(self.feature_clusters[cid], dtype=np.int64)
                if len(feats) > 0:
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
            _save_json(
                cache_dir / "example_scores_meta.json",
                {"tau": tau, "cluster_ids": cluster_ids, "matrices": self._member_cache_meta(mats)},
            )

            self._example_u = u
            self._example_s = s
            self._example_cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
            logger.info(f"Built per-example scores u/s ({u.shape}) and persisted under {cache_dir}.")

    # ==========================================================================
    # SECTION 5: Cluster & Feature Detail Interpretations (B_k, T_m, A_k, R_m, f)
    # ==========================================================================

    def _cached_info(self, key: str, build_fn) -> Dict[str, Any]:
        """Memoize formatted per-cluster UI payloads in the in-memory info cache."""
        cached = self._cluster_info_cache.get(key)
        if cached is not None:
            return cached
        res = build_fn()
        self._cluster_info_cache[key] = res
        return res

    def _top_cluster_features(self, m: int, top_n: int = 8) -> List[Dict[str, Any]]:
        """Return top member SAE features for feature cluster T_m ranked by global firing activity."""
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
        valid_feats = [f for f in feats if 0 <= f < len(tot)]
        if not valid_feats:
            return []
        firings = tot[valid_feats]
        order = np.argsort(firings)[-top_n:][::-1]
        return [{
            "feature_index": int(valid_feats[j]),
            "firing": float(firings[j]),
            "neuronpedia_url": self._neuronpedia_url(int(valid_feats[j])),
        } for j in order]

    def _cluster_top_examples(self, m: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """Return top dataset examples firing feature cluster T_m using dense RAM matrices."""
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

        self._ensure_example_scores(mats)
        scores = np.zeros(len(examples), dtype=np.float32)
        if self._all_member_cols is not None:
            slots = np.searchsorted(self._all_member_cols, top_f)
            valid_mask = (slots < len(self._all_member_cols)) & (self._all_member_cols[slots] == top_f)
            slots = slots[valid_mask]
            if len(slots) > 0:
                for attr in ("C_max", "R_max"):
                    M = self._member_matrix(mats, attr)
                    if M is not None:
                        scores += M[:, slots].sum(axis=1)
        return self._top_examples(scores, examples, top_n)

    def _feature_cluster_info(self, m: int, top_n_examples: int = 5) -> Dict[str, Any]:
        """Whole-cluster interpretation for T_m: LLM label, top features, and representative examples."""
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
        self._prewarm_neuronpedia_features([tf["feature_index"] for tf in res["top_features"]])
        return res

    def _data_cluster_info(self, k: int, top_n_examples: int = 5) -> Dict[str, Any]:
        """Interpretation for data cluster B_k: title, description, and sampled prompt exemplars."""
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

    @staticmethod
    def _csr_col(mat: Any, col_idx: int) -> np.ndarray:
        """Extract a single column from a sparse or dense matrix as a dense 1-D array."""
        if hasattr(mat, "toarray"):
            return np.asarray(mat[:, col_idx].toarray()).ravel()
        return np.asarray(mat[:, col_idx]).ravel()

    def _feature_act(self, mats, feature_index: int) -> np.ndarray:
        """Extract per-example total activation (C_max + R_max) for individual SAE feature `feature_index`."""
        if self._all_member_cols is not None:
            slot = int(np.searchsorted(self._all_member_cols, feature_index))
            if slot < len(self._all_member_cols) and int(self._all_member_cols[slot]) == feature_index:
                act = np.zeros(mats.C_max.shape[0], dtype=np.float32)
                for attr in ("C_max", "R_max"):
                    M = self._member_matrices.get(attr)
                    if M is not None:
                        act += M[:, slot]
                return act
        return self._csr_col(mats.C_max, feature_index) + self._csr_col(mats.R_max, feature_index)

    def _feature_detail(self, feature_index: int, top_n: int = 5) -> Dict[str, Any]:
        """Per-feature interpretation: firing statistics, top examples, and Neuronpedia web metadata."""
        f_int = int(feature_index)
        cache_key = f"feat_{f_int}_{top_n}"

        def build() -> Dict[str, Any]:
            mats = self._load_feature_matrices()
            examples = self._load_examples()
            out: Dict[str, Any] = {"feature_index": f_int}
            if mats is None:
                out["error"] = "feature matrices not cached for this run"
                return out
            d_sae = mats.C_max.shape[1]
            if not (0 <= f_int < d_sae):
                out["error"] = f"feature index out of range (d_sae={d_sae})"
                return out

            act = self._feature_act(mats, f_int)
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

            url = self._neuronpedia_url(f_int)
            if url:
                out["neuronpedia_url"] = url
                np_data = self._get_neuronpedia_feature(f_int)
                if np_data:
                    out["neuronpedia"] = np_data
            return out

        return self._cached_info(cache_key, build)

    # ==========================================================================
    # SECTION 6: Live Neural Inspector (Mode A Prompt & Mode B Pair Predictions)
    # ==========================================================================

    def get_inspector(self):
        """Get or initialize the live NeuralInspector configured for this run's Model and SAE."""
        if self.inspector is None:
            from pdd.neural_inspector import get_neural_inspector
            model_cfg = self.summary.get("config", {}).get("model", {})
            sae_cfg = self.summary.get("config", {}).get("sae", {})

            self.inspector = get_neural_inspector(
                model_path=model_cfg.get("path", "google/gemma-2-2b"),
                sae_repo=sae_cfg.get("repo", "gemma-scope-2b-pt-res-canonical"),
                sae_id=sae_cfg.get("sae_id"),
                layer=sae_cfg.get("layer", 12),
                d_in=sae_cfg.get("d_in"),
                d_sae=sae_cfg.get("d_sae"),
                k=sae_cfg.get("k"),
            )
        return self.inspector

    def _cluster_signals(self, activations: np.ndarray, mode: str = "sum") -> Dict[int, float]:
        """Aggregate a live (d_sae,) activation vector into per-feature-cluster signals T_m."""
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
        """Score hypotheses by |delta| multiplied by live per-cluster signal strength."""
        ev: List[Tuple[Dict[str, Any], float]] = []
        for h in hypos:
            m = h.get("m")
            sig = signals.get(m, 0.0) if m is not None else 0.0
            ev.append((h, abs(float(h.get("delta", 0.0))) * abs(sig)))
        return ev

    def _cluster_keywords(self, activations: np.ndarray, feature_ms: Sequence[int], top_n: int = 3) -> List[str]:
        """Return top individual SAE features by live activation within specified feature clusters."""
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
        """Rank data clusters B_k by live-evidence-weighted hypotheses."""
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
            feature_ms = [h.get("m") for h in hypos if h.get("m") is not None]
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
        """Extract top Prompt-Conditioned hypotheses (A_k x R_m) matching live prompt tokens."""
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
        for k, _ in scored_ak[:top_k]:
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
                    f"indicating that post-training will likely {direction_word} these response features."
                )

                pc_shifts.append({
                    "prompt_cluster_k": k,
                    "response_cluster_m": m,
                    "pipeline_type": "prompt_conditioned",
                    "delta": delta,
                    "effect_direction": "Amplified after DPO" if is_amplified else "Suppressed after DPO",
                    "z_score": z,
                    "cohens_d": d,
                    "prompt_cluster_tokens": p_tokens[:6],
                    "response_cluster_tokens": r_tokens[:6],
                    "interpretation": interpretation,
                })

        return pc_shifts[:top_k]

    @staticmethod
    def _project_clusters(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip internal full hypothesis arrays from scored clusters for lightweight JSON responses."""
        return [{k: v for k, v in c.items() if k != "hypos"} for c in scored]

    def _sae_feature_item(self, i: int, val: float, m: Optional[int]) -> Dict[str, Any]:
        """Format an active SAE feature item with its cluster assignment and Neuronpedia link."""
        return {
            "feature_index": i,
            "activation": round(val, 4),
            "cluster_m": m,
            "neuronpedia_url": self._neuronpedia_url(i),
        }

    def _top_sae_features(self, activations: np.ndarray, top_n: int = 8, min_cluster_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract top active SAE features sorted by magnitude."""
        if activations is None or len(activations) == 0:
            return []
        min_sz = self.min_partition_cluster_size if min_cluster_size is None else min_cluster_size
        feat_to_cluster = self.feature_to_cluster_map
        pos_idx = np.flatnonzero(activations > 0)
        if len(pos_idx) == 0:
            return []
        sorted_pos = pos_idx[np.argsort(-activations[pos_idx])]
        out = []
        for idx in sorted_pos:
            i = int(idx)
            val = float(activations[i])
            m = feat_to_cluster.get(i)
            if m is not None and len(self.feature_clusters.get(m, [])) < min_sz:
                m = None
            out.append(self._sae_feature_item(i, val, m))
            if len(out) >= top_n:
                break
        return out

    def _inspect_feature_samples(self, m: int, k: int, side: str = "amplify") -> Dict[str, Any]:
        """Top preference examples whose labels amplify (u>0) or suppress (u<0) feature cluster T_m."""
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
                i_int = int(i)
                if i_int >= len(examples):
                    continue
                ex = examples[i_int]
                u_i = float(u[i_int])
                samples.append({
                    "index": i_int,
                    "u": u_i,
                    "s": float(s[i_int]),
                    "effect_direction": "Amplified after DPO" if u_i > 0 else ("Suppressed after DPO" if u_i < 0 else "Neutral"),
                    **self._example_view(ex),
                })
            return {**base, "total_matching": int(len(present)), "samples": samples}

        return self._cached_info(cache_key, build)

    def _inspect_compound_samples(self, conditions: List[Tuple[int, str, float]], k: int) -> Dict[str, Any]:
        """Top preference examples satisfying EVERY compound condition: [(m, direction, tau), ...]."""
        k = max(1, min(int(k), 200))
        conds: List[Tuple[int, str, float]] = []
        for m, direction, tau in conditions:
            direction = direction if direction in ("amplify", "suppress") else "amplify"
            tau_val = float(tau) if tau is not None and float(tau) > 0 else 0.1
            conds.append((int(m), direction, tau_val))
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
                i_int = int(i)
                if i_int >= len(examples):
                    continue
                ex = examples[i_int]
                u_map = {str(m): float(u[i_int]) for (m, _, _), u in zip(conds, col_u)}
                samples.append({
                    "index": i_int,
                    "u": u_map,
                    "s": {str(m): float(s[i_int]) for (m, _, _), s in zip(conds, col_s)},
                    "score": float(sum(abs(u_map[str(m)]) / t for (m, _, t) in conds)),
                    "effect_directions": {
                        str(m): "Amplified after DPO" if u[i_int] > 0 else "Suppressed after DPO"
                        for (m, _, _), u in zip(conds, col_u)
                    },
                    **self._example_view(ex),
                })
            return {**base, "total_matching": int(len(idxs)), "conditions": cond_list, "samples": samples}

        return self._cached_info(cache_key, build)

    # ==========================================================================
    # SECTION 7: Neuronpedia Integration & Web Metadata Subsystem
    # ==========================================================================

    @staticmethod
    def _neuronpedia_sae_set(sae_cfg: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Map run SAE configuration to Neuronpedia (model_slug, sae_slug)."""
        repo = str(sae_cfg.get("repo", "")).lower()
        sae_id = str(sae_cfg.get("sae_id", "")).lower()
        layer = sae_cfg.get("layer", 12)
        if "gemma-scope" in repo or "gemma-2-2b" in repo or "canonical" in repo:
            model_id = "gemma-2-2b"
            if "canonical" in repo or "res" in repo:
                sae_set = f"{layer}-gemmascope-res-16k"
            elif "mlp" in repo:
                sae_set = f"{layer}-gemmascope-mlp-16k"
            else:
                sae_set = f"{layer}-gemmascope-res-16k"
            return model_id, sae_set
        return None

    @staticmethod
    def _neuronpedia_verified(model_id: str, sae_set: str) -> bool:
        """Probe Neuronpedia API to verify whether the model/SAE slug pair exists."""
        import urllib.request
        url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_set}/0"
        req = urllib.request.Request(url, headers={"User-Agent": "PDD-Viewer/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"Neuronpedia slug probe for {model_id}/{sae_set} failed ({e}).")
            return False

    def _neuronpedia_cache_dir(self) -> Path:
        """Return path to `<run_dir>/viewer_cache/neuronpedia/`."""
        d = self.run_dir / "viewer_cache" / "neuronpedia"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _get_neuronpedia_feature(self, feature_index: int) -> Optional[Dict[str, Any]]:
        """Fetch feature metadata from Neuronpedia with local disk caching."""
        np_set = self._neuronpedia_set()
        if not np_set:
            return None
        cache_file = self._neuronpedia_cache_dir() / f"feat_{feature_index}.json"
        if cache_file.exists():
            try:
                data = _read_json(cache_file)
                if data:
                    return data
            except Exception as e:
                logger.warning(f"Corrupt Neuronpedia cache file {cache_file.name}; refetching: {e}")

        data = self._neuronpedia_feature(np_set[0], np_set[1], feature_index)
        if data is not None:
            try:
                _save_json(cache_file, data)
            except Exception as e:
                logger.debug(f"Failed to persist Neuronpedia cache for feature {feature_index}: {e}")
        return data

    def _prewarm_neuronpedia_features(self, feature_indices: List[int]) -> None:
        """Asynchronously pre-warm Neuronpedia cache in the background for active features."""
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
    def _neuronpedia_feature(model_id: str, sae_set: str, feature_index: int) -> Optional[Dict[str, Any]]:
        """Fetch JSON feature card directly from Neuronpedia over HTTPS."""
        import urllib.request
        url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_set}/{feature_index}"
        req = urllib.request.Request(url, headers={"User-Agent": "PDD-Viewer/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    exs = data.get("activations", [])[:3]
                    return {
                        "model": model_id,
                        "sae": sae_set,
                        "feature_index": feature_index,
                        "description": data.get("description", ""),
                        "label": data.get("label", ""),
                        "top_tokens": [t.get("token") for t in data.get("top_tokens", [])[:6] if t.get("token")],
                        "examples_count": len(data.get("activations", [])),
                        "top_examples": [
                            {"maxValue": ex.get("maxValue"), "tokens": ex.get("tokens", [])[:30]} for ex in exs
                        ],
                    }
        except Exception as e:
            logger.debug(f"Neuronpedia API fetch failed for feature {feature_index} ({e}).")
        return None

    def _neuronpedia_set(self) -> Optional[Tuple[str, str]]:
        """Return verified Neuronpedia (model_slug, sae_slug) pair, verified in the background."""
        if self._np_set is not None:
            return self._np_set
        sae_cfg = self.summary.get("config", {}).get("sae", {})
        pair = self._neuronpedia_sae_set(sae_cfg)
        if not pair:
            return None
        with self._np_verify_lock:
            if self._np_set is not None:
                return self._np_set
            if not self._np_verifying:
                self._np_verifying = True
                threading.Thread(target=self._verify_neuronpedia_slug, daemon=True, name="PDD-NPVerify").start()
        return pair

    def _verify_neuronpedia_slug(self) -> None:
        """Background worker verifying Neuronpedia slug resolution."""
        sae_cfg = self.summary.get("config", {}).get("sae", {})
        pair = self._neuronpedia_sae_set(sae_cfg)
        if pair and self._neuronpedia_verified(pair[0], pair[1]):
            self._np_set = pair
            logger.info(f"Neuronpedia verified for {pair[0]}/{pair[1]}.")
        else:
            logger.info("Neuronpedia integration disabled for this run (no verified slug).")

    def _neuronpedia_url(self, feature_index: int) -> Optional[str]:
        """Generate web browser URL to the feature card on neuronpedia.org."""
        np_set = self._neuronpedia_set()
        if not np_set:
            return None
        return f"https://www.neuronpedia.org/{np_set[0]}/{np_set[1]}/{feature_index}"


# ==============================================================================
# Global State Accessor
# ==============================================================================

_STATE: Optional[ViewerState] = None


def get_state() -> ViewerState:
    """Lazy global state accessor for the targeted run (configured via --run_dir)."""
    global _STATE
    if _STATE is None:
        _STATE = ViewerState()
    return _STATE


# ==============================================================================
# FastAPI REST Endpoints
# ==============================================================================

@app.get("/api/runs")
def list_runs() -> Dict[str, Any]:
    """Return metadata summary for the active run directory (single-run mode)."""
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
    """Retrieve run summary, validation metrics, cluster labels, and top hypotheses."""
    state = get_state()
    val_file = state.run_dir / "p4_validation" / "p4_r2_metrics.json"
    val_data = _read_json(val_file)
    validation_metrics = val_data if val_data is not None else {}

    # Ensure full hypothesis lists are loaded into memory
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
def get_feature_cluster_info(
    m: int = Query(..., description="Feature cluster community ID T_m"),
    top_n: int = Query(5, description="Number of top examples to return")
) -> Dict[str, Any]:
    """Whole-cluster interpretation for SAE feature cluster community T_m."""
    state = get_state()
    return state._feature_cluster_info(m, top_n_examples=top_n)


@app.get("/api/inspect_feature_samples")
def get_inspect_feature_samples(
    m: Optional[int] = Query(None, description="Feature cluster community ID T_m (single-cluster query)"),
    k: int = Query(50, ge=1, le=200, description="Number of top samples to return"),
    side: str = Query("amplify", description="'amplify' (chosen fires concept) or 'suppress' (rejected fires concept)"),
    conditions: Optional[str] = Query(None, description="Compound query: 'm:amplify|suppress[:tau],...'")
) -> Dict[str, Any]:
    """Inverse search (Tab 4): find training preference pairs whose labels drive feature cluster T_m."""
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
def get_pc_cluster_examples(
    cluster_type: str = Query(..., description="'prompt' for A_k or 'response' for R_m"),
    cid: int = Query(..., description="Cluster ID integer"),
    top_n: int = Query(5, description="Number of top examples to return")
) -> Dict[str, Any]:
    """Retrieve representative tokens and real examples expressing prompt cluster A_k or response cluster R_m."""
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
def get_cluster_detail(
    type: str = Query(..., description="'data' (B_k), 'feature' (T_m), 'prompt' (A_k), or 'response' (R_m)"),
    id: int = Query(..., description="Cluster ID integer"),
    top_n: int = Query(5, description="Number of top examples to return")
) -> Dict[str, Any]:
    """Unified polymorphic lookup endpoint for all 4 cluster families (B_k, T_m, A_k, R_m)."""
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
def get_feature_detail(
    f: int = Query(..., ge=0, description="SAE feature index"),
    top_n: int = Query(5, description="Number of top firing examples to return")
) -> Dict[str, Any]:
    """Per-feature interpretation: firing statistics, top examples, and Neuronpedia web metadata."""
    state = get_state()
    return state._feature_detail(f, top_n=top_n)


@app.post("/api/inspect_prompt")
def inspect_prompt(req: PromptInspectionRequest) -> Dict[str, Any]:
    """Mode A: Live prompt forward pass through GPU Model + SAE to predict downstream behavioral shifts."""
    state = get_state()
    prompt_text = req.prompt.strip()
    if not prompt_text:
        return {"prompt": "", "matched_clusters": [], "predicted_behavior_shifts": []}

    # 1. Real GPU Forward Pass -> Prompt Features P(x)
    inspector = state.get_inspector()
    p_feat = inspector.extract_prompt_features(prompt_text)

    # 2. Per-feature-cluster activity of the live prompt (T_m <- sum of P(x) members)
    act = state._cluster_signals(p_feat, mode="sum")

    scored_clusters = state._score_data_clusters(act, p_feat)[:req.top_k]
    matched_clusters = state._project_clusters(scored_clusters)

    # 3. Extract Feature-Conditioned Predicted Shifts (B_k x T_m)
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
    """Mode B: Batched GPU forward pass on preference pair to measure exact SAE disparity u = 1(C>0.01) - 1(R>0.01)."""
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


# ==============================================================================
# Frontend Static Asset Mounting & HTML Servicing
# ==============================================================================

if VIEWER_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(VIEWER_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serve the single-page interactive viewer frontend (`viewer/index.html`)."""
    index_file = VIEWER_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"status": "PDD Viewer Backend Running", "frontend_path": str(index_file)})


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    """CLI launcher for the PDD Interactive Web Viewer."""
    parser = argparse.ArgumentParser(description="Launch PDD Interactive Web Viewer")
    parser.add_argument("--run_dir", type=str, default="runs/qwen3_1.7b_dolci", help="Path to target PDD run directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7000, help="Port to bind server")
    parser.add_argument("--no-prewarm", action="store_true", help="Skip loading/building the SAE member cache at startup")
    args = parser.parse_args()

    os.environ["PDD_RUN_DIR"] = args.run_dir
    global _STATE
    _STATE = ViewerState(run_dir=args.run_dir)

    # Load (or build + persist once) the dense SAE member cache before serving
    if not args.no_prewarm:
        _STATE.prewarm()

    # Pre-warm cached examples in a background daemon thread to eliminate first-click disk latency
    threading.Thread(target=_STATE._load_examples, daemon=True).start()

    logger.info(f"Starting PDD Viewer for '{args.run_dir}' at http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
