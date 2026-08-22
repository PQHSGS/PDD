"""FastAPI server for the Predictive Data Debugging (PDD) Interactive Viewer.

Serves run metadata, feature-conditioned (B.1) & prompt-conditioned (B.2) hypotheses,
cluster statistics, and live prompt/preference pair neural inspection endpoints.
Points directly to a target run directory and its linked checkpoint artifacts.
"""
from __future__ import annotations

import argparse
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
from . import inspection
from .neuronpedia import NeuronpediaClient

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


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for NumPy scalar and array types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _save_json(path: Path, data: Any, indent: Optional[int] = None) -> None:
    """Atomically write a JSON file using a thread-safe process/thread-unique temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}_{os.getpid()}_{threading.get_ident()}.tmp.json"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=_json_default)
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

        self._val_metrics_raw: Optional[Any] = None
        """Raw cache of `p4_validation/p4_r2_metrics.json` (per-cluster R^2 / Pearson r)."""

        self._val_metrics_mtime: float = 0.0
        """File modification time of `p4_r2_metrics.json` for hot reload (mechanical HDD: stat-only on hits)."""

        # ----------------------------------------------------------------------
        # 6. Dense Cluster Member Lookup Caches
        # ----------------------------------------------------------------------
        self._all_member_cols: Optional[np.ndarray] = None
        """Sorted array of unique SAE feature indices belonging to ANY community >= min_size."""

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
        """Mutex protecting lazy dataset example parsing and caching."""

        self._member_cache_lock: threading.Lock = threading.Lock()
        """Mutex protecting compilation and memory-mapping of dense member matrices."""

        self.np_client: NeuronpediaClient = NeuronpediaClient(self.run_dir / "viewer_cache" / "neuronpedia")
        """Self-contained Neuronpedia subsystem: slug verification (own lock), LRU-capped
        RAM card cache, per-run disk cache, and one shared prewarm thread pool."""

        self.inspector = None
        """Lazy-loaded NeuralInspector instance for live GPU model/SAE forward passes."""

        self._inspector_lock: threading.Lock = threading.Lock()
        """Mutex serializing live Mode A/B GPU forward passes through the shared NeuralInspector."""

        # ----------------------------------------------------------------------
        # 8. Checkpoint Artifact & Validation Matrices Caches
        # ----------------------------------------------------------------------
        self._fc_result: Optional[Any] = None
        """Lazy-loaded FeatureConditionedResult containing cluster_assignments, u_matrix, s_matrix."""

        self._pc_prompt_clusters: Optional[Dict[int, List[int]]] = None
        """Lazy-loaded A_k prompt feature cluster membership (from `prompt_conditioned.npz` only)."""

        self._cluster_validation_metrics: Optional[Dict[str, Any]] = None
        """Aggregated per-feature predicted vs observed post-DPO deltas across feature clusters."""

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

    def _load_validation_metrics(self) -> Dict[str, Any]:
        """P4 validation metrics, re-read when `p4_r2_metrics.json` updates on disk (empty dict if absent)."""
        if self._val_metrics_raw is None and not getattr(self, "_val_metrics_warned", False):
            if not (self.run_dir / "p4_validation" / "p4_r2_metrics.json").exists():
                self._val_metrics_warned = True
                logger.warning(
                    f"No p4_validation/p4_r2_metrics.json under '{self.run_dir}'; "
                    "R^2 metrics will read as empty until a P4 validation run writes it."
                )
        raw = self._reload_if_changed(
            self.run_dir / "p4_validation" / "p4_r2_metrics.json", "_val_metrics_raw", "_val_metrics_mtime"
        )
        if raw is None:
            return {}
        return raw

    def prewarm(self) -> None:
        """Prewarm the dense member cache at server startup for zero-latency lookups."""
        self._ensure_member_cache()

    def prefetch_background(self) -> None:
        """Sequentially prewarm all lazily-loaded disk singletons in a background daemon thread.

        Pre-warms cached examples (examples.json), feature-conditioned matrices (u_matrix/s_matrix),
        P4 validation metrics, and prompt-conditioned clusters. Sequential execution keeps mechanical
        HDD contention low while ensuring instant first-click responses across all tabs.
        """
        def _bg_prefetch() -> None:
            try:
                self._load_examples()
                self._load_fc_result()
                self._get_cluster_validation_metrics()
                self._load_pc_prompt_clusters()
            except Exception as e:
                logger.warning(f"Background prefetch failed ({e}); lazy loaders will retry on demand.")

        threading.Thread(target=_bg_prefetch, daemon=True, name="PDD-StatePrefetch").start()

    def _ensure_member_cache(self) -> None:
        """Lazily build/load the dense member cache when skipped (e.g. --no-prewarm boots).

        Thread-safe: ``_load_member_cache`` holds ``_member_cache_lock`` and is a
        no-op once ``_all_member_cols`` is populated, so concurrent requests converge
        on the first build.
        """
        if self._all_member_cols is not None or not self.feature_clusters:
            return
        mats = self._load_feature_matrices()
        if mats is not None:
            self._load_member_cache(mats)

    @property
    def _fc_cfg(self) -> Dict[str, Any]:
        """Feature-conditioned pipeline configuration block from run metadata (warn once when absent)."""
        cfg = self.summary.get("config", {}).get("feature_conditioned", {})
        if not cfg and not getattr(self, "_fc_cfg_warned", False):
            self._fc_cfg_warned = True
            logger.warning(
                "No 'feature_conditioned' block in pdd_summary.json config; "
                "hypothesis filters fall back to built-in defaults."
            )
        return cfg

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

    def _feature_delta_tau(self) -> float:
        """The B.1 tau threshold used to extract per-feature deltas (from config JSON)."""
        return float(self._fc_cfg.get("tau", 0.01))

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

    def _load_pc_cluster_examples(self) -> Optional[Dict[str, Any]]:
        """Prompt-cluster (A_k / R_m) examples, re-read when `prompt_conditioned_cluster_examples.json` updates."""
        return self._reload_if_changed(
            Path(pc_cluster_examples_path(str(self.run_dir))),
            "_pc_cluster_examples",
            "_pc_cluster_examples_mtime",
        )

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
            elif not self.pc_hypos and self.summary:
                # Fallback to the truncated top-k copy embedded in pdd_summary.json.
                logger.warning(
                    "prompt_conditioned_hypotheses.json missing under '%s'; "
                    "falling back to summary top hypotheses.",
                    self.run_dir,
                )
                self.pc_hypos, self.k_to_pc = self._parse_hypotheses(
                    {"hypotheses": self.summary.get("top_prompt_conditioned_hypotheses", [])}
                )
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

    def _load_fc_result(self) -> Optional[Any]:
        """Lazy-load FeatureConditionedResult containing cluster_assignments, u_matrix, and s_matrix."""
        if self._fc_result is not None:
            return self._fc_result
        if not self.checkpoint_dir:
            return None
        fc_file = self.checkpoint_dir / "feature_conditioned.npz"
        if not fc_file.exists():
            return None
        try:
            from .feature_conditioned import FeatureConditionedResult
            self._fc_result = FeatureConditionedResult.load_checkpoint(str(fc_file))
            return self._fc_result
        except Exception as e:
            logger.warning(f"Failed to load feature_conditioned.npz: {e}")
            return None

    def _load_pc_prompt_clusters(self) -> Optional[Dict[int, List[int]]]:
        """Lazy-load ONLY the small `prompt_clusters_json` key from the (multi-GB) prompt_conditioned.npz.

        Reading a single key via ``np.load(mmap_mode="r")`` avoids memory-mapping the huge
        c_matrix/u_matrix arrays and parsing the ~1M hypothesis objects the full
        `PromptConditionedResult.load_checkpoint` would touch.
        """
        if self._pc_prompt_clusters is not None:
            return self._pc_prompt_clusters
        if not self.checkpoint_dir:
            return None
        pc_file = self.checkpoint_dir / "prompt_conditioned.npz"
        if not pc_file.exists():
            return None
        try:
            with np.load(pc_file, mmap_mode="r", allow_pickle=True) as data:
                if "prompt_clusters_json" not in data:
                    logger.warning("prompt_conditioned.npz has no prompt_clusters_json key.")
                    return None
                clusters = {int(k): v for k, v in json.loads(str(data["prompt_clusters_json"])).items()}
            self._pc_prompt_clusters = clusters
            return clusters
        except Exception as e:
            logger.warning(f"Failed to load prompt_clusters from prompt_conditioned.npz: {e}")
            return None

    def _get_cluster_validation_metrics(self) -> Dict[str, Any]:
        """Aggregate per-feature predicted vs observed post-DPO deltas into per-cluster metrics."""
        if self._cluster_validation_metrics is not None:
            return self._cluster_validation_metrics
        from .validation import cluster_validation_metrics
        self._cluster_validation_metrics = cluster_validation_metrics(
            self.feature_clusters, self.run_dir / "p4_validation"
        )
        return self._cluster_validation_metrics

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
                from .data import DatasetLoader
                logger.info(f"Loading cached examples from {ex_file}...")
                self._examples = DatasetLoader.load_json_cache(str(ex_file))
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
        return inspection.top_examples(scores, examples, self._example_view, top_n)

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
        member_indices = pc_ex.get(key, {}).get(str(cid), [])
        out = []
        for i in member_indices[:top_n]:
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

    def _member_matrix(self, mats, attr: str) -> Optional[np.ndarray]:
        """Return the dense member matrix (N, len(all_member_cols)) for C_max or R_max.

        RAM note: on a COLD cache this materializes a full dense (N x n_members)
        float32 array in RAM via one BLAS product; only after `_persist_member_cache`
        runs are subsequent boots mmap-backed from `<run>/viewer_cache/`.
        """
        cached = self._member_matrices.get(attr)
        if cached is not None:
            return cached
        M = getattr(mats, attr)
        cols = self._all_member_cols
        if M is None or cols is None or len(cols) == 0:
            return None
        if hasattr(M, "tocsr"):
            import scipy.sparse as sp
            d_sae = M.shape[1]
            n_cols = len(cols)
            # Fast BLAS column slice via sparse indicator matrix
            E = sp.csr_matrix((np.ones(n_cols, dtype=np.float32), (cols, np.arange(n_cols))), shape=(d_sae, n_cols))
            dense = (M @ E).toarray().astype(np.float32)
        else:
            dense = np.asarray(M[:, cols], dtype=np.float32)
        self._member_matrices[attr] = dense
        return dense

    def _feature_firings(self) -> Optional[np.ndarray]:
        """Compute and cache total firing counts across chosen and rejected responses for member features."""
        if self._feature_totals is not None:
            return self._feature_totals
        mats = self._load_feature_matrices()
        if mats is None or self._all_member_cols is None:
            return None
        d_sae = mats.C_max.shape[1]
        tot = np.zeros(d_sae, dtype=np.float32)
        M_c = self._member_matrices.get("C_max")
        M_r = self._member_matrices.get("R_max")
        if M_c is not None and M_r is not None:
            tot[self._all_member_cols] = M_c.sum(axis=0) + M_r.sum(axis=0)
        self._feature_totals = tot
        return self._feature_totals

    def _member_cache_meta(self, mats) -> Dict[str, Any]:
        """Generate fingerprint metadata to validate cache integrity against source matrices."""
        c_shape = [int(x) for x in mats.C_max.shape] if mats and mats.C_max is not None else []
        n_members = len(self._all_member_cols) if self._all_member_cols is not None else len(self._expected_member_cols())
        return {
            "c_shape": c_shape,
            "min_cluster_size": int(self.min_partition_cluster_size),
            "n_clusters": int(len(self.feature_clusters)),
            "n_members": int(n_members),
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

    def _persist_member_cache(self, mats) -> None:
        """Atomically persist member matrices under `<run>/viewer_cache/`."""
        cache_dir = self.run_dir / "viewer_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "member_cols.npy": self._all_member_cols,
            "member_matrix_C_max.npy": self._member_matrices.get("C_max"),
            "member_matrix_R_max.npy": self._member_matrices.get("R_max"),
            "feature_totals.npy": self._feature_totals,
        }
        for name, arr in payload.items():
            if arr is not None:
                _save_npy(cache_dir / name, arr)
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
                "member_cols.npy",
                "member_matrix_C_max.npy",
                "member_matrix_R_max.npy",
                "feature_totals.npy",
            ]
            meta = _read_json(cache_dir / "meta.json")
            if all((cache_dir / f).exists() for f in cache_files) and self._member_cache_valid(mats, meta):
                try:
                    self._all_member_cols = np.load(cache_dir / "member_cols.npy")
                    self._member_matrices = {
                        "C_max": np.load(cache_dir / "member_matrix_C_max.npy", mmap_mode="r"),
                        "R_max": np.load(cache_dir / "member_matrix_R_max.npy", mmap_mode="r"),
                    }
                    self._feature_totals = np.load(cache_dir / "feature_totals.npy")
                    logger.info(f"Loaded member cache from {cache_dir} (mmap).")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load member cache ({e}); rebuilding.")
            
            # Rebuild member cache
            self._all_member_cols = self._expected_member_cols()
            self._member_matrices = {
                "C_max": self._member_matrix(mats, "C_max"),
                "R_max": self._member_matrix(mats, "R_max"),
            }
            self._feature_firings()
            self._persist_member_cache(mats)

    # ==========================================================================
    # SECTION 5: Cluster & Feature Detail Interpretations (B_k, T_m, A_k, R_m, f)
    # ==========================================================================

    _CLUSTER_INFO_CACHE_MAX = 512
    """FIFO cap on memoized per-cluster UI payloads (bounded RAM across many m/top_n/sort keys)."""

    def _cached_info(self, key: str, build_fn) -> Dict[str, Any]:
        """Memoize formatted per-cluster UI payloads in the in-memory info cache."""
        cached = self._cluster_info_cache.get(key)
        if cached is not None:
            return cached
        res = build_fn()
        self._cluster_info_cache[key] = res
        if len(self._cluster_info_cache) > self._CLUSTER_INFO_CACHE_MAX:
            # Drop the oldest entry (dict preserves insertion order).
            self._cluster_info_cache.pop(next(iter(self._cluster_info_cache)))
        return res

    def _top_cluster_features(self, m: int, top_n: int = 8) -> List[Dict[str, Any]]:
        """Return top member SAE features for feature cluster T_m ranked by global firing activity."""
        m_int = int(m)
        feats = self.feature_clusters.get(m_int, [])
        if not feats:
            return []
        label_obj = self._load_feature_cluster_labels().get(m_int, {})
        c_title = label_obj.get("title", f"Feature Cluster T_{m_int}")
        c_keywords = label_obj.get("keywords", [])

        def _make_entry(f_idx: int, firing_val: float = 0.0) -> Dict[str, Any]:
            np_data = self._get_neuronpedia_feature(f_idx, allow_network=False)
            label = (np_data.get("label") or np_data.get("description") or f"{c_title} Latent") if np_data else f"{c_title} Latent"
            desc = (np_data.get("description") or f"Member of {c_title} (T_{m_int})") if np_data else f"Member of {c_title} (T_{m_int})"
            toks = (np_data.get("top_tokens") or c_keywords[:4]) if np_data else c_keywords[:4]
            return {
                "feature_index": f_idx,
                "firing": float(firing_val),
                "neuronpedia_url": self._neuronpedia_url(f_idx),
                "description": desc,
                "label": label,
                "top_tokens": toks,
                "cluster_m": m_int,
            }

        tot = self._feature_firings()
        if tot is None or len(tot) == 0:
            self._ensure_member_cache()
            tot = self._feature_firings()
        if tot is None or len(tot) == 0:
            logger.warning(f"Feature firing totals unavailable; returning T_{m} members unranked.")
            return [_make_entry(int(f), 0.0) for f in feats[:top_n]]

        valid_feats = [f for f in feats if 0 <= f < len(tot)]
        if not valid_feats:
            return []
        firings = tot[valid_feats]
        order = np.argsort(firings)[-top_n:][::-1]
        return [_make_entry(int(valid_feats[j]), firings[j]) for j in order]

    def _cluster_top_examples(self, m: int, top_n: int = 12, sort: str = "activation") -> List[Dict[str, Any]]:
        """Return top dataset examples firing feature cluster T_m across all its member features.

        ``sort="activation"`` ranks by summed C_max+R_max activation mass (presence);
        ``sort="disparity"`` ranks by |u| per-example preference disparity against T_m.
        """
        mats = self._load_feature_matrices()
        examples = self._load_examples()
        m_int = int(m)
        feats = self.feature_clusters.get(m_int, [])
        if mats is None or examples is None or not feats:
            return []

        if sort == "disparity":
            fc = self._load_fc_result()
            if fc is not None and fc.u_matrix is not None:
                cluster_ids = sorted(int(c) for c in self.feature_clusters.keys())
                if m_int in cluster_ids:
                    u_col = fc.u_matrix[:, cluster_ids.index(m_int)]
                    scores = np.abs(np.asarray(u_col, dtype=np.float32))
                    return self._top_examples(scores, examples, top_n)
                logger.warning(f"T_{m_int} not found in u_matrix columns; falling back to activation order.")
            else:
                logger.warning("Disparity ranking unavailable (feature_conditioned.npz missing); falling back to activation order.")

        self._ensure_member_cache()
        if self._all_member_cols is not None:
            feats_arr = np.asarray(feats, dtype=np.int64)
            slots = np.searchsorted(self._all_member_cols, feats_arr)
            valid_mask = (slots < len(self._all_member_cols)) & (self._all_member_cols[slots] == feats_arr)
            slots = slots[valid_mask]
            if len(slots) > 0:
                scores = np.zeros(len(examples), dtype=np.float32)
                for attr in ("C_max", "R_max"):
                    M = self._member_matrix(mats, attr)
                    if M is not None:
                        scores += M[:, slots].sum(axis=1)
                return self._top_examples(scores, examples, top_n)
        return []

    def _feature_cluster_info(self, m: int, top_n_examples: int = 12, sort: str = "activation") -> Dict[str, Any]:
        """Whole-cluster interpretation for T_m: LLM label, top features, and representative examples."""
        m_int = int(m)
        sort_mode = "disparity" if sort == "disparity" else "activation"
        cache_key = f"T_{m_int}_{top_n_examples}_{sort_mode}"

        def build() -> Dict[str, Any]:
            feats = self.feature_clusters.get(m_int, [])
            lbl = self._load_feature_cluster_labels().get(m_int) or {
                "title": f"Feature cluster T_{m}", "description": "", "keywords": [],
            }
            val_metrics = self._get_cluster_validation_metrics().get("clusters", {}).get(m_int)
            # Disparity view: prefer the Pass 2b contrastive label when present
            use_disp = sort_mode == "disparity" and lbl.get("disparity_title")
            return {
                "cluster_m": m_int,
                "label": lbl,
                "title": lbl.get("disparity_title") if use_disp else lbl.get("title"),
                "description": lbl.get("disparity_description") if use_disp else lbl.get("description"),
                "keywords": (lbl.get("disparity_keywords") if use_disp else lbl.get("keywords")) or [],
                "n_features": len(feats),
                "validation": val_metrics,
                "top_features": self._top_cluster_features(m_int, top_n=8),
                "examples": self._cluster_top_examples(m_int, top_n=top_n_examples, sort=sort_mode),
                "sort": sort_mode,
            }

        res = self._cached_info(cache_key, build)
        self._prewarm_neuronpedia_features([tf["feature_index"] for tf in res["top_features"]])
        return res

    def _data_cluster_info(self, k: int, top_n_examples: int = 10) -> Dict[str, Any]:
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

            parent_m = self.feature_to_cluster_map.get(f_int)
            if parent_m is not None:
                lab = self._load_feature_cluster_labels().get(parent_m, {})
                c_title = lab.get("title", f"Feature Cluster T_{parent_m}")
                c_kw = lab.get("keywords", [])
                out["parent_cluster"] = {
                    "m": parent_m,
                    "title": c_title,
                    "keywords": c_kw,
                }
                if "neuronpedia" not in out:
                    out["local_interpretation"] = {
                        "label": f"{c_title} Latent #{f_int}",
                        "description": f"Constituent SAE latent of Feature Community T_{parent_m} ({c_title}).",
                        "keywords": c_kw,
                    }
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

    def _score_prompt_conditioned_hypotheses(self, prompt_text: str, p_feat: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Extract top Prompt-Conditioned hypotheses (A_k x R_m) via exact SAE feature-space co-activation."""
        pc_clusters = self._load_pc_prompt_clusters()
        if not pc_clusters:
            logger.warning("prompt_conditioned.npz unavailable; A_k scoring falls back to summary-order prompt clusters.")
        return inspection.score_prompt_conditioned(
            p_feat, pc_clusters or {}, self.prompt_hypotheses_map,
            self._pc_cluster_tokens, top_k=top_k,
        )

    def _sae_feature_item(self, i: int, val: float, m: Optional[int]) -> Dict[str, Any]:
        """Format an active SAE feature item with its cluster assignment and Neuronpedia link."""
        return inspection.sae_feature_item(i, val, m, self._neuronpedia_url)

    def _top_sae_features(self, activations: np.ndarray, top_n: int = 8) -> List[Dict[str, Any]]:
        """Extract top active SAE features sorted by magnitude."""
        return inspection.top_sae_features(
            activations, self.feature_to_cluster_map, self.feature_clusters,
            self.min_partition_cluster_size, top_n, self._neuronpedia_url,
        )

    def _inspect_feature_samples(self, m: int, k: int, side: str = "amplify") -> Dict[str, Any]:
        """Inverse search (Tab 4 single-cluster): rank samples by per-example disparity u against T_m."""
        cluster_ids = sorted(int(c) for c in self.feature_clusters.keys())
        lbl = dict(self._load_feature_cluster_labels().get(int(m)) or {})
        if lbl.get("disparity_title"):
            lbl["title"] = lbl["disparity_title"]
            if lbl.get("disparity_description"):
                lbl["description"] = lbl["disparity_description"]
        return inspection.rank_cluster_samples(
            m=int(m), side=side, top_n=k,
            fc=self._load_fc_result(),
            examples=self._load_examples(),
            mats=self._load_feature_matrices(),
            feature_clusters=self.feature_clusters,
            cluster_ids=cluster_ids,
            example_view_fn=self._example_view,
            neuronpedia_url_fn=self._neuronpedia_url,
            label=lbl or None,
        )

    def _inspect_compound_samples(self, conditions: List[Tuple[int, str, float]], k: int) -> Dict[str, Any]:
        """Inverse search (Tab 4 compound): rank samples satisfying ALL directional conditions."""
        cluster_ids = sorted(int(c) for c in self.feature_clusters.keys())
        raw_labels = self._load_feature_cluster_labels() or {}
        disp_labels = {
            mid: ({**lbl_entry, "title": lbl_entry["disparity_title"]} if isinstance(lbl_entry, dict) and lbl_entry.get("disparity_title") else lbl_entry)
            for mid, lbl_entry in raw_labels.items()
        }
        return inspection.rank_compound_samples(
            conditions=conditions, top_n=k,
            fc=self._load_fc_result(),
            examples=self._load_examples(),
            mats=self._load_feature_matrices(),
            feature_clusters=self.feature_clusters,
            cluster_ids=cluster_ids,
            example_view_fn=self._example_view,
            neuronpedia_url_fn=self._neuronpedia_url,
            feature_cluster_labels=disp_labels,
        )

    # ==========================================================================
    # SECTION 7: Neuronpedia Integration (delegates to pdd.neuronpedia.NeuronpediaClient)
    # ==========================================================================

    def _summary_np_cfg(self) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """(sae_cfg, model_cfg) blocks from run metadata, handed to the client per call."""
        cfg = self.summary.get("config", {})
        return cfg.get("sae", {}), cfg.get("model")

    def _neuronpedia_set(self) -> Optional[Tuple[str, str]]:
        """Verified (model_slug, sae_slug); first call spawns one background verifier."""
        sae_cfg, model_cfg = self._summary_np_cfg()
        return self.np_client.resolved_set(sae_cfg, model_cfg)

    def _neuronpedia_url(self, feature_index: int) -> Optional[str]:
        """Browser URL for the feature card on neuronpedia.org (None when disabled)."""
        return self.np_client.url(self._neuronpedia_set(), feature_index)

    def _get_neuronpedia_feature(self, feature_index: int, allow_network: bool = True) -> Optional[Dict[str, Any]]:
        """Feature card via RAM LRU cache -> disk cache -> optional network fetch."""
        sae_cfg, model_cfg = self._summary_np_cfg()
        return self.np_client.get_feature(
            feature_index, allow_network=allow_network, sae_cfg=sae_cfg, model_cfg=model_cfg
        )

    def _prewarm_neuronpedia_features(self, feature_indices: List[int]) -> None:
        """Fire-and-forget background prewarm through the client's shared pool."""
        sae_cfg, model_cfg = self._summary_np_cfg()
        self.np_client.prewarm(feature_indices, sae_cfg=sae_cfg, model_cfg=model_cfg)


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
    validation_metrics = state._load_validation_metrics()

    # Ensure full hypothesis lists are loaded into memory
    state.feature_hypotheses_map
    state.prompt_hypotheses_map

    return {
        "summary": state.summary,
        "validation_metrics": validation_metrics,
        "tau": state._feature_delta_tau(),
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


@app.get("/api/cluster_validation")
def get_cluster_validation(
    m: Optional[int] = Query(None, description="Feature cluster ID T_m (optional)")
) -> Dict[str, Any]:
    """Retrieve aggregated predicted vs observed post-DPO deltas across feature clusters."""
    state = get_state()
    metrics = state._get_cluster_validation_metrics()
    if m is not None:
        return {
            "cluster_m": m,
            "validation": metrics.get("clusters", {}).get(int(m)),
            "global_r2": metrics.get("r2", 0.0),
            "global_pearson_r": metrics.get("pearson_r", 0.0),
        }
    return metrics


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
        parsed = inspection.parse_conditions(conditions, default_thresh=0.1)
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
    cluster_type: str = Query(..., alias="type", description="'data' (B_k), 'feature' (T_m), 'prompt' (A_k), or 'response' (R_m)"),
    cluster_id: int = Query(..., alias="id", description="Cluster ID integer"),
    top_n: int = Query(5, description="Number of top examples to return"),
    sort: str = Query("activation", description="T_m example ranking: 'activation' (C+R mass) or 'disparity' (|u|)")
) -> Dict[str, Any]:
    """Unified polymorphic lookup endpoint for all 4 cluster families (B_k, T_m, A_k, R_m)."""
    state = get_state()
    family_key = cluster_type.lower().strip()
    if family_key in ("data", "b", "bk"):
        return {"cluster_family": "B", "cluster_type": "data", "id": cluster_id, **state._data_cluster_info(cluster_id, top_n_examples=top_n)}
    elif family_key in ("feature", "t", "tm"):
        return {"cluster_family": "T", "cluster_type": "feature", "id": cluster_id, **state._feature_cluster_info(cluster_id, top_n_examples=top_n, sort=sort)}
    elif family_key in ("prompt", "a", "ak", "response", "r", "rm"):
        is_prompt = family_key in ("prompt", "a", "ak")
        pc_type = "prompt" if is_prompt else "response"
        return {
            "cluster_family": "A" if is_prompt else "R",
            "cluster_type": pc_type,
            "id": cluster_id,
            "tokens": state._pc_cluster_tokens(pc_type, cluster_id),
            "examples": state._pc_cluster_top_examples(pc_type, cluster_id, top_n=top_n),
        }
    raise HTTPException(400, f"Unsupported cluster type: '{cluster_type}'. Allowed: data (B), feature (T), prompt (A), response (R).")


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
    with state._inspector_lock:
        inspector = state.get_inspector()
        p_feat = inspector.extract_prompt_features(prompt_text)

    # 2. Per-feature-cluster activity of the live prompt (T_m <- sum of P(x) members)
    act = inspection.cluster_signals(p_feat, state.feature_clusters, mode="sum")

    data_labels_map = {int(label_entry.get("cluster_id")): label_entry for label_entry in state._load_data_cluster_labels() if "cluster_id" in label_entry}
    scored_clusters = inspection.score_data_clusters(
        act, p_feat, state.feature_hypotheses_map, data_labels_map, state.feature_clusters
    )[:req.top_k]
    matched_clusters = inspection.project_clusters(scored_clusters)

    # 3. Extract Feature-Conditioned Predicted Shifts (B_k x T_m)
    predicted_shifts = inspection.predicted_behavior_shifts(
        scored_clusters, act, state._load_feature_cluster_labels(), state._load_data_cluster_labels()
    )

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
    with state._inspector_lock:
        inspector = state.get_inspector()
        c_p, r_p, u = inspector.extract_pair_features(prompt_text, chosen_text, rejected_text, tau=state._feature_delta_tau())

    # 2. Per-feature-cluster live disparity: u_m = sum over T_m members of u
    u_sig = inspection.cluster_signals(u, state.feature_clusters, mode="sum")
    pair_act = c_p + r_p

    # 3. Score Data Clusters B_k by live-evidence-weighted hypotheses
    data_labels_map = {int(label_entry.get("cluster_id")): label_entry for label_entry in state._load_data_cluster_labels() if "cluster_id" in label_entry}
    scored_clusters = inspection.score_data_clusters(
        u_sig, pair_act, state.feature_hypotheses_map, data_labels_map, state.feature_clusters
    )[:req.top_k]
    matched_clusters = inspection.project_clusters(scored_clusters)

    # 4. Extract Promoted vs. Suppressed Concepts from the LIVE disparity
    promoted_concepts, suppressed_concepts = inspection.pair_concepts(
        u_sig, state._best_hypothesis_by_m(), state.feature_clusters, state.min_feat_cluster_size,
        state._load_feature_cluster_labels(), state._load_data_cluster_labels(),
    )

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

    # Pre-warm all lazily-loaded singletons in background thread
    _STATE.prefetch_background()

    logger.info(f"Starting PDD Viewer for '{args.run_dir}' at http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
