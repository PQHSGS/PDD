"""Neuronpedia web-metadata client (extracted from viewer_server SECTION 7).

Self-contained subsystem: slug resolution + background verification, feature-card
fetching/normalization, an LRU-capped RAM card cache, a per-run disk cache under
``<run>/viewer_cache/neuronpedia/``, and ONE persistent ThreadPoolExecutor shared
by all prewarm jobs (previously every request spawned its own thread + pool).

The client never touches run artifacts; it only reads the summary config block it
is handed and writes into its own cache directory.
"""
from __future__ import annotations

import collections
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .logger import get_logger

logger = get_logger("PDD.Neuronpedia")

# RAM bound for the in-memory card cache: d_sae can be 32k-65k, but UI sessions
# touch at most a few hundred features; LRU keeps hot cards while bounding RAM.
CARD_CACHE_MAX = 2048
PREWARM_BATCH = 16

_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()


def _shared_executor() -> ThreadPoolExecutor:
    """One process-wide pool for all Neuronpedia prewarm work (no per-request churn)."""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="PDD-NP")
        return _EXECUTOR


def _read_json_file(path: Path) -> Optional[Any]:
    """Read JSON, returning None on any failure (caller decides fallback)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error reading {path.name}: {e}")
        return None


def _write_json_file(path: Path, data: Any) -> None:
    """Atomic JSON write (tmp + replace) so a crash never corrupts the disk cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        Path.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


class NeuronpediaClient:
    """Slug resolution, verification, card fetching, and two-level caching."""

    def __init__(self, cache_dir: Path):
        self._cache_dir = cache_dir
        self._np_set: Optional[Tuple[str, str]] = None
        self._verifying = False
        self._verify_lock = threading.Lock()
        self._cards: "collections.OrderedDict[int, Dict[str, Any]]" = collections.OrderedDict()
        self._cards_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Slug mapping & verification
    # ------------------------------------------------------------------

    @staticmethod
    def sae_set_from_cfg(sae_cfg: Dict[str, Any], model_cfg: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, str]]:
        """Map run SAE configuration to Neuronpedia (model_slug, sae_slug)."""
        repo = str(sae_cfg.get("repo", "")).lower()
        sae_id = str(sae_cfg.get("sae_id", "")).lower()
        model_path = str((model_cfg or {}).get("path", "")).lower()
        layer = sae_cfg.get("layer", 14)
        k = sae_cfg.get("k", 80)

        # 1. Qwen3-1.7B (e.g. 14-resid-batchtopk-65k__l0-80)
        if "qwen3" in repo or "qwen3" in model_path or "qwen_qwen3" in sae_id:
            return "qwen3-1.7b", f"{layer}-resid-batchtopk-65k__l0-{k}"

        # 2. Gemma-2-2B / Gemma-Scope
        if "gemma-scope" in repo or "gemma-2-2b" in repo or "canonical" in repo or "gemma" in model_path:
            sae_set = f"{layer}-gemmascope-mlp-16k" if "mlp" in repo else f"{layer}-gemmascope-res-16k"
            return "gemma-2-2b", sae_set

        return None

    @staticmethod
    def slug_verified(model_id: str, sae_set: str) -> bool:
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

    def resolved_set(self, sae_cfg: Dict[str, Any], model_cfg: Optional[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
        """Return the verified (model_slug, sae_slug) pair.

        The first caller after boot spawns one background verifier; later calls get
        the unverified candidate pair immediately (URLs degrade gracefully) and the
        set is upgraded once verification succeeds. A failed verification resets the
        flag so a later call may retry.
        """
        if self._np_set is not None:
            return self._np_set
        pair = self.sae_set_from_cfg(sae_cfg, model_cfg)
        if not pair:
            return None
        spawn = False
        with self._verify_lock:
            if self._np_set is None and not self._verifying:
                self._verifying = True
                spawn = True
        if spawn:
            threading.Thread(target=self._verify_worker, args=(pair,), daemon=True, name="PDD-NPVerify").start()
        return pair

    def _verify_worker(self, pair: Tuple[str, str]) -> None:
        try:
            if self.slug_verified(pair[0], pair[1]):
                self._np_set = pair
                logger.info(f"Neuronpedia verified for {pair[0]}/{pair[1]}.")
            else:
                logger.warning(
                    f"Neuronpedia slug verification failed for {pair[0]}/{pair[1]}; "
                    "will retry on the next request."
                )
        except Exception as e:
            logger.warning(f"Neuronpedia slug verification crashed ({e}).")
        finally:
            with self._verify_lock:
                self._verifying = False

    # ------------------------------------------------------------------
    # Card cache (RAM LRU + disk)
    # ------------------------------------------------------------------

    def cache_dir(self) -> Path:
        """Return `<run>/viewer_cache/neuronpedia/`, creating it on demand."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir

    def _remember_card(self, f_idx: int, data: Dict[str, Any]) -> None:
        with self._cards_lock:
            self._cards[f_idx] = data
            self._cards.move_to_end(f_idx)
            while len(self._cards) > CARD_CACHE_MAX:
                self._cards.popitem(last=False)

    def _peek_card(self, f_idx: int) -> Optional[Dict[str, Any]]:
        with self._cards_lock:
            if f_idx in self._cards:
                self._cards.move_to_end(f_idx)
                return self._cards[f_idx]
        return None

    def get_feature(self, feature_index: int, allow_network: bool = True,
                    sae_cfg: Optional[Dict[str, Any]] = None,
                    model_cfg: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch feature metadata with RAM fast-path, disk cache, then network."""
        f_idx = int(feature_index)

        hit = self._peek_card(f_idx)
        if hit is not None:
            return hit

        np_set = self.resolved_set(sae_cfg or {}, model_cfg or {})
        if not np_set:
            return None

        cache_file = self.cache_dir() / f"feat_{f_idx}.json"
        if cache_file.exists():
            data = _read_json_file(cache_file)
            if data and (data.get("description") or data.get("label")):
                self._remember_card(f_idx, data)
                return data

        if not allow_network:
            return None

        data = self.fetch_feature(np_set[0], np_set[1], f_idx)
        if data is not None:
            self._remember_card(f_idx, data)
            try:
                _write_json_file(cache_file, data)
            except Exception as e:
                logger.debug(f"Failed to persist Neuronpedia cache for feature {f_idx}: {e}")
        return data

    # ------------------------------------------------------------------
    # URL helper & background prewarm
    # ------------------------------------------------------------------

    def url(self, np_set: Optional[Tuple[str, str]], feature_index: int) -> Optional[str]:
        """Browser URL for the feature card on neuronpedia.org (None when disabled)."""
        if not np_set:
            return None
        return f"https://www.neuronpedia.org/{np_set[0]}/{np_set[1]}/{int(feature_index)}"

    def prewarm(self, feature_indices: List[int],
                sae_cfg: Optional[Dict[str, Any]] = None,
                model_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Submit uncached feature fetches to the shared pool (fire-and-forget)."""
        np_set = self.resolved_set(sae_cfg or {}, model_cfg or {})
        if not np_set or not feature_indices:
            return
        uncached = [int(f) for f in feature_indices[:PREWARM_BATCH]
                    if int(f) not in self._cards]
        if not uncached:
            return
        executor = _shared_executor()

        def _job(f_idx: int) -> None:
            try:
                self.get_feature(f_idx, allow_network=True, sae_cfg=sae_cfg, model_cfg=model_cfg)
            except Exception as e:
                logger.warning(f"Neuronpedia prewarm fetch failed for feature {f_idx}: {e}")

        for f in uncached:
            executor.submit(_job, f)

    # ------------------------------------------------------------------
    # Raw HTTP fetch & response normalization
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_feature(model_id: str, sae_set: str, feature_index: int) -> Optional[Dict[str, Any]]:
        """Fetch JSON feature card directly from Neuronpedia over HTTPS."""
        import urllib.request
        url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_set}/{feature_index}"
        req = urllib.request.Request(url, headers={"User-Agent": "PDD-Viewer/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    exs = data.get("activations", [])[:3]
                    explanations = data.get("explanations", [])
                    desc = (explanations[0].get("description") if explanations else "") or data.get("description", "")
                    label = (explanations[0].get("description") if explanations else "") or data.get("label", "") or desc
                    model_name = explanations[0].get("explanationModelName", "") if explanations else ""

                    pos_tokens = data.get("pos_str", [])
                    if not pos_tokens and data.get("top_tokens"):
                        pos_tokens = [t.get("token") for t in data.get("top_tokens", []) if t.get("token")]
                    neg_tokens = data.get("neg_str", [])

                    return {
                        "model": model_id,
                        "sae": sae_set,
                        "feature_index": feature_index,
                        "description": desc,
                        "label": label,
                        "explanation_model": model_name,
                        "pos_tokens": [{"token": str(tok)} for tok in pos_tokens[:8]],
                        "top_tokens": [str(tok) for tok in pos_tokens[:6]],
                        "neg_tokens": [{"token": str(tok)} for tok in neg_tokens[:6]],
                        "max_act_approx": data.get("maxActApprox"),
                        "correlated_features": data.get("correlated_features_indices", [])[:10],
                        "examples_count": len(data.get("activations", [])),
                        "top_examples": [
                            {"maxValue": ex.get("maxValue"), "tokens": ex.get("tokens", [])[:30]} for ex in exs
                        ],
                    }
        except Exception as e:
            logger.debug(f"Neuronpedia API fetch failed for feature {feature_index} ({e}).")
        return None
