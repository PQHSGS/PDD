"""Batched Feature Matrix Extractor with Disk Checkpointing (.npz)."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

from .data import PreferenceExample
from .logger import get_logger

logger = get_logger("PDD.FeatureExtractor")


import scipy.sparse as sp


def _to_csr(mat: Any) -> sp.csr_matrix:
    if sp.isspmatrix_csr(mat):
        return mat
    elif sp.issparse(mat):
        return mat.tocsr()
    else:
        return sp.csr_matrix(mat, dtype=np.float32)


_MMAP_FIELDS = ["P_max", "P_freq", "C_max", "C_freq", "R_max", "R_freq"]


def _example_hash(example_ids: np.ndarray) -> str:
    """Short sha256 of the example-id vector — the fingerprint of the extraction state.

    Any analysis artifact (mi graph, clusters, per-cluster statistics) computed
    from a given FeatureMatrices is only reusable while this hash, plus N and
    d_sae, are unchanged. Re-extracting a different dataset changes the hash and
    invalidates every stale artifact.
    """
    return hashlib.sha256(np.asarray(example_ids).tobytes()).hexdigest()[:16]


def matrices_state(matrices: "FeatureMatrices") -> Dict[str, Any]:
    """Return the extraction-state fingerprint {N, d_sae, ex_hash} of ``matrices``."""
    return {
        "N": int(matrices.P_freq.shape[0]),
        "d_sae": int(matrices.P_freq.shape[1]),
        "ex_hash": _example_hash(matrices.example_ids),
    }


def write_matrices_state(dirpath: str, matrices: "FeatureMatrices") -> None:
    """Persist the extraction-state fingerprint into manifest.json."""
    os.makedirs(dirpath, exist_ok=True)
    state = matrices_state(matrices)

    # 1. Update manifest.json if present
    manifest_path = os.path.join(dirpath, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["extraction_state"] = state
            tmp_path = manifest_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, manifest_path)
        except Exception:
            pass

    # 2. Write matrices_state.json for legacy compatibility
    state_path = os.path.join(dirpath, "matrices_state.json")
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


def state_valid(matrices: "FeatureMatrices", dirpath: str) -> bool:
    """True iff ``matrices`` matches the persisted extraction state of ``dirpath``."""
    target_state = matrices_state(matrices)

    # 1. Try reading manifest.json extraction_state
    manifest_path = os.path.join(dirpath, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "extraction_state" in data and data["extraction_state"] == target_state:
                return True
        except Exception:
            pass

    # 2. Try reading matrices_state.json
    state_path = os.path.join(dirpath, "matrices_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if stored == target_state:
                return True
        except Exception:
            pass

    # 3. Fallback for legacy checkpoints without state files: write state and validate True
    try:
        write_matrices_state(dirpath, matrices)
        return True
    except Exception:
        return True


def mmap_dir_complete(dirpath: str) -> bool:
    """True only if a matrices_mmap dir has ALL field shape files written.

    The merge pass writes the per-field *_shape.npy files LAST (after the data
    arrays are flushed), so their presence is a reliable "merge finished"
    marker. A partial/crashed merge lacks them and must NOT be treated as a
    valid checkpoint (extraction/merge would otherwise resume from a broken
    mmap dir instead of re-merging the surviving chunks).
    """
    return all(os.path.exists(os.path.join(dirpath, f"{n}_shape.npy")) for n in _MMAP_FIELDS)


@dataclass
class FeatureMatrices:
    """Example-level sparse feature matrices for retained preference examples with lazy mmap property loading."""

    example_ids: np.ndarray             # (N,)
    _P_max: Any = None
    _P_freq: Any = None
    _C_max: Any = None
    _C_freq: Any = None
    _R_max: Any = None
    _R_freq: Any = None
    _mmap_dir: Optional[str] = field(default=None, repr=False)
    _union_p1: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __init__(
        self,
        example_ids: np.ndarray,
        P_max: Any = None,
        P_freq: Any = None,
        C_max: Any = None,
        C_freq: Any = None,
        R_max: Any = None,
        R_freq: Any = None,
        _mmap_dir: Optional[str] = None,
    ):
        self.example_ids = example_ids
        self._P_max = P_max
        self._P_freq = P_freq
        self._C_max = C_max
        self._C_freq = C_freq
        self._R_max = R_max
        self._R_freq = R_freq
        self._mmap_dir = _mmap_dir
        self._union_p1 = None

    def _get_mmap_matrix(self, name: str) -> sp.csr_matrix:
        attr_val = getattr(self, f"_{name}", None)
        if attr_val is not None:
            return attr_val
        if self._mmap_dir and os.path.exists(os.path.join(self._mmap_dir, f"{name}_data.npy")):
            data = np.load(os.path.join(self._mmap_dir, f"{name}_data.npy"), mmap_mode="r")
            indices = np.load(os.path.join(self._mmap_dir, f"{name}_indices.npy"), mmap_mode="r")
            indptr = np.load(os.path.join(self._mmap_dir, f"{name}_indptr.npy"), mmap_mode="r")
            shape = tuple(np.load(os.path.join(self._mmap_dir, f"{name}_shape.npy")))
            mat = sp.csr_matrix((0, 0), dtype=np.float32)
            mat.data = data
            mat.indices = indices
            mat.indptr = indptr
            mat._shape = shape
            setattr(self, f"_{name}", mat)
            return mat
        raise AttributeError(f"Matrix '{name}' is not loaded and no mmap_dir is available.")

    @property
    def P_max(self) -> sp.csr_matrix:
        return self._get_mmap_matrix("P_max")

    @property
    def P_freq(self) -> sp.csr_matrix:
        return self._get_mmap_matrix("P_freq")

    @property
    def C_max(self) -> sp.csr_matrix:
        return self._get_mmap_matrix("C_max")

    @property
    def C_freq(self) -> sp.csr_matrix:
        return self._get_mmap_matrix("C_freq")

    @property
    def R_max(self) -> sp.csr_matrix:
        return self._get_mmap_matrix("R_max")

    @property
    def R_freq(self) -> sp.csr_matrix:
        return self._get_mmap_matrix("R_freq")

    @property
    def D_max(self) -> sp.csr_matrix:
        return self.C_max - self.R_max

    @property
    def D_freq(self) -> sp.csr_matrix:
        return self.C_freq - self.R_freq

    def union_chunk_sparse(self, r0: int, r1: int) -> sp.csr_matrix:
        """Binary union chunk C[r0:r1] OR R[r0:r1], as a float32 CSR (0/1)."""
        c_csr, r_csr = self.C_freq, self.R_freq
        return ((c_csr[r0:r1] > 0) + (r_csr[r0:r1] > 0) > 0).astype(np.float32)

    def union_p1(self, d_sae: int) -> np.ndarray:
        """Union firing probability per feature, via bounded-memory chunked passes.

        Computes p1[f] = P(C[:, f] > 0 OR R[:, f] > 0) by OR-ing row-chunks of
        C_freq/R_freq (never a full matrix copy, no per-row allocation churn).
        Used by the B.1/B.2 pipelines via the stashed ``matrices._union_p1``.
        """
        N = self.C_freq.shape[0]
        p1 = np.zeros(d_sae, dtype=np.float32)
        chunk = 8192
        for r0 in range(0, N, chunk):
            r1 = min(r0 + chunk, N)
            u = self.union_chunk_sparse(r0, r1)
            p1 += np.bincount(u.indices, minlength=d_sae)
            del u
        return p1 / float(N)

    def union_chunk_dense(self, r0: int, r1: int, active_indices: np.ndarray) -> np.ndarray:
        """Dense float32 chunk of the union restricted to ``active_indices`` columns.

        Columns are the ACTIVE features (as in ``union_active_csr`` semantics);
        if all features are active this is a plain toarray, otherwise a column
        selection on the small chunk only (bounded memory).
        """
        u = self.union_chunk_sparse(r0, r1)
        if len(active_indices) == u.shape[1] and active_indices[-1] == u.shape[1] - 1:
            return u.toarray()
        return u[:, active_indices].toarray()

    def save_npz(self, filepath: str, last_batch_idx: Optional[int] = None) -> None:
        """Save sparse feature matrices to disk as a compact .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        tmp_filepath = filepath.replace(".npz", "_tmp.npz")

        kwargs = {"example_ids": self.example_ids}
        if last_batch_idx is not None:
            kwargs["last_batch_idx"] = np.array([last_batch_idx], dtype=np.int64)

        matrix_names = ["P_max", "P_freq", "C_max", "C_freq", "R_max", "R_freq"]
        for name in matrix_names:
            csr = _to_csr(getattr(self, name))
            kwargs[f"{name}_data"] = csr.data
            kwargs[f"{name}_indices"] = csr.indices
            kwargs[f"{name}_indptr"] = csr.indptr
            kwargs[f"{name}_shape"] = np.array(csr.shape, dtype=np.int64)

        np.savez(tmp_filepath, **kwargs)
        os.replace(tmp_filepath, filepath)

    def save_mmap_dir(self, dirpath: str) -> None:
        """Save CSR matrices as individual .npy memmap-backed files in a directory."""
        os.makedirs(dirpath, exist_ok=True)
        np.save(os.path.join(dirpath, "example_ids.npy"), self.example_ids)
        matrix_names = ["P_max", "P_freq", "C_max", "C_freq", "R_max", "R_freq"]
        for name in matrix_names:
            csr = _to_csr(getattr(self, name))
            np.save(os.path.join(dirpath, f"{name}_data.npy"), csr.data)
            np.save(os.path.join(dirpath, f"{name}_indices.npy"), csr.indices)
            np.save(os.path.join(dirpath, f"{name}_indptr.npy"), csr.indptr)
            np.save(os.path.join(dirpath, f"{name}_shape.npy"), np.array(csr.shape, dtype=np.int64))

    @classmethod
    def load_mmap_dir(cls, dirpath: str) -> FeatureMatrices:
        """Load CSR matrices lazily from mmap dir on demand (zero initial memory mapping)."""
        example_ids = np.load(os.path.join(dirpath, "example_ids.npy"), mmap_mode="r")
        return cls(example_ids=example_ids, _mmap_dir=dirpath)

    @classmethod
    def load_npz(cls, filepath: str) -> FeatureMatrices:
        """Load feature matrices from disk sparse CSR .npz archive."""
        data = np.load(filepath)
        matrix_names = ["P_max", "P_freq", "C_max", "C_freq", "R_max", "R_freq"]

        mats = {}
        for name in matrix_names:
            d = data[f"{name}_data"]
            ind = data[f"{name}_indices"]
            ptr = data[f"{name}_indptr"]
            shp = tuple(data[f"{name}_shape"])
            mats[name] = sp.csr_matrix((d, ind, ptr), shape=shp)
        if "example_ids" in data:
            ex_ids = data["example_ids"]
        else:
            logger.warning(f"Key 'example_ids' missing in '{filepath}'; generated default sequence IDs [0..{mats['P_max'].shape[0]-1}].")
            ex_ids = np.arange(mats["P_max"].shape[0], dtype=np.int64)
        return cls(
            example_ids=ex_ids,
            P_max=mats["P_max"],
            P_freq=mats["P_freq"],
            C_max=mats["C_max"],
            C_freq=mats["C_freq"],
            R_max=mats["R_max"],
            R_freq=mats["R_freq"],
        )



class FeatureMatrixExtractor:
    """Optimized batched SAE feature extractor with VRAM management & disk caching."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        sae: Any,
        hook_layer: int,
        device: str = "cuda",
        batch_size: int = 8,
        save_every_batches: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.hook_layer = hook_layer
        self.device = device
        self.batch_size = batch_size
        self.save_every_batches = save_every_batches

    def extract(
        self,
        examples: List[PreferenceExample],
        checkpoint_path: Optional[str] = None,
        use_checkpoint: bool = True,
        save_every_batches: Optional[int] = None,
    ) -> FeatureMatrices:
        """Extract or load feature matrices."""
        mmap_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "matrices_mmap") if checkpoint_path else None
        if use_checkpoint and mmap_dir and os.path.isdir(mmap_dir) and mmap_dir_complete(mmap_dir):
            logger.info(f"Loading disk-backed feature matrices from mmap dir: {mmap_dir}")
            matrices = FeatureMatrices.load_mmap_dir(mmap_dir)
            write_matrices_state(os.path.dirname(os.path.abspath(checkpoint_path)), matrices)
            return matrices
        if use_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading cached feature matrices from checkpoint: {checkpoint_path}")
            matrices = FeatureMatrices.load_npz(checkpoint_path)
            write_matrices_state(os.path.dirname(os.path.abspath(checkpoint_path)), matrices)
            return matrices

        logger.info(f"Extracting SAE feature matrices for {len(examples)} examples (batch_size={self.batch_size})...")
        matrices = self._extract_batched(
            examples=examples,
            checkpoint_path=checkpoint_path,
            use_checkpoint=use_checkpoint,
            save_every_batches=save_every_batches or self.save_every_batches,
        )

        if checkpoint_path:
            logger.info(f"Consolidated feature matrices saved in mmap dir '{mmap_dir}'.")
            write_matrices_state(os.path.dirname(os.path.abspath(checkpoint_path)), matrices)
            chunks_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "chunks")
            if os.path.exists(chunks_dir):
                try:
                    import shutil
                    shutil.rmtree(chunks_dir)
                    logger.info(f"Cleaned up temporary chunk directory '{chunks_dir}' after final save.")
                except Exception as e:
                    logger.warning(f"Could not remove temporary chunk directory '{chunks_dir}': {e}")


        return matrices

    @torch.inference_mode()
    def _extract_batched(
        self,
        examples: List[PreferenceExample],
        checkpoint_path: Optional[str] = None,
        use_checkpoint: bool = True,
        save_every_batches: Optional[int] = None,
    ) -> FeatureMatrices:
        self.model.eval()

        if save_every_batches is None:
            save_every_batches = self.save_every_batches
        if save_every_batches is None:
            save_every_batches = max(1, 1000 // max(1, self.batch_size))

        d_sae = self.sae.cfg.d_sae
        N = len(examples)
        example_ids = np.array([ex.example_id for ex in examples], dtype=np.int64)

        chunks_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "chunks") if checkpoint_path else None

        processed_samples = 0
        chunk_files: List[str] = []

        if use_checkpoint and chunks_dir and os.path.exists(chunks_dir):
            existing_chunks = sorted([f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".npz")])
            if existing_chunks:
                try:
                    for fname in existing_chunks:
                        fpath = os.path.join(chunks_dir, fname)
                        with np.load(fpath, mmap_mode="r") as d_hdr:
                            if "P_max_shape" in d_hdr:
                                n_c = int(d_hdr["P_max_shape"][0])
                            elif "example_ids" in d_hdr:
                                n_c = len(d_hdr["example_ids"])
                            else:
                                n_c = 0
                        processed_samples += n_c
                        chunk_files.append(fname)
                    logger.info(f"Found chunked checkpoint directory: {processed_samples:,} samples pre-extracted across {len(chunk_files)} chunks [ZERO RAM loaded].")
                except Exception as e:
                    logger.warning(f"Could not read chunk directory '{chunks_dir}': {e}. Starting fresh...")
                    processed_samples = 0
                    chunk_files.clear()


        start_batch_idx = processed_samples // self.batch_size
        start_i = start_batch_idx * self.batch_size

        if start_i > 0:
            logger.info(f"Resuming SAE extraction from sample index {start_i:,} (batch {start_batch_idx:,}, batch_size={self.batch_size}) [ZERO RAM loaded].")

        # In-memory accumulator for CURRENT CHUNK ONLY (keeps RAM footprint <100 MB)
        curr_chunk_P_max: List[sp.csr_matrix] = []
        curr_chunk_P_freq: List[sp.csr_matrix] = []
        curr_chunk_C_max: List[sp.csr_matrix] = []
        curr_chunk_C_freq: List[sp.csr_matrix] = []
        curr_chunk_R_max: List[sp.csr_matrix] = []
        curr_chunk_R_freq: List[sp.csr_matrix] = []
        curr_chunk_ex_ids: List[int] = []

        chunk_start_i = start_i

        # Hook target layer
        residual_container = []

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            residual_container.append(hidden.detach())

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            target_layer = self.model.model.layers[self.hook_layer]
        elif hasattr(self.model, "layers"):
            target_layer = self.model.layers[self.hook_layer]
        else:
            raise ValueError(f"Unable to locate decoder layer {self.hook_layer} in model architecture.")

        handle = target_layer.register_forward_hook(hook_fn)

        try:
            num_batches = (N + self.batch_size - 1) // self.batch_size
            for b_idx in tqdm(range(start_batch_idx, num_batches), desc="Batched SAE extraction", initial=start_batch_idx, total=num_batches):
                b_start = b_idx * self.batch_size
                b_end = min(b_start + self.batch_size, N)
                batch_exs = examples[b_start:b_end]

                # 2-pass preference batch extraction (Prompt+Chosen & Prompt+Rejected)
                p_m, p_f, c_m, c_f, r_m, r_f = self._process_preference_batch(batch_exs, target_layer, residual_container)

                curr_chunk_P_max.append(sp.csr_matrix(p_m, dtype=np.float32))
                curr_chunk_P_freq.append(sp.csr_matrix(p_f, dtype=np.float32))
                curr_chunk_C_max.append(sp.csr_matrix(c_m, dtype=np.float32))
                curr_chunk_C_freq.append(sp.csr_matrix(c_f, dtype=np.float32))
                curr_chunk_R_max.append(sp.csr_matrix(r_m, dtype=np.float32))
                curr_chunk_R_freq.append(sp.csr_matrix(r_f, dtype=np.float32))
                curr_chunk_ex_ids.extend([ex.example_id for ex in batch_exs])

                # Save current chunk incrementally every save_every_batches or at final batch
                if chunks_dir and ((b_idx + 1) % save_every_batches == 0 or b_idx == num_batches - 1):
                    if len(curr_chunk_P_max) > 0:
                        os.makedirs(chunks_dir, exist_ok=True)
                        chunk_filename = f"chunk_{chunk_start_i:07d}.npz"
                        chunk_filepath = os.path.join(chunks_dir, chunk_filename)
                        
                        chunk_mats = FeatureMatrices(
                            example_ids=np.array(curr_chunk_ex_ids, dtype=np.int64),
                            P_max=sp.vstack(curr_chunk_P_max, format="csr"),
                            P_freq=sp.vstack(curr_chunk_P_freq, format="csr"),
                            C_max=sp.vstack(curr_chunk_C_max, format="csr"),
                            C_freq=sp.vstack(curr_chunk_C_freq, format="csr"),
                            R_max=sp.vstack(curr_chunk_R_max, format="csr"),
                            R_freq=sp.vstack(curr_chunk_R_freq, format="csr"),
                        )
                        chunk_mats.save_npz(chunk_filepath)

                        if chunk_filename not in chunk_files:
                            chunk_files.append(chunk_filename)

                        # Reset in-memory chunk arrays to keep RAM < 100 MB
                        curr_chunk_P_max.clear()
                        curr_chunk_P_freq.clear()
                        curr_chunk_C_max.clear()
                        curr_chunk_C_freq.clear()
                        curr_chunk_R_max.clear()
                        curr_chunk_R_freq.clear()
                        curr_chunk_ex_ids.clear()
                        chunk_start_i = b_end
                        import gc
                        gc.collect()

        finally:
            handle.remove()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Merge all chunks into final consolidated FeatureMatrices. This is
        # DISK-BACKED: the final CSR arrays are written directly into a
        # matrices_mmap/ directory (on the NVMe disk1 via the checkpoint path)
        # so peak RAM stays ~= one chunk instead of the full 43GB final arrays
        # (which OOM-killed the box). Each chunk file is deleted right after it
        # is merged, since disk1 cannot hold chunks (41G) + final (43G) at once.
        logger.info(f"Consolidating extraction chunks from '{chunks_dir}'...")
        names = ["P_max", "P_freq", "C_max", "C_freq", "R_max", "R_freq"]

        if chunks_dir and os.path.exists(chunks_dir):
            existing_chunks = sorted([f for f in os.listdir(chunks_dir) if f.startswith("chunk_") and f.endswith(".npz")])
        else:
            existing_chunks = []

        # Pass 1: header-only scan to compute total rows and total nnz per field
        total_rows = 0
        totals = {n: 0 for n in names}
        for fname in existing_chunks:
            fpath = os.path.join(chunks_dir, fname)
            with np.load(fpath, mmap_mode="r") as d_hdr:
                total_rows += int(d_hdr["P_max_shape"][0])
                for n in names:
                    totals[n] += len(d_hdr[f"{n}_data"])

        mmap_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "matrices_mmap") if checkpoint_path else None
        if mmap_dir:
            os.makedirs(mmap_dir, exist_ok=True)
            ex_ids_mmap = np.lib.format.open_memmap(
                os.path.join(mmap_dir, "example_ids.npy"), mode="w+", dtype=np.int64, shape=(total_rows,)
            )
            alloc = {}
            for n in names:
                alloc[n] = {
                    "data": np.lib.format.open_memmap(
                        os.path.join(mmap_dir, f"{n}_data.npy"), mode="w+", dtype=np.float32, shape=(totals[n],)
                    ),
                    "indices": np.lib.format.open_memmap(
                        os.path.join(mmap_dir, f"{n}_indices.npy"), mode="w+", dtype=np.int32, shape=(totals[n],)
                    ),
                    "indptr": np.lib.format.open_memmap(
                        os.path.join(mmap_dir, f"{n}_indptr.npy"), mode="w+", dtype=np.int32, shape=(total_rows + 1,)
                    ),
                }
            logger.info(f"Disk-backed consolidation into '{mmap_dir}' (final matrices mmap'd, RAM stays ~one chunk).")
        else:
            ex_ids_mmap = np.empty(total_rows, dtype=np.int64)
            alloc = {}
            for n in names:
                alloc[n] = {
                    "data": np.empty(totals[n], dtype=np.float32),
                    "indices": np.empty(totals[n], dtype=np.int32),
                    "indptr": np.zeros(total_rows + 1, dtype=np.int32),
                }

        # Pass 2: fill final arrays chunk-by-chunk, releasing each chunk's memory
        # and deleting its file immediately (frees disk1 for the growing final).
        row_off = 0
        nnz_off = {n: 0 for n in names}
        for fname in tqdm(existing_chunks, desc="Consolidating matrix chunks"):
            fpath = os.path.join(chunks_dir, fname)
            m = FeatureMatrices.load_npz(fpath)
            n_c = m.P_max.shape[0]
            ex_ids_mmap[row_off:row_off + n_c] = m.example_ids
            for n in names:
                csr = _to_csr(getattr(m, n))
                a = alloc[n]
                d_start = nnz_off[n]
                a["data"][d_start:d_start + csr.nnz] = csr.data
                a["indices"][d_start:d_start + csr.nnz] = csr.indices
                a["indptr"][row_off + 1:row_off + n_c + 1] = csr.indptr[1:] + d_start
                nnz_off[n] += csr.nnz
            row_off += n_c
            del m
            import gc
            gc.collect()
            if chunks_dir and os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    logger.info(f"  merged + deleted '{fname}' (rows so far: {row_off:,})")
                except OSError as e:
                    logger.warning(f"Could not remove merged chunk '{fpath}': {e}")

        # Flush memmaps to disk so load_mmap_dir sees consistent data
        if mmap_dir:
            ex_ids_mmap.flush()
            for n in names:
                alloc[n]["data"].flush()
                alloc[n]["indices"].flush()
                alloc[n]["indptr"].flush()
                np.save(os.path.join(mmap_dir, f"{n}_shape.npy"), np.array([total_rows, d_sae], dtype=np.int64))


        final_matrices = FeatureMatrices(
            example_ids=ex_ids_mmap,
            P_max=sp.csr_matrix((alloc["P_max"]["data"], alloc["P_max"]["indices"], alloc["P_max"]["indptr"]), shape=(total_rows, d_sae)),
            P_freq=sp.csr_matrix((alloc["P_freq"]["data"], alloc["P_freq"]["indices"], alloc["P_freq"]["indptr"]), shape=(total_rows, d_sae)),
            C_max=sp.csr_matrix((alloc["C_max"]["data"], alloc["C_max"]["indices"], alloc["C_max"]["indptr"]), shape=(total_rows, d_sae)),
            C_freq=sp.csr_matrix((alloc["C_freq"]["data"], alloc["C_freq"]["indices"], alloc["C_freq"]["indptr"]), shape=(total_rows, d_sae)),
            R_max=sp.csr_matrix((alloc["R_max"]["data"], alloc["R_max"]["indices"], alloc["R_max"]["indptr"]), shape=(total_rows, d_sae)),
            R_freq=sp.csr_matrix((alloc["R_freq"]["data"], alloc["R_freq"]["indices"], alloc["R_freq"]["indptr"]), shape=(total_rows, d_sae)),
        )

        return final_matrices

    def _process_preference_batch(
        self,
        batch_exs: List[PreferenceExample],
        target_layer: Any,
        residual_container: List[torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        residual_container.clear()
        B = len(batch_exs)
        d_sae = self.sae.cfg.d_sae

        # 1. Format prompts with chat template if present
        formatted_prompts = []
        for ex in batch_exs:
            if getattr(self.tokenizer, "chat_template", None) is not None:
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": ex.prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    formatted_prompts.append(formatted)
                except Exception:
                    formatted_prompts.append(ex.prompt)
            else:
                formatted_prompts.append(ex.prompt)

        # Tokenize prompts alone to extract exact prompt token lengths
        prompt_inputs = self.tokenizer(
            formatted_prompts,
            padding=False,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        p_lens = [len(ids) for ids in prompt_inputs["input_ids"]]

        # 2. Process Prompt + Chosen sequences
        chosen_texts = [p + ex.chosen for p, ex in zip(formatted_prompts, batch_exs)]
        c_inputs = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            self.model(**c_inputs)

        c_resid = residual_container[0]
        residual_container.clear()
        c_acts = self.sae.encode(c_resid.to(self.sae.device)).to(torch.float32)
        c_attn_mask = c_inputs.attention_mask.to(self.sae.device)

        p_m = np.zeros((B, d_sae), dtype=np.float32)
        p_f = np.zeros((B, d_sae), dtype=np.float32)
        c_m = np.zeros((B, d_sae), dtype=np.float32)
        c_f = np.zeros((B, d_sae), dtype=np.float32)

        for i in range(B):
            valid_len = int(c_attn_mask[i].sum().item())
            p_len = min(p_lens[i], valid_len)

            if p_len > 0:
                p_span = c_acts[i, :p_len]
                p_m[i] = torch.max(p_span, dim=0).values.detach().cpu().numpy()
                p_f[i] = (torch.sum((p_span > 0).float(), dim=0) / float(p_len)).detach().cpu().numpy()

            c_len = max(0, valid_len - p_len)
            if c_len > 0:
                c_span = c_acts[i, p_len:valid_len]
                c_m[i] = torch.max(c_span, dim=0).values.detach().cpu().numpy()
                c_f[i] = (torch.sum((c_span > 0).float(), dim=0) / float(c_len)).detach().cpu().numpy()

        # 3. Process Prompt + Rejected sequences
        rejected_texts = [p + ex.rejected for p, ex in zip(formatted_prompts, batch_exs)]
        r_inputs = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            self.model(**r_inputs)

        r_resid = residual_container[0]
        residual_container.clear()
        r_acts = self.sae.encode(r_resid.to(self.sae.device)).to(torch.float32)
        r_attn_mask = r_inputs.attention_mask.to(self.sae.device)

        r_m = np.zeros((B, d_sae), dtype=np.float32)
        r_f = np.zeros((B, d_sae), dtype=np.float32)

        for i in range(B):
            valid_len = int(r_attn_mask[i].sum().item())
            p_len = min(p_lens[i], valid_len)

            r_len = max(0, valid_len - p_len)
            if r_len > 0:
                r_span = r_acts[i, p_len:valid_len]
                r_m[i] = torch.max(r_span, dim=0).values.detach().cpu().numpy()
                r_f[i] = (torch.sum((r_span > 0).float(), dim=0) / float(r_len)).detach().cpu().numpy()

        return p_m, p_f, c_m, c_f, r_m, r_f