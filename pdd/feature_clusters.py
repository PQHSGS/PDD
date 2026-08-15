"""Leiden Feature Clusterer with disk checkpointing (.json)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import glob
import json
import os
import tempfile
import igraph as ig
import leidenalg as la
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any

from .logger import get_logger

logger = get_logger("PDD.FeatureClusterer")


@dataclass
class FeatureClusterMap:
    """Mapping of feature clusters T_m and feature assignments."""

    clusters: Dict[int, List[int]]         # m (1..K_r) -> list of SAE feature indices T_m
    feature_to_cluster: Dict[int, int]    # feature index g -> cluster_id m (0 if unassigned)

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)

    def save_json(self, filepath: str) -> None:
        """Save cluster mapping to disk as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "clusters": {str(k): v for k, v in self.clusters.items()},
            "feature_to_cluster": {str(k): v for k, v in self.feature_to_cluster.items()},
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> FeatureClusterMap:
        """Load cluster mapping from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        clusters = {int(k): v for k, v in data["clusters"].items()}
        feature_to_cluster = {int(k): v for k, v in data["feature_to_cluster"].items()}
        return cls(clusters=clusters, feature_to_cluster=feature_to_cluster)


class LeidenFeatureClusterer:
    """Normalized binary mutual-information graph builder & Leiden community detector."""

    def __init__(
        self,
        min_community_size: int,
        top_pct: float = 1.0,
        min_firing_freq: float = 1e-4,
        block_size: int = 2048,
        resolution_parameter: float = 1.5,
    ):
        self.min_community_size = min_community_size
        self.top_pct = top_pct
        self.min_firing_freq = min_firing_freq
        self.block_size = block_size
        self.resolution_parameter = resolution_parameter


    def cluster(
        self,
        matrices: Any,
        seed: int = 0,
        checkpoint_path: Optional[str] = None,
        use_checkpoint: bool = True,
    ) -> FeatureClusterMap:
        """Cluster SAE features using binary MI graph and Leiden algorithm.

        ``matrices`` is a FeatureMatrices; the binary activation union is streamed
        from the mmap'd C_freq/R_freq (two row passes, ~0 RAM) so neither the full
        ~15GB C/R pair nor the ~19GB union matrix is ever materialized in RAM.
        """
        ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path)) if checkpoint_path else None
        mi_graph_path = os.path.join(ckpt_dir, "mi_graph.npz") if ckpt_dir else None

        # Cached clusters/mi_graph are only valid while the extraction state
        # (N, d_sae, example-id hash) is unchanged; re-extracting different data
        # invalidates them and forces recomputation.
        from .feature_matrices import state_valid
        cached_ok = state_valid(matrices, ckpt_dir) if ckpt_dir else False

        if use_checkpoint and checkpoint_path and os.path.exists(checkpoint_path) and cached_ok:
            logger.info(f"Loading cached feature clusters from checkpoint: {checkpoint_path}")
            return FeatureClusterMap.load_json(checkpoint_path)
        if mi_graph_path and os.path.exists(mi_graph_path) and not cached_ok:
            logger.warning(f"Ignoring stale MI graph '{mi_graph_path}' (extraction state changed); recomputing.")
            os.remove(mi_graph_path)

        # Clean up stale memmaps left by previous crashed clustering runs. They are
        # scratch artifacts of dead processes (no longer referenced), and removing
        # them frees space on the tight root filesystem where /tmp lives.
        try:
            for stale in glob.glob(os.path.join(tempfile.gettempdir(), "pdd_normMI_*.npy")):
                try:
                    os.remove(stale)
                    logger.info(f"Removed stale clustering memmap '{stale}'")
                except OSError:
                    pass
        except OSError:
            pass

        cluster_map = self._build_clusters(matrices, seed, mi_graph_path=mi_graph_path)

        if checkpoint_path:
            logger.info(f"Saving feature cluster mapping to checkpoint: {checkpoint_path}")
            cluster_map.save_json(checkpoint_path)

        return cluster_map

    def _build_clusters(self, matrices: Any, seed: int, mi_graph_path: Optional[str] = None) -> FeatureClusterMap:
        """Cluster features using binary MI, computed memory-safely.

        Two streaming row passes over the mmap'd C_freq/R_freq build (a) the per-
        feature union firing probability and (b) the binary union matrix U over
        active features (~1.6GB). The MI co-occurrence is then computed with dense
        BLAS on 4096-row chunks (never the full 34GB dense U), and the norm-MI is
        streamed to a disk memmap. Peak RAM stays < ~5GB.
        """
        import gc

        from .feature_matrices import state_valid

        d_sae = matrices.C_freq.shape[1]
        cached_ok = state_valid(matrices, os.path.dirname(mi_graph_path)) if mi_graph_path else False

        if mi_graph_path and os.path.exists(mi_graph_path) and cached_ok:
            logger.info(f"Found cached MI graph at '{mi_graph_path}'. Skipping MI block computation!")
            data = np.load(mi_graph_path)
            global_i = data["global_i"]
            global_j = data["global_j"]
            edge_weights = data["weights"].tolist()
            edges = list(zip(global_i.tolist(), global_j.tolist()))
            g = ig.Graph(n=d_sae, edges=edges, edge_attrs={"weight": edge_weights})
            
            logger.info(f"Running Leiden community detection (RBConfiguration, res={self.resolution_parameter}) on graph with {len(edges):,} edges...")
            partition = la.find_partition(
                g,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=self.resolution_parameter,
                seed=seed,
            )

            clusters: Dict[int, List[int]] = {}
            feature_to_cluster: Dict[int, int] = {feat: 0 for feat in range(d_sae)}

            cluster_id_counter = 1
            for comm in partition:
                if len(comm) >= self.min_community_size:
                    clusters[cluster_id_counter] = list(comm)
                    for feat in comm:
                        feature_to_cluster[feat] = cluster_id_counter
                    cluster_id_counter += 1

            logger.info(f"Extracted {len(clusters)} retained Leiden feature communities.")
            return FeatureClusterMap(clusters=clusters, feature_to_cluster=feature_to_cluster)

        N, d_sae = matrices.C_freq.shape
        block = self.block_size

        # 1. Union firing probabilities via a single streaming row pass (peak RAM ~0).
        p1 = matrices.union_p1(d_sae)

        # Stash the union firing probabilities for reuse by the B.2 pipeline's
        # response-feature retention filter (r_counts = p1 * N), avoiding a
        # second full union computation. Persisted so separate runs at the same
        # extraction state reuse it too.
        matrices._union_p1 = p1
        if mi_graph_path:
            np.savez(
                os.path.join(os.path.dirname(mi_graph_path), "union_p1.npz"),
                p1=p1.astype(np.float32),
                N=np.array([N], dtype=np.int64),
                d_sae=np.array([d_sae], dtype=np.int64),
            )

        p0 = 1.0 - p1

        active_indices = np.where(p1 > self.min_firing_freq)[0]
        D_act = len(active_indices)

        if D_act < self.min_community_size:
            logger.warning(f"Only {D_act} active features found. Returning empty cluster map.")
            return FeatureClusterMap(clusters={}, feature_to_cluster={g: 0 for g in range(d_sae)})

        p1_act = p1[active_indices]
        p0_act = p0[active_indices]

        H = - np.where(p0_act > 0, p0_act * np.log(p0_act + 1e-12), 0.0) \
            - np.where(p1_act > 0, p1_act * np.log(p1_act + 1e-12), 0.0)

        logger.info(f"Computing block-wise binary MI over D_act={D_act} active features (N={N})...")
        n_pairs = D_act * (D_act - 1) // 2

        # Upper-triangle (i<j) linear index: idx = i*(2*D - i - 1)//2 + (j - i - 1)
        def tri_idx(i, j):
            return i * (2 * D_act - i - 1) // 2 + (j - i - 1)

        # Write the norm-MI memmap on the NVMe scratch/checkpoint disk (the root
        # fs where /tmp lives is ~99% full); it is deleted after the edge pass.
        scratch_dir = os.environ.get("PDD_SCRATCH_DIR")
        if not scratch_dir and mi_graph_path:
            scratch_dir = os.path.dirname(os.path.abspath(mi_graph_path))
        if scratch_dir:
            os.makedirs(scratch_dir, exist_ok=True)
        mmap_path = os.path.join(scratch_dir or tempfile.gettempdir(), f"pdd_normMI_{os.getpid()}.npy")
        norm_mi_mmap = np.lib.format.open_memmap(mmap_path, mode="w+", dtype=np.float32, shape=(n_pairs,))

        n_blocks = (D_act + block - 1) // block
        logger.info(f"Building binary MI graph over {n_blocks} feature blocks (block_size={block})...")

        # 2. Co-occurrence p11 = U^T U / N via dense BLAS over row-chunks. The
        #    binary union is streamed one row-chunk at a time from the mmap'd
        #    C_freq/R_freq (never materialized in full, ~34GB dense or ~15GB
        #    sparse), and per-block-pair products accumulate into float32 buffers.
        chunk_rows = 4096
        widths = [min(block, D_act - b * block) for b in range(n_blocks)]
        p11 = [
            [np.zeros((widths[bi], widths[bj]), dtype=np.float32) for bj in range(bi, n_blocks)]
            for bi in range(n_blocks)
        ]
        for r0 in tqdm(range(0, N, chunk_rows), desc="MI co-occurrence (dense BLAS)"):
            r1 = min(r0 + chunk_rows, N)
            Uc = matrices.union_chunk_dense(r0, r1, active_indices)
            UcT = np.ascontiguousarray(Uc.T)
            for bi in range(n_blocks):
                i0, i1 = bi * block, min((bi + 1) * block, D_act)
                Ai = UcT[i0:i1]
                for bj in range(bi, n_blocks):
                    j0, j1 = bj * block, min((bj + 1) * block, D_act)
                    p11[bi][bj - bi] += Ai @ Uc[:, j0:j1]
        del Uc, UcT
        gc.collect()

        # 4. Per block-pair: normalize, compute MI / norm-MI, sorted write to memmap.
        for bi in tqdm(range(n_blocks), desc="Writing norm-MI blocks"):
            i0, i1 = bi * block, min((bi + 1) * block, D_act)
            p1_i = p1_act[i0:i1][:, None]
            p0_i = p0_act[i0:i1][:, None]
            Hi = H[i0:i1][:, None]

            for bj in range(bi, n_blocks):
                j0, j1 = bj * block, min((bj + 1) * block, D_act)
                p11_blk = p11[bi][bj - bi] / np.float32(N)

                p1_j = p1_act[j0:j1][None, :]
                p0_j = p0_act[j0:j1][None, :]

                # Upper-triangle mask: only keep pairs where global i < global j.
                if bi == bj:
                    ii = np.arange(i0, i1)[:, None]
                    jj = np.arange(j0, j1)[None, :]
                    keep = (ii < jj)
                else:
                    keep = None

                p10 = p1_i - p11_blk
                p01 = p1_j - p11_blk
                p00 = 1.0 - p11_blk - p10 - p01

                p11c = np.clip(p11_blk, 1e-7, 1.0)
                p10c = np.clip(p10, 1e-7, 1.0)
                p01c = np.clip(p01, 1e-7, 1.0)
                p00c = np.clip(p00, 1e-7, 1.0)

                MI = p11c * np.log(p11c / (p1_i * p1_j + 1e-7)) \
                   + p10c * np.log(p10c / (p1_i * p0_j + 1e-7)) \
                   + p01c * np.log(p01c / (p0_i * p1_j + 1e-7)) \
                   + p00c * np.log(p00c / (p0_i * p0_j + 1e-7))

                Hj = H[j0:j1][None, :]
                norm_MI = MI / (np.sqrt(Hi * Hj) + 1e-7)

                rows, cols = np.where(keep) if keep is not None else np.where(np.ones_like(norm_MI, dtype=bool))
                gidx = tri_idx(i0 + rows, j0 + cols)
                vals = norm_MI[rows, cols]
                # Write in ascending index order -> sequential-ish pages on the
                # memmap, avoiding random scattered writes on slow storage.
                order = np.argsort(gidx)
                norm_mi_mmap[gidx[order]] = vals[order]

                del p11_blk, norm_MI, vals
                gc.collect()

        norm_mi_mmap.flush()
        logger.info(f"Computed {n_pairs:,} normalized-MI values (upper triangle) into memmap.")

        chunk = 2 ** 26

        # Pass A: collect strictly-positive weights (mirrors original > 0 filter)
        # to compute the top-{top_pct}% cutoff.
        pos_vals = []
        for s in tqdm(range(0, n_pairs, chunk), desc="Scanning MI percentiles"):
            e = min(s + chunk, n_pairs)
            seg = norm_mi_mmap[s:e]
            sel = np.where(seg > 0)[0]
            if len(sel):
                pos_vals.append(seg[sel].astype(np.float64))
        if not pos_vals:
            try:
                os.remove(mmap_path)
            except OSError:
                pass
            logger.warning("No positive MI pairs found. Returning empty cluster map.")
            return FeatureClusterMap(clusters={}, feature_to_cluster={g: 0 for g in range(d_sae)})
        all_pos = np.concatenate(pos_vals)
        cutoff = np.percentile(all_pos, 100.0 - self.top_pct)
        del all_pos, pos_vals
        logger.info(f"Top-{self.top_pct}% MI cutoff over {n_pairs:,} pairs = {cutoff:.6f}")

        # Pass B: collect entries >= cutoff (streaming; invert the triu index)
        # Exact inversion: S(i) = i*(2D - i - 1)//2 is the offset of row i's first
        # element. row_starts[k] = S(k+1), so i = # of starts <= flat_idx.
        row_starts = ((np.arange(1, D_act, dtype=np.int64) * (2 * D_act - np.arange(1, D_act) - 1)) // 2)
        top_i_list, top_j_list, top_w_list = [], [], []
        for s in tqdm(range(0, n_pairs, chunk), desc="Extracting top MI edges"):

            e = min(s + chunk, n_pairs)
            seg = norm_mi_mmap[s:e]
            sel = np.where(seg >= cutoff)[0]
            if len(sel) == 0:
                continue
            flat_idx = s + sel
            i_global = np.searchsorted(row_starts, flat_idx, side="right").astype(np.int64)
            i_global = np.minimum(i_global, D_act - 2)
            j_global = flat_idx - ((i_global * (2 * D_act - i_global - 1)) // 2) + i_global + 1
            j_global = np.minimum(j_global, D_act - 1)
            top_i_list.append(i_global)
            top_j_list.append(j_global)
            top_w_list.append(seg[sel].astype(np.float64))

        try:
            os.remove(mmap_path)
        except OSError:
            pass

        if not top_i_list:
            logger.warning("No MI pairs above cutoff. Returning empty cluster map.")
            return FeatureClusterMap(clusters={}, feature_to_cluster={g: 0 for g in range(d_sae)})

        triu_i = np.concatenate(top_i_list)
        triu_j = np.concatenate(top_j_list)
        weights = np.concatenate(top_w_list)

        global_i = active_indices[triu_i]
        global_j = active_indices[triu_j]

        if mi_graph_path:
            logger.info(f"Saving computed MI graph to '{mi_graph_path}' for fast caching...")
            np.savez_compressed(mi_graph_path, global_i=global_i, global_j=global_j, weights=weights)

        edges = list(zip(global_i.tolist(), global_j.tolist()))
        edge_weights = weights.tolist()

        g = ig.Graph(n=d_sae, edges=edges, edge_attrs={"weight": edge_weights})

        logger.info(f"Running Leiden community detection (RBConfiguration, res={self.resolution_parameter}) on graph with {len(edges):,} edges...")
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=self.resolution_parameter,
            seed=seed,
        )

        clusters: Dict[int, List[int]] = {}
        feature_to_cluster: Dict[int, int] = {feat: 0 for feat in range(d_sae)}

        cluster_id_counter = 1
        for comm in partition:
            if len(comm) >= self.min_community_size:
                clusters[cluster_id_counter] = list(comm)
                for feat in comm:
                    feature_to_cluster[feat] = cluster_id_counter
                cluster_id_counter += 1

        logger.info(f"Extracted {len(clusters)} retained Leiden feature communities.")
        return FeatureClusterMap(clusters=clusters, feature_to_cluster=feature_to_cluster)
