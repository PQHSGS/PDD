"""FastAPI server for the Predictive Data Debugging (PDD) Interactive Viewer.

Serves run metadata, feature-conditioned & prompt-conditioned hypotheses,
cluster statistics, and live prompt/preference pair inspection endpoints.
Points directly to a target run directory and its linked checkpoint artifacts.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VIEWER_DIR = Path(__file__).parent.parent / "viewer"


class PromptInspectionRequest(BaseModel):
    prompt: str
    run_dir: Optional[str] = None
    top_k: int = 5
    apply_chat_template: bool = False
    template_type: str = "gemma"


class PreferencePairInspectionRequest(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    run_dir: Optional[str] = None
    top_k: int = 5


class ViewerState:
    """Manages the target run directory, linked checkpoints, and hypothesis indices."""

    def __init__(self, run_dir: str = "runs/gemma2_2b_dolci"):
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
            with open(sum_path, "r", encoding="utf-8") as f:
                self.summary = json.load(f)

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
        lbl_file = self.run_dir / "p6_autolabeling" / "cluster_labels.json"
        if lbl_file.exists():
            try:
                with open(lbl_file, "r", encoding="utf-8") as f:
                    self.cluster_labels = json.load(f).get("labels", [])
            except Exception:
                pass

        # 4. Load & Pre-Index Hypotheses
        fc_file = self.run_dir / "feature_conditioned_hypotheses.json"
        if fc_file.exists():
            try:
                with open(fc_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.fc_hypos = data.get("hypotheses", data) if isinstance(data, dict) else data
                    self.k_to_fc = {}
                    for h in self.fc_hypos:
                        k = h.get("k")
                        if k is not None:
                            self.k_to_fc.setdefault(k, []).append(h)
            except Exception as e:
                logger.warning(f"Error reading fc hypotheses: {e}")

        pc_file = self.run_dir / "prompt_conditioned_hypotheses.json"
        if pc_file.exists():
            try:
                with open(pc_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.pc_hypos = data.get("hypotheses", data) if isinstance(data, dict) else data
                    self.k_to_pc = {}
                    for h in self.pc_hypos:
                        k = h.get("k")
                        if k is not None:
                            self.k_to_pc.setdefault(k, []).append(h)
            except Exception as e:
                logger.warning(f"Error reading pc hypotheses: {e}")

        logger.info(
            f"ViewerState initialized for '{self.run_dir.name}': "
            f"{len(self.fc_hypos)} FC hypotheses, {len(self.pc_hypos)} PC hypotheses, {len(self.feature_clusters)} feature clusters."
        )

    def get_inspector(self):
        """Get or initialize the NeuralInspector configured for this target run."""
        if self.inspector is None:
            from pdd.neural_inspector import get_neural_inspector
            model_cfg = self.summary.get("config", {}).get("model", {})
            sae_cfg = self.summary.get("config", {}).get("sae", {})

            model_path = model_cfg.get("path", "google/gemma-2-2b")
            sae_repo = sae_cfg.get("repo", "gemma-scope-2b-pt-res-canonical")
            sae_id = sae_cfg.get("sae_id", "layer_12/width_16k/canonical")
            layer = sae_cfg.get("layer", 12)

            self.inspector = get_neural_inspector(model_path=model_path, sae_repo=sae_repo, sae_id=sae_id, layer=layer)
        return self.inspector


# Global target state
STATE = ViewerState()


@app.get("/api/runs")
def list_runs() -> Dict[str, Any]:
    """Return the active target run details."""
    metrics = STATE.summary.get("metrics", {})
    return {
        "runs": [{
            "name": STATE.run_dir.name,
            "path": str(STATE.run_dir),
            "timestamp": STATE.summary.get("timestamp", "N/A"),
            "config_name": STATE.summary.get("config", {}).get("name", STATE.run_dir.name),
            "model": STATE.summary.get("config", {}).get("model", {}).get("path", "N/A"),
            "sae": STATE.summary.get("config", {}).get("sae", {}).get("repo", "N/A"),
            "num_examples": metrics.get("num_examples", 0),
            "num_clusters": metrics.get("num_sae_feature_clusters", len(STATE.feature_clusters)),
        }]
    }


@app.get("/api/run_data")
def get_run_data(run_dir: Optional[str] = Query(None)) -> Dict[str, Any]:
    """Retrieve summary, validation metrics, cluster labels, and top hypotheses for the active run."""
    val_file = STATE.run_dir / "p4_validation" / "p4_r2_metrics.json"
    validation_metrics = {}
    if val_file.exists():
        try:
            with open(val_file, "r", encoding="utf-8") as f:
                validation_metrics = json.load(f)
        except Exception:
            pass

    return {
        "summary": STATE.summary,
        "validation_metrics": validation_metrics,
        "cluster_labels": STATE.cluster_labels,
        "top_feature_conditioned_hypotheses": STATE.fc_hypos[:100],
        "top_prompt_conditioned_hypotheses": STATE.pc_hypos[:100],
    }


@app.post("/api/inspect_prompt")
def inspect_prompt(req: PromptInspectionRequest) -> Dict[str, Any]:
    """Mode A: Inspect prompt through live GPU Model + SAE forward pass and predict downstream behavioral shifts."""
    prompt_text = req.prompt.strip()
    if not prompt_text:
        return {"prompt": "", "matched_clusters": [], "predicted_behavior_shifts": []}

    # 1. Real GPU Forward Pass -> SAE Features P(x)
    inspector = STATE.get_inspector()
    p_feat = inspector.extract_prompt_features(prompt_text)

    # 2. Score Prompt Clusters A_k
    label_map = {cl.get("cluster_id"): cl for cl in STATE.cluster_labels}
    top_sae_indices = np.argsort(p_feat)[-5:][::-1]
    top_sae_kws = [f"SAE-Feat_{idx} (act={p_feat[idx]:.2f})" for idx in top_sae_indices]

    scored_clusters = []
    for k, hypos in STATE.k_to_pc.items():
        max_delta = max(abs(float(h.get("delta", 0.0))) for h in hypos)
        cl_info = label_map.get(k, {})
        title = cl_info.get("title", f"Prompt Cluster A_{k}")
        desc = cl_info.get("description", f"Active prompt cluster with {len(hypos)} verified hypotheses.")
        scored_clusters.append({
            "cluster_id": k,
            "title": title,
            "description": desc,
            "matched_keywords": top_sae_kws,
            "relevance_score": float(max_delta),
            "hypos": hypos,
        })

    scored_clusters = sorted(scored_clusters, key=lambda x: x["relevance_score"], reverse=True)[:req.top_k]

    matched_clusters = [{
        "cluster_id": c["cluster_id"],
        "title": c["title"],
        "description": c["description"],
        "matched_keywords": c["matched_keywords"],
        "relevance_score": c["relevance_score"],
    } for c in scored_clusters]

    # 3. Extract Predicted Shifts from Genuine SAE Hypotheses
    predicted_shifts = []
    for c in scored_clusters:
        for h in c["hypos"][:2]:
            k = h.get("k")
            m = h.get("m")
            delta = float(h.get("delta", 0.0))
            z = float(h.get("z_score", 0.0))
            d = float(h.get("cohens_d", 0.0))
            is_amplified = delta > 0

            direction_word = "AMPLIFIED (Boosted)" if is_amplified else "SUPPRESSED (Inhibited)"
            interpretation = (
                f"When given this prompt type (A_{k}), the DPO model will produce Response Concept R_{m} "
                f"significantly MORE ({direction_word}) compared to the base SFT model "
                f"(Predicted Shift Δ = {delta:+.5f}, Welch z = {z:.2f})."
                if is_amplified else
                f"When given this prompt type (A_{k}), the DPO model will produce Response Concept R_{m} "
                f"significantly LESS ({direction_word}) compared to the base SFT model "
                f"(Predicted Shift Δ = {delta:+.5f}, Welch z = {z:.2f})."
            )

            predicted_shifts.append({
                "prompt_cluster_k": k,
                "response_cluster_m": m,
                "data_cluster_k": k,
                "feature_cluster_m": m,
                "delta": delta,
                "effect_direction": "Amplified after DPO" if is_amplified else "Suppressed after DPO",
                "z_score": z,
                "cohens_d": d,
                "interpretation": interpretation,
            })

    predicted_shifts = sorted(predicted_shifts, key=lambda s: abs(s["delta"]), reverse=True)[:10]

    return {
        "prompt": prompt_text,
        "matched_clusters": matched_clusters,
        "predicted_behavior_shifts": predicted_shifts,
    }


@app.post("/api/inspect_preference_pair")
def inspect_preference_pair(req: PreferencePairInspectionRequest) -> Dict[str, Any]:
    """Mode B: Batched GPU forward pass on preference pair to measure exact SAE disparity u = 1(C>0.01) - 1(R>0.01)."""
    prompt_text = req.prompt.strip()
    chosen_text = req.chosen.strip()
    rejected_text = req.rejected.strip()

    if not prompt_text or not chosen_text or not rejected_text:
        return {"matched_clusters": [], "promoted_concepts": [], "suppressed_concepts": []}

    # 1. Batched GPU Forward Pass -> Chosen (C), Rejected (R), and Disparity (u)
    inspector = STATE.get_inspector()
    c_p, r_p, u = inspector.extract_pair_features(prompt_text, chosen_text, rejected_text)

    # 2. Score Data Clusters B_k
    label_map = {cl.get("cluster_id"): cl for cl in STATE.cluster_labels}
    top_pair_indices = np.argsort(c_p + r_p)[-5:][::-1]
    top_pair_kws = [f"SAE-Feat_{idx} (act={(c_p+r_p)[idx]:.2f})" for idx in top_pair_indices]

    scored_clusters = []
    for k, hypos in STATE.k_to_fc.items():
        max_delta = max(abs(float(h.get("delta", 0.0))) for h in hypos)
        cl_info = label_map.get(k, {})
        title = cl_info.get("title", f"Data Topic B_{k}")
        desc = cl_info.get("description", f"Active dataset topic cluster with {len(hypos)} verified hypotheses.")
        scored_clusters.append({
            "cluster_id": k,
            "title": title,
            "description": desc,
            "matched_keywords": top_pair_kws,
            "relevance_score": float(max_delta),
            "hypos": hypos,
        })

    scored_clusters = sorted(scored_clusters, key=lambda x: x["relevance_score"], reverse=True)[:req.top_k]

    matched_clusters = [{
        "cluster_id": c["cluster_id"],
        "title": c["title"],
        "description": c["description"],
        "matched_keywords": c["matched_keywords"],
        "relevance_score": c["relevance_score"],
    } for c in scored_clusters]

    # 3. Extract Promoted vs. Suppressed Concepts
    promoted_concepts = []
    suppressed_concepts = []

    for c in scored_clusters:
        for h in c["hypos"]:
            k = h.get("k")
            m = h.get("m")
            delta = float(h.get("delta", 0.0))
            z = float(h.get("z_score", 0.0))
            is_chosen = h.get("is_chosen_leaning", delta > 0)

            item = {
                "feature_cluster_m": m,
                "data_cluster_k": k,
                "delta": delta,
                "z_score": z,
                "signal_strength": "Strong" if abs(z) > 3.0 else "Moderate",
                "explanation": (
                    f"Real SAE Disparity u > 0: Chosen response fires concept features significantly more than rejected."
                    if is_chosen else
                    f"Real SAE Disparity u < 0: Rejected response fires concept features, teaching the model to suppress it."
                ),
            }
            if is_chosen:
                promoted_concepts.append(item)
            else:
                suppressed_concepts.append(item)

    promoted_concepts = sorted(promoted_concepts, key=lambda x: x["delta"], reverse=True)[:5]
    suppressed_concepts = sorted(suppressed_concepts, key=lambda x: x["delta"])[:5]

    return {
        "prompt": prompt_text,
        "chosen_length": len(chosen_text),
        "rejected_length": len(rejected_text),
        "promoted_sae_features_count": int((u > 0).sum()),
        "suppressed_sae_features_count": int((u < 0).sum()),
        "matched_clusters": matched_clusters,
        "promoted_concepts": promoted_concepts,
        "suppressed_concepts": suppressed_concepts,
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
    parser.add_argument("--run_dir", type=str, default="runs/gemma2_2b_dolci", help="Path to target PDD run directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7000, help="Port to bind server")
    args = parser.parse_args()

    # Re-initialize state with user-specified run directory
    global STATE
    STATE = ViewerState(run_dir=args.run_dir)

    logger.info(f"Starting PDD Viewer for '{args.run_dir}' at http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
