"""Pure algorithm layer for the Tab 3/4 behavioral explorer (Mode A/B + inverse search).

Extracted from `pdd/viewer_server.py` so the interactive algorithms are unit-testable
without FastAPI or a `ViewerState`. Every function here is a pure function: data goes in,
data comes out. The viewer/server is responsible for lazy loading, memory-mapping, caching,
and passing its artifacts in as arguments.

Math notes:
- `u` (per-example disparity) and `s` (per-example firing) follow Appendix B.1.
- `score_prompt_conditioned` ranks prompt clusters A_k by the MEAN feature-expression
  score over members, matching the paper's `c_{i,k} = (P @ S_p) / |A_k|` normalization
  (NOT a raw sum, which would bias large clusters).
- Tab 4 (inverse search) ranks individual training samples by their per-example
  disparity `u_i` against feature cluster T_m. No arbitrary thresholds: the
  directional filter (u > 0 for amplify, u < 0 for suppress) IS the faithful
  condition from B.1. Ranking by |u| surfaces the strongest drivers first.
  Compound queries AND multiple directional conditions and score by total
  absolute disparity sum(|u_m|).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .logger import get_logger

logger = get_logger("PDD.Inspection")


def project_clusters(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip internal full hypothesis arrays from scored clusters for lightweight JSON responses."""
    return [{k: v for k, v in c.items() if k != "hypos"} for c in scored]


def parse_conditions(conditions: str, default_thresh: float = 0.01) -> List[Tuple[int, str, float]]:
    """Parse a compound query string into `(m, direction, min_delta)` tuples.

    Format: ``'m:amplify|suppress[:min_delta],...'`` (e.g. ``'7:amplify:0.03,4:suppress'``).
    Malformed parts are skipped; an invalid threshold falls back to ``default_thresh``.
    """
    parsed: List[Tuple[int, str, float]] = []
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
            d_thresh = float(fields[2]) if len(fields) > 2 and fields[2].strip() else default_thresh
        except ValueError:
            d_thresh = default_thresh
        parsed.append((cm, direction, d_thresh))
    return parsed


def cluster_signals(activations: np.ndarray, feature_clusters: Dict[int, List[int]], mode: str = "sum") -> Dict[int, float]:
    """Aggregate a live (d_sae,) activation vector into per-feature-cluster signals T_m.

    ``mode="sum"`` sums member activations; ``mode="mean"`` averages them.
    """
    signals: Dict[int, float] = {}
    if activations is None or len(activations) == 0:
        return signals
    for m, feats in feature_clusters.items():
        idx = np.asarray(feats, dtype=np.int64)
        idx = idx[idx < len(activations)]
        if len(idx) == 0:
            signals[m] = 0.0
            continue
        vals = activations[idx]
        signals[m] = float(vals.mean()) if mode == "mean" else float(vals.sum())
    return signals


def hypothesis_evidence(hypos: List[Dict[str, Any]], signals: Dict[int, float]) -> List[Tuple[Dict[str, Any], float]]:
    """Score hypotheses by |delta| multiplied by live per-cluster signal strength."""
    ev: List[Tuple[Dict[str, Any], float]] = []
    for h in hypos:
        m = h.get("m")
        sig = signals.get(m, 0.0) if m is not None else 0.0
        ev.append((h, abs(float(h.get("delta", 0.0))) * abs(sig)))
    return ev


def cluster_keywords(
    activations: np.ndarray,
    feature_ms: Sequence[int],
    feature_clusters: Dict[int, List[int]],
    top_n: int = 3,
) -> List[str]:
    """Return top individual SAE features by live activation within the specified feature clusters."""
    if activations is None or len(activations) == 0:
        return []
    feats: List[int] = []
    for m in feature_ms:
        feats.extend(feature_clusters.get(m, []))
    if not feats:
        return []
    idx = np.asarray(feats, dtype=np.int64)
    idx = idx[idx < len(activations)]
    if len(idx) == 0:
        return []
    vals = activations[idx]
    order = np.argsort(vals)[-top_n:][::-1]
    return [f"SAE-Feat_{int(idx[i])} (act={vals[i]:.2f})" for i in order]


def score_data_clusters(
    signal: Dict[int, float],
    feat_for_keywords: np.ndarray,
    hypos_map: Dict[int, List[Dict[str, Any]]],
    data_labels_map: Dict[int, Dict[str, Any]],
    feature_clusters: Dict[int, List[int]],
    top_n_keywords: int = 3,
) -> List[Dict[str, Any]]:
    """Rank data clusters B_k by live-evidence-weighted hypotheses (Tab 3 Mode A/B)."""
    scored = []
    for k, hypos in hypos_map.items():
        ev = hypothesis_evidence(hypos, signal)
        if not ev:
            continue
        best_h, best_ev = max(ev, key=lambda t: t[1])
        if best_ev <= 0:
            continue
        cl_info = data_labels_map.get(k, {})
        title = cl_info.get("title", f"Data Topic B_{k}")
        desc = cl_info.get("description", f"Active dataset topic cluster with {len(hypos)} verified hypotheses.")
        feature_ms = [h.get("m") for h in hypos if h.get("m") is not None]
        scored.append({
            "cluster_id": k,
            "title": title,
            "description": desc,
            "matched_keywords": cluster_keywords(feat_for_keywords, feature_ms, feature_clusters, top_n_keywords),
            "relevance_score": float(best_ev),
            "best_hypothesis": best_h,
            "hypos": hypos,
        })
    return sorted(scored, key=lambda x: x["relevance_score"], reverse=True)


def score_prompt_conditioned(
    p_feat: np.ndarray,
    prompt_clusters: Dict[int, List[int]],
    prompt_hypos_map: Dict[int, List[Dict[str, Any]]],
    pc_tokens_fn: Callable[[str, int], List[str]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Extract top Prompt-Conditioned hypotheses (A_k x R_m) via live SAE feature-space co-activation.

    A_k scores use the MEAN feature expression ``mean over A_k members of P`` to match the
    pipeline's normalized `c_{i,k}` (paper Appendix B.2); raw sums would inflate large clusters.
    """
    if not prompt_hypos_map:
        return []

    scored_ak: List[Tuple[int, float]] = []
    if p_feat is not None and len(p_feat) > 0 and prompt_clusters:
        c_scores: Dict[int, float] = {}
        for k, mems in prompt_clusters.items():
            k_int = int(k)
            idx = np.asarray(mems, dtype=np.int64)
            idx = idx[idx < len(p_feat)]
            if len(idx) > 0:
                c_scores[k_int] = float(p_feat[idx].mean())
        scored_ak = sorted(c_scores.items(), key=lambda x: x[1], reverse=True)

    if not scored_ak or scored_ak[0][1] == 0.0:
        scored_ak = [(int(k), 0.0) for k in list(prompt_hypos_map.keys())[:top_k]]

    pc_shifts: List[Dict[str, Any]] = []
    for k, ak_score in scored_ak[:top_k]:
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

            p_tokens = pc_tokens_fn("prompt", k)
            r_tokens = pc_tokens_fn("response", m)

            direction_word = "AMPLIFIED (Boosted)" if is_amplified else "SUPPRESSED (Inhibited)"
            interpretation = (
                f"Prompt expresses prompt-feature cluster A_{k} (feature activation score: {ak_score:.2f}, represented by: {', '.join(p_tokens[:4]) if p_tokens else 'local prompt subspace'}). "
                f"In local preference data, this condition shifts response-delta cluster R_{m} "
                f"(represented by: {', '.join(r_tokens[:4]) if r_tokens else 'response disparity features'}) "
                f"with local effect size Cohen's d = {d:.2f} (Δ = {delta:+.5f}, Welch z = {z:.2f}), "
                f"indicating that post-training will likely {direction_word} these response features."
            )

            pc_shifts.append({
                "prompt_cluster_k": k,
                "prompt_score": ak_score,
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


def _matrix_row(mat: Any, row: int) -> np.ndarray:
    """Extract a single row from a sparse or dense matrix as a dense 1-D array."""
    if hasattr(mat, "toarray"):
        return np.asarray(mat[row].toarray()).ravel()
    return np.asarray(mat[row]).ravel()


def _sample_firing(
    mats: Any,
    feats: List[int],
    global_idx: int,
    neuronpedia_url_fn: Callable[[int], Optional[str]],
    max_display: int = 20,
) -> List[Dict[str, Any]]:
    """Scan full cluster T_m member features for firing on example `global_idx`.

    Returns per-feature C_freq/R_freq and which side (chosen/rejected/both) the
    feature fires on, capped at ``max_display``. Used by rank_cluster_samples and
    rank_compound_samples to annotate sample cards with member-firing detail.
    """
    firing_feats: List[Dict[str, Any]] = []
    if mats is None or len(feats) == 0:
        return firing_feats
    c_row = _matrix_row(mats.C_freq, global_idx)
    r_row = _matrix_row(mats.R_freq, global_idx)
    for f in feats:
        f_int = int(f)
        c_act = float(c_row[f_int]) if f_int < len(c_row) else 0.0
        r_act = float(r_row[f_int]) if f_int < len(r_row) else 0.0
        if c_act > 0 or r_act > 0:
            firing_feats.append({
                "feature_index": f_int,
                "c_freq": round(c_act, 4),
                "r_freq": round(r_act, 4),
                "active_in": (
                    "chosen" if (c_act > 0 and r_act <= 0)
                    else ("rejected" if (r_act > 0 and c_act <= 0) else "both")
                ),
                "neuronpedia_url": neuronpedia_url_fn(f_int),
            })
    if len(firing_feats) > 1:
        firing_feats.sort(key=lambda x: x["c_freq"] + x["r_freq"], reverse=True)
    return firing_feats[:max_display]


def rank_cluster_samples(
    m: int,
    side: str,
    top_n: int,
    fc: Any,
    examples: Sequence[Any],
    mats: Any,
    feature_clusters: Dict[int, List[int]],
    cluster_ids: List[int],
    example_view_fn: Callable[[Any], Dict[str, str]],
    neuronpedia_url_fn: Callable[[int], Optional[str]],
    label: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rank individual training samples by per-example disparity u against feature cluster T_m.

    Faithful to B.1: the directional filter (u > 0 for amplify, u < 0 for suppress) IS
    the condition — no arbitrary thresholds.  Ranking by |u| surfaces the strongest
    preference-shift drivers first.  ``total_matching`` counts ALL direction-satisfying
    examples; ``samples`` contains only the top_k for display.
    """
    m_int = int(m)
    side = side if side in ("amplify", "suppress") else "amplify"
    top_n = max(1, min(int(top_n), 200))
    feats = feature_clusters.get(m_int, [])

    base: Dict[str, Any] = {
        "cluster_m": m_int,
        "label": label or {"title": f"Feature cluster T_{m_int}", "description": "", "keywords": []},
        "n_features": len(feats),
        "side": side,
        "total_matching": 0,
        "samples": [],
    }
    if fc is None or examples is None or fc.u_matrix is None:
        return base
    if fc.u_matrix.shape[1] != len(cluster_ids):
        logger.warning(
            f"u_matrix shape {fc.u_matrix.shape} does not match "
            f"{len(cluster_ids)} feature clusters; cannot rank samples."
        )
        return base
    if m_int not in cluster_ids:
        return base

    pos = cluster_ids.index(m_int)
    u = fc.u_matrix[:, pos]
    s = fc.s_matrix[:, pos]

    if side == "amplify":
        mask = u > 0
        order = np.argsort(-u)
    else:
        mask = u < 0
        order = np.argsort(u)

    satisfying = order[mask[order]]
    total_matching = int(len(satisfying))

    samples: List[Dict[str, Any]] = []
    for i in satisfying[:top_n]:
        i_int = int(i)
        if i_int >= len(examples):
            continue
        ex = examples[i_int]
        u_i = float(u[i_int])
        context_k = int(fc.cluster_assignments[i_int]) if fc.cluster_assignments is not None else -1

        samples.append({
            "index": i_int,
            "u": round(u_i, 4),
            "s": round(float(s[i_int]), 4),
            "context_k": context_k,
            "effect_direction": "Amplified after DPO" if u_i > 0 else "Suppressed after DPO",
            "member_firing": _sample_firing(mats, feats, i_int, neuronpedia_url_fn),
            **example_view_fn(ex),
        })

    return {**base, "total_matching": total_matching, "samples": samples}


def rank_compound_samples(
    conditions: List[Tuple[int, str, float]],
    top_n: int,
    fc: Any,
    examples: Sequence[Any],
    mats: Any,
    feature_clusters: Dict[int, List[int]],
    cluster_ids: List[int],
    example_view_fn: Callable[[Any], Dict[str, str]],
    neuronpedia_url_fn: Callable[[int], Optional[str]],
    feature_cluster_labels: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Rank individual samples satisfying ALL compound directional conditions.

    Each condition is (m, direction, _) — the filter is purely directional (u > 0
    for amplify, u < 0 for suppress) with no arbitrary thresholds.  Score is
    ``sum |u_m|`` across conditions — total absolute disparity, faithful to the
    paper's per-pair primitives.  ``total_matching`` counts the full AND-mask;
    ``samples`` is top_k only.
    """
    conds: List[Tuple[int, str]] = []
    for m, direction, _ in conditions:
        direction = direction if direction in ("amplify", "suppress") else "amplify"
        conds.append((int(m), direction))

    if not conds:
        return {"compound": True, "conditions": [], "total_matching": 0, "samples": []}

    t_label_map: Dict[int, Dict[str, Any]] = feature_cluster_labels or {}

    if fc is None or examples is None or fc.u_matrix is None:
        return {"compound": True, "conditions": [], "total_matching": 0, "samples": []}
    if fc.u_matrix.shape[1] != len(cluster_ids):
        logger.warning(
            f"u_matrix shape {fc.u_matrix.shape} does not match "
            f"{len(cluster_ids)} feature clusters; cannot rank compound samples."
        )
        return {"compound": True, "conditions": [], "total_matching": 0, "samples": []}

    col_u: Dict[int, np.ndarray] = {}
    col_s: Dict[int, np.ndarray] = {}
    mask = np.ones(len(examples), dtype=bool)
    cond_meta: List[Dict[str, Any]] = []

    for m_int, direction in conds:
        if m_int not in cluster_ids:
            continue
        pos = cluster_ids.index(m_int)
        u = fc.u_matrix[:, pos]
        s = fc.s_matrix[:, pos]
        col_u[m_int] = u
        col_s[m_int] = s

        n_dir = int(np.sum(u > 0)) if direction == "amplify" else int(np.sum(u < 0))
        cl_info = t_label_map.get(m_int, {})
        cond_meta.append({
            "m": m_int,
            "direction": direction,
            "label": cl_info.get("title", f"Feature cluster T_{m_int}"),
            "n_total": n_dir,
        })

        if direction == "amplify":
            mask &= (u > 0)
        else:
            mask &= (u < 0)

    satisfying = np.flatnonzero(mask)
    total_matching = int(len(satisfying))

    scores = np.zeros(len(examples), dtype=np.float32)
    for m_int in col_u:
        scores += np.abs(col_u[m_int])

    order = satisfying[np.argsort(-scores[satisfying])]

    all_feats: List[int] = []
    for m_int, _ in conds:
        all_feats.extend(feature_clusters.get(m_int, []))
    all_feats = list(set(all_feats))

    samples: List[Dict[str, Any]] = []
    for i in order[:top_n]:
        i_int = int(i)
        if i_int >= len(examples):
            continue
        ex = examples[i_int]
        u_map = {m_int: round(float(col_u[m_int][i]), 4) for m_int in col_u}
        s_map = {m_int: round(float(col_s[m_int][i]), 4) for m_int in col_s}
        effects = []
        for m_int, direction in conds:
            u_val = float(col_u[m_int][i]) if m_int in col_u else 0.0
            if u_val > 0:
                effects.append("Amplified")
            elif u_val < 0:
                effects.append("Suppressed")
            else:
                effects.append("Neutral")

        context_k = int(fc.cluster_assignments[i_int]) if fc.cluster_assignments is not None else -1
        samples.append({
            "index": i_int,
            "u": u_map,
            "s": s_map,
            "score": round(float(scores[i]), 4),
            "effect_directions": effects,
            "context_k": context_k,
            "member_firing": _sample_firing(mats, all_feats, i_int, neuronpedia_url_fn),
            **example_view_fn(ex),
        })

    return {
        "compound": True,
        "conditions": cond_meta,
        "total_matching": total_matching,
        "samples": samples,
    }


def top_examples(
    scores: np.ndarray,
    examples: Sequence[Any],
    example_view_fn: Callable[[Any], Dict[str, str]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
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
            **example_view_fn(examples[i]),
        })
    return out


def sae_feature_item(i: int, val: float, m: Optional[int], neuronpedia_url_fn: Optional[Callable[[int], Optional[str]]]) -> Dict[str, Any]:
    """Format an active SAE feature item with its cluster assignment and optional Neuronpedia link."""
    return {
        "feature_index": i,
        "activation": round(val, 4),
        "cluster_m": m,
        "neuronpedia_url": neuronpedia_url_fn(i) if neuronpedia_url_fn is not None else None,
    }


def top_sae_features(
    activations: np.ndarray,
    feat_to_cluster: Dict[int, int],
    feature_clusters: Dict[int, List[int]],
    min_partition_size: int,
    top_n: int = 8,
    neuronpedia_url_fn: Optional[Callable[[int], Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """Extract top active SAE features sorted by magnitude, tagging their cluster community."""
    if activations is None or len(activations) == 0:
        return []
    pos_idx = np.flatnonzero(activations > 0)
    if len(pos_idx) == 0:
        return []
    sorted_pos = pos_idx[np.argsort(-activations[pos_idx])]
    out = []
    for idx in sorted_pos:
        i = int(idx)
        val = float(activations[i])
        m = feat_to_cluster.get(i)
        if m is not None and len(feature_clusters.get(m, [])) < min_partition_size:
            m = None
        out.append(sae_feature_item(i, val, m, neuronpedia_url_fn))
        if len(out) >= top_n:
            break
    return out


def _data_cluster_title_map(data_cluster_labels: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    """Map data cluster id k -> LLM label title (shared by Mode A shifts and Mode B concepts)."""
    return {
        int(lab.get("cluster_id")): lab.get("title", f"Data Cluster B_{lab.get('cluster_id')}")
        for lab in data_cluster_labels if "cluster_id" in lab
    }


def predicted_behavior_shifts(
    scored_clusters: List[Dict[str, Any]],
    act: Dict[int, float],
    feature_cluster_labels: Dict[int, Dict[str, Any]],
    data_cluster_labels: Sequence[Dict[str, Any]],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Build Feature-Conditioned predicted shifts (B_k x T_m) from scored data clusters (Mode A)."""
    predicted_shifts = []
    b_title_map = _data_cluster_title_map(data_cluster_labels)
    for c in scored_clusters:
        ordered = sorted(
            c["hypos"],
            key=lambda h: abs(float(h.get("delta", 0.0))) * abs(act.get(h.get("m"), 0.0)),
            reverse=True,
        )
        for h in ordered[:2]:
            k = h.get("k")
            m = h.get("m")
            delta = float(h.get("delta", 0.0))
            z = float(h.get("z_score", 0.0))
            d = float(h.get("cohens_d", 0.0))
            evidence = act.get(m, 0.0)
            is_amplified = delta > 0

            t_info = feature_cluster_labels.get(int(m), {}) if m is not None else {}
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
    return predicted_shifts[:limit]


def pair_concepts(
    u_sig: Dict[int, float],
    best_by_m: Dict[int, Dict[str, Any]],
    feature_clusters: Dict[int, List[int]],
    min_feat_cluster_size: int,
    feature_cluster_labels: Dict[int, Dict[str, Any]],
    data_cluster_labels: Sequence[Dict[str, Any]],
    top_n: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build Promoted vs. Suppressed concept lists from live Mode B disparity (per-pair u)."""
    promoted: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    b_title_map = _data_cluster_title_map(data_cluster_labels)
    for m, uval in sorted(u_sig.items(), key=lambda t: abs(t[1]), reverse=True):
        if uval == 0.0:
            continue
        if len(feature_clusters.get(m, [])) < min_feat_cluster_size or m not in best_by_m:
            continue
        h = best_by_m.get(m, {})
        k = h.get("k")
        z = float(h.get("z_score", 0.0))
        is_chosen = uval > 0
        t_info = feature_cluster_labels.get(int(m), {}) if m is not None else {}
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
            promoted.append(item)
        else:
            suppressed.append(item)

    promoted.sort(key=lambda x: x["delta"], reverse=True)
    suppressed.sort(key=lambda x: x["delta"])
    return promoted[:top_n], suppressed[:top_n]