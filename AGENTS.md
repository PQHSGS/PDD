# AGENTS.md — PDD (Predictive Data Debugging)

Re-implementation of Goodfire's *Anatomy of Post-Training* (arXiv:2606.12360).
Source of truth for the algorithm: `docs/paper/main.tex` (the downloaded arXiv
LaTeX source) — read the relevant appendix section before touching a module.

## Environment

- Conda env: **`pdd`** (clone of `sae_circuit`). Activate with
  `source /mnt/disk1/miniconda3/etc/profile.d/conda.sh && conda activate pdd`.
- Versions pinned in `requirements.txt`. **Watch out:** the env has
  transformers 4.49 / torch 2.4 / sae-lens 5.3. Do NOT `pip install peft/trl`
  normally — they pull transformers>=5 which needs torch>=2.5 and breaks the
  env. If SFT/DPO deps are needed, install with `--no-deps` and pinned
  versions.

## Model / SAE

- Base: `Qwen/Qwen3-1.7B-Base`.
- SAE: Qwen-Scope TopK (`Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`, k=50),
  per-layer `layer{N}.sae.pt`. sae-lens 5.3.0 has **no** native `qwen_scope`
  loader — load via `SAE.from_dict` (see `pdd/sae.py`; `W_enc`/`W_dec` need
  transposing to sae-lens convention).
- The paper's SAE is a BatchTopK trained on the SFT model; Qwen-Scope (TopK,
  base-model) is our stand-in. Document any fidelity deviations in code.

## Paper recipe (implementation checklist)

- Feature-conditioned pipeline: `docs/paper/main.tex` §"Feature-Conditioned
  Pipeline" (Appendix B.1): per-pair primitives `s` / `u` (τ=0.01) / `v`;
  silent bucket = 5th pct of ‖s‖₂; spherical k-means K=512 on normalized s
  (MiniBatchKMeans); Welch inside-vs-outside Δ/z + Cohen's d; split-half
  (row-index parity) SC flag + Δ^min; filters |T_m|≥10, n_k≥25, SC=1; rank by
  Δ^min.
- Prompt-conditioned pipeline (Appendix B.2): P^q/D^q (max & freq), feature
  retention (resp: n≥200, σ≥1e-3, mean|D|≥1e-4; prompt: n≥200), feature
  embeddings (randomized SVD-128 on 30k sample, ℓ2-norm), MiniBatchKMeans →
  A_k / R_m, scores c_{i,k} / u_{i,m}, top-n_top selection, Δ + Cohen's d + z.
- Feature clusters: binary-MI graph (top 1% off-diagonal pairs, normalized MI),
  Leiden, keep communities ≥4 features.

## Dev rules

- Keep headless/reproducible: `python -m pdd.cli --config configs/qwen3_1.7b_base.json` with dataclass/JSON config in
  `configs/` and `pdd/config.py`, checkpoints in `checkpoints/`, run outputs under `runs/`.
- CPU-first for analysis (numpy/sklearn). Only use the 4090 for model/SAE
  forward passes and training — check `nvidia-smi` first; never kill running
  processes.
- Commit only when asked. Update README status checklist as phases land.
- `.opencode/` inherited from EM (agents + notebooklm skill). Note anything
  SAESteeringBench-specific there is stale context; PDD is self-contained.