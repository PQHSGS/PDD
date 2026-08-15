# AGENTS.md — PDD (Predictive Data Debugging)

Re-implementation of Goodfire's *Anatomy of Post-Training* (arXiv:2606.12360).
Source of truth for the algorithm: `docs/paper/main.tex` (the downloaded arXiv
LaTeX source) — read the relevant appendix section before touching a module.

## Environment

- Conda env: **`pdd`** (clone of `sae_circuit`). Activate with
  `conda activate pdd` (or `source ~/.bashrc && conda activate pdd`). **CRITICAL RULE**: ALWAYS use standard `conda activate pdd`. NEVER use hardcoded paths like `/mnt/disk1/miniconda3/...` or `/mnt/disk1/miniconda3/condabin/conda`!
- Versions pinned in `requirements.txt`. **Watch out:** the env has
  transformers 4.49 / torch 2.4 / sae-lens 5.3. Do NOT `pip install peft/trl`
  normally — they pull transformers>=5 which needs torch>=2.5 and breaks the
  env. If SFT/DPO deps are needed, install with `--no-deps` and pinned
  versions.

## Model / SAE

- Base: `Qwen/Qwen3-1.7B-Base` or `google/gemma-2-2b`.
- SAE: Qwen-Scope TopK (`Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`, k=50) or Gemma-Scope TopK (`gemma-scope-2b-pt-res-canonical`, layer 12).
- The paper's SAE is a BatchTopK trained on the SFT model; Qwen-Scope / Gemma-Scope is our stand-in. Document any fidelity deviations in code.

## Strict Dev Rules & Coding Guidelines

- **CRITICAL RULE: IMMEDIATE & PROACTIVE USER COMMUNICATION.** ALWAYS answer user questions and provide status updates in visible natural text FIRST before executing background tool calls or diagnostic scripts. NEVER remain silent or execute background tool loops without informing the user.
- **CRITICAL RULE: DATA PRESERVATION & SAFETY.** NEVER delete, overwrite, or touch pre-extracted feature matrix checkpoints, activation datasets (`matrices_mmap`), or long-running feature extraction artifacts without explicit user confirmation.
- **CRITICAL RULE: FILE I/O & CHECKPOINT MUTATION CONFIRMATION.** ALWAYS ask the user for explicit confirmation before making any code modifications or running scripts that alter saved checkpoint files, metadata schemas, directory structures, or file I/O formatting. Explicitly ask the user if they want to update any specific checkpoint subfolders.
- **STRICT RULE: ZERO MOCK OR SYNTHETIC CODE.** NEVER write placeholder, random, or mock code under any circumstances. All pipeline stages and experiment scripts (`p4_dpo_validation.py`, `p5_interventions.py`, `p6_autolabeling.py`) MUST execute 100% real GPU model training, real SAE rollout extraction, real dataset inoculation, and real data analysis.
- **STRICT RULE: TQDM PROGRESS BARS.** ALWAYS use `tqdm` progress bars for all batch precomputation, DPO training epochs, dataset feature extraction, and text rollout generation loops across all experiment scripts.
- **STRICT RULE: GPU VRAM & CPU RAM SAFETY.** ALWAYS call `torch.cuda.empty_cache()` periodically during training and rollout extraction loops. Use sparse matrix operations for feature primitive calculations to keep RAM allocation < 1 MB and prevent OS OOM killer crashes. Store precomputed reference logps in 1D CPU numpy arrays (40 KB).
- **Headless & Reproducible:** Run via `python -m pdd.cli --config configs/gemma2_2b_base.json` with dataclass/JSON configs in `configs/` and `pdd/config.py`, checkpoints in `checkpoints/`, run outputs under `runs/`.
- **CPU-First for Analysis:** Use numpy/sklearn on CPU for statistical analysis. Use RTX 4090 GPU strictly for model/SAE forward passes and DPO training — check `nvidia-smi` first; never kill running processes.

## Paper Recipe (Implementation Checklist)

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