# AGENTS.md — PDD (Predictive Data Debugging)

Re-implementation of Goodfire's *Anatomy of Post-Training* (arXiv:2606.12360).
Source of truth for the algorithm: `docs/paper/main.tex` (the downloaded arXiv
LaTeX source) — read the relevant appendix section before touching a module.

## Environment

- Conda env: **`pdd`** (clone of `sae_circuit`). Activate with
  `conda activate pdd` (or `source ~/.bashrc && conda activate pdd`). **CRITICAL RULE**: ALWAYS use standard `conda activate pdd`. NEVER use hardcoded paths like `/mnt/disk1/miniconda3/...` or `/mnt/disk1/miniconda3/condabin/conda`!
- Versions pinned in `requirements.txt`. The env currently runs
  transformers 4.51.0 / torch 2.4 / sae-lens 5.3. Upgrading is allowed and
  expected when the task needs newer models (Qwen3/Gemma3 need transformers
  >=4.51, which is already installed). Note: `pip install peft/trl` plain will
  pull transformers>=5 + torch>=2.5 — install with `--no-deps` and pinned
  versions if only those are needed, or upgrade the whole env deliberately and
  re-run the viewer/pipeline smoke tests afterwards.

## Model / SAE

- Base: `Qwen/Qwen3-1.7B-Base` or `google/gemma-2-2b`.
- SAE: Qwen-Scope TopK (`Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`, k=50) or Gemma-Scope TopK (`gemma-scope-2b-pt-res-canonical`, layer 12).
## Server & Hardware Context (CPU Load & Mechanical HDD)

- **High System Load (50+)**: The server frequently runs heavy background training jobs (e.g. `ST_train.py`, meteorological ViT models), causing CPU run queues and load averages to reach 50–60.
- **Mechanical HDD on `/mnt/disk4`**: Conda environment (`pdd`), models, checkpoints, and datasets reside on `/mnt/disk4` (`/dev/sdb1`), which is a mechanical hard drive under heavy I/O contention. Python imports and file reads can take 15–30s due to disk wait (`wait_on_page_bit_common`, `STAT=Dl+`).
- **RAM & Swap Pressure**: System RAM cache is near capacity with ~2GB pushed to swap.
- **DIAGNOSTIC FIRST PRINCIPLE**: If a command or server startup is slow, hanging, or delayed, **ALWAYS check system load, disk I/O wait, and swap FIRST (`uptime`, `vmstat 1 2`, `ps aux`) before assuming there is a code bug**. Do NOT immediately assume code is broken or rewrite logic when the bottleneck is server I/O contention. Keep startup reads lazy and lightweight to prevent I/O blocking.

## Strict Dev Rules & Coding Guidelines

- **CRITICAL RULE: IMMEDIATE & PROACTIVE USER COMMUNICATION.** ALWAYS answer user questions and provide status updates in visible natural text FIRST before executing background tool calls or diagnostic scripts. NEVER remain silent or execute background tool loops without informing the user.
- **STRICT RULE: NO POLLING OR REPETITIVE STATUS CHECK LOOPS.** When a background task (e.g. conda environment cloning, package installation, long-running data preprocessing, model training) is running, NEVER poll or execute repetitive status checking loops (`manage_task status`, schedule timers, repeated `ls`, `ps aux`). Simply stop calling tools and wait quietly for the system notification when the task completes.
- **CRITICAL RULE: DATA PRESERVATION & SAFETY.** NEVER delete, overwrite, or touch pre-extracted feature matrix checkpoints, activation datasets (`matrices_mmap`), or long-running feature extraction artifacts without explicit user confirmation.
- **CRITICAL RULE: FILE I/O & CHECKPOINT MUTATION CONFIRMATION.** ALWAYS ask the user for explicit confirmation before making any code modifications or running scripts that alter saved checkpoint files, metadata schemas, directory structures, or file I/O formatting. Explicitly ask the user if they want to update any specific checkpoint subfolders.
- **STRICT RULE: ZERO MOCK OR SYNTHETIC CODE.** NEVER write placeholder, random, or mock code under any circumstances. All pipeline stages and experiment scripts (`p4_dpo_validation.py`, `p5_interventions.py`, `pdd/autolabeling.py`) MUST execute 100% real GPU model training, real SAE rollout extraction, real dataset inoculation, and real data analysis.
- **STRICT RULE: TQDM PROGRESS BARS.** ALWAYS use `tqdm` progress bars for all batch precomputation, DPO training epochs, dataset feature extraction, and text rollout generation loops across all experiment scripts.
- **STRICT RULE: GPU VRAM & CPU RAM SAFETY.** ALWAYS call `torch.cuda.empty_cache()` periodically during training and rollout extraction loops. Use sparse matrix operations for feature primitive calculations to keep RAM allocation < 1 MB and prevent OS OOM killer crashes. Store precomputed reference logps in 1D CPU numpy arrays (40 KB).
- **Headless & Reproducible:** Run via `python -m pdd.cli --config configs/gemma2_2b_base.json` with dataclass/JSON configs in `configs/` and `pdd/config.py`, checkpoints in `checkpoints/`, run outputs under `runs/`. The pipeline ends with the auto-label stage (`auto_label` config block, default on): B_k LLM labels, T_m whole-cluster labels, and A_k/R_m example indices are written directly under `<run>/` (cluster_labels.json, feature_cluster_labels.json, prompt_conditioned_cluster_examples.json) so the viewer is fully interpretable after a single run. Launch the viewer with `python -m pdd.viewer --run_dir runs/<name>`; the viewer reads run artifacts lazily and never mutates checkpoints.
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
- Auto-labeling stage (B.1.7 + viewer interpretation, `pdd/autolabeling.py`, final
  pipeline stage): Pass 1 labels data clusters B_k from real centroid/random
  sampled prompts via the B.1 `s_matrix`; Pass 2 labels feature clusters T_m from
  the real response examples firing them (C_max+R_max), independent of Neuronpedia;
  Pass 3 maps prompt clusters A_k / response-delta clusters R_m to their strongest
  real examples (c_matrix / |u_matrix|). Reuses the in-memory fc/pc results — no
  recompute. Artifact paths are shared with the viewer via the helpers in
  `pdd/autolabeling.py` (cluster_labels_path / feature_cluster_labels_path /
  pc_cluster_examples_path).
- Viewer (`pdd/viewer_server.py`, `viewer/`): single-run FastAPI + JS UI. T_m tags
  (inspector + B.1 table) open a whole-cluster dropdown (LLM label + Neuronpedia
  member links + real examples); A_k / R_m links show their strongest examples.