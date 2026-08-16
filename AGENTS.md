# AGENTS.md

Project `survey` = **PDD** (Predictive Data Debugging), re-implementation of Goodfire's *Anatomy of Post-Training* (arXiv:2606.12360; paper in `docs/paper/main.tex`). Runs on UF HiPerGator.

## STRICT RULES (NEVER BREAK)

### R1. Save data ONLY here `[RULE]`

- Save in `NguyenDo3/survey/`. Max boundary: `NguyenDo3/` = `/blue/rc-rse/vanminh.nguyen/NguyenDo/NguyenDo3/`. Nothing outside it.
- NEVER save in `$HOME` (40 GB, mostly full), `/tmp`, login caches, node local disk. `$HOME/storage/blue/...` = same disk as `/blue/rc-rse/...`, use the `/blue/...` path.
- `/blue/rc-rse/vanminh.nguyen/` holds old data/envs — read only, don't add.

### R2. Project stuff goes in the project folder `[RULE]`

- Env, checkpoints, private data, models → `NguyenDo3/survey/` only.
- Env: `NguyenDo3/survey/.conda_envs/pdd`. Use absolute path: `conda activate /blue/rc-rse/vanminh.nguyen/NguyenDo/NguyenDo3/survey/.conda_envs/pdd`.
- Old paths `/mnt/disk1/...`, `/mnt/disk4/...` don't exist here — never use.

### R3. Redirect caches before every run/download/install `[RULE]`

```bash
PROJECT_DIR="/blue/rc-rse/vanminh.nguyen/NguyenDo/NguyenDo3/survey"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/huggingface/datasets"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache/xdg"
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.cache/uv/python"
export TMPDIR="$PROJECT_DIR/.tmp"
mkdir -p "$PROJECT_DIR/.cache" "$PROJECT_DIR/.tmp"
```

Shared caches (HF hub, wandb) → `NguyenDo3/.cache/...`, never `$HOME`. `/blue/rc-rse` is ~95% full (~2.7 TB free) — download little, purge old checkpoints.

## Server: HiPerGator / SLURM

Login node has **no GPU**. Use `sbatch`/`srun`.

- **GPU partitions:** `hpg-b200` (B200, 8/node) and `hpg-turin` (L4, 3/node), both `--gres=gpu:N`, 14-day limit. Others exist; don't use them.
- **Jobs:** `squeue --me` (status), `scancel <jobid>` (kill), `tail -f logs/<name>_<jobid>.out` (output).
- **Submit** from repo root:

```bash
#!/bin/bash
#SBATCH --job-name=pdd
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=hpg-b200      # or hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --time=12:00:00
```

- **Interactive GPU:** `srun --partition=hpg-b200 --gres=gpu:1 --cpus-per-task=4 --mem=40gb --time=02:00:00 --pty bash -i`
- **Env/modules:** `module load conda` then `conda activate <abs path>`. CUDA: `module load cuda/12.8.1`.
- **Gotchas:** SLURM spools your script — use `$SLURM_SUBMIT_DIR`, not `${BASH_SOURCE[0]}`. Only allocated GPUs visible → always `cuda:0`. Pin `export UV_PYTHON="$(command -v python)"` so `uv run` uses the conda env. `--time` < partition limit or job dies.
- Docs: https://docs.hpc.ufl.edu/

## PDD rules

- **Algorithm source:** `docs/paper/main.tex` — read the appendix before editing a module.
- **Env:** `pdd` (py 3.11, from `requirements.txt`): transformers 4.51 / torch 2.4.1 / sae-lens 5.3. Do NOT `pip install peft/trl` — pulls transformers>=5 needing torch>=2.5, breaks env. If needed: `--no-deps` + pinned.
- **Model/SAE:** `Qwen/Qwen3-1.7B-Base` + `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` (k=50), or `google/gemma-2-2b` + `gemma-scope-2b-pt-res-canonical` (layer 12). These are stand-ins for the paper's BatchTopK — note deviations in code.
- **Run:** `PYTHONPATH=. python -m pdd.cli --config configs/<model>.json`; checkpoints/ and runs/ in repo. Data: `allenai/Dolci-Instruct-DPO/SFT`. Experiments: `PYTHONPATH=. python experiments/p4_dpo_validation.py --config configs/qwen3_1.7b_base.json` (also p5, p6).
- **Dev rules:**
  - Communicate: status updates in visible text FIRST, never silent background loops.
  - Never delete/overwrite feature matrix checkpoints, `matrices_mmap`, or extraction artifacts without asking.
  - Confirm before code that alters checkpoints/schemas/file formats.
  - No mock/synthetic code — everything runs real GPU training, real SAE extraction, real analysis.
  - `tqdm` on all batch/epoch/extraction/rollout loops.
  - Call `torch.cuda.empty_cache()` periodically; sparse ops for feature primitives (RAM < 1 MB); reference logps as 1D CPU numpy arrays.
  - CPU (numpy/sklearn) for analysis; GPU only for model/SAE forwards + DPO training. Check `squeue --me` before launching; never kill running jobs.

## Paper recipe

- Feature-conditioned (B.1): primitives `s`/`u` (τ=0.01)/`v`; silent bucket = 5th pct ‖s‖₂; spherical k-means K=512 (MiniBatchKMeans); Welch Δ/z + Cohen's d; split-half SC + Δ^min; filters |T_m|≥10, n_k≥25, SC=1; rank Δ^min.
- Prompt-conditioned (B.2): P^q/D^q (max & freq); retention (resp: n≥200, σ≥1e-3, mean|D|≥1e-4; prompt: n≥200); SVD-128 on 30k, ℓ2; MiniBatchKMeans → A_k/R_m, c_{i,k}/u_{i,m}; Δ + Cohen's d + z.
- Clusters: binary-MI graph (top 1% off-diagonal), Leiden, keep ≥4 features.
