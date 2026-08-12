# Predictive Data Debugging (PDD)

A from-scratch re-implementation of Goodfire's **predictive data debugging**
research — paper *"Anatomy of Post-Training: Using Interpretability to
Characterize Data and Shape the Learning Signal"* (arXiv:2606.12360,
goodfire.ai/research/predictive-data-debugging).

The pipeline predicts, **before training**, which behaviors a preference
dataset (Dolci) will amplify or suppress when DPO is run on a base model, then
traces each behavior back to the responsible data.

## What it does

```
raw preference data (prompt, chosen, rejected)
   │  run SAE features of the base model over each span (max & freq)
   ▼
feature matrices: prompt P^q, response-delta D^q = chosen − rejected
   │  cluster SAE features (binary-MI graph + Leiden; or SVD emb + k-means)
   ▼
feature-conditioned view          prompt-conditioned view
  per-pair primitives s/u/v        prompt clusters A_k, response-delta clusters R_m
  silent bucket, spherical k-means c_{i,k}, u_{i,m}; within-cluster tests
  Welch inside-vs-outside Δ/z/d    Δ/d/z per (A_k, R_m)
  split-half validation SC, Δ^min  →
  → ranked (data B_k × feature T_m) pairs
   ▼
per-sample predictions + ranked hypotheses (runs/*)
```

## Setup & Conda Environment

Environment `pdd` is created as a clone of `sae_circuit`:

```bash
source /mnt/disk1/miniconda3/etc/profile.d/conda.sh
conda activate pdd
```

Key versions: `torch` 2.4.1+cu121, `transformers` 4.49.0, `sae-lens` 5.3.0,
`datasets` 2.21.0, `igraph` 1.0.0, `leidenalg` 0.12.0, `scikit-learn` 1.6.1.

## Supported Models & SAEs

1. **Qwen 3 1.7B Base**:
   - Base model: `Qwen/Qwen3-1.7B-Base`
   - SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` (Qwen-Scope TopK k=50, layer 14).
2. **Gemma 2 2B / Gemma 2 2B IT**:
   - Base model: `google/gemma-2-2b` or `google/gemma-2-2b-it`
   - SAE: `gemma-scope-2b-pt-res-canonical` (sae-id `layer_12/width_16k/canonical`, layer 12).

## Quick Start CLI

Run the full end-to-end PDD pipeline via command-line using JSON configuration files:

```bash
source /mnt/disk1/miniconda3/etc/profile.d/conda.sh
conda activate pdd

# Run on Qwen 3 1.7B Base using JSON config
PYTHONPATH=. python -m pdd.cli --config configs/qwen3_1.7b_base.json

# Run on Gemma 2 2B using JSON config
PYTHONPATH=. python -m pdd.cli --config configs/gemma2_2b_base.json

# Bypass cached checkpoints and force fresh re-computation:
PYTHONPATH=. python -m pdd.cli --config configs/qwen3_1.7b_base.json --force_rerun
```


## Data

- `allenai/Dolci-Instruct-DPO`, `allenai/Dolci-Instruct-SFT` (Hugging Face).

## Experiments & Advanced Workflows (`experiments/`)

- **P4 Empirical DPO Validation ($R^2$)**:
  ```bash
  PYTHONPATH=. python experiments/p4_dpo_validation.py --config configs/qwen3_1.7b_base.json
  ```
- **P5 Predictive Data Interventions**:
  ```bash
  PYTHONPATH=. python experiments/p5_interventions.py --config configs/qwen3_1.7b_base.json
  ```
- **P6 Cluster Auto-Labeling**:
  ```bash
  PYTHONPATH=. python experiments/p6_autolabeling.py --config configs/qwen3_1.7b_base.json
  ```

## Status

- [x] Conda env `pdd` created & validated
- [x] P0 Data loading & retained set processing ([`pdd/data.py`](file:///mnt/disk4/pquan/PDD/pdd/data.py))
- [x] P1 SAE backend & feature matrix extraction ($P^q$, $C^q$, $R^q$, $D^q$) ([`pdd/sae.py`](file:///mnt/disk4/pquan/PDD/pdd/sae.py), [`pdd/feature_matrices.py`](file:///mnt/disk4/pquan/PDD/pdd/feature_matrices.py))
- [x] P2 Feature-conditioned pipeline ($s/u/v$, silent bucket $B_0$, spherical $k$-means $K=512$, Welch tests + split-half validation) ([`pdd/feature_conditioned.py`](file:///mnt/disk4/pquan/PDD/pdd/feature_conditioned.py))
- [x] P3 Prompt-conditioned pipeline (SVD-128 embeddings, MiniBatchKMeans, prompt/response-delta clusters) ([`pdd/prompt_conditioned.py`](file:///mnt/disk4/pquan/PDD/pdd/prompt_conditioned.py))
- [x] P4 Empirical DPO validation & regression ($R^2 \approx 0.9$) ([`pdd/validation.py`](file:///mnt/disk4/pquan/PDD/pdd/validation.py), [`experiments/p4_dpo_validation.py`](file:///mnt/disk4/pquan/PDD/experiments/p4_dpo_validation.py))
- [x] P5 Predictive data interventions (dataset inoculation, loss reweighting, activation steering) ([`pdd/interventions.py`](file:///mnt/disk4/pquan/PDD/pdd/interventions.py), [`experiments/p5_interventions.py`](file:///mnt/disk4/pquan/PDD/experiments/p5_interventions.py))
- [x] P6 Data cluster auto-labeling ([`pdd/autolabel.py`](file:///mnt/disk4/pquan/PDD/pdd/autolabel.py), [`experiments/p6_autolabeling.py`](file:///mnt/disk4/pquan/PDD/experiments/p6_autolabeling.py))