# GEMINI.md — PDD (Predictive Data Debugging)

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
- **STRICT RULE: ZERO MOCK OR SYNTHETIC CODE.** NEVER write placeholder, random, or mock code under any circumstances. All pipeline stages and experiment scripts (`p4_dpo_validation.py`, `p5_interventions.py`, `pdd/autolabel.py`) MUST execute 100% real GPU model training, real SAE rollout extraction, real dataset inoculation, and real data analysis.
- **STRICT RULE: TQDM PROGRESS BARS.** ALWAYS use `tqdm` progress bars for all batch precomputation, DPO training epochs, dataset feature extraction, and text rollout generation loops across all experiment scripts.
- **STRICT RULE: GPU VRAM & CPU RAM SAFETY.** ALWAYS call `torch.cuda.empty_cache()` periodically during training and rollout extraction loops. Use sparse matrix operations for feature primitive calculations to keep RAM allocation < 1 MB and prevent OS OOM killer crashes. Store precomputed reference logps in 1D CPU numpy arrays (40 KB).
- **CRITICAL RULE: CODE POSITIONING & FUNCTION GROUPING.** ALWAYS group related functions, helpers, and handlers into clearly delineated, named sections (e.g. `SECTION 1: Lifecycle`, `SECTION 2: Hypothesis Maps`, `SECTION 3: Dataset Examples`, `SECTION 4: Member Cache`, `SECTION 5: Cluster Interpretations`, `SECTION 6: Live Inspector`, `SECTION 7: Neuronpedia`). NEVER scatter related helper methods across distant lines in a file. Position low-level internal helpers immediately adjacent to their public/calling methods or inside their designated functional section.
- **CRITICAL RULE: STANDARDIZED NAMING & MATHEMATICAL CONSISTENCY.** Maintain strict, unambiguous variable and parameter naming across all pipeline stages, viewer servers, and experiment scripts:
  - `m`: Feature Cluster community ID ($T_m$).
  - `k`: Data Cluster ID ($B_k$) or Prompt Cluster ID ($A_k$).
  - `feature_index` or `f`: Individual SAE feature index ($0 \le f < d_{\text{sae}}$).
  - `top_n` / `top_k`: Number of top items to retrieve.
  - `mats`: `FeatureMatrices` container.
  - `examples`: Sequence of `DatasetExample` instances.
  - NEVER use ambiguous names like `c`, `cid`, `cl`, `item`, `idx` when they could refer to multiple distinct mathematical entities ($B_k$ vs $T_m$ vs $A_k$ vs $R_m$).
- **CRITICAL RULE: COMPREHENSIVE ATTRIBUTE & FUNCTION DOCUMENTATION.** Every class attribute (especially in stateful containers like `ViewerState`) MUST have an explicit type annotation and a multi-line or inline docstring explaining its mathematical meaning, lifecycle (boot vs lazy load), and thread-safety invariants. Every public/internal helper must have a clear docstring so human readers and future AI coding agents can instantly understand data flows without guesswork.
- **Headless & Reproducible:** Run via `python -m pdd.cli --config configs/gemma2_2b_base.json` with dataclass/JSON configs in `configs/` and `pdd/config.py`, checkpoints in `checkpoints/`, run outputs under `runs/`. The pipeline ends with the auto-label stage (`auto_label` config block, default on): B_k LLM labels, T_m whole-cluster labels, and A_k/R_m example indices are written directly under `<run>/` (cluster_labels.json, feature_cluster_labels.json, prompt_conditioned_cluster_examples.json) so the viewer is fully interpretable after a single run. Launch the viewer with `python -m pdd.viewer --run_dir runs/<name>`; the viewer reads run artifacts lazily and never mutates checkpoints.
- **CPU-First for Analysis:** Use numpy/sklearn on CPU for statistical analysis. Use RTX 4090 GPU strictly for model/SAE forward passes and DPO training — check `nvidia-smi` first; never kill running processes.

## Shell & Tooling Quirks

- `source ~/.bashrc` prints harmless `juliaup ... command not found: complete` errors — ignore them. After it, chain commands with `;` (NOT `&&`, which aborts on the julia error exit code): `source ~/.bashrc; conda activate pdd; python ...`
- `rtk grep/rg/sed/read` intermittently fail (server I/O contention). On failure, fall back to the built-in `grep`/`read` tools.
- Keep startup imports lazy (import numpy inside functions where it's only needed for one branch); module-level heavy imports slow interactive/CLI use on the mechanical HDD.

## Codebase Map & Module Responsibilities

- `pdd/config.py` + `configs/*.json`: pipeline config dataclasses. All thresholds are config-driven, never hardcoded. Key feature-conditioned filters: `min_feat_cluster_size`=10, `min_data_cluster_size`=25, `sign_consistent`=1 (SC).
- `pdd/cli.py`: entry point (`python -m pdd.cli --config configs/...`), arg parsing, stdout reconfigure (must run AFTER `get_logger`).
- `pdd/pipeline.py`: orchestrator; `_resolve_checkpoint_subfolder` ranks resume candidates by progress score: complete matrices.npz / full `matrices_mmap` dir (N + 1M, via `_completed_matrices_score`) > surviving `chunks/` (sum of chunk sizes) > `examples.json` (1) > partial mmap dir (0). Resume logic reads chunk sizes with `np.load(..., mmap_mode="r")`.
- `pdd/feature_matrices.py`: matrix extraction state (`manifest.json` / legacy `matrices_state.json`) + `FeatureMatrixExtractor`. Write/read failures must log warnings, never silently pass.
- `pdd/neural_inspector.py`: model+SAE stack. `_encode_inputs` shared by `extract_prompt_features`/`extract_pair_features`; `_retry_load_on_cpu`; `_component_device`. Note `sae.encode` returns a tuple in some versions → always index `[0]`.
- `pdd/sae.py`: `SAEBackend`; `_hf_download` = local-cache-first hub download helper (shared by `_load_qwen_scope`/`_load_batch_topk`).
- `pdd/autolabel.py`: LLM auto-labeling (`LLMClusterLabeler`, `_strip_prefixes` cleans titles/descs). Bare `except Exception` at the JSON-parse point is intentional (break on malformed LLM output, propagates None to logged fallbacks).
- `pdd/inspection.py`: Pure functional algorithm layer for behavioral exploration (Mode A/B inference, Tab 4 single & compound sample ranking, normalized prompt-conditioned scoring). Fully decoupled from HTTP server routing and state containers for isolated testing.
- `pdd/validation.py`: Prediction validation metrics & cluster aggregation (`compute_prediction_validation_metrics`, `cluster_validation_metrics`). Auto-detects the highest epoch checkpoint array (`delta_all_epoch*.npy`) and computes per-cluster $R^2$ and Pearson $r$.
- `pdd/viewer_server.py` + `viewer/app.js`: single-run FastAPI + JS UI; `python -m pdd.viewer --run_dir runs/<name>` (optionally `--no-prewarm`). Reads run artifacts lazily, never mutates checkpoints; Neuronpedia cache under `<run>/viewer_cache/`.
- Pipeline math modules (feature-conditioned / prompt-conditioned / feature-cluster algorithms) live in `pdd/feature_conditioned.py`, `pdd/prompt_conditioned.py`, `pdd/feature_clusters.py` — treat as the paper's math, keep stable.

## Viewer API & Run Artifacts (for wiring)

- `GET /api/runs`, `/api/run_data` (returns `summary`, `validation_metrics`, `top_feature_conditioned_hypotheses`, `top_prompt_conditioned_hypotheses`, `cluster_labels`, `feature_cluster_labels`), `/api/feature_cluster_info?m=<k>&top_n=...`, `/api/cluster_detail?type=feature|data&id=...`, `/api/cluster_validation?m=<T_m>`, `/api/inspect_feature_samples?m=<T_m>&k=50&side=amplify|suppress` (single) or `?conditions=<m:amplify|suppress[:tau],...>` (compound: top samples satisfying EVERY condition, ranked by total disparity `sum|u_m|`), `/api/neuronpedia_*`.
- Tab 4 (Behavior → Prompt) = the inverse search: pick a T_m, get the top preference pairs whose labels amplify (u>0, chosen fires the cluster more) or suppress (u<0, rejected fires it more) it. `_inspect_feature_samples` reads per-example u/s from `u_matrix` and `s_matrix` directly via `inspection.rank_cluster_samples` and `inspection.rank_compound_samples`. Directional filtering ($u > 0$ vs $u < 0$) and disparity ranking ($|u_i|$) surface the strongest drivers first. B.1 T_m dropdown examples are presence-ranked (C_max+R_max); tab 4 examples are label-disparity-ranked — keep the two axes distinct.
- Run artifact files under `<run>/`: `cluster_labels.json` (B_k), `feature_cluster_labels.json` (T_m), `prompt_conditioned_cluster_examples.json` (A_k/R_m), `feature_conditioned_hypotheses.json`, `pdd_summary.json`; under `<run>/p4_validation/`: `u_feature.npy` + `delta_all_epoch<N>.npy` (per-feature predicted and empirical post-DPO deltas).
- Server-side helper conventions: `_cached_info(key, build)` memoizes per-cluster lookups (call `build` even on cache hit ONLY for side-effects like prewarm); `_reload_if_changed(path, cache_attr, mtime_attr)` is the shared mtime-reload for the three label artifact loaders (`_load_data_cluster_labels`, `_load_feature_cluster_labels`, `_load_pc_cluster_examples`) — don't re-expand them, and never bypass it with direct cache-attr reads; `_parse_hypotheses` builds `k -> list_of_hypotheses`; `_neuronpedia_verified`/`_sae_feature_item`/`_worker` keep handlers thin. Endpoint families: `/api/inspect_*` = derivation/inverse-search (prompt → predicted shifts, cluster → driving samples); `/api/feature_cluster_info` + `/api/cluster_detail` + `/api/feature_detail` + `/api/cluster_validation` + `/api/pc_cluster_examples` = lookup. Config key `min_feat_cluster_size` abbreviates "feature" — rename would silently change old-run defaults, so it stays.

## Experiment Scripts (`experiments/`)

- `test_dpo_validation.py`: DPO validation. `load_validated_cluster_ids()` builds the hypothesis-set cluster IDs from the config thresholds — this IS the validation universe (R² is measured ONLY over these clusters, never the whole Leiden partition; falls back to the full partition with a warning if `feature_conditioned_hypotheses.json` is missing); `compute_cluster_validation(..., valid_ids=...)` computes the metrics; `eval_epoch` emits per-epoch metrics saved to `p4_r2_metrics.json`/`p4_r2_by_epoch.json`. Observed on the 65k run: hypothesis-set R²=0.0171 (29 of 118 clusters) — still noise-level, so the low R² is a signal-quality issue (u_bar vs empirical Δ mismatch), not a cluster-count/threshold one.
- `test_feature_clusters_audit.py`: Standalone diagnostic tool for Leiden cluster size distributions, SAE feature retention rates, and hypothesis coverage across `--config`, `--run_dir`, or `--clusters_json`.
- `test_mode_a_prediction.py`: Combined Mode A (Prompt -> Predicted Shifts) + Tab 4 (Behavior -> Driving Dataset Pairs) causal test tool. Evaluates live prompt activations on the GPU and fetches ground-truth training pairs that explain each predicted shift (`--url http://localhost:9000`, `--prompt "..."`, or `--test_cluster <m>`). Includes 10 cross-domain benchmark test cases.
- `test_mode_b_prediction.py`: Mode B (Preference Pair Audit) live disparity test tool. Executes live batched GPU forward passes on $(x, y_c, y_r)$ to measure exact SAE feature disparity $u = \mathbf{1}\{C > 0.01\} - \mathbf{1}\{R > 0.01\}$, surfacing ▲ Promoted vs ▼ Suppressed concepts across 8 diverse domains.
- `test_data_bottlenecks.py`: Exhaustive behavioral interference & data bottleneck discovery tool. Runs vectorized BLAS matrix products ($\mathbf{C} = \mathbf{M}_A^T \times \mathbf{M}_B$) over 17,000+ cluster pairs in <2.5s across 4 interference regimes: `amp_sup` (+A, -B decoupling), `amp_amp` (+A, +B multi-skill synthesis), `sup_sup` (-A, -B de-biasing), and `amp_neutral` (+A, 0B isolation). Supports `--mode`, `--cluster <m>`, `--export_json <path>`, `--tau <f>`, and `--top_k <int>`.
- `test_data_sufficiency.py`: Data sufficiency & latent coverage diagnostic tool. Evaluates whether a candidate pair subset $\mathcal{S}$ has sufficient latent feature coverage ($\text{Cov}(T_A) \ge 95\%$) and gradient rank across any interference mode. Supports automated batch audits over top bottlenecks (`--top_k <int>` or `--bottlenecks_json <path>`), single-pair audits (`--target_cluster <m> --interference_cluster <m>`), and master synthetic blueprint exports (`--export_inoculation_spec <path>`).
- Known tokenizer facts: Qwen3 has NO BOS and right-pads; Gemma2 has BOS + left-pads. Robust positional mask = `attention_mask.argmax() + prompt_len + tokenizer_offset`.
- LoRA gradient graph: `enable_input_require_grads()` is called for **both** full and LoRA modes — for LoRA, frozen base embeddings don't require grad by default, so without this hook the backward graph is severed and `loss.backward()` raises "element 0 of tensors does not require grad". `gradient_checkpointing_enable()` is gated on `lora_rank == 0` only — it breaks LoRA by detaching intermediate activations across checkpointed segments. Batched SAE encodes: zero-pad the window, do ONE `sae.encode`, then slice back — matches per-window output.

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
- Auto-labeling stage (B.1.7 + viewer interpretation, `pdd/autolabel.py`, final
  pipeline stage): Powered by `google/gemma-3-4b-it` (128k context, 90.2% IFEval)
  for structured semantic synthesis. Pass 1 labels data clusters B_k from real
  centroid/random sampled prompts via the B.1 `s_matrix` (30 centroid + 20 random prompts, 600 chars);
  Pass 2 labels feature clusters T_m using structured Prompt + Chosen (Promoted) + Rejected (Suppressed)
  exemplars firing them (C_max+R_max), capturing the full preference delta contrast;
  Pass 3 maps prompt clusters A_k / response-delta clusters R_m to their strongest
  real examples (c_matrix / |u_matrix|). Reuses the in-memory fc/pc results — no
  recompute. Artifact paths are shared with the viewer via the helpers in
  `pdd/autolabel.py` (cluster_labels_path / feature_cluster_labels_path /
  pc_cluster_examples_path).
- Viewer (`pdd/viewer_server.py`, `viewer/`): single-run FastAPI + JS UI. T_m tags
  (inspector + B.1 table) open a whole-cluster dropdown (LLM label + Neuronpedia
  member links + real examples); A_k / R_m links show their strongest examples.

## Dev Workflow — Verify Before You Report

- **Mandatory pre-report checks after editing Python/JS**: `python -m py_compile <changed files>` (syntax) + `ruff check pdd/` (lint) + `node --check viewer/app.js` (JS) + quick module import test.
- **Viewer smoke test**: boot `python -m pdd.viewer --run_dir runs/<name> --port <free> --no-prewarm` in background, wait ~20–30s (mechanical HDD), `curl /api/run_data` + `/api/feature_cluster_info?m=...` + `/api/cluster_detail`, then kill the server. This catches runtime breakage in `_cached_info`/handler refactors that py_compile/ruff cannot.
- **Functional tests**: exercise refactored helpers directly (e.g. checkpoint-subfolder ranking across npz/chunks/examples/partial-mmap scenarios) rather than trusting a refactor by inspection.
- **Pitfall**: `pkill -f "pdd.viewer --run_dir ..."` ALSO matches the user's own long-running viewer instance (different port). Kill by PID (from `pgrep -af "pdd.viewer"`) instead, and never kill the user's viewer without asking.

## Refactor/Cleanup Guidelines (behavior-preserving)

- Work in strict category order: 1) redundant/duplicate code, 2) implicit fallbacks (silent `except: pass` / `or {}` MUST become `logger.warning`), 3) un-needed complex if-else / dead paths, 4) function sizing/overlap (no forced splits; merge only honest duplicates). Run the full verify battery after EACH pass.
- Per-function `import numpy as np` is function-scoped: when extracting a helper that uses np, the OUTER function loses np if it still references it — re-add the import to the outer scope or the refactor silently breaks the non-tested branch.
- Keep math modules (`feature_conditioned.py`, `prompt_conditioned.py`, `feature_clusters.py`) untouched; never alter checkpoint/data artifacts (see Data Preservation rule).

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
