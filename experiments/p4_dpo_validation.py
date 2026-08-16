"""Phase P4 Experiment: 100% Paper-Accurate DPO Fine-Tuning & SAE Rollout Validation (R^2).

Replicates Goodfire's paper (arXiv:2606.12360, §3, §4 & App. B):
1. Fine-tunes the base model using DPO loss (Rafailov et al., 2023) for one or more epochs on GPU to produce pi_DPO.
2. Samples text rollouts y_SFT from pre-DPO model and y_DPO from post-DPO model over held-out prompts.
3. Encodes rollouts through the SAE to measure real empirical rollout activation shifts:
   delta_empirical = mean(a(y_DPO)) - mean(a(y_SFT))
4. Computes R^2 regression and Pearson correlation against predicted hypotheses delta_predicted.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict
import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pdd.config import PipelineConfig
from pdd.data import PreferenceExample
from pdd.feature_clusters import FeatureClusterMap
from pdd.feature_conditioned import FeatureConditionedPipeline
from pdd.feature_matrices import FeatureMatrices
from pdd.logger import get_logger
from pdd.sae import ModelBackend, SAEBackend
from pdd.validation import ValidationMetrics, compute_prediction_validation_metrics

logger = get_logger("PDD.Exp.P4")


class DPODataset(Dataset):
    """PyTorch Dataset wrapper for preference pairs."""

    def __init__(self, examples: list[PreferenceExample], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        prompt = ex.prompt
        chosen = f"{prompt} {ex.chosen}"
        rejected = f"{prompt} {ex.rejected}"

        c_enc = self.tokenizer(chosen, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        r_enc = self.tokenizer(rejected, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

        p_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        c_labels = c_enc.input_ids.squeeze(0).clone()
        c_labels[:p_len] = -100
        c_labels[c_enc.attention_mask.squeeze(0) == 0] = -100

        r_labels = r_enc.input_ids.squeeze(0).clone()
        r_labels[:p_len] = -100
        r_labels[r_enc.attention_mask.squeeze(0) == 0] = -100

        return {
            "chosen_ids": c_enc.input_ids.squeeze(0),
            "chosen_mask": c_enc.attention_mask.squeeze(0),
            "chosen_labels": c_labels,
            "rejected_ids": r_enc.input_ids.squeeze(0),
            "rejected_mask": r_enc.attention_mask.squeeze(0),
            "rejected_labels": r_labels,
            "prompt": prompt,
        }


def compute_sequence_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-sequence log-probabilities using memory-efficient logsumexp."""
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        targets = labels[:, 1:]

        loss_mask = targets != -100
        targets_clamped = targets.clone()
        targets_clamped[~loss_mask] = 0

        log_z = torch.logsumexp(logits, dim=-1)
        token_logits = torch.gather(logits, dim=2, index=targets_clamped.unsqueeze(2)).squeeze(2)
        per_token_logps = token_logits - log_z

        return (per_token_logps.float() * loss_mask.float()).sum(dim=1)


class LoRALinear(torch.nn.Module):
    """Pure PyTorch LoRA linear projection wrapper (zero-dependency, low-VRAM)."""

    def __init__(self, original_linear: torch.nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = original_linear
        for p in self.linear.parameters():
            p.requires_grad = False

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.scaling = alpha / r

        self.lora_A = torch.nn.Parameter(torch.zeros((r, in_dim), dtype=original_linear.weight.dtype, device=original_linear.weight.device))
        self.lora_B = torch.nn.Parameter(torch.zeros((out_dim, r), dtype=original_linear.weight.dtype, device=original_linear.weight.device))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.linear(x)
        lora_res = (F.linear(x, self.lora_A) @ self.lora_B.T) * self.scaling
        return res + lora_res


def apply_pure_pytorch_lora(model: torch.nn.Module, r: int = 8, alpha: float = 16.0) -> torch.nn.Module:
    """Wraps target query/value projections across ALL layers with pure PyTorch LoRA adapters."""
    for p in model.parameters():
        p.requires_grad = False

    count = 0
    for name, module in model.named_modules():
        if any(target in name for target in ["q_proj", "v_proj", "k_proj", "o_proj"]) and isinstance(module, torch.nn.Linear):
            parent_name = name.rsplit(".", 1)[0]
            child_name = name.rsplit(".", 1)[1]
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha))
            count += 1

    logger.info(f"Applied Pure PyTorch LoRA to {count} linear projections across all layers.")
    return model


def train_dpo_model(model, tokenizer, dataset, device: str, batch_size: int = 1, grad_accum: int = 4, beta: float = 0.1, lr: float = 1e-5, epochs: int = 1, lora_rank: int = 16, on_epoch_end=None):
    """Fine-tunes the base model using DPO loss in PyTorch with VRAM offloading & gradient accumulation.

    on_epoch_end(epoch_num, model) is invoked after each completed epoch (e.g. to sample rollouts
    and compute per-epoch validation metrics). The model is set back to train() mode afterwards.
    """
    logger.info(f"=== Starting Real DPO Fine-Tuning on GPU ({len(dataset):,} examples, batch_size={batch_size}, grad_accum={grad_accum}, lr={lr}, beta={beta}) ===")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 1. Precompute reference model logps (stored in CPU numpy arrays)
    logger.info("Precomputing reference model logps...")
    model.eval()
    ref_c_arr = np.zeros(len(dataset), dtype=np.float32)
    ref_r_arr = np.zeros(len(dataset), dtype=np.float32)

    ref_batch_size = max(4, min(16, batch_size * 8))
    ref_dataloader = DataLoader(dataset, batch_size=ref_batch_size, shuffle=False)

    with torch.inference_mode():
        for idx, batch in enumerate(tqdm(ref_dataloader, desc="Precomputing reference logps")):
            c_ids = batch["chosen_ids"].to(device)
            c_mask = batch["chosen_mask"].to(device)
            c_lbls = batch["chosen_labels"].to(device)

            r_ids = batch["rejected_ids"].to(device)
            r_mask = batch["rejected_mask"].to(device)
            r_lbls = batch["rejected_labels"].to(device)

            ref_c = compute_sequence_logps(model, c_ids, c_mask, c_lbls)
            ref_r = compute_sequence_logps(model, r_ids, r_mask, r_lbls)

            bs = c_ids.size(0)
            start_i = idx * ref_batch_size
            end_i = start_i + bs
            ref_c_arr[start_i:end_i] = ref_c.detach().cpu().numpy()
            ref_r_arr[start_i:end_i] = ref_r.detach().cpu().numpy()

            if (idx + 1) % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Wrap policy model with LoRA across ALL layers
    model = apply_pure_pytorch_lora(model, r=lora_rank, alpha=16.0)
    model.train()

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    total_loss = 0.0
    step = 0
    t0 = time.time()
    optimizer.zero_grad()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"DPO Training (Epoch {epoch+1}/{epochs})")
        for idx, batch in enumerate(pbar):
            c_ids = batch["chosen_ids"].to(device)
            c_mask = batch["chosen_mask"].to(device)
            c_lbls = batch["chosen_labels"].to(device)

            r_ids = batch["rejected_ids"].to(device)
            r_mask = batch["rejected_mask"].to(device)
            r_lbls = batch["rejected_labels"].to(device)

            bs = c_ids.size(0)
            ref_c_logps = torch.from_numpy(ref_c_arr[idx * bs : (idx + 1) * bs]).to(device)
            ref_r_logps = torch.from_numpy(ref_r_arr[idx * bs : (idx + 1) * bs]).to(device)

            policy_c_logps = compute_sequence_logps(model, c_ids, c_mask, c_lbls)
            policy_r_logps = compute_sequence_logps(model, r_ids, r_mask, r_lbls)

            pi_logratios = policy_c_logps - policy_r_logps
            ref_logratios = ref_c_logps - ref_r_logps

            logits = pi_logratios - ref_logratios
            loss = -F.logsigmoid(beta * logits).mean() / grad_accum
            loss.backward()

            if (idx + 1) % grad_accum == 0 or (idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            current_loss = loss.item() * grad_accum
            total_loss += current_loss
            step += 1
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "avg_loss": f"{total_loss/step:.4f}"})

            if step % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, model)
            model.train()

    t1 = time.time()
    logger.info(f"=== DPO Fine-Tuning Complete in {t1-t0:.2f}s! Final Loss: {total_loss/max(1, step):.4f} ===")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def sample_rollout_activations(model, tokenizer, sae, hook_layer: int, prompts: list[str], device: str, max_new_tokens: int = 128, tau: float = 0.01, seed: int = 0):
    """Samples paired stochastic text rollouts and returns (mean_freq, per_prompt_freq).

    per_prompt_freq is (n_prompts, d_sae) mean firing frequency per rollout. Each prompt index
    gets a fixed seed and the pre- and post-DPO passes iterate the same prompt list, so both draw
    identical sampling noise; the paired difference isolates the model's learned change rather
    than decoding noise. Per-prompt arrays are kept so the measurement noise floor and all
    statistics can be computed offline (no extra GPU passes).
    """
    model.eval()
    all_feature_means = []
    base_model = getattr(model, "model", getattr(model, "transformer", model))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    target_layer = base_model.layers[hook_layer] if hasattr(base_model, "layers") else model.layers[hook_layer]
    activations = []

    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        activations.append(act)

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        for i, p in enumerate(tqdm(prompts, desc="Sampling text rollouts")):
            inputs = tokenizer(p, max_length=256, truncation=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                torch.manual_seed(seed + i)
                torch.cuda.manual_seed(seed + i)
                gen_tokens = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=pad_id,
                )

            activations.clear()
            with torch.inference_mode():
                base_model(gen_tokens)

            if not activations:
                continue

            res_act = activations[0].squeeze(0)[len(inputs.input_ids[0]):]  # Generated tokens only
            if len(res_act) == 0:
                continue

            sae_acts = sae.encode(res_act.to(sae.device))  # (seq_len, d_sae)
            mean_freq = (sae_acts > tau).float().mean(dim=0).detach().cpu().numpy()
            all_feature_means.append(mean_freq)

            del gen_tokens, res_act, sae_acts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    if not all_feature_means:
        return None, None
    per_prompt_freq = np.stack(all_feature_means, axis=0)  # (n_prompts, d_sae)
    return per_prompt_freq.mean(axis=0), per_prompt_freq


def compute_reward_margin(model, tokenizer, examples, device: str, n: int = 200, seed: int = 0, batch_size: int = 8) -> tuple[float, float, int]:
    """Mean (logp_chosen - logp_rejected) over n held-out pairs.

    Positive control: if DPO learned the preference signal, this margin widens after training.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(examples), size=min(n, len(examples)), replace=False)
    subset = [examples[i] for i in idx]
    ds = DPODataset(subset, tokenizer)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    margins = []
    with torch.inference_mode():
        for batch in dl:
            c_ids = batch["chosen_ids"].to(device)
            c_mask = batch["chosen_mask"].to(device)
            c_lbls = batch["chosen_labels"].to(device)
            r_ids = batch["rejected_ids"].to(device)
            r_mask = batch["rejected_mask"].to(device)
            r_lbls = batch["rejected_labels"].to(device)
            c_lp = compute_sequence_logps(model, c_ids, c_mask, c_lbls)
            r_lp = compute_sequence_logps(model, r_ids, r_mask, r_lbls)
            margins.append((c_lp - r_lp).detach().cpu().numpy())
    margins = np.concatenate(margins)
    return float(margins.mean()), float(margins.std()), int(len(margins))


def compute_cluster_validation(delta_empirical_all: np.ndarray, cluster_ids, cluster_map, u_bar_global: np.ndarray, num_features: int) -> tuple[ValidationMetrics, np.ndarray, np.ndarray]:
    """Ranks feature clusters by predicted disparity and returns R^2/Pearson plus the ranked arrays."""
    sorted_cluster_indices = np.argsort(-np.abs(u_bar_global))
    if num_features > 0:
        sorted_cluster_indices = sorted_cluster_indices[:num_features]

    delta_predicted = []
    delta_empirical = []
    for col_idx in sorted_cluster_indices:
        cid = cluster_ids[col_idx]
        feats = cluster_map.clusters[cid]
        if not feats:
            continue
        delta_predicted.append(float(u_bar_global[col_idx]))
        delta_empirical.append(float(np.mean(delta_empirical_all[feats])))

    delta_pred_arr = np.array(delta_predicted, dtype=np.float64)
    delta_emp_arr = np.array(delta_empirical, dtype=np.float64)
    metrics = compute_prediction_validation_metrics(delta_pred_arr, delta_emp_arr)
    return metrics, delta_pred_arr, delta_emp_arr


def main():
    parser = argparse.ArgumentParser(description="Phase P4: 100% Real DPO Training & Rollout Validation")
    parser.add_argument("--config", type=str, default="configs/gemma2_2b_base.json", help="Path to JSON config")
    parser.add_argument("--num_features", type=int, default=50, help="Number of top feature clusters to evaluate (0 = all clusters)")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per GPU step")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for DPO training")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta regularization parameter")
    parser.add_argument("--train_samples", type=int, default=10000, help="Number of samples to train DPO on (must be in (0, dataset_size) so held-out eval prompts exist)")
    parser.add_argument("--eval_prompts", type=int, default=200, help="Number of evaluation prompts for rollout comparison")
    parser.add_argument("--epochs", type=int, default=1, help="Number of DPO training epochs")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens generated per rollout")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank for DPO fine-tuning")
    parser.add_argument("--margin_pairs", type=int, default=200, help="Number of held-out pairs for the reward-margin positive control")
    args = parser.parse_args()

    cfg = PipelineConfig.load_json(args.config)
    logger.info(f"=== Phase P4: 100% Real DPO Training & SAE Rollout Experiment for '{cfg.name}' ===")

    run_dir = cfg.output_dir
    summary_path = os.path.join(run_dir, "pdd_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"PDD summary file not found at '{summary_path}'. Please run PDD pipeline first.")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    subfolder = summary_data.get("checkpoint_subfolder")

    # 1. Load actual cached examples
    ex_path = os.path.join(subfolder, "examples.json")
    with open(ex_path, "r", encoding="utf-8") as f:
        ex_dicts = json.load(f)
    examples = [PreferenceExample.from_dict(d) for d in ex_dicts]

    if args.train_samples <= 0 or args.train_samples + args.eval_prompts > len(examples):
        raise ValueError(f"--train_samples ({args.train_samples}) + --eval_prompts ({args.eval_prompts}) must be within the {len(examples)} cached examples so eval prompts are held out")

    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(len(examples))
    train_indices = perm[: args.train_samples]
    eval_indices = perm[args.train_samples : args.train_samples + args.eval_prompts]
    train_examples = [examples[i] for i in train_indices]
    eval_examples = [examples[i] for i in eval_indices]
    eval_prompts = [ex.prompt for ex in eval_examples]

    output_dir = os.path.join(cfg.output_dir, "p4_validation")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Model & SAE
    logger.info("Loading Base Model and SAE for DPO Fine-Tuning & Rollout extraction...")
    model_backend = ModelBackend(cfg.model)
    model, tokenizer = model_backend.load()

    sae_backend = SAEBackend(cfg.sae)
    sae = sae_backend.load()

    # 3. Positive control (pre-DPO): reward margin on held-out pairs
    pre_margin_mean, pre_margin_std, n_margin = compute_reward_margin(model, tokenizer, eval_examples, cfg.model.device, n=args.margin_pairs, seed=cfg.seed)
    logger.info(f"Pre-DPO reward margin (held-out): {pre_margin_mean:.4f} +/- {pre_margin_std:.4f} (n={n_margin})")

    # 4. Capture Pre-DPO rollout SAE feature activations (paired sampling; per-prompt saved)
    logger.info(f"Sampling Pre-DPO text rollouts over {len(eval_prompts)} held-out evaluation prompts...")
    sft_act_mean, sft_per_prompt = sample_rollout_activations(model, tokenizer, sae, cfg.sae.layer, eval_prompts, cfg.model.device, tau=cfg.feature_conditioned.tau, seed=cfg.seed)
    np.save(os.path.join(output_dir, "per_prompt_pre.npy"), sft_per_prompt)

    # 5. Compute Global Feature Cluster Disparity u_bar_m once (TRAIN examples only; static dataset signal)
    clusters_path = os.path.join(subfolder, "clusters.json")
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)
    feat_to_cluster = {int(k): int(v) for k, v in clusters_data.get("feature_to_cluster", {}).items()}
    retained_clusters = {int(k): [int(x) for x in v] for k, v in clusters_data.get("clusters", {}).items()}
    cluster_map = FeatureClusterMap(clusters=retained_clusters, feature_to_cluster=feat_to_cluster)

    mmap_dir = os.path.join(subfolder, "matrices_mmap")
    npz_path = os.path.join(subfolder, "matrices.npz")
    if os.path.isdir(mmap_dir):
        mats = FeatureMatrices.load_mmap_dir(mmap_dir)
    elif os.path.exists(npz_path):
        mats = FeatureMatrices.load_npz(npz_path)
    else:
        raise FileNotFoundError(f"No feature matrices found in '{subfolder}'.")

    fc_pipeline = FeatureConditionedPipeline(cfg.feature_conditioned)
    fc_res = fc_pipeline.run(mats, cluster_map, seed=cfg.seed)
    u_bar_global = np.mean(fc_res.u_matrix[train_indices], axis=0)  # (K_r,) mean disparity over TRAIN examples only
    cluster_ids = sorted(cluster_map.clusters.keys())
    with open(os.path.join(output_dir, "cluster_ids.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_ids, f)
    np.save(os.path.join(output_dir, "u_bar_global.npy"), u_bar_global)

    # Per-feature predicted disparity over train examples (feature-level Spearman in offline analysis)
    tau = cfg.feature_conditioned.tau
    c_bin = mats.C_freq[train_indices] > tau
    r_bin = mats.R_freq[train_indices] > tau
    u_feature = np.asarray(c_bin.mean(axis=0) - r_bin.mean(axis=0)).ravel().astype(np.float32)
    del c_bin, r_bin
    gc.collect()
    np.save(os.path.join(output_dir, "u_feature.npy"), u_feature)

    # 6. Fine-Tune Model on DPO Loss, computing R^2 after every epoch (move SAE to CPU to free VRAM during training)
    logger.info(f"Training DPO model on all {len(train_examples):,} preference examples...")
    if hasattr(sae, "to"):
        sae.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dpo_dataset = DPODataset(train_examples, tokenizer)
    per_epoch_metrics = []

    def eval_epoch(epoch: int, current_model):
        if hasattr(sae, "to"):
            sae.to(cfg.sae.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Sampling Post-DPO text rollouts for epoch {epoch}/{args.epochs} over {len(eval_prompts)} held-out evaluation prompts...")
        dpo_act_mean, dpo_per_prompt = sample_rollout_activations(current_model, tokenizer, sae, cfg.sae.layer, eval_prompts, cfg.model.device, tau=cfg.feature_conditioned.tau, seed=cfg.seed)
        np.save(os.path.join(output_dir, f"per_prompt_post_epoch{epoch}.npy"), dpo_per_prompt)
        delta_empirical_all = dpo_act_mean - sft_act_mean  # mean(f(y_DPO)) - mean(f(y_SFT))
        logger.info(f"Empirical shift magnitude: max|Δ|={np.abs(delta_empirical_all).max():.2e}, mean|Δ|={np.abs(delta_empirical_all).mean():.2e}")

        # Full per-cluster empirical shift over ALL retained clusters (for offline top-k / negative control)
        delta_emp_full = np.full(len(cluster_ids), np.nan, dtype=np.float32)
        for k, cid in enumerate(cluster_ids):
            feats = cluster_map.clusters[cid]
            if feats:
                delta_emp_full[k] = float(np.mean(delta_empirical_all[feats]))
        np.save(os.path.join(output_dir, f"delta_emp_full_epoch{epoch}.npy"), delta_emp_full)
        np.save(os.path.join(output_dir, f"delta_all_epoch{epoch}.npy"), delta_empirical_all)

        metrics, _, _ = compute_cluster_validation(delta_empirical_all, cluster_ids, cluster_map, u_bar_global, args.num_features)
        margin_mean, margin_std, _ = compute_reward_margin(current_model, tokenizer, eval_examples, cfg.model.device, n=args.margin_pairs, seed=cfg.seed)
        per_epoch_metrics.append({"epoch": epoch, **asdict(metrics), "reward_margin": margin_mean, "reward_margin_std": margin_std})
        logger.info(f"Epoch {epoch}/{args.epochs} R^2 = {metrics.r2_score:.4f} | Pearson r: {metrics.pearson_r:.4f} | reward margin: {margin_mean:.4f} (pre={pre_margin_mean:.4f})")
        if hasattr(sae, "to"):
            sae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_dpo_model(
        model, tokenizer, dpo_dataset, cfg.model.device,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        beta=args.beta, lr=args.lr, epochs=args.epochs, lora_rank=args.lora_rank, on_epoch_end=eval_epoch
    )

    # 7. Save per-epoch R^2 track, reward margins, and final metrics
    with open(os.path.join(output_dir, "p4_r2_by_epoch.json"), "w", encoding="utf-8") as f:
        json.dump(per_epoch_metrics, f, indent=2)
    with open(os.path.join(output_dir, "reward_margin.json"), "w", encoding="utf-8") as f:
        json.dump({
            "pre_dpo": {"mean": pre_margin_mean, "std": pre_margin_std, "n": n_margin},
            "per_epoch": [{"epoch": m["epoch"], "mean": m["reward_margin"], "std": m["reward_margin_std"]} for m in per_epoch_metrics],
        }, f, indent=2)

    final_metrics = ValidationMetrics(**{k: v for k, v in per_epoch_metrics[-1].items() if k not in ("epoch", "reward_margin", "reward_margin_std")})
    metrics_file = os.path.join(output_dir, "p4_r2_metrics.json")
    final_metrics.save_json(metrics_file)

    track = ", ".join(f"ep{m['epoch']}: R2={m['r2_score']:.4f}" for m in per_epoch_metrics)
    logger.info(f"=== [Phase P4 100% Real DPO Validation Completed!] ===")
    logger.info(f"Per-epoch track -> {track}. Final R^2: {final_metrics.r2_score:.4f}. Saved to '{output_dir}'")


if __name__ == "__main__":
    main()
