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
import json
import os
import sys
import time
from dataclasses import asdict
from typing import List, Optional, Set

# Kaggle/Colab compatibility: prevent broken preinstalled torchvision/torchaudio C++ ABI from crashing transformers
for _pkg in ("torchvision", "torchaudio"):
    try:
        __import__(_pkg)
    except Exception:
        sys.modules[_pkg] = None
        sys.modules[f"{_pkg}.io"] = None
        sys.modules[f"{_pkg}.ops"] = None

import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pdd.config import PipelineConfig
from pdd.data import PreferenceExample
from pdd.feature_clusters import FeatureClusterMap
from pdd.feature_matrices import FeatureMatrices
from pdd.logger import get_logger
from pdd.sae import ModelBackend, SAEBackend
from pdd.validation import ValidationMetrics, compute_prediction_validation_metrics

logger = get_logger("PDD.Exp.P4")


class DPODataset(Dataset):
    """PyTorch Dataset wrapper for preference pairs."""

    def __init__(self, examples: list[PreferenceExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._prompt_offset = self._leading_special_tokens()

    def _leading_special_tokens(self) -> int:
        """Count special tokens the tokenizer prepends to a plain string (<bos> for Gemma2, 0 for Qwen3)."""
        probe = self.tokenizer.encode("probe prompt text")
        n = 0
        while n < len(probe) and probe[n] in self.tokenizer.all_special_ids:
            n += 1
        return n

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
        c_labels[: int(c_enc.attention_mask.squeeze(0).argmax()) + p_len + self._prompt_offset] = -100
        c_labels[c_enc.attention_mask.squeeze(0) == 0] = -100

        r_labels = r_enc.input_ids.squeeze(0).clone()
        r_labels[: int(r_enc.attention_mask.squeeze(0).argmax()) + p_len + self._prompt_offset] = -100
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
    """Compute per-sequence log-probabilities using PyTorch fused C++ cross-entropy (zero 9GB transient buffers)."""
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :].contiguous()
        targets = labels[:, 1:].contiguous()

        loss_mask = (targets != -100)
        targets_clamped = targets.clone()
        targets_clamped[~loss_mask] = 0

        # PyTorch fused CUDA cross-entropy computes -log P(token) in-kernel with zero intermediate memory allocation
        bs, seq_len, vocab_size = logits.shape
        loss_flat = F.cross_entropy(logits.view(-1, vocab_size), targets_clamped.view(-1), reduction="none").view(bs, seq_len)
        per_token_logps = -loss_flat

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


def train_dpo_model(
    model, tokenizer, dataset, device: str,
    batch_size: int = 2, grad_accum: int = 32, beta: float = 0.1,
    lr: float = 1e-6, epochs: int = 1, lora_rank: int = 0,
    warmup_ratio: float = 0.1, on_epoch_end=None
):
    """Fine-tunes the base model using DPO loss in PyTorch with VRAM offloading & gradient accumulation.

    When lora_rank == 0, performs 100% Full-Parameter fine-tuning with gradient checkpointing (paper-exact).
    When lora_rank > 0, applies pure PyTorch LoRA adapters with specified rank.
    """
    mode_str = "Full-Parameter Fine-Tuning" if lora_rank == 0 else f"LoRA Fine-Tuning (rank={lora_rank})"
    eff_bs = batch_size * grad_accum
    logger.info(
        f"=== Starting Real DPO {mode_str} on GPU ({len(dataset):,} examples, "
        f"micro_bs={batch_size}, grad_accum={grad_accum}, eff_bs={eff_bs}, lr={lr:.2e}, beta={beta}) ==="
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 1. Precompute reference model logps (stored in CPU numpy arrays to prevent GPU OOM)
    logger.info("Precomputing reference model logps...")
    model.eval()
    ref_c_arr = np.zeros(len(dataset), dtype=np.float32)
    ref_r_arr = np.zeros(len(dataset), dtype=np.float32)

    ref_batch_size = max(4, min(8, batch_size * 2))
    ref_dataloader = DataLoader(dataset, batch_size=ref_batch_size, shuffle=False)

    with torch.inference_mode():
        pbar = tqdm(ref_dataloader, desc="Precomputing reference logps")
        for idx, batch in enumerate(pbar):
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

            if (idx + 1) % 2000 == 0:
                logger.info(f"Precomputed reference logps for {end_i:,} / {len(dataset):,} examples ({(end_i/len(dataset))*100:.1f}%)...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Setup Trainable Model: Full-Parameter or LoRA
    if lora_rank > 0:
        model = apply_pure_pytorch_lora(model, r=lora_rank, alpha=16.0)
    else:
        for p in model.parameters():
            p.requires_grad = True
        logger.info("Configured 100% Full-Parameter DPO Fine-Tuning (all weights trainable).")

    model.train()

    # enable_input_require_grads is needed for BOTH full and LoRA modes.
    # For LoRA: frozen base embeddings don't require grad; without this hook the backward
    # graph is severed at the embedding layer → loss.backward() raises "element 0 of tensors
    # does not require grad and does not have a grad_fn".
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception as e:
            logger.warning(f"enable_input_require_grads unsupported ({e}); continuing without input grads.")

    # gradient_checkpointing BREAKS LoRA: it detaches intermediate activations across
    # checkpointed segments, severing the path to LoRA adapter gradients.
    # Only enable for full fine-tuning (lora_rank == 0) where all weights require grad.
    if lora_rank == 0 and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for full fine-tuning.")
        except Exception as e:
            logger.warning(f"Gradient checkpointing unavailable ({e}); continuing without it.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        logger.info("Using 8-bit AdamW (bnb.optim.AdamW8bit) for low-VRAM 100% Full-Parameter Training.")
    except Exception:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        logger.info("Using standard AdamW optimizer.")

    # 3. Linear Warmup Scheduler (matching paper Table 8)
    total_steps = (len(dataloader) * epochs) // grad_accum
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.1, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    total_loss = 0.0
    accum_step = 0
    opt_step = 0
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

            accum_step += 1
            if accum_step % grad_accum == 0 or (idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

            current_loss = loss.item() * grad_accum
            total_loss += current_loss
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "avg_loss": f"{total_loss/accum_step:.4f}", "lr": f"{current_lr:.2e}"})

            if accum_step % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, model)
            model.train()

    t1 = time.time()
    logger.info(f"=== DPO Fine-Tuning Complete in {t1-t0:.2f}s! Final Loss: {total_loss/max(1, accum_step):.4f} ===")
def sample_rollout_activations(
    model, tokenizer, sae, hook_layer: int, prompts: list[str], device: str,
    max_new_tokens: int = 128, tau: float = 0.01, seed: int = 0, temperature: float = 0.0,
    batch_size: int = 16
):
    """Samples text rollouts in batches (10x faster) and returns (mean_freq, per_prompt_freq).

    temperature=0.0 uses greedy decoding (zero sampling noise, pure empirical shift measurement).
    temperature>0.0 uses stochastic sampling with top_p=0.9.
    """
    model.eval()
    all_feature_means = []
    base_model = getattr(model, "model", getattr(model, "transformer", model))
    
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    target_layer = base_model.layers[hook_layer] if hasattr(base_model, "layers") else model.layers[hook_layer]
    activations = []

    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        activations.append(act)

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        pbar = tqdm(range(0, len(prompts), batch_size), desc="Sampling text rollouts (batched)")
        for start_idx in pbar:
            batch_prompts = prompts[start_idx : start_idx + batch_size]
            inputs = tokenizer(batch_prompts, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
            
            with torch.inference_mode():
                torch.manual_seed(seed + start_idx)
                torch.cuda.manual_seed(seed + start_idx)
                if temperature is None or temperature <= 0.0:
                    gen_tokens = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=False, pad_token_id=pad_id,
                    )
                else:
                    gen_tokens = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=True, temperature=float(temperature), top_p=0.9, pad_token_id=pad_id,
                    )

            activations.clear()
            with torch.inference_mode():
                gen_mask = (gen_tokens != pad_id).long()
                base_model(gen_tokens, attention_mask=gen_mask)

            if not activations:
                continue

            full_acts = activations[0]
            inp_len = inputs.input_ids.shape[1]

            windows: List[torch.Tensor] = []
            lengths: List[int] = []
            for b in range(len(batch_prompts)):
                sample_gen_tokens = gen_tokens[b, inp_len:]
                valid_mask = (sample_gen_tokens != pad_id)
                num_valid = valid_mask.sum().item()
                if num_valid == 0:
                    continue
                windows.append(full_acts[b, inp_len : inp_len + num_valid])
                lengths.append(num_valid)

            if windows:
                max_len = max(lengths)
                d = windows[0].shape[-1]
                padded = torch.zeros(len(windows), max_len, d, dtype=windows[0].dtype, device=windows[0].device)
                for i, (w, ln) in enumerate(zip(windows, lengths)):
                    padded[i, :ln] = w
                sae_acts = sae.encode(padded.to(sae.device))
                if isinstance(sae_acts, tuple):
                    sae_acts = sae_acts[0]
                for i, ln in enumerate(lengths):
                    mean_freq = (sae_acts[i, :ln] > tau).float().mean(dim=0).detach().cpu().numpy()
                    all_feature_means.append(mean_freq)

            del gen_tokens, full_acts
    finally:
        handle.remove()
        tokenizer.padding_side = orig_padding_side

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


def compute_cluster_validation(delta_empirical_all: np.ndarray, cluster_ids, cluster_map, u_bar_global: np.ndarray, num_features: int, valid_ids: Optional[Set[int]] = None) -> tuple[ValidationMetrics, np.ndarray, np.ndarray]:
    """Ranks feature clusters by predicted disparity and returns R^2/Pearson plus the ranked arrays.

    ``valid_ids`` optionally limits validation to a subset of cluster ids — e.g. the
    clusters referenced by hypotheses passing the paper's B.1 viewer filters — so the R^2
    measures the predictions the paper actually presents (see load_validated_cluster_ids).
    """
    if valid_ids is not None:
        keep = [i for i, cid in enumerate(cluster_ids) if cid in valid_ids]
    else:
        keep = list(range(len(cluster_ids)))
    sorted_cluster_indices = sorted(keep, key=lambda i: -abs(u_bar_global[i]))
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


def load_validated_cluster_ids(run_dir: str, fc_cfg, cluster_map) -> Optional[Set[int]]:
    """Feature clusters T_m referenced by hypotheses passing the paper's B.1 viewer filters.

    Filters are read from config (never hardcoded):
      |T_m| >= fc_cfg.min_feat_cluster_size, n_k >= fc_cfg.min_data_cluster_size, SC == 1.
    Returns None (=> full-partition validation) when the hypotheses artifact is missing.
    """
    path = os.path.join(run_dir, "feature_conditioned_hypotheses.json")
    if not os.path.exists(path):
        logger.warning("feature_conditioned_hypotheses.json not found; validating on the full partition only.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    hypos = data.get("hypotheses", data) if isinstance(data, dict) else data
    valid: Set[int] = set()
    for h in hypos:
        m = h.get("m")
        if m is None:
            continue
        if int(h.get("t_m", 0) or 0) < fc_cfg.min_feat_cluster_size:
            continue
        if int(h.get("n_k", 0) or 0) < fc_cfg.min_data_cluster_size:
            continue
        if not bool(h.get("sign_consistent", False)):
            continue
        valid.add(int(m))
    n_in = sum(1 for m in valid if m in cluster_map.clusters)
    logger.info(
        f"Validated cluster set (|T_m|>={fc_cfg.min_feat_cluster_size}, "
        f"n_k>={fc_cfg.min_data_cluster_size}, SC=1): {n_in} clusters."
    )
    return valid


def main():
    parser = argparse.ArgumentParser(description="Phase P4: 100% Real DPO Training & Rollout Validation")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config (e.g. configs/qwen3_1.7b_base.json)")
    args = parser.parse_args()

    cfg = PipelineConfig.load_json(args.config)
    v = cfg.dpo_validation

    logger.info(f"=== Phase P4: 100% Real DPO Training & SAE Rollout Experiment for '{cfg.name}' ===")

    run_dir = cfg.output_dir
    summary_path = os.path.join(run_dir, "pdd_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"PDD summary file not found at '{summary_path}'. Please run PDD pipeline first.")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    subfolder = summary_data.get("checkpoint_subfolder")
    if subfolder and not os.path.exists(subfolder):
        # Fallback for cloud/portable environments (e.g. Kaggle/Colab) where absolute paths differ:
        # Check relative subfolder path inside cfg.checkpoint_dir
        base_sub = os.path.basename(subfolder.rstrip("/\\"))
        candidate = os.path.join(cfg.checkpoint_dir, base_sub)
        if os.path.exists(candidate):
            subfolder = candidate
        elif os.path.exists(cfg.checkpoint_dir):
            subfolder = cfg.checkpoint_dir

    # 1. Load actual cached examples (or fallback to HF dataset if the 1.5GB examples.json was not uploaded)
    # NOTE: Keep this fallback active for portable cloud/Kaggle runs where giant json files are omitted from Git.
    ex_path = os.path.join(subfolder, "examples.json") if subfolder else None
    if ex_path and os.path.exists(ex_path):
        with open(ex_path, "r", encoding="utf-8") as f:
            ex_dicts = json.load(f)
        examples = [PreferenceExample.from_dict(d) for d in ex_dicts]
    else:
        from pdd.data import DatasetLoader
        logger.info(f"Local 'examples.json' not found; loading preference dataset directly from Hugging Face Hub ({cfg.data.path})...")
        examples = DatasetLoader(cfg.data).load(use_checkpoint=False)

    train_samples = v.train_samples
    eval_prompts_count = v.eval_prompts

    if train_samples is None or train_samples <= 0 or train_samples >= len(examples):
        train_samples = len(examples) - eval_prompts_count
        logger.info(f"Using all available remaining data for DPO training: {train_samples:,} pairs (held-out eval prompts: {eval_prompts_count:,}).")

    if train_samples + eval_prompts_count > len(examples):
        raise ValueError(f"train_samples ({train_samples}) + eval_prompts ({eval_prompts_count}) exceeds total dataset size ({len(examples)})")

    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(len(examples))
    train_indices = perm[: train_samples]
    eval_indices = perm[train_samples : train_samples + eval_prompts_count]
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
    pre_margin_mean, pre_margin_std, n_margin = compute_reward_margin(model, tokenizer, eval_examples, cfg.model.device, n=200, seed=cfg.seed)
    logger.info(f"Pre-DPO reward margin (held-out): {pre_margin_mean:.4f} +/- {pre_margin_std:.4f} (n={n_margin})")

    # 4. Capture Pre-DPO rollout SAE feature activations (paired sampling; per-prompt saved)
    logger.info(f"Sampling Pre-DPO text rollouts over {len(eval_prompts)} held-out evaluation prompts (temp={v.temperature})...")
    sft_act_mean, sft_per_prompt = sample_rollout_activations(
        model, tokenizer, sae, cfg.sae.layer, eval_prompts, cfg.model.device,
        tau=cfg.feature_conditioned.tau, seed=cfg.seed, temperature=v.temperature,
    )
    np.save(os.path.join(output_dir, "per_prompt_pre.npy"), sft_per_prompt)

    # 5. Compute Global Feature Cluster Disparity u_bar_m once (TRAIN examples only; static dataset signal)
    clusters_path = os.path.join(subfolder, "clusters.json")
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)
    feat_to_cluster = {int(k): int(v) for k, v in clusters_data.get("feature_to_cluster", {}).items()}
    retained_clusters = {int(k): [int(x) for x in v] for k, v in clusters_data.get("clusters", {}).items()}
    cluster_map = FeatureClusterMap(clusters=retained_clusters, feature_to_cluster=feat_to_cluster)
    cluster_ids = sorted(cluster_map.clusters.keys())

    u_bar_path = os.path.join(output_dir, "u_bar_global.npy")
    if os.path.exists(u_bar_path):
        logger.info(f"Loading precomputed global cluster disparity from '{u_bar_path}' (0.01s)...")
        u_bar_global = np.load(u_bar_path)
    else:
        logger.info("Computing global feature cluster disparity from activation dataset...")
        mmap_dir = os.path.join(subfolder, "matrices_mmap")
        npz_path = os.path.join(subfolder, "matrices.npz")
        if os.path.isdir(mmap_dir):
            mats = FeatureMatrices.load_mmap_dir(mmap_dir)
        elif os.path.exists(npz_path):
            mats = FeatureMatrices.load_npz(npz_path)
        else:
            raise FileNotFoundError(f"No feature matrices found in '{subfolder}'.")

        tau = cfg.feature_conditioned.tau
        active_feats = sorted(list(cluster_map.feature_to_cluster.keys()))

        def compute_sparse_freq_mean(mat, indices, feat_cols, tau_val: float) -> np.ndarray:
            if hasattr(mat, "tocsr") or hasattr(mat, "indptr"):
                sub = mat[indices][:, feat_cols]
                data_gt = (sub.data > tau_val).astype(np.float32)
                sub_bin = sub.__class__((data_gt, sub.indices, sub.indptr), shape=sub.shape)
                return np.asarray(sub_bin.mean(axis=0)).ravel().astype(np.float32)
            else:
                total = np.zeros(len(feat_cols), dtype=np.float64)
                chunk_size = 20000
                for start in range(0, len(indices), chunk_size):
                    end = min(start + chunk_size, len(indices))
                    total += (mat[indices[start:end], :][:, feat_cols] > tau_val).sum(axis=0)
                return (total / max(1, len(indices))).astype(np.float32)

        c_mean = compute_sparse_freq_mean(mats.C_freq, train_indices, active_feats, tau)
        r_mean = compute_sparse_freq_mean(mats.R_freq, train_indices, active_feats, tau)
        feat_to_u = dict(zip(active_feats, c_mean - r_mean))

        u_bar_global = np.zeros(len(cluster_ids), dtype=np.float32)
        for k, cid in enumerate(cluster_ids):
            feats = cluster_map.clusters[cid]
            if feats:
                u_bar_global[k] = float(np.mean([feat_to_u[f] for f in feats if f in feat_to_u]))

        np.save(u_bar_path, u_bar_global)

    with open(os.path.join(output_dir, "cluster_ids.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_ids, f)

    # 5b. Feature clusters T_m referenced by hypotheses passing the paper's B.1 viewer
    # filters (|T_m| >= min_feat_cluster_size, n_k >= min_data_cluster_size, SC=1), read
    # from config — the set the author's viewer would actually present as predictions.
    # This hypothesis set is THE validation universe: R^2 is measured only over the
    # clusters the pipeline would actually present, not the whole Leiden partition.
    valid_cluster_ids = load_validated_cluster_ids(run_dir, cfg.feature_conditioned, cluster_map)

    # 6. Fine-Tune Model on DPO Loss, computing R^2 after every epoch (move SAE to CPU to free VRAM during training)
    logger.info(f"Training DPO model on all {len(train_examples):,} preference examples...")
    if hasattr(sae, "to"):
        sae.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dpo_dataset = DPODataset(train_examples, tokenizer, max_length=v.max_length)
    per_epoch_metrics = []

    def eval_epoch(epoch: int, current_model):
        if hasattr(sae, "to"):
            sae.to(cfg.sae.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Sampling Post-DPO text rollouts for epoch {epoch}/{v.epochs} over {len(eval_prompts)} held-out evaluation prompts (temp={v.temperature})...")
        dpo_act_mean, dpo_per_prompt = sample_rollout_activations(
            current_model, tokenizer, sae, cfg.sae.layer, eval_prompts, cfg.model.device,
            tau=cfg.feature_conditioned.tau, seed=cfg.seed, temperature=v.temperature,
        )
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

        metrics, _, _ = compute_cluster_validation(
            delta_empirical_all, cluster_ids, cluster_map, u_bar_global, v.num_features, valid_ids=valid_cluster_ids
        )
        margin_mean, margin_std, _ = compute_reward_margin(current_model, tokenizer, eval_examples, cfg.model.device, n=200, seed=cfg.seed)
        per_epoch_metrics.append({
            "epoch": epoch,
            **asdict(metrics),
            "reward_margin": margin_mean,
            "reward_margin_std": margin_std,
        })
        logger.info(
            f"Epoch {epoch}/{v.epochs} R^2 = {metrics.r2_score:.4f} | "
            f"Pearson r: {metrics.pearson_r:.4f} | reward margin: {margin_mean:.4f} (pre={pre_margin_mean:.4f})"
        )
        if hasattr(sae, "to"):
            sae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_dpo_model(
        model, tokenizer, dpo_dataset, cfg.model.device,
        batch_size=v.batch_size, grad_accum=v.grad_accum,
        beta=v.beta, lr=v.lr, epochs=v.epochs, lora_rank=v.lora_rank,
        warmup_ratio=v.warmup_ratio, on_epoch_end=eval_epoch,
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
    logger.info("=== [Phase P4 100% Real DPO Validation Completed!] ===")
    logger.info(f"Per-epoch track -> {track}. Final R^2: {final_metrics.r2_score:.4f}. Saved to '{output_dir}'")


if __name__ == "__main__":
    main()
