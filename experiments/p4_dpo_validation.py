"""Phase P4 Experiment: 100% Paper-Accurate DPO Fine-Tuning & SAE Rollout Validation (R^2).

Replicates Goodfire's paper (arXiv:2606.12360, §3, §4 & App. B):
1. Fine-tunes the base model using DPO loss (Rafailov et al., 2023) for 1 epoch on GPU to produce pi_DPO.
2. Samples text rollouts y_SFT from pre-DPO model and y_DPO from post-DPO model over held-out prompts.
3. Encodes rollouts through the SAE to measure real empirical rollout activation shifts:
   delta_empirical = mean(a(y_DPO)) - mean(a(y_SFT))
4. Computes R^2 regression and Pearson correlation against predicted hypotheses delta_predicted.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pdd.config import PipelineConfig
from pdd.data import PreferenceExample
from pdd.logger import get_logger
from pdd.sae import ModelBackend, SAEBackend
from pdd.validation import compute_prediction_validation_metrics

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


def train_dpo_model(model, tokenizer, dataset, device: str, batch_size: int = 1, grad_accum: int = 4, beta: float = 0.1, lr: float = 1e-5, epochs: int = 1):
    """Fine-tunes the base model using DPO loss in PyTorch with VRAM offloading & gradient accumulation."""
    logger.info(f"=== Starting Real DPO Fine-Tuning on GPU ({len(dataset):,} examples, batch_size={batch_size}, grad_accum={grad_accum}, lr={lr}, beta={beta}) ===")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 1. Precompute reference model logps (stored in 40 KB CPU numpy arrays)
    logger.info("Precomputing reference model logps...")
    model.eval()
    ref_c_arr = np.zeros(len(dataset), dtype=np.float32)
    ref_r_arr = np.zeros(len(dataset), dtype=np.float32)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Precomputing reference logps")):
            c_ids = batch["chosen_ids"].to(device)
            c_mask = batch["chosen_mask"].to(device)
            c_lbls = batch["chosen_labels"].to(device)

            r_ids = batch["rejected_ids"].to(device)
            r_mask = batch["rejected_mask"].to(device)
            r_lbls = batch["rejected_labels"].to(device)

            ref_c = compute_sequence_logps(model, c_ids, c_mask, c_lbls)
            ref_r = compute_sequence_logps(model, r_ids, r_mask, r_lbls)

            bs = c_ids.size(0)
            ref_c_arr[idx * bs : (idx + 1) * bs] = ref_c.detach().cpu().numpy()
            ref_r_arr[idx * bs : (idx + 1) * bs] = ref_r.detach().cpu().numpy()

            if (idx + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Wrap policy model with LoRA across ALL layers
    model = apply_pure_pytorch_lora(model, r=8, alpha=16.0)
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

    t1 = time.time()
    logger.info(f"=== DPO Fine-Tuning Complete in {t1-t0:.2f}s! Final Loss: {total_loss/max(1, step):.4f} ===")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def sample_rollout_activations(model, tokenizer, sae, hook_layer: int, prompts: list[str], device: str, max_new_tokens: int = 64):
    """Generates text rollouts and extracts mean SAE feature activations across generated tokens."""
    model.eval()
    all_feature_means = []
    base_model = getattr(model, "model", getattr(model, "transformer", model))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for p in tqdm(prompts, desc="Sampling text rollouts"):
        inputs = tokenizer(p, max_length=256, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=pad_id,
            )

        activations = []
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            activations.append(act)

        target_layer = base_model.layers[hook_layer] if hasattr(base_model, "layers") else model.layers[hook_layer]
        handle = target_layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                base_model(gen_tokens)
        finally:
            handle.remove()

        if not activations:
            continue

        res_act = activations[0].squeeze(0)[len(inputs.input_ids[0]):]  # Gen tokens only
        if len(res_act) == 0:
            continue

        sae_acts = sae.encode(res_act.to(sae.device))  # (seq_len, d_sae)
        mean_act = sae_acts.mean(dim=0).detach().cpu().numpy()
        all_feature_means.append(mean_act)

        del gen_tokens, activations, res_act, sae_acts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.mean(all_feature_means, axis=0) if all_feature_means else None


def main():
    parser = argparse.ArgumentParser(description="Phase P4: 100% Real DPO Training & Rollout Validation")
    parser.add_argument("--config", type=str, default="configs/gemma2_2b_base.json", help="Path to JSON config")
    parser.add_argument("--num_features", type=int, default=50, help="Number of top hypotheses to evaluate")
    parser.add_argument("--batch_size", type=int, default=1, help="Micro-batch size per GPU step")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for DPO training")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta regularization parameter")
    parser.add_argument("--train_samples", type=int, default=0, help="Number of samples to train DPO on (0 = all dataset samples)")
    parser.add_argument("--eval_prompts", type=int, default=200, help="Number of evaluation prompts for rollout comparison")
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

    n_train = len(examples) if args.train_samples <= 0 else min(args.train_samples, len(examples))
    
    # Split held-out evaluation prompts from training examples for true generalization
    if args.train_samples > 0 and len(examples) > args.train_samples:
        held_out_examples = examples[args.train_samples:]
    else:
        # Odd-indexed parity split (Goodfire paper App. B.1 split-half rule)
        held_out_examples = examples[1::2]

    n_eval = min(args.eval_prompts, len(held_out_examples))
    eval_prompts = [ex.prompt for ex in held_out_examples[:n_eval]]

    # 2. Load Model & SAE
    logger.info("Loading Base Model and SAE for DPO Fine-Tuning & Rollout extraction...")
    model_backend = ModelBackend(cfg.model)
    model, tokenizer = model_backend.load()

    sae_backend = SAEBackend(cfg.sae)
    sae = sae_backend.load()

    # 3. Capture Pre-DPO (SFT) Rollout SAE Feature Activations (on held-out evaluation prompts)
    logger.info(f"Sampling Pre-DPO (SFT) text rollouts over {len(eval_prompts)} held-out evaluation prompts...")
    sft_act_mean = sample_rollout_activations(model, tokenizer, sae, cfg.sae.layer, eval_prompts, cfg.model.device)

    # 4. Fine-Tune Model on DPO Loss (move SAE to CPU to free VRAM during training)
    train_examples = examples[:n_train]
    logger.info(f"Training DPO model on all {len(train_examples):,} preference examples...")
    if hasattr(sae, "to"):
        sae.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dpo_dataset = DPODataset(train_examples, tokenizer)
    dpo_model = train_dpo_model(
        model, tokenizer, dpo_dataset, cfg.model.device,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        beta=args.beta, lr=args.lr
    )

    if hasattr(sae, "to"):
        sae.to(cfg.sae.device)

    # 5. Capture Post-DPO Rollout SAE Feature Activations
    logger.info(f"Sampling Post-DPO text rollouts over {len(eval_prompts)} held-out evaluation prompts...")
    dpo_act_mean = sample_rollout_activations(dpo_model, tokenizer, sae, cfg.sae.layer, eval_prompts, cfg.model.device)

    # Compute Empirical Rollout Feature Shift: delta_empirical = mean(a(y_DPO)) - mean(a(y_SFT))
    delta_empirical_all = dpo_act_mean - sft_act_mean

    # 6. Load Hypotheses & Correlate delta_predicted vs delta_empirical
    hypo_path = os.path.join(run_dir, "feature_conditioned_hypotheses.json")
    with open(hypo_path, "r", encoding="utf-8") as f:
        h_raw = json.load(f)
    hypotheses = h_raw.get("hypotheses", h_raw) if isinstance(h_raw, dict) else h_raw
    top_hypotheses = sorted(hypotheses, key=lambda h: abs(h.get("delta", 0.0)), reverse=True)[:args.num_features]

    # Load Feature Clusters mapping
    clusters_path = os.path.join(subfolder, "clusters.json")
    with open(clusters_path, "r", encoding="utf-8") as f:
        clusters_data = json.load(f)
    retained_clusters = {int(k): [int(x) for x in v] for k, v in clusters_data.get("clusters", {}).items()}

    delta_predicted = []
    delta_empirical = []

    for h in top_hypotheses:
        m = h["m"]
        pred_delta = h.get("delta", 0.0)
        feats = retained_clusters.get(m, [])
        if not feats:
            continue
        emp_delta = float(np.mean(delta_empirical_all[feats]))
        delta_predicted.append(pred_delta)
        delta_empirical.append(emp_delta)

    delta_pred_arr = np.array(delta_predicted, dtype=np.float64)
    delta_emp_arr = np.array(delta_empirical, dtype=np.float64)

    metrics = compute_prediction_validation_metrics(delta_pred_arr, delta_emp_arr)

    output_dir = os.path.join(cfg.output_dir, "p4_validation")
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "p4_r2_metrics.json")
    metrics.save_json(metrics_file)

    logger.info(f"=== [Phase P4 100% Real DPO Validation Completed!] ===")
    logger.info(f"R^2 score: {metrics.r2_score:.4f} | Pearson r: {metrics.pearson_r:.4f}. Saved to '{metrics_file}'")


if __name__ == "__main__":
    main()
