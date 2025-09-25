#!/usr/bin/env python3
"""Training script for the DPM multi-modal model.

Features:
- Config-driven training via YAML (see configs/default_config.yaml)
- Differential learning rates for pre-trained encoders (MRI/Pathology)
- Multi-task loss (Diagnosis CE, Treatment BCEWithLogits, Prognosis Cox PH)
- Mixed precision (AMP) support
- TensorBoard logging
- Checkpointing and simple validation metrics per epoch
- Runtime and memory usage reporting
"""
from __future__ import annotations

import argparse
import os
import time
import random
import logging
from pathlib import Path
from typing import Dict, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, roc_auc_score

from dpm.models.dpm import DPM, DPMConfig
from dpm.data.dataset import PADTSDataset, DataConfig
from dpm.utils.losses import DiagnosisLoss, TreatmentLoss, PrognosisLoss


# ------------------------------ Utilities ------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    data_cfg = DataConfig(
        clinical_input_dim=cfg["model"]["clinical_input_dim"],
        omics_feature_dim=cfg["model"]["omics_feature_dim"],
        omics_seq_len=cfg["data"].get("omics_seq_len", 128),
        mri_in_channels=cfg["model"]["mri_in_channels"],
        mri_size=tuple(cfg["data"].get("mri_size", [64, 128, 128])),
        image_size=cfg["data"].get("image_size", 224),
    )

    if cfg["data"].get("synthetic", True):
        total_len = int(cfg["data"].get("synthetic_len", 200))
        train_len = max(1, int(0.8 * total_len))
        val_len = max(1, total_len - train_len)
        ds_train = PADTSDataset(
            manifest_path=None,
            mode="train",
            synthetic=True,
            synthetic_len=train_len,
            data_cfg=data_cfg,
        )
        ds_val = PADTSDataset(
            manifest_path=None,
            mode="val",
            synthetic=True,
            synthetic_len=val_len,
            data_cfg=data_cfg,
        )
    else:
        manifest = cfg["data"]["manifest_path"]
        if manifest is None:
            raise ValueError("For real data, 'data.manifest_path' must be provided in config.")
        ds_train = PADTSDataset(manifest, mode="train", synthetic=False, data_cfg=data_cfg)
        ds_val = PADTSDataset(manifest, mode="val", synthetic=False, data_cfg=data_cfg)

    batch_size = int(cfg["data"].get("batch_size", 4))
    num_workers = int(cfg["data"].get("num_workers", 2))

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val


def build_model(cfg: Dict, device: torch.device) -> DPM:
    model_cfg = DPMConfig(
        embed_dim=cfg["model"]["embed_dim"],
        diag_num_classes=cfg["model"]["diag_num_classes"],
        clinical_input_dim=cfg["model"]["clinical_input_dim"],
        omics_feature_dim=cfg["model"]["omics_feature_dim"],
        mri_in_channels=cfg["model"]["mri_in_channels"],
        pathology_backbone=cfg["model"]["pathology_backbone"],
        fusion_layers=cfg["model"]["fusion_layers"],
        fusion_heads=cfg["model"]["fusion_heads"],
        fusion_dropout=cfg["model"]["fusion_dropout"],
    )
    model = DPM(model_cfg)
    model.to(device)
    return model


def build_optimizer(model: DPM, cfg: Dict) -> torch.optim.Optimizer:
    eta_main = float(cfg["optim"]["eta_main"])  # higher LR for randomly init modules
    eta_base = float(cfg["optim"]["eta_base"])  # lower LR for pre-trained encoders
    wd = float(cfg["optim"].get("weight_decay", 1e-4))

    base_modules = [model.mri_encoder, model.pathology_encoder]
    base_params = []
    for m in base_modules:
        base_params += list(p for p in m.parameters() if p.requires_grad)

    all_params = list(p for p in model.parameters() if p.requires_grad)
    main_params = [p for p in all_params if p not in base_params]

    optim = torch.optim.AdamW(
        [
            {"params": base_params, "lr": eta_base},
            {"params": main_params, "lr": eta_main},
        ],
        weight_decay=wd,
    )
    return optim


def train_one_epoch(
    model: DPM,
    dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    losses: Dict[str, nn.Module],
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    log_interval: int,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "diag": 0.0, "treat": 0.0, "prog": 0.0}
    n_steps = 0

    pbar = tqdm(dl, desc=f"Train Epoch {epoch}")
    for step, batch in enumerate(pbar, start=1):
        clinical = batch["clinical"].to(device, non_blocking=True)
        omics = batch["omics"].to(device, non_blocking=True)
        mri = batch["mri"].to(device, non_blocking=True)
        path = batch["pathology"].to(device, non_blocking=True)
        pm = {k: v.to(device, non_blocking=True) for k, v in batch["present_mask"].items()}

        y_diag = batch["labels"]["diag"].to(device)
        y_treat = batch["labels"]["treat"].to(device)
        y_time = batch["labels"]["time"].to(device)
        y_event = batch["labels"]["event"].to(device)

        optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and device.type == "cuda"
        ctx = torch.cuda.amp.autocast(enabled=amp_enabled) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
        with ctx:
            diag_logits, treat_logits, risk_score = model.forward_with_logits(
                {
                    "clinical": clinical,
                    "omics": omics,
                    "mri": mri,
                    "pathology": path,
                },
                present_mask=pm,
            )
            L_diag = losses["diag"](diag_logits, y_diag)
            L_treat = losses["treat"](treat_logits, y_treat)
            L_prog = losses["prog"](risk_score, y_time, y_event)
            lw = cfg_cache["loss_weights"]
            loss = lw["diag"] * L_diag + lw["treat"] * L_treat + lw["prog"] * L_prog

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running["loss"] += float(loss.detach().cpu().item())
        running["diag"] += float(L_diag.detach().cpu().item())
        running["treat"] += float(L_treat.detach().cpu().item())
        running["prog"] += float(L_prog.detach().cpu().item())
        n_steps += 1

        if step % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{running['loss']/n_steps:.4f}",
                "diag": f"{running['diag']/n_steps:.4f}",
                "treat": f"{running['treat']/n_steps:.4f}",
                "prog": f"{running['prog']/n_steps:.4f}",
            })

    # Averages
    avg = {k: v / max(n_steps, 1) for k, v in running.items()}
    return avg


def evaluate(model: DPM, dl: DataLoader, device: torch.device, epoch: int) -> Dict[str, float]:
    model.eval()
    y_diag_all, y_diag_pred = [], []
    y_treat_all, y_treat_score = [], []
    val_loss_diag = 0.0
    val_steps = 0

    with torch.no_grad():
        for batch in tqdm(dl, desc="Validate"):
            clinical = batch["clinical"].to(device, non_blocking=True)
            omics = batch["omics"].to(device, non_blocking=True)
            mri = batch["mri"].to(device, non_blocking=True)
            path = batch["pathology"].to(device, non_blocking=True)
            pm = {k: v.to(device, non_blocking=True) for k, v in batch["present_mask"].items()}

            y_diag = batch["labels"]["diag"].to(device)
            y_treat = batch["labels"]["treat"].to(device)

            diag_logits, treat_logits, risk_score = model.forward_with_logits(
                {
                    "clinical": clinical,
                    "omics": omics,
                    "mri": mri,
                    "pathology": path,
                },
                present_mask=pm,
            )
            val_loss_diag += nn.functional.cross_entropy(diag_logits, y_diag).item()
            val_steps += 1

            p_diag = torch.softmax(diag_logits, dim=-1)
            pred = p_diag.argmax(dim=-1)
            y_diag_all.extend(y_diag.cpu().tolist())
            y_diag_pred.extend(pred.cpu().tolist())

            y_treat_all.extend(y_treat.cpu().tolist())
            y_treat_score.extend(torch.sigmoid(treat_logits).view(-1).cpu().tolist())

    # Metrics
    diag_acc = accuracy_score(y_diag_all, y_diag_pred) if len(y_diag_all) > 0 else 0.0
    try:
        treat_auc = roc_auc_score(y_treat_all, y_treat_score)
    except Exception:
        treat_auc = float('nan')

    return {"diag_acc": float(diag_acc), "treat_auc": float(treat_auc)}


# Global cache for loss weights inside train loop without passing cfg repeatedly
cfg_cache: Dict = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DPM model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--save_ckpt",
        type=str,
        default="",
        help=(
            "Optional path template to save checkpoints. Use a string like 'ckpt_epoch_{epoch}.pt'. "
            "If provided without '{epoch}', the epoch number will be appended before file extension."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    global cfg_cache
    cfg_cache = cfg

    setup_logging()
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logging.info("Using CUDA")
    else:
        logging.info("Using CPU")

    dl_train, dl_val = build_dataloaders(cfg)
    model = build_model(cfg, device)

    # Optional multi-GPU support (DataParallel) for simplicity
    use_multi_gpu = torch.cuda.device_count() > 1
    if device.type == "cuda" and use_multi_gpu:
        logging.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    optimizer = build_optimizer(model, cfg)
    if device.type == "cuda" and use_multi_gpu:
        # Wrap after building optimizer so parameter groups are created on the base model
        model = torch.nn.DataParallel(model)

    epochs = int(cfg["optim"].get("epochs", 3))
    log_interval = int(cfg.get("log_interval", 10))
    use_amp = bool(cfg["optim"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    losses = {
        "diag": DiagnosisLoss(),
        "treat": TreatmentLoss(),
        "prog": PrognosisLoss(),
    }

    best_acc = -1.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        e_start = time.time()
        train_stats = train_one_epoch(model, dl_train, optimizer, losses, device, scaler, log_interval, epoch)
        val_stats = evaluate(model, dl_val, device, epoch)
        e_dur = time.time() - e_start

        # Memory usage (GPU if available)
        if device.type == "cuda":
            mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            mem = 0.0

        # Pretty print epoch summary
        table = PrettyTable()
        table.title = f"Epoch {epoch} Summary"
        table.field_names = ["metric", "value"]
        table.add_row(["train_loss", f"{train_stats['loss']:.4f}"])
        table.add_row(["train_diag", f"{train_stats['diag']:.4f}"])
        table.add_row(["train_treat", f"{train_stats['treat']:.4f}"])
        table.add_row(["train_prog", f"{train_stats['prog']:.4f}"])
        table.add_row(["val_diag_acc", f"{val_stats['diag_acc']:.4f}"])
        table.add_row(["val_treat_auc", f"{val_stats['treat_auc']:.4f}"])
        table.add_row(["epoch_time_s", f"{e_dur:.2f}"])
        table.add_row(["gpu_mem_GB", f"{mem:.2f}"])
        logging.info("\n" + str(table))

        # Reset CUDA memory stat for next epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Optional checkpoint saving (no default directories used)
        if args.save_ckpt:
            # Prepare checkpoint dict
            ckpt = {
                "model": model.state_dict() if not isinstance(model, torch.nn.DataParallel) else model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": cfg,
            }
            path_tmpl = args.save_ckpt
            if "{epoch}" in path_tmpl:
                ckpt_path = path_tmpl.format(epoch=epoch)
            else:
                # Insert epoch before extension
                p = Path(path_tmpl)
                if p.suffix:
                    ckpt_path = str(p.with_name(p.stem + f"_epoch_{epoch}" + p.suffix))
                else:
                    ckpt_path = str(p) + f"_epoch_{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            logging.info(f"Saved checkpoint to {ckpt_path}")

    t_dur = time.time() - t0
    logging.info(f"Training completed in {t_dur:.2f} seconds.")
    # No TensorBoard writer used; training complete


if __name__ == "__main__":
    main()
