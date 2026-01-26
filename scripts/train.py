#!/usr/bin/env python3
"""Training script for DPM multi-modal temporal model.

This script supports:
- Multi-task learning with configurable loss weights
- Differential learning rates for pre-trained vs randomly initialized components
- Cosine annealing learning rate scheduler with warmup
- Early stopping based on validation loss
- Optional pre-trained weight loading for MRI and Pathology encoders
- Automatic mixed precision (AMP) training

Usage:
    python scripts/train.py --config configs/default_config.yaml
    python scripts/train.py --config configs/default_config.yaml --save_ckpt model.pt
"""
from __future__ import annotations

import argparse
import os
import time
import random
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score

from dpm.models.dpm import DPM, DPMConfig
from dpm.data.dataset import PADTSDataset, DataConfig
from dpm.data.collate import collate_patient_batch
from dpm.utils.losses import DiagnosisLoss, TreatmentLoss, PrognosisLoss, StagingLoss


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
        ds_train = PADTSDataset(None, mode="train", synthetic=True, synthetic_len=train_len, data_cfg=data_cfg)
        ds_val = PADTSDataset(None, mode="val", synthetic=True, synthetic_len=val_len, data_cfg=data_cfg)
    else:
        manifest = cfg["data"]["manifest_path"]
        ds_train = PADTSDataset(manifest, mode="train", synthetic=False, data_cfg=data_cfg)
        ds_val = PADTSDataset(manifest, mode="val", synthetic=False, data_cfg=data_cfg)

    batch_size = int(cfg["data"].get("batch_size", 4))
    num_workers = int(cfg["data"].get("num_workers", 0))

    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_patient_batch
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_patient_batch
    )
    return dl_train, dl_val


def build_model(cfg: Dict, device: torch.device) -> DPM:
    """Build the DPM model and optionally load pre-trained weights.

    Pre-trained weights for MRI (MRI-PTPCa) and Pathology (CONCH) encoders
    are loaded if paths are specified in the config. If paths are not provided
    or files don't exist, the model uses random initialization.

    Args:
        cfg: Configuration dictionary.
        device: Target device for the model.

    Returns:
        Initialized DPM model.
    """
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
        temporal_layers=cfg["model"].get("temporal_layers", 4),
        temporal_heads=cfg["model"].get("temporal_heads", 8),
        temporal_dropout=cfg["model"].get("temporal_dropout", 0.1),
        max_time_gap=cfg["model"].get("max_time_gap", 1825.0),
        isup_num_classes=cfg["model"].get("isup_num_classes", 5),
        ipss_num_classes=cfg["model"].get("ipss_num_classes", 3),
        nih_cpsi_num_classes=cfg["model"].get("nih_cpsi_num_classes", 3),
        num_treatments=cfg["model"].get("num_treatments", 10),
    )
    model = DPM(model_cfg)

    # Load pre-trained weights if specified
    pretrained_cfg = cfg["model"].get("pretrained", {})
    if pretrained_cfg:
        # MRI-PTPCa weights
        mri_weights = pretrained_cfg.get("mri_ptpca_weights")
        if mri_weights:
            loaded = model.mri_encoder.load_pretrained_mri_ptpca_weights(mri_weights)
            if loaded:
                logging.info(f"Loaded MRI-PTPCa weights from {mri_weights}")
            else:
                logging.info("MRI encoder using random initialization")

        # CONCH weights
        conch_weights = pretrained_cfg.get("conch_weights")
        if conch_weights:
            loaded = model.pathology_encoder.load_pretrained_conch_weights(conch_weights)
            if loaded:
                logging.info(f"Loaded CONCH weights from {conch_weights}")
            else:
                logging.info("Pathology encoder using random initialization")

    model.to(device)
    return model


def build_optimizer(model: DPM, cfg: Dict) -> torch.optim.Optimizer:
    eta_main = float(cfg["optim"]["eta_main"])
    eta_base = float(cfg["optim"]["eta_base"])
    wd = float(cfg["optim"].get("weight_decay", 1e-4))

    base_modules = [model.mri_encoder, model.pathology_encoder]
    base_params = []
    for m in base_modules:
        base_params += list(p for p in m.parameters() if p.requires_grad)

    all_params = list(p for p in model.parameters() if p.requires_grad)
    main_params = [p for p in all_params if not any(p is bp for bp in base_params)]

    optim = torch.optim.AdamW([
        {"params": base_params, "lr": eta_base},
        {"params": main_params, "lr": eta_main},
    ], weight_decay=wd)
    return optim


cfg_cache: Dict = {}


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict,
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build learning rate scheduler based on config.

    Supports:
    - cosine: Cosine annealing with optional warmup
    - step: Step decay (not implemented in this example)
    - none: No scheduling

    Args:
        optimizer: The optimizer to schedule.
        cfg: Configuration dictionary.
        steps_per_epoch: Number of training steps per epoch.

    Returns:
        Learning rate scheduler or None.
    """
    scheduler_type = cfg["optim"].get("scheduler", "none")
    if scheduler_type == "none":
        return None

    epochs = int(cfg["optim"].get("epochs", 50))
    warmup_epochs = int(cfg["optim"].get("warmup_epochs", 5))

    if scheduler_type == "cosine":
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler

    else:
        logging.warning(f"Unknown scheduler type: {scheduler_type}, using none")
        return None


class EarlyStopping:
    """Early stopping handler to prevent overfitting.

    Monitors a metric (default: validation loss) and stops training
    if no improvement is seen for `patience` epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: "min" for loss (lower is better), "max" for metrics.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, value: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value.
            epoch: Current epoch number.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_one_epoch(
    model: DPM,
    dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    losses: Dict[str, nn.Module],
    device: torch.device,
    scaler,
    log_interval: int,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running = {"loss": 0.0, "diag": 0.0, "treat": 0.0, "prog": 0.0, "isup": 0.0, "ipss": 0.0, "nih_cpsi": 0.0}
    n_steps = 0
    lw = cfg_cache["loss_weights"]

    pbar = tqdm(dl, desc=f"Train Epoch {epoch}")
    for step, batch in enumerate(pbar, start=1):
        # Move to device
        x = {
            "clinical": batch["clinical"].to(device),
            "omics": batch["omics"].to(device),
            "mri": batch["mri"].to(device),
            "pathology": batch["pathology"].to(device),
        }
        pm = {k: v.to(device) for k, v in batch["present_mask"].items()}
        time_gaps = batch["time_gaps"].to(device)
        visit_mask = batch["visit_mask"].to(device)

        labels = {k: v.to(device) for k, v in batch["labels"].items()}

        optimizer.zero_grad(set_to_none=True)

        amp_enabled = scaler is not None and device.type == "cuda"
        ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else torch.amp.autocast("cpu", enabled=False)

        with ctx:
            outputs = model.forward_with_logits(x, pm, time_gaps, visit_mask)

            L_diag = losses["diag"](outputs["diag"], labels["diag"])
            L_treat = losses["treat"](outputs["treat"], labels["treat"])
            L_prog = losses["prog"](outputs["risk"], labels["time"], labels["event"])
            L_isup = losses["isup"](outputs["isup"], labels["isup"])
            L_ipss = losses["ipss"](outputs["ipss"], labels["ipss"])
            L_nih_cpsi = losses["nih_cpsi"](outputs["nih_cpsi"], labels["nih_cpsi"])

            loss = (
                lw["diag"] * L_diag +
                lw["treat"] * L_treat +
                lw["prog"] * L_prog +
                lw.get("isup", 0.5) * L_isup +
                lw.get("ipss", 0.5) * L_ipss +
                lw.get("nih_cpsi", 0.5) * L_nih_cpsi
            )

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running["loss"] += float(loss.item())
        running["diag"] += float(L_diag.item())
        running["treat"] += float(L_treat.item())
        running["prog"] += float(L_prog.item())
        running["isup"] += float(L_isup.item())
        running["ipss"] += float(L_ipss.item())
        running["nih_cpsi"] += float(L_nih_cpsi.item())
        n_steps += 1

        if step % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{running['loss']/n_steps:.4f}",
                "diag": f"{running['diag']/n_steps:.4f}",
            })

    return {k: v / max(n_steps, 1) for k, v in running.items()}


def evaluate(
    model: DPM,
    dl: DataLoader,
    losses: Dict[str, nn.Module],
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Evaluate the model on validation data.

    Computes both accuracy metrics and validation loss for early stopping.

    Args:
        model: The DPM model.
        dl: Validation DataLoader.
        losses: Dictionary of loss functions.
        device: Target device.
        epoch: Current epoch number.

    Returns:
        Dictionary with validation metrics including loss and accuracy.
    """
    model.eval()
    y_diag_all, y_diag_pred = [], []
    running_loss = 0.0
    n_batches = 0
    lw = cfg_cache["loss_weights"]

    with torch.no_grad():
        for batch in tqdm(dl, desc="Validate"):
            x = {
                "clinical": batch["clinical"].to(device),
                "omics": batch["omics"].to(device),
                "mri": batch["mri"].to(device),
                "pathology": batch["pathology"].to(device),
            }
            pm = {k: v.to(device) for k, v in batch["present_mask"].items()}
            time_gaps = batch["time_gaps"].to(device)
            visit_mask = batch["visit_mask"].to(device)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}

            outputs = model.forward_with_logits(x, pm, time_gaps, visit_mask)
            pred = outputs["diag"].argmax(dim=-1)

            # Compute validation loss
            L_diag = losses["diag"](outputs["diag"], labels["diag"])
            L_treat = losses["treat"](outputs["treat"], labels["treat"])
            L_prog = losses["prog"](outputs["risk"], labels["time"], labels["event"])
            L_isup = losses["isup"](outputs["isup"], labels["isup"])
            L_ipss = losses["ipss"](outputs["ipss"], labels["ipss"])
            L_nih_cpsi = losses["nih_cpsi"](outputs["nih_cpsi"], labels["nih_cpsi"])

            loss = (
                lw["diag"] * L_diag +
                lw["treat"] * L_treat +
                lw["prog"] * L_prog +
                lw.get("isup", 0.5) * L_isup +
                lw.get("ipss", 0.5) * L_ipss +
                lw.get("nih_cpsi", 0.5) * L_nih_cpsi
            )
            running_loss += loss.item()
            n_batches += 1

            y_diag_all.extend(batch["labels"]["diag"].cpu().tolist())
            y_diag_pred.extend(pred.cpu().tolist())

    diag_acc = accuracy_score(y_diag_all, y_diag_pred) if len(y_diag_all) > 0 else 0.0
    val_loss = running_loss / max(n_batches, 1)

    return {"diag_acc": float(diag_acc), "val_loss": float(val_loss)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DPM model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--save_ckpt", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    global cfg_cache
    cfg_cache = cfg

    setup_logging()
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Log configuration summary
    logging.info(f"Model config: embed_dim={cfg['model']['embed_dim']}, "
                 f"fusion_layers={cfg['model']['fusion_layers']}, "
                 f"temporal_layers={cfg['model'].get('temporal_layers', 4)}")

    dl_train, dl_val = build_dataloaders(cfg)
    model = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg)

    epochs = int(cfg["optim"].get("epochs", 50))
    log_interval = int(cfg.get("log_interval", 10))
    use_amp = bool(cfg["optim"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    # Build learning rate scheduler
    scheduler = build_scheduler(optimizer, cfg, len(dl_train))
    if scheduler:
        logging.info(f"Using {cfg['optim'].get('scheduler', 'cosine')} learning rate scheduler")

    # Initialize early stopping
    early_stop_cfg = cfg["optim"].get("early_stopping", {})
    early_stopper = None
    if early_stop_cfg.get("enabled", False):
        patience = early_stop_cfg.get("patience", 10)
        early_stopper = EarlyStopping(patience=patience, mode="min")
        logging.info(f"Early stopping enabled with patience={patience}")

    losses = {
        "diag": DiagnosisLoss(),
        "treat": TreatmentLoss(),
        "prog": PrognosisLoss(),
        "isup": StagingLoss(ignore_index=-1),
        "ipss": StagingLoss(ignore_index=-1),
        "nih_cpsi": StagingLoss(ignore_index=-1),
    }

    best_val_loss = float("inf")
    best_model_state = None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        e_start = time.time()
        train_stats = train_one_epoch(model, dl_train, optimizer, losses, device, scaler, log_interval, epoch)
        val_stats = evaluate(model, dl_val, losses, device, epoch)
        e_dur = time.time() - e_start

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        table = PrettyTable()
        table.title = f"Epoch {epoch} Summary"
        table.field_names = ["metric", "value"]
        table.add_row(["train_loss", f"{train_stats['loss']:.4f}"])
        table.add_row(["train_diag", f"{train_stats['diag']:.4f}"])
        table.add_row(["train_treat", f"{train_stats['treat']:.4f}"])
        table.add_row(["train_prog", f"{train_stats['prog']:.4f}"])
        table.add_row(["train_isup", f"{train_stats['isup']:.4f}"])
        table.add_row(["train_ipss", f"{train_stats['ipss']:.4f}"])
        table.add_row(["train_nih_cpsi", f"{train_stats['nih_cpsi']:.4f}"])
        table.add_row(["val_loss", f"{val_stats['val_loss']:.4f}"])
        table.add_row(["val_diag_acc", f"{val_stats['diag_acc']:.4f}"])
        table.add_row(["learning_rate", f"{current_lr:.6f}"])
        table.add_row(["epoch_time_s", f"{e_dur:.2f}"])
        logging.info("\n" + str(table))

        # Save best model
        if val_stats["val_loss"] < best_val_loss:
            best_val_loss = val_stats["val_loss"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logging.info(f"New best model at epoch {epoch} with val_loss={best_val_loss:.4f}")

        # Save checkpoint per epoch if requested
        if args.save_ckpt:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": cfg,
                "val_loss": val_stats["val_loss"],
            }
            if scheduler is not None:
                ckpt["scheduler"] = scheduler.state_dict()
            p = Path(args.save_ckpt)
            ckpt_path = str(p.with_name(p.stem + f"_epoch_{epoch}" + p.suffix)) if p.suffix else str(p) + f"_epoch_{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            logging.info(f"Saved checkpoint to {ckpt_path}")

        # Check early stopping
        if early_stopper is not None:
            if early_stopper(val_stats["val_loss"], epoch):
                logging.info(f"Early stopping triggered at epoch {epoch}. "
                             f"Best epoch was {early_stopper.best_epoch} with val_loss={early_stopper.best_value:.4f}")
                break

    # Save best model at the end
    if best_model_state is not None and args.save_ckpt:
        p = Path(args.save_ckpt)
        best_path = str(p.with_name(p.stem + "_best" + p.suffix)) if p.suffix else str(p) + "_best.pt"
        torch.save({"model": best_model_state, "config": cfg, "val_loss": best_val_loss}, best_path)
        logging.info(f"Saved best model to {best_path}")

    logging.info(f"Training completed in {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":
    main()
