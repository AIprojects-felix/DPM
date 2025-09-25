#!/usr/bin/env python3
"""Standalone evaluation script for the DPM model.

This script loads a configuration file and an optional checkpoint to evaluate the
model on the validation split. It reports basic metrics (diagnosis accuracy and
AUC for treatment) along with runtime.
"""
from __future__ import annotations

import argparse
import time
import logging
from typing import Dict

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, roc_auc_score

from dpm.models.dpm import DPM, DPMConfig
from dpm.data.dataset import PADTSDataset, DataConfig


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


def build_dataloader(cfg: Dict) -> DataLoader:
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
        val_len = max(1, int(0.2 * total_len))
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
        ds_val = PADTSDataset(manifest, mode="val", synthetic=False, data_cfg=data_cfg)

    batch_size = int(cfg["data"].get("batch_size", 4))
    num_workers = int(cfg["data"].get("num_workers", 2))

    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_val


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
    model = DPM(model_cfg).to(device)
    return model


def evaluate(model: DPM, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_diag_all, y_diag_pred = [], []
    y_treat_all, y_treat_score = [], []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Evaluate"):
            clinical = batch["clinical"].to(device, non_blocking=True)
            omics = batch["omics"].to(device, non_blocking=True)
            mri = batch["mri"].to(device, non_blocking=True)
            path = batch["pathology"].to(device, non_blocking=True)
            pm = {k: v.to(device, non_blocking=True) for k, v in batch["present_mask"].items()}

            diag_logits, treat_logits, risk_score = model.forward_with_logits(
                {
                    "clinical": clinical,
                    "omics": omics,
                    "mri": mri,
                    "pathology": path,
                },
                present_mask=pm,
            )
            p_diag = torch.softmax(diag_logits, dim=-1)
            pred = p_diag.argmax(dim=-1)
            y_diag_all.extend(batch["labels"]["diag"].cpu().tolist())
            y_diag_pred.extend(pred.cpu().tolist())

            y_treat_all.extend(batch["labels"]["treat"].cpu().tolist())
            y_treat_score.extend(torch.sigmoid(treat_logits).view(-1).cpu().tolist())

    diag_acc = accuracy_score(y_diag_all, y_diag_pred) if len(y_diag_all) > 0 else 0.0
    try:
        treat_auc = roc_auc_score(y_treat_all, y_treat_score)
    except Exception:
        treat_auc = float("nan")

    return {"diag_acc": float(diag_acc), "treat_auc": float(treat_auc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DPM model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint path to load model weights")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logging.info("Using CUDA")
    else:
        logging.info("Using CPU")

    dl_val = build_dataloader(cfg)
    model = build_model(cfg, device)

    if args.ckpt:
        logging.info(f"Loading checkpoint from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)

    t0 = time.time()
    metrics = evaluate(model, dl_val, device)
    dur = time.time() - t0

    table = PrettyTable()
    table.title = "Evaluation Summary"
    table.field_names = ["metric", "value"]
    table.add_row(["val_diag_acc", f"{metrics['diag_acc']:.4f}"])
    table.add_row(["val_treat_auc", f"{metrics['treat_auc']:.4f}"])
    table.add_row(["eval_time_s", f"{dur:.2f}"])
    logging.info("\n" + str(table))


if __name__ == "__main__":
    main()
