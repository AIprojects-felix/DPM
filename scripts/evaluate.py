#!/usr/bin/env python3
"""Evaluation script for DPM model.

Computes comprehensive metrics as described in the paper:
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- ECE (Expected Calibration Error)
- Brier Score
- Accuracy
- C-index (for survival/prognosis)
"""
from __future__ import annotations

import argparse
import time
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from dpm.models.dpm import DPM, DPMConfig
from dpm.data.dataset import PADTSDataset, DataConfig
from dpm.data.collate import collate_patient_batch


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloader(cfg: Dict, mode: str = "val") -> DataLoader:
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
        ds = PADTSDataset(None, mode=mode, synthetic=True, synthetic_len=max(1, int(0.2 * total_len)), data_cfg=data_cfg)
    else:
        ds = PADTSDataset(cfg["data"]["manifest_path"], mode=mode, synthetic=False, data_cfg=data_cfg)

    return DataLoader(
        ds, batch_size=int(cfg["data"].get("batch_size", 4)),
        shuffle=False, num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=True, collate_fn=collate_patient_batch
    )


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
        temporal_layers=cfg["model"].get("temporal_layers", 2),
        temporal_heads=cfg["model"].get("temporal_heads", 4),
        temporal_dropout=cfg["model"].get("temporal_dropout", 0.1),
        max_time_gap=cfg["model"].get("max_time_gap", 1825.0),
        isup_num_classes=cfg["model"].get("isup_num_classes", 5),
        ipss_num_classes=cfg["model"].get("ipss_num_classes", 3),
        nih_cpsi_num_classes=cfg["model"].get("nih_cpsi_num_classes", 3),
        num_treatments=cfg["model"].get("num_treatments", 10),
    )
    return DPM(model_cfg).to(device)


# ============================================================================
# Metric Computation Functions
# ============================================================================


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy.
    Lower is better (0 = perfectly calibrated).

    Args:
        probs: Predicted probabilities [N, C] for C classes
        labels: True labels [N]
        n_bins: Number of bins for calibration

    Returns:
        ECE score (float)
    """
    if len(probs) == 0:
        return 0.0

    # Get predicted class and confidence
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    # Bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Samples in this bin
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return float(ece)


def compute_multiclass_auroc(
    probs: np.ndarray,
    labels: np.ndarray,
    average: str = "macro"
) -> float:
    """Compute AUROC for multi-class classification using one-vs-rest.

    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
        average: Averaging method ('macro', 'weighted', 'micro')

    Returns:
        AUROC score
    """
    if len(probs) == 0 or len(np.unique(labels)) < 2:
        return 0.0

    n_classes = probs.shape[1]

    # One-hot encode labels
    labels_onehot = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        if 0 <= label < n_classes:
            labels_onehot[i, label] = 1

    try:
        # Check if all classes have at least one sample
        classes_present = labels_onehot.sum(axis=0) > 0
        if classes_present.sum() < 2:
            return 0.0

        return roc_auc_score(
            labels_onehot, probs,
            average=average,
            multi_class="ovr"
        )
    except ValueError:
        # Handle edge cases (e.g., single class in batch)
        return 0.0


def compute_multiclass_auprc(
    probs: np.ndarray,
    labels: np.ndarray,
    average: str = "macro"
) -> float:
    """Compute AUPRC for multi-class classification using one-vs-rest.

    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
        average: Averaging method ('macro', 'weighted', 'micro')

    Returns:
        AUPRC score
    """
    if len(probs) == 0 or len(np.unique(labels)) < 2:
        return 0.0

    n_classes = probs.shape[1]

    # One-hot encode labels
    labels_onehot = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        if 0 <= label < n_classes:
            labels_onehot[i, label] = 1

    try:
        # Compute AP for each class and average
        ap_scores = []
        weights = []
        for c in range(n_classes):
            if labels_onehot[:, c].sum() > 0:  # Class has positive samples
                ap = average_precision_score(labels_onehot[:, c], probs[:, c])
                ap_scores.append(ap)
                weights.append(labels_onehot[:, c].sum())

        if not ap_scores:
            return 0.0

        if average == "macro":
            return float(np.mean(ap_scores))
        elif average == "weighted":
            weights = np.array(weights)
            return float(np.average(ap_scores, weights=weights))
        else:  # micro - flatten and compute
            return average_precision_score(labels_onehot.ravel(), probs.ravel())
    except ValueError:
        return 0.0


def compute_multiclass_brier(
    probs: np.ndarray,
    labels: np.ndarray
) -> float:
    """Compute multi-class Brier score.

    Brier score measures the mean squared difference between predicted
    probabilities and one-hot encoded true labels.
    Lower is better (0 = perfect).

    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]

    Returns:
        Brier score
    """
    if len(probs) == 0:
        return 0.0

    n_classes = probs.shape[1]

    # One-hot encode labels
    labels_onehot = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        if 0 <= label < n_classes:
            labels_onehot[i, label] = 1

    # Brier score = mean squared error between probs and one-hot labels
    return float(np.mean(np.sum((probs - labels_onehot) ** 2, axis=1)))


def compute_c_index(
    risk_scores: np.ndarray,
    times: np.ndarray,
    events: np.ndarray
) -> float:
    """Compute concordance index (C-index) for survival analysis.

    C-index measures the probability that predictions are concordant
    with actual outcomes. Higher is better (0.5 = random, 1.0 = perfect).

    Args:
        risk_scores: Predicted risk scores [N]
        times: Survival times [N]
        events: Event indicators (1=event occurred, 0=censored) [N]

    Returns:
        C-index score
    """
    if len(risk_scores) == 0 or events.sum() == 0:
        return 0.5  # Random baseline

    n = len(risk_scores)
    concordant = 0
    permissible = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Only consider pairs where at least one had an event
            # and we can determine ordering
            if times[i] != times[j]:
                if events[i] == 1 and times[i] < times[j]:
                    # i had event before j's time
                    permissible += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5
                elif events[j] == 1 and times[j] < times[i]:
                    # j had event before i's time
                    permissible += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5

    if permissible == 0:
        return 0.5

    return float(concordant / permissible)


def compute_multilabel_metrics(
    probs: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for multi-label classification (treatment prediction).

    Args:
        probs: Predicted probabilities [N, T] for T treatments
        labels: True labels [N, T] binary

    Returns:
        Dictionary with auroc, auprc, brier scores
    """
    if len(probs) == 0:
        return {"auroc": 0.0, "auprc": 0.0, "brier": 0.0}

    try:
        # AUROC and AUPRC per label, then macro average
        aurocs = []
        auprcs = []
        for t in range(probs.shape[1]):
            if labels[:, t].sum() > 0 and labels[:, t].sum() < len(labels):
                aurocs.append(roc_auc_score(labels[:, t], probs[:, t]))
                auprcs.append(average_precision_score(labels[:, t], probs[:, t]))

        auroc = np.mean(aurocs) if aurocs else 0.0
        auprc = np.mean(auprcs) if auprcs else 0.0

        # Brier score for multi-label
        brier = np.mean((probs - labels) ** 2)

        return {"auroc": float(auroc), "auprc": float(auprc), "brier": float(brier)}
    except ValueError:
        return {"auroc": 0.0, "auprc": 0.0, "brier": 0.0}


# ============================================================================
# Main Evaluation Function
# ============================================================================


def evaluate(model: DPM, dl: DataLoader, device: torch.device, cfg: Dict) -> Dict[str, float]:
    """Evaluate model with comprehensive metrics.

    Collects predictions and computes:
    - Diagnosis: Accuracy, AUROC, AUPRC, ECE, Brier Score
    - Staging (ISUP, IPSS, NIH-CPSI): Accuracy, AUROC, AUPRC, ECE, Brier
    - Treatment: AUROC, AUPRC, Brier Score (multi-label)
    - Prognosis: C-index

    Args:
        model: DPM model
        dl: DataLoader
        device: torch device
        cfg: Configuration dictionary

    Returns:
        Dictionary of metric names to values
    """
    model.eval()

    # Collect predictions for all tasks
    # Diagnosis (multi-class)
    diag_probs_all: List[np.ndarray] = []
    diag_labels_all: List[int] = []

    # Staging tasks (multi-class with ignore_index=-1)
    isup_probs_all: List[np.ndarray] = []
    isup_labels_all: List[int] = []
    ipss_probs_all: List[np.ndarray] = []
    ipss_labels_all: List[int] = []
    nih_cpsi_probs_all: List[np.ndarray] = []
    nih_cpsi_labels_all: List[int] = []

    # Treatment (multi-label)
    treat_probs_all: List[np.ndarray] = []
    treat_labels_all: List[np.ndarray] = []

    # Prognosis (survival)
    prog_risk_all: List[float] = []
    prog_time_all: List[float] = []
    prog_event_all: List[int] = []

    with torch.no_grad():
        for batch in tqdm(dl, desc="Evaluate"):
            x = {
                "clinical": batch["clinical"].to(device),
                "omics": batch["omics"].to(device),
                "mri": batch["mri"].to(device),
                "pathology": batch["pathology"].to(device),
            }
            pm = {k: v.to(device) for k, v in batch["present_mask"].items()}
            time_gaps = batch["time_gaps"].to(device)
            visit_mask = batch["visit_mask"].to(device)

            outputs = model.forward_with_logits(x, pm, time_gaps, visit_mask)

            # Diagnosis - softmax to get probabilities
            diag_logits = outputs["diag"]  # [B, C]
            diag_probs = F.softmax(diag_logits, dim=-1).cpu().numpy()
            diag_labels = batch["labels"]["diag"].cpu().numpy()

            for i in range(len(diag_labels)):
                diag_probs_all.append(diag_probs[i])
                diag_labels_all.append(int(diag_labels[i]))

            # Staging tasks - only include valid samples (not -1)
            if "isup" in outputs:
                isup_logits = outputs["isup"]
                isup_probs = F.softmax(isup_logits, dim=-1).cpu().numpy()
                isup_labels = batch["labels"]["isup"].cpu().numpy()
                for i in range(len(isup_labels)):
                    if isup_labels[i] != -1:
                        isup_probs_all.append(isup_probs[i])
                        isup_labels_all.append(int(isup_labels[i]))

            if "ipss" in outputs:
                ipss_logits = outputs["ipss"]
                ipss_probs = F.softmax(ipss_logits, dim=-1).cpu().numpy()
                ipss_labels = batch["labels"]["ipss"].cpu().numpy()
                for i in range(len(ipss_labels)):
                    if ipss_labels[i] != -1:
                        ipss_probs_all.append(ipss_probs[i])
                        ipss_labels_all.append(int(ipss_labels[i]))

            if "nih_cpsi" in outputs:
                nih_cpsi_logits = outputs["nih_cpsi"]
                nih_cpsi_probs = F.softmax(nih_cpsi_logits, dim=-1).cpu().numpy()
                nih_cpsi_labels = batch["labels"]["nih_cpsi"].cpu().numpy()
                for i in range(len(nih_cpsi_labels)):
                    if nih_cpsi_labels[i] != -1:
                        nih_cpsi_probs_all.append(nih_cpsi_probs[i])
                        nih_cpsi_labels_all.append(int(nih_cpsi_labels[i]))

            # Treatment - sigmoid to get probabilities
            if "treat" in outputs:
                treat_logits = outputs["treat"]
                treat_probs = torch.sigmoid(treat_logits).cpu().numpy()
                treat_labels = batch["labels"]["treat"].cpu().numpy()
                for i in range(len(treat_labels)):
                    treat_probs_all.append(treat_probs[i])
                    treat_labels_all.append(treat_labels[i])

            # Prognosis - collect risk scores and survival info
            # Model outputs "risk" key for prognosis
            if "risk" in outputs:
                prog_risk = outputs["risk"].cpu().numpy().flatten()
                # Labels use "time" and "event" keys
                prog_time = batch["labels"]["time"].cpu().numpy().flatten()
                prog_event = batch["labels"]["event"].cpu().numpy().flatten()
                for i in range(len(prog_risk)):
                    prog_risk_all.append(float(prog_risk[i]))
                    prog_time_all.append(float(prog_time[i]))
                    prog_event_all.append(int(prog_event[i]))

    # Compute metrics
    metrics: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # Diagnosis Metrics
    # -------------------------------------------------------------------------
    if diag_probs_all:
        diag_probs_arr = np.array(diag_probs_all)
        diag_labels_arr = np.array(diag_labels_all)
        diag_preds = np.argmax(diag_probs_arr, axis=1)

        metrics["diag_acc"] = accuracy_score(diag_labels_arr, diag_preds)
        metrics["diag_auroc"] = compute_multiclass_auroc(diag_probs_arr, diag_labels_arr)
        metrics["diag_auprc"] = compute_multiclass_auprc(diag_probs_arr, diag_labels_arr)
        metrics["diag_ece"] = compute_ece(diag_probs_arr, diag_labels_arr)
        metrics["diag_brier"] = compute_multiclass_brier(diag_probs_arr, diag_labels_arr)

    # -------------------------------------------------------------------------
    # Staging Metrics (ISUP)
    # -------------------------------------------------------------------------
    if isup_probs_all:
        isup_probs_arr = np.array(isup_probs_all)
        isup_labels_arr = np.array(isup_labels_all)
        isup_preds = np.argmax(isup_probs_arr, axis=1)

        metrics["isup_acc"] = accuracy_score(isup_labels_arr, isup_preds)
        metrics["isup_auroc"] = compute_multiclass_auroc(isup_probs_arr, isup_labels_arr)
        metrics["isup_auprc"] = compute_multiclass_auprc(isup_probs_arr, isup_labels_arr)
        metrics["isup_ece"] = compute_ece(isup_probs_arr, isup_labels_arr)
        metrics["isup_brier"] = compute_multiclass_brier(isup_probs_arr, isup_labels_arr)

    # -------------------------------------------------------------------------
    # Staging Metrics (IPSS)
    # -------------------------------------------------------------------------
    if ipss_probs_all:
        ipss_probs_arr = np.array(ipss_probs_all)
        ipss_labels_arr = np.array(ipss_labels_all)
        ipss_preds = np.argmax(ipss_probs_arr, axis=1)

        metrics["ipss_acc"] = accuracy_score(ipss_labels_arr, ipss_preds)
        metrics["ipss_auroc"] = compute_multiclass_auroc(ipss_probs_arr, ipss_labels_arr)
        metrics["ipss_auprc"] = compute_multiclass_auprc(ipss_probs_arr, ipss_labels_arr)
        metrics["ipss_ece"] = compute_ece(ipss_probs_arr, ipss_labels_arr)
        metrics["ipss_brier"] = compute_multiclass_brier(ipss_probs_arr, ipss_labels_arr)

    # -------------------------------------------------------------------------
    # Staging Metrics (NIH-CPSI)
    # -------------------------------------------------------------------------
    if nih_cpsi_probs_all:
        nih_cpsi_probs_arr = np.array(nih_cpsi_probs_all)
        nih_cpsi_labels_arr = np.array(nih_cpsi_labels_all)
        nih_cpsi_preds = np.argmax(nih_cpsi_probs_arr, axis=1)

        metrics["nih_cpsi_acc"] = accuracy_score(nih_cpsi_labels_arr, nih_cpsi_preds)
        metrics["nih_cpsi_auroc"] = compute_multiclass_auroc(nih_cpsi_probs_arr, nih_cpsi_labels_arr)
        metrics["nih_cpsi_auprc"] = compute_multiclass_auprc(nih_cpsi_probs_arr, nih_cpsi_labels_arr)
        metrics["nih_cpsi_ece"] = compute_ece(nih_cpsi_probs_arr, nih_cpsi_labels_arr)
        metrics["nih_cpsi_brier"] = compute_multiclass_brier(nih_cpsi_probs_arr, nih_cpsi_labels_arr)

    # -------------------------------------------------------------------------
    # Treatment Metrics (multi-label)
    # -------------------------------------------------------------------------
    if treat_probs_all:
        treat_probs_arr = np.array(treat_probs_all)
        treat_labels_arr = np.array(treat_labels_all)
        treat_metrics = compute_multilabel_metrics(treat_probs_arr, treat_labels_arr)
        metrics["treat_auroc"] = treat_metrics["auroc"]
        metrics["treat_auprc"] = treat_metrics["auprc"]
        metrics["treat_brier"] = treat_metrics["brier"]

    # -------------------------------------------------------------------------
    # Prognosis Metrics (survival)
    # -------------------------------------------------------------------------
    if prog_risk_all:
        prog_risk_arr = np.array(prog_risk_all)
        prog_time_arr = np.array(prog_time_all)
        prog_event_arr = np.array(prog_event_all)
        metrics["prog_c_index"] = compute_c_index(prog_risk_arr, prog_time_arr, prog_event_arr)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DPM model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--mode", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dl = build_dataloader(cfg, mode=args.mode)
    model = build_model(cfg, device)

    if args.ckpt:
        logging.info(f"Loading checkpoint from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))

    t0 = time.time()
    metrics = evaluate(model, dl, device, cfg)
    dur = time.time() - t0

    # -------------------------------------------------------------------------
    # Display Results
    # -------------------------------------------------------------------------

    # Diagnosis metrics table
    diag_table = PrettyTable()
    diag_table.title = f"Diagnosis Metrics ({args.mode})"
    diag_table.field_names = ["Metric", "Value"]
    diag_keys = ["diag_acc", "diag_auroc", "diag_auprc", "diag_ece", "diag_brier"]
    for key in diag_keys:
        if key in metrics:
            diag_table.add_row([key, f"{metrics[key]:.4f}"])
    logging.info("\n" + str(diag_table))

    # Staging metrics table
    staging_table = PrettyTable()
    staging_table.title = f"Staging Metrics ({args.mode})"
    staging_table.field_names = ["Task", "Accuracy", "AUROC", "AUPRC", "ECE", "Brier"]
    for task in ["isup", "ipss", "nih_cpsi"]:
        if f"{task}_acc" in metrics:
            staging_table.add_row([
                task.upper(),
                f"{metrics.get(f'{task}_acc', 0):.4f}",
                f"{metrics.get(f'{task}_auroc', 0):.4f}",
                f"{metrics.get(f'{task}_auprc', 0):.4f}",
                f"{metrics.get(f'{task}_ece', 0):.4f}",
                f"{metrics.get(f'{task}_brier', 0):.4f}",
            ])
    if staging_table.rowcount > 0:
        logging.info("\n" + str(staging_table))

    # Treatment metrics table
    treat_table = PrettyTable()
    treat_table.title = f"Treatment Metrics ({args.mode})"
    treat_table.field_names = ["Metric", "Value"]
    treat_keys = ["treat_auroc", "treat_auprc", "treat_brier"]
    for key in treat_keys:
        if key in metrics:
            treat_table.add_row([key, f"{metrics[key]:.4f}"])
    if treat_table.rowcount > 0:
        logging.info("\n" + str(treat_table))

    # Prognosis metrics table
    prog_table = PrettyTable()
    prog_table.title = f"Prognosis Metrics ({args.mode})"
    prog_table.field_names = ["Metric", "Value"]
    if "prog_c_index" in metrics:
        prog_table.add_row(["C-index", f"{metrics['prog_c_index']:.4f}"])
    if prog_table.rowcount > 0:
        logging.info("\n" + str(prog_table))

    # Summary table
    summary_table = PrettyTable()
    summary_table.title = "Evaluation Summary"
    summary_table.field_names = ["Info", "Value"]
    summary_table.add_row(["Mode", args.mode])
    summary_table.add_row(["Eval Time (s)", f"{dur:.2f}"])
    summary_table.add_row(["Total Metrics", len(metrics)])
    logging.info("\n" + str(summary_table))


if __name__ == "__main__":
    main()
