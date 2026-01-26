"""Custom collate function for variable-length visit sequences."""

import torch
from typing import Dict, List, Any


def collate_patient_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length visit sequences to same length.

    Args:
        batch: List of samples, each with 'visits', 'time_gaps', 'labels'

    Returns:
        Batched tensors with shape [B, T_max, ...]
    """
    B = len(batch)
    T_max = max(len(b["visits"]) for b in batch)

    # Get dimensions from first sample's first visit
    sample_visit = batch[0]["visits"][0]
    clinical_dim = sample_visit["clinical"].shape[0]
    omics_shape = sample_visit["omics"].shape
    mri_shape = sample_visit["mri"].shape
    path_shape = sample_visit["pathology"].shape

    # Initialize tensors
    clinical = torch.zeros(B, T_max, clinical_dim)
    omics = torch.zeros(B, T_max, *omics_shape)
    mri = torch.zeros(B, T_max, *mri_shape)
    pathology = torch.zeros(B, T_max, *path_shape)

    present_mask = {
        "clinical": torch.zeros(B, T_max, dtype=torch.bool),
        "omics": torch.zeros(B, T_max, dtype=torch.bool),
        "mri": torch.zeros(B, T_max, dtype=torch.bool),
        "pathology": torch.zeros(B, T_max, dtype=torch.bool),
    }

    time_gaps = torch.zeros(B, T_max)
    visit_mask = torch.zeros(B, T_max, dtype=torch.bool)

    # Fill in data
    for i, sample in enumerate(batch):
        T_i = len(sample["visits"])
        visit_mask[i, :T_i] = True
        time_gaps[i, :T_i] = sample["time_gaps"][:T_i]

        for t, visit in enumerate(sample["visits"]):
            clinical[i, t] = visit["clinical"]
            omics[i, t] = visit["omics"]
            mri[i, t] = visit["mri"]
            pathology[i, t] = visit["pathology"]
            present_mask["clinical"][i, t] = visit["present_mask"]["clinical"]
            present_mask["omics"][i, t] = visit["present_mask"]["omics"]
            present_mask["mri"][i, t] = visit["present_mask"]["mri"]
            present_mask["pathology"][i, t] = visit["present_mask"]["pathology"]

    # Stack labels
    labels = {}
    for key in batch[0]["labels"]:
        labels[key] = torch.stack([b["labels"][key] for b in batch])

    return {
        "clinical": clinical,
        "omics": omics,
        "mri": mri,
        "pathology": pathology,
        "present_mask": present_mask,
        "time_gaps": time_gaps,
        "visit_mask": visit_mask,
        "labels": labels,
    }
