"""Dataset for multi-modal multi-visit patient data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import os
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from PIL import Image

try:
    from torchvision import transforms as T
except Exception:
    T = None


@dataclass
class DataConfig:
    """Data configuration."""
    clinical_input_dim: int = 64
    omics_feature_dim: int = 128
    omics_seq_len: int = 128
    mri_in_channels: int = 1
    mri_size: Tuple[int, int, int] = (64, 128, 128)
    image_size: int = 224


class PADTSDataset(Dataset):
    """Dataset returning all visits per patient."""

    def __init__(
        self,
        manifest_path: Optional[str],
        mode: str = "train",
        synthetic: bool = False,
        synthetic_len: int = 200,
        data_cfg: Optional[DataConfig] = None,
    ) -> None:
        super().__init__()
        assert mode in {"train", "val", "test"}
        self.mode = mode
        self.synthetic = synthetic
        self.synthetic_len = synthetic_len
        self.cfg = data_cfg or DataConfig()

        self.df: Optional[pd.DataFrame] = None
        self.patient_ids: List[str] = []

        if not synthetic:
            if manifest_path is None or not os.path.exists(manifest_path):
                raise FileNotFoundError("Manifest path must exist unless synthetic=True.")
            df = pd.read_csv(manifest_path)

            # Filter by split
            if "split" in df.columns:
                df = df[df["split"].astype(str).str.lower() == mode]

            self.df = df.reset_index(drop=True)
            self.patient_ids = self.df["patient_id"].unique().tolist()

        # Image transform
        if T is not None:
            self.img_tf = T.Compose([
                T.Resize((self.cfg.image_size, self.cfg.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.img_tf = None

    def __len__(self) -> int:
        if self.synthetic:
            return self.synthetic_len
        return len(self.patient_ids)

    def _load_clinical(self, path: Optional[str]) -> Tuple[torch.Tensor, bool]:
        dim = self.cfg.clinical_input_dim
        if not path or not isinstance(path, str) or pd.isna(path) or not os.path.exists(path):
            return torch.zeros(dim, dtype=torch.float32), False
        try:
            if path.endswith(".npy"):
                arr = np.load(path).astype(np.float32).reshape(-1)
            else:
                arr = pd.read_csv(path).to_numpy(dtype=np.float32).reshape(-1)
            if arr.shape[0] < dim:
                arr = np.concatenate([arr, np.zeros(dim - arr.shape[0], dtype=np.float32)])
            return torch.from_numpy(arr[:dim]), True
        except Exception:
            return torch.zeros(dim, dtype=torch.float32), False

    def _load_omics(self, path: Optional[str]) -> Tuple[torch.Tensor, bool]:
        Tlen, Fdim = self.cfg.omics_seq_len, self.cfg.omics_feature_dim
        if not path or not isinstance(path, str) or pd.isna(path) or not os.path.exists(path):
            return torch.zeros(Tlen, Fdim, dtype=torch.float32), False
        try:
            if path.endswith(".npy"):
                arr = np.load(path).astype(np.float32)
            else:
                arr = pd.read_csv(path).to_numpy(dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, Fdim)
            # Adjust feature dim
            if arr.shape[1] > Fdim:
                arr = arr[:, :Fdim]
            elif arr.shape[1] < Fdim:
                arr = np.concatenate([arr, np.zeros((arr.shape[0], Fdim - arr.shape[1]), dtype=np.float32)], axis=1)
            # Adjust seq len
            if arr.shape[0] >= Tlen:
                arr = arr[:Tlen]
            else:
                arr = np.concatenate([arr, np.zeros((Tlen - arr.shape[0], Fdim), dtype=np.float32)], axis=0)
            return torch.from_numpy(arr), True
        except Exception:
            return torch.zeros(Tlen, Fdim, dtype=torch.float32), False

    def _load_mri(self, path: Optional[str]) -> Tuple[torch.Tensor, bool]:
        C, (D, H, W) = self.cfg.mri_in_channels, self.cfg.mri_size
        if not path or not isinstance(path, str) or pd.isna(path) or not os.path.exists(path):
            return torch.zeros(C, D, H, W, dtype=torch.float32), False
        try:
            if path.endswith(".npy"):
                vol = np.load(path).astype(np.float32)
                vol_t = torch.from_numpy(vol)
                # Ensure shape [C, D, H, W]
                if vol_t.ndim == 3:
                    vol_t = vol_t.unsqueeze(0)
                # Resize if needed
                if vol_t.shape[1:] != (D, H, W):
                    vol_t = F.interpolate(vol_t.unsqueeze(0), size=(D, H, W), mode="trilinear", align_corners=False).squeeze(0)
                return vol_t, True
            else:
                # NIfTI format (optional)
                try:
                    import nibabel as nib
                    img = nib.load(path)
                    vol = img.get_fdata().astype(np.float32)
                    if np.nanmax(vol) > 0:
                        vol = (vol - np.nanmin(vol)) / (np.nanmax(vol) - np.nanmin(vol) + 1e-8)
                    vol = np.nan_to_num(vol, nan=0.0)
                    vol_t = torch.from_numpy(vol)[None, None, ...]
                    vol_t = F.interpolate(vol_t, size=(D, H, W), mode="trilinear", align_corners=False).squeeze(0)
                    return vol_t, True
                except Exception:
                    return torch.zeros(C, D, H, W, dtype=torch.float32), False
        except Exception:
            return torch.zeros(C, D, H, W, dtype=torch.float32), False

    def _load_pathology(self, path: Optional[str]) -> Tuple[torch.Tensor, bool]:
        H = W = self.cfg.image_size
        if not path or not isinstance(path, str) or pd.isna(path) or not os.path.exists(path):
            return torch.zeros(3, H, W, dtype=torch.float32), False
        try:
            if path.endswith(".npy"):
                arr = np.load(path).astype(np.float32)
                img_t = torch.from_numpy(arr)
                if img_t.shape[1:] != (H, W):
                    img_t = F.interpolate(img_t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
                return img_t, True
            else:
                img = Image.open(path).convert("RGB")
                if self.img_tf is not None:
                    return self.img_tf(img), True
                else:
                    img = img.resize((W, H))
                    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1), True
        except Exception:
            return torch.zeros(3, H, W, dtype=torch.float32), False

    def _load_single_visit(self, row) -> Dict[str, Any]:
        """Load data for a single visit."""
        clin, p_clin = self._load_clinical(getattr(row, "clinical_path", None))
        omic, p_omic = self._load_omics(getattr(row, "omics_path", None))
        mri, p_mri = self._load_mri(getattr(row, "mri_path", None))
        path, p_path = self._load_pathology(getattr(row, "pathology_path", None))

        return {
            "clinical": clin,
            "omics": omic,
            "mri": mri,
            "pathology": path,
            "present_mask": {
                "clinical": torch.tensor(p_clin, dtype=torch.bool),
                "omics": torch.tensor(p_omic, dtype=torch.bool),
                "mri": torch.tensor(p_mri, dtype=torch.bool),
                "pathology": torch.tensor(p_path, dtype=torch.bool),
            },
        }

    def _extract_labels(self, row) -> Dict[str, torch.Tensor]:
        """Extract labels from last visit row."""
        # Treatment labels (JSON string)
        treat_str = getattr(row, "treat_labels", "[0,0,0,0,0,0,0,0,0,0]")
        if pd.isna(treat_str):
            treat_str = "[0,0,0,0,0,0,0,0,0,0]"
        treat = json.loads(treat_str)

        return {
            "diag": torch.tensor(int(getattr(row, "diag_label", 0)), dtype=torch.long),
            "treat": torch.tensor(treat, dtype=torch.float32),
            "time": torch.tensor(float(getattr(row, "prog_time", 1.0)), dtype=torch.float32),
            "event": torch.tensor(int(getattr(row, "prog_event", 0)), dtype=torch.long),
            "isup": torch.tensor(int(getattr(row, "isup_grade", -1)), dtype=torch.long),
            "ipss": torch.tensor(int(getattr(row, "ipss_score", -1)), dtype=torch.long),
            "nih_cpsi": torch.tensor(int(getattr(row, "nih_cpsi_score", -1)), dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.synthetic:
            return self._get_synthetic_item(idx)

        patient_id = self.patient_ids[idx]
        patient_df = self.df[self.df["patient_id"] == patient_id].copy()
        patient_df["visit_date"] = pd.to_datetime(patient_df["visit_date"])
        patient_df = patient_df.sort_values("visit_date")

        visits = []
        time_gaps = [0.0]
        prev_date = None

        for row in patient_df.itertuples():
            visit = self._load_single_visit(row)
            visits.append(visit)

            if prev_date is not None:
                gap = (row.visit_date - prev_date).days
                time_gaps.append(float(gap))
            prev_date = row.visit_date

        # Labels from last visit
        last_row = patient_df.iloc[-1]
        labels = self._extract_labels(last_row)

        return {
            "visits": visits,
            "time_gaps": torch.tensor(time_gaps, dtype=torch.float32),
            "labels": labels,
        }

    def _get_synthetic_item(self, idx: int) -> Dict[str, Any]:
        """Generate synthetic sample."""
        rng = np.random.default_rng(seed=idx)
        num_visits = rng.integers(3, 6)

        visits = []
        time_gaps = [0.0]

        for v in range(num_visits):
            clin = torch.from_numpy(rng.normal(0, 1, size=(self.cfg.clinical_input_dim,)).astype(np.float32))
            omic = torch.from_numpy(rng.normal(0, 1, size=(self.cfg.omics_seq_len, self.cfg.omics_feature_dim)).astype(np.float32))
            mri = torch.from_numpy(rng.normal(0, 1, size=(self.cfg.mri_in_channels, *self.cfg.mri_size)).astype(np.float32))
            path = torch.from_numpy(rng.normal(0, 1, size=(3, self.cfg.image_size, self.cfg.image_size)).astype(np.float32))

            pm = {
                "clinical": torch.tensor(rng.random() > 0.0, dtype=torch.bool),
                "omics": torch.tensor(rng.random() > 0.95, dtype=torch.bool),
                "mri": torch.tensor(rng.random() > 0.4, dtype=torch.bool),
                "pathology": torch.tensor(rng.random() > 0.75, dtype=torch.bool),
            }

            visits.append({
                "clinical": clin,
                "omics": omic,
                "mri": mri,
                "pathology": path,
                "present_mask": pm,
            })

            if v > 0:
                time_gaps.append(float(rng.integers(30, 180)))

        # Labels
        diag = int(rng.integers(0, 8))
        treat = [int(rng.integers(0, 2)) for _ in range(10)]
        isup = int(rng.integers(0, 5)) if diag >= 3 else -1
        ipss = int(rng.integers(0, 3)) if diag == 2 else -1
        nih_cpsi = int(rng.integers(0, 3)) if diag == 1 else -1

        labels = {
            "diag": torch.tensor(diag, dtype=torch.long),
            "treat": torch.tensor(treat, dtype=torch.float32),
            "time": torch.tensor(float(rng.integers(100, 2000)), dtype=torch.float32),
            "event": torch.tensor(int(rng.integers(0, 2)), dtype=torch.long),
            "isup": torch.tensor(isup, dtype=torch.long),
            "ipss": torch.tensor(ipss, dtype=torch.long),
            "nih_cpsi": torch.tensor(nih_cpsi, dtype=torch.long),
        }

        return {
            "visits": visits,
            "time_gaps": torch.tensor(time_gaps, dtype=torch.float32),
            "labels": labels,
        }
