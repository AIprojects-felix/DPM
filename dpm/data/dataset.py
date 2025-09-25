"""Dataset and data loading utilities for PADTS multi-modal data.

This module defines the `PADTSDataset` that loads clinical tabular data, omics
sequences, MRI volumes, and pathology images from a manifest CSV, with robust
handling for missing modalities. It also supports a synthetic data mode for quick
prototyping and debugging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from PIL import Image

try:  # Optional dependencies are declared in requirements
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover
    nib = None

try:
    from torchvision import transforms as T  # type: ignore
except Exception:  # pragma: no cover
    T = None


@dataclass
class DataConfig:
    """Lightweight data config for dataset construction.

    Only the relevant keys used by the dataset are included.
    """

    clinical_input_dim: int = 64
    omics_feature_dim: int = 128
    omics_seq_len: int = 128
    mri_in_channels: int = 1
    mri_size: Tuple[int, int, int] = (64, 128, 128)  # (D, H, W)
    image_size: int = 224


class PADTSDataset(Dataset):
    """Dataset for multi-modal PADTS data.

    Either provide a `manifest_path` CSV or set `synthetic=True` to generate
    synthetic samples. The manifest is expected to contain the following columns:
    - patient_id: str/int
    - clinical_path: path to a single-row CSV or a .npy file of shape [clinical_input_dim]
    - omics_path: path to a CSV or .npy file of shape [T, omics_feature_dim]
    - mri_path: path to a NIfTI image (.nii/.nii.gz)
    - pathology_path: path to an RGB image (e.g., .tiff/.png/.jpg)
    - diag_label: int class index
    - treat_label: 0/1 float or int
    - prog_time: float time-to-event or censoring time
    - prog_event: 0/1 event indicator

    Missing values in those paths (empty/NaN) are treated as missing modalities.
    """

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
        if not synthetic:
            if manifest_path is None or not os.path.exists(manifest_path):
                raise FileNotFoundError(
                    "Manifest path must exist unless synthetic=True."
                )
            df = pd.read_csv(manifest_path)
            # Optional split column to filter rows
            if "split" in df.columns:
                df = df[df["split"].astype(str).str.lower() == mode]
            self.df = df.reset_index(drop=True)

        # Image transform
        if T is not None:
            self.img_tf = T.Compose(
                [
                    T.Resize((self.cfg.image_size, self.cfg.image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:  # minimal fallback
            self.img_tf = None

    def __len__(self) -> int:
        if self.synthetic:
            return self.synthetic_len
        assert self.df is not None
        return len(self.df)

    # ------------------------------ Loaders ------------------------------
    def _load_clinical(self, path: Optional[str]) -> Tuple[torch.Tensor, bool]:
        dim = self.cfg.clinical_input_dim
        if path is None or not isinstance(path, str) or not os.path.exists(path):
            return torch.zeros(dim, dtype=torch.float32), False
        try:
            if path.endswith(".npy"):
                arr = np.load(path)
                vec = np.asarray(arr, dtype=np.float32).reshape(-1)
            else:
                df = pd.read_csv(path)
                vec = df.to_numpy(dtype=np.float32).reshape(-1)
            if vec.shape[0] < dim:
                pad = np.zeros(dim - vec.shape[0], dtype=np.float32)
                vec = np.concatenate([vec, pad], axis=0)
            vec = vec[:dim]
            return torch.from_numpy(vec), True
        except Exception:
            return torch.zeros(dim, dtype=torch.float32), False

    def _load_omics(self, path: Optional[str]) -> Tuple[torch.Tensor, bool]:
        Tlen = self.cfg.omics_seq_len
        Fdim = self.cfg.omics_feature_dim
        if path is None or not isinstance(path, str) or not os.path.exists(path):
            return torch.zeros(Tlen, Fdim, dtype=torch.float32), False
        try:
            if path.endswith(".npy"):
                arr = np.load(path)
            else:
                arr = pd.read_csv(path).to_numpy(dtype=np.float32)
            arr = np.asarray(arr, dtype=np.float32)
            # Ensure 2D
            if arr.ndim == 1:
                arr = arr.reshape(-1, Fdim)
            # Truncate/pad to fixed length
            if arr.shape[1] != Fdim:
                # If feature dim mismatched, project/crop/pad to Fdim
                if arr.shape[1] > Fdim:
                    arr = arr[:, :Fdim]
                else:
                    pad_feat = np.zeros((arr.shape[0], Fdim - arr.shape[1]), dtype=np.float32)
                    arr = np.concatenate([arr, pad_feat], axis=1)
            if arr.shape[0] >= Tlen:
                arr = arr[:Tlen]
            else:
                pad_tok = np.zeros((Tlen - arr.shape[0], Fdim), dtype=np.float32)
                arr = np.concatenate([arr, pad_tok], axis=0)
            return torch.from_numpy(arr), True
        except Exception:
            return torch.zeros(Tlen, Fdim, dtype=torch.float32), False

    def _load_mri(self, path: Optional[str]) -> Tuple[torch.Tensor, bool]:
        C = self.cfg.mri_in_channels
        D, H, W = self.cfg.mri_size
        if path is None or not isinstance(path, str) or not os.path.exists(path) or nib is None:
            return torch.zeros(C, D, H, W, dtype=torch.float32), False
        try:
            img = nib.load(path)
            vol = img.get_fdata().astype(np.float32)
            # Normalize to [0,1]
            if np.nanmax(vol) > 0:
                vol = (vol - np.nanmin(vol)) / (np.nanmax(vol) - np.nanmin(vol) + 1e-8)
            vol = np.nan_to_num(vol, nan=0.0)
            vol_t = torch.from_numpy(vol)[None, None, ...]  # [1,1,D,H,W or whatever]
            vol_t = F.interpolate(vol_t, size=(D, H, W), mode="trilinear", align_corners=False)
            vol_t = vol_t.squeeze(0)  # [1, D, H, W]
            if C == 1:
                return vol_t, True
            # Tile channels if requested >1
            vol_t = vol_t.repeat(C, 1, 1, 1)
            return vol_t, True
        except Exception:
            return torch.zeros(C, D, H, W, dtype=torch.float32), False

    def _load_pathology(self, path: Optional[str]) -> Tuple[torch.Tensor, bool]:
        H = W = self.cfg.image_size
        if path is None or not isinstance(path, str) or not os.path.exists(path):
            return torch.zeros(3, H, W, dtype=torch.float32), False
        try:
            img = Image.open(path).convert("RGB")
            if self.img_tf is not None:
                img_t = self.img_tf(img)
            else:
                img = img.resize((W, H))
                img_t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
            return img_t, True
        except Exception:
            return torch.zeros(3, H, W, dtype=torch.float32), False

    # ------------------------------ Get item ------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.synthetic:
            rng = np.random.default_rng(seed=idx)
            clin = torch.from_numpy(rng.normal(0, 1, size=(self.cfg.clinical_input_dim,)).astype(np.float32))
            omic = torch.from_numpy(rng.normal(0, 1, size=(self.cfg.omics_seq_len, self.cfg.omics_feature_dim)).astype(np.float32))
            mri = torch.from_numpy(rng.normal(0, 1, size=(self.cfg.mri_in_channels, *self.cfg.mri_size)).astype(np.float32))
            path = torch.from_numpy(rng.normal(0, 1, size=(3, self.cfg.image_size, self.cfg.image_size)).astype(np.float32))

            # Randomly drop modalities (20% chance each)
            pm = {
                "clinical": rng.random() > 0.2,
                "omics": rng.random() > 0.2,
                "mri": rng.random() > 0.2,
                "pathology": rng.random() > 0.2,
            }
            if not pm["clinical"]:
                clin = torch.zeros_like(clin)
            if not pm["omics"]:
                omic = torch.zeros_like(omic)
            if not pm["mri"]:
                mri = torch.zeros_like(mri)
            if not pm["pathology"]:
                path = torch.zeros_like(path)

            y_diag = int(rng.integers(0, 3))
            y_treat = float(rng.integers(0, 2))
            y_time = float(rng.integers(1, 1000))
            y_event = int(rng.integers(0, 2))
        else:
            assert self.df is not None
            row = self.df.iloc[idx]
            clin, p_clin = self._load_clinical(row.get("clinical_path"))
            omic, p_omic = self._load_omics(row.get("omics_path"))
            mri, p_mri = self._load_mri(row.get("mri_path"))
            path, p_path = self._load_pathology(row.get("pathology_path"))
            pm = {
                "clinical": bool(p_clin),
                "omics": bool(p_omic),
                "mri": bool(p_mri),
                "pathology": bool(p_path),
            }
            # Labels
            y_diag = int(row.get("diag_label", 0))
            y_treat = float(row.get("treat_label", 0.0))
            y_time = float(row.get("prog_time", 1.0))
            y_event = int(row.get("prog_event", 0))

        sample: Dict[str, Any] = {
            "clinical": clin.to(dtype=torch.float32),
            "omics": omic.to(dtype=torch.float32),
            "mri": mri.to(dtype=torch.float32),
            "pathology": path.to(dtype=torch.float32),
            "present_mask": {
                "clinical": torch.tensor(pm["clinical"], dtype=torch.bool),
                "omics": torch.tensor(pm["omics"], dtype=torch.bool),
                "mri": torch.tensor(pm["mri"], dtype=torch.bool),
                "pathology": torch.tensor(pm["pathology"], dtype=torch.bool),
            },
            "labels": {
                "diag": torch.tensor(y_diag, dtype=torch.long),
                "treat": torch.tensor(y_treat, dtype=torch.float32),
                "time": torch.tensor(y_time, dtype=torch.float32),
                "event": torch.tensor(y_event, dtype=torch.long),
            },
        }
        return sample
