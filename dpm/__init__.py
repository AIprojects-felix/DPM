"""DPM: A modular multi-modal learning framework for PADTS tasks.

This package contains the core model, encoders, dataset utilities, and loss functions.
All code and comments are strictly in English to ensure portability and clarity.
"""

from .models.dpm import DPM  # re-export primary model

__all__ = ["DPM"]
