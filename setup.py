"""Setup script for DPM package."""
from setuptools import setup, find_packages

setup(
    name="dpm",
    version="0.1.0",
    description="Digital Prostate Model - Multi-modal temporal fusion for prostate disease diagnosis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "timm",
        "numpy",
        "pandas",
        "pyyaml",
        "scikit-learn",
        "tqdm",
        "prettytable",
        "pillow",
    ],
)
