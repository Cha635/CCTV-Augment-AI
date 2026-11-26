"""
Configuration and constants for CCTV Augment AI platform.
"""

from pathlib import Path
import os
import sys
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Dataset paths
dataset_env = os.environ.get("CCTV_DATASET_DIR")
if dataset_env and Path(dataset_env).exists():
    DEFAULT_DATASET_DIR = Path(dataset_env)
elif (BASE_DIR / "yolo-person-vehicle-dataset").exists():
    DEFAULT_DATASET_DIR = BASE_DIR / "yolo-person-vehicle-dataset"
elif (PROJECT_ROOT / "yolo-person-vehicle-dataset").exists():
    DEFAULT_DATASET_DIR = PROJECT_ROOT / "yolo-person-vehicle-dataset"
elif (BASE_DIR / "dataset" / "original").exists():
    DEFAULT_DATASET_DIR = BASE_DIR / "dataset" / "original"
else:
    DEFAULT_DATASET_DIR = BASE_DIR / "dataset" / "splits"

PROCESSED_DATA_DIR = BASE_DIR / "dataset" / "processed"
AUGMENTED_DATA_DIR = BASE_DIR / "dataset" / "augmented"
SPLITS_DIR = BASE_DIR / "dataset" / "splits"

# Models paths
MODELS_DIR = BASE_DIR / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
TRAINED_DIR = MODELS_DIR / "trained"

# Outputs paths
OUTPUTS_DIR = BASE_DIR / "outputs"
AUGMENTED_IMAGES_DIR = OUTPUTS_DIR / "augmented_images"
DETECTIONS_DIR = OUTPUTS_DIR / "detections"
REPORTS_DIR = OUTPUTS_DIR / "reports"
COMPARISONS_DIR = OUTPUTS_DIR / "comparisons"
BATCHES_DIR = OUTPUTS_DIR / "batches"

# Logs
LOGS_DIR = BASE_DIR / "logs"

# Ensure all critical directories exist
for directory in [
    PROCESSED_DATA_DIR,
    AUGMENTED_DATA_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    PRETRAINED_DIR,
    TRAINED_DIR,
    OUTPUTS_DIR,
    AUGMENTED_IMAGES_DIR,
    DETECTIONS_DIR,
    REPORTS_DIR,
    COMPARISONS_DIR,
    BATCHES_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Classes
CLASS_NAMES = {0: "person", 1: "vehicle"}
CLASS_COLORS = {
    0: (0, 216, 255),    # Cyan/Neon Blue for person in BGR
    1: (255, 128, 0),    # Orange/Amber for vehicle in BGR
}
CLASS_HEX_COLORS = {
    0: "#00D8FF",
    1: "#FF8000"
}

# Hardware check
def get_hardware_info():
    cuda_available = torch.cuda.is_available()
    device_name = "CPU"
    device_id = "cpu"
    vram_gb = 0.0
    
    if cuda_available:
        device_id = "0"
        device_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        
    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "device_id": device_id,
        "vram_gb": vram_gb,
    }

# Theme Palette (Accessible High-Contrast Cyber-Surveillance)
THEME = {
    "bg_dark": "#0B1120",
    "card_bg": "#151F32",
    "bg_secondary": "#1D2B42",
    "card_border": "#2E4166",
    "accent_cyan": "#22D3EE",
    "accent_blue": "#3B82F6",
    "success": "#22C55E",
    "warning": "#FBBF24",
    "error": "#F87171",
    "text_primary": "#F8FAFC",
    "text_secondary": "#CBD5E1",
    "text_muted": "#A3B3C7",
}
