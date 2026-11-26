"""
Model Training Module for CCTV Augment AI.
Provides process control, status polling, and log streaming
for YOLO training experiments.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import sys
import json
import psutil
import config

TRAINING_PROCESS: Optional[subprocess.Popen] = None

def get_training_status_file(exp_name: str = "cctv_train") -> Path:
    return config.LOGS_DIR / f"{exp_name}_status.json"

def get_training_log_file(exp_name: str = "cctv_train") -> Path:
    return config.LOGS_DIR / f"{exp_name}.log"

def start_training_process(
    yaml_path: str,
    exp_name: str = "cctv_exp_original",
    epochs: int = 15,
    batch_size: int = 16,
    imgsz: int = 640,
    model_name: str = "yolov8n.pt",
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Spawns background Python training process using train_runner.py.
    """
    global TRAINING_PROCESS
    
    status_file = get_training_status_file(exp_name)
    log_file = get_training_log_file(exp_name)
    
    # Check if already running
    if is_training_active(exp_name):
        return {"status": "error", "message": "A training process is already active!"}
        
    runner_script = config.BASE_DIR / "train_runner.py"
    
    cmd = [
        sys.executable,
        str(runner_script.resolve()),
        "--data", str(Path(yaml_path).resolve()),
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--imgsz", str(imgsz),
        "--model", model_name,
        "--project", str(config.TRAINED_DIR.resolve()),
        "--name", exp_name,
        "--status-file", str(status_file.resolve()),
        "--log-file", str(log_file.resolve()),
        "--device", device,
    ]
    
    try:
        # Launch subprocess detached from main thread
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(config.BASE_DIR)
        )
        TRAINING_PROCESS = proc
        
        return {
            "status": "started",
            "pid": proc.pid,
            "experiment": exp_name,
            "status_file": str(status_file),
            "log_file": str(log_file)
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to spawn training process: {str(e)}"}

def is_training_active(exp_name: str = "cctv_train") -> bool:
    """Checks whether the training process is currently active."""
    status_file = get_training_status_file(exp_name)
    if not status_file.exists():
        return False
        
    try:
        with open(status_file, "r") as f:
            data = json.load(f)
            return bool(data.get("is_training", False))
    except Exception:
        return False

def get_training_progress(exp_name: str = "cctv_train") -> Dict[str, Any]:
    """Reads latest JSON training status."""
    status_file = get_training_status_file(exp_name)
    if not status_file.exists():
        return {
            "is_training": False,
            "completed": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "progress_pct": 0.0,
            "history": []
        }
        
    try:
        with open(status_file, "r") as f:
            return json.load(f)
    except Exception:
        return {"is_training": False, "error": "Could not read status file"}

def get_recent_training_logs(exp_name: str = "cctv_train", max_lines: int = 40) -> str:
    """Reads last N lines of training log file."""
    log_file = get_training_log_file(exp_name)
    if not log_file.exists():
        return "No training logs available yet."
        
    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return "".join(lines[-max_lines:])
    except Exception as e:
        return f"Error reading logs: {str(e)}"

def stop_training_process() -> bool:
    """Safely terminates active training process."""
    global TRAINING_PROCESS
    if TRAINING_PROCESS and TRAINING_PROCESS.poll() is None:
        try:
            TRAINING_PROCESS.terminate()
            TRAINING_PROCESS.wait(timeout=5)
            return True
        except Exception:
            TRAINING_PROCESS.kill()
            return True
    return False

def list_trained_models() -> List[Dict[str, Any]]:
    """Discovers all fine-tuned YOLO models in models/trained/."""
    models = []
    if not config.TRAINED_DIR.exists():
        return models
        
    for exp_dir in config.TRAINED_DIR.iterdir():
        if exp_dir.is_dir():
            weights_dir = exp_dir / "weights"
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"
            
            if best_pt.exists():
                models.append({
                    "name": exp_dir.name,
                    "type": "best",
                    "path": str(best_pt.resolve()),
                    "size_mb": round(best_pt.stat().st_size / (1024 * 1024), 2),
                    "modified": best_pt.stat().st_mtime
                })
            elif last_pt.exists():
                models.append({
                    "name": exp_dir.name,
                    "type": "last",
                    "path": str(last_pt.resolve()),
                    "size_mb": round(last_pt.stat().st_size / (1024 * 1024), 2),
                    "modified": last_pt.stat().st_mtime
                })
    return models
