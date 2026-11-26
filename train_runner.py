"""
Standalone YOLO Training Worker for CCTV Augment AI.
Can run in background process or command line.
Writes real-time status and logs to JSON/log files for UI consumption.
"""

import sys
import os
from pathlib import Path
import json
import argparse
import time
import yaml

def run_training(
    data_yaml: str,
    epochs: int = 15,
    batch_size: int = 16,
    imgsz: int = 640,
    model_name: str = "yolov8n.pt",
    project: str = "runs/detect",
    name: str = "cctv_train",
    status_file: str = "logs/training_status.json",
    log_file: str = "logs/training.log",
    device: str = "auto"
):
    from ultralytics import YOLO
    import torch
    
    os.makedirs(Path(status_file).parent, exist_ok=True)
    os.makedirs(Path(log_file).parent, exist_ok=True)
    
    # Check and ensure dataset YAML paths resolve correctly
    try:
        yaml_p = Path(data_yaml).resolve()
        if yaml_p.exists():
            with open(yaml_p, "r", encoding="utf-8") as yf:
                ydata = yaml.safe_load(yf)
            # If path is "." or missing, resolve to YAML directory
            if isinstance(ydata, dict) and (ydata.get("path") == "." or not ydata.get("path")):
                ydata["path"] = str(yaml_p.parent.resolve()).replace("\\", "/")
                with open(yaml_p, "w", encoding="utf-8") as yf:
                    yaml.dump(ydata, yf, default_flow_style=False)
    except Exception as ye:
        pass

    # Check device
    if device == "auto":
        dev = "0" if torch.cuda.is_available() else "cpu"
    else:
        dev = device
        
    status = {
        "is_training": True,
        "completed": False,
        "error": None,
        "experiment_name": name,
        "current_epoch": 0,
        "total_epochs": epochs,
        "progress_pct": 0.0,
        "start_time": time.time(),
        "train_loss": 0.0,
        "val_loss": 0.0,
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "weights_path": None,
        "history": []
    }
    
    with open(status_file, "w") as sf:
        json.dump(status, sf, indent=2)
        
    class StatusCallback:
        def on_train_epoch_end(self, trainer):
            current_ep = trainer.epoch + 1
            tot_ep = trainer.epochs
            pct = round((current_ep / tot_ep) * 100, 1)
            
            # Extract losses and metrics safely
            train_loss = 0.0
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                try:
                    train_loss = float(trainer.loss_items.sum().item())
                except Exception:
                    pass
            elif hasattr(trainer, 'loss') and trainer.loss is not None:
                try:
                    train_loss = float(trainer.loss.item() if hasattr(trainer.loss, 'item') else trainer.loss)
                except Exception:
                    pass
            
            # Check metrics
            metrics = getattr(trainer, 'metrics', {})
            p = float(metrics.get("metrics/precision(B)", 0.0))
            r = float(metrics.get("metrics/recall(B)", 0.0))
            map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
            map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))
            
            epoch_data = {
                "epoch": current_ep,
                "train_loss": round(train_loss, 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "mAP50": round(map50, 4),
                "mAP50_95": round(map50_95, 4),
            }
            
            status["current_epoch"] = current_ep
            status["progress_pct"] = pct
            status["train_loss"] = epoch_data["train_loss"]
            status["precision"] = epoch_data["precision"]
            status["recall"] = epoch_data["recall"]
            status["mAP50"] = epoch_data["mAP50"]
            status["mAP50_95"] = epoch_data["mAP50_95"]
            status["history"].append(epoch_data)
            
            with open(status_file, "w") as f:
                json.dump(status, f, indent=2)
                
            with open(log_file, "a") as lf:
                lf.write(f"[EPOCH {current_ep}/{tot_ep}] loss={train_loss:.4f} P={p:.4f} R={r:.4f} mAP50={map50:.4f}\n")
                
    try:
        model = YOLO(model_name)
        cb = StatusCallback()
        model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
        
        with open(log_file, "a") as lf:
            lf.write(f"--- Starting Training [{name}] at {time.ctime()} ---\n")
            lf.write(f"Device: {dev}, Batch: {batch_size}, Imgsz: {imgsz}, Epochs: {epochs}\n")
            
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=dev,
            project=project,
            name=name,
            exist_ok=True,
            verbose=True,
            workers=0 if sys.platform.startswith("win") else 2,
            plots=True
        )
        
        # Save completion status
        weights_path = Path(project) / name / "weights" / "best.pt"
        status["is_training"] = False
        status["completed"] = True
        status["weights_path"] = str(weights_path.resolve()) if weights_path.exists() else None
        status["end_time"] = time.time()
        status["duration_sec"] = round(status["end_time"] - status["start_time"], 2)
        
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
            
        with open(log_file, "a") as lf:
            lf.write(f"--- Training Completed Successfully! Best weights: {status['weights_path']} ---\n")
            
    except Exception as e:
        status["is_training"] = False
        status["completed"] = False
        status["error"] = str(e)
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
        with open(log_file, "a") as lf:
            lf.write(f"--- TRAINING ERROR: {str(e)} ---\n")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training Runner for CCTV Augment AI")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="cctv_train")
    parser.add_argument("--status-file", default="logs/training_status.json")
    parser.add_argument("--log-file", default="logs/training.log")
    parser.add_argument("--device", default="auto")
    
    args = parser.parse_args()
    run_training(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        model_name=args.model,
        project=args.project,
        name=args.name,
        status_file=args.status_file,
        log_file=args.log_file,
        device=args.device
    )
