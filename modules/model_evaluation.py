"""
Model Evaluation and Robustness Benchmark Module for CCTV Augment AI.
Evaluates YOLO models on held-out test datasets, computes mAP@50, mAP@50-95,
Precision, Recall, and stress-tests robustness under environmental perturbations.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

import config
from modules.lighting_augmentation import simulate_nighttime
from modules.weather_augmentation import simulate_rain, simulate_fog
from modules.occlusion_augmentation import apply_occlusion

def evaluate_yolo_model(
    model_path: str,
    data_yaml: str,
    split: str = "test",
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Runs standard Ultralytics YOLO validation/evaluation on the specified split.
    """
    if not ULTRALYTICS_AVAILABLE:
        return {"error": "Ultralytics is not installed."}
        
    model = YOLO(model_path)
    
    # Resolve device
    if device == "auto":
        hw = config.get_hardware_info()
        dev = "0" if hw["cuda_available"] else "cpu"
    else:
        dev = device
        
    start_time = time.time()
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        device=dev,
        verbose=False
    )
    duration = time.time() - start_time
    
    # Extract results
    results_dict = metrics.results_dict
    
    # Precision, Recall, mAP50, mAP50-95
    p = float(results_dict.get("metrics/precision(B)", 0.0))
    r = float(results_dict.get("metrics/recall(B)", 0.0))
    map50 = float(results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(results_dict.get("metrics/mAP50-95(B)", 0.0))
    
    # Speed
    speed = metrics.speed
    inference_ms = float(speed.get("inference", 0.0))
    
    return {
        "model_path": model_path,
        "split": split,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "f1_score": round(2 * p * r / max(1e-5, p + r), 4),
        "inference_ms": round(inference_ms, 2),
        "eval_duration_sec": round(duration, 2),
    }

def run_environmental_robustness_benchmark(
    model_orig_path: str,
    model_aug_path: str,
    test_img_dir: Path,
    max_samples: int = 50,
    conf_thresh: float = 0.3
) -> Dict[str, Any]:
    """
    Stress-tests both models across 5 conditions:
    1. Clean Test Images
    2. Rain Simulation
    3. Fog Simulation
    4. Night Simulation
    5. Occlusion Simulation
    
    Compares detection count retention and average confidence.
    """
    if not ULTRALYTICS_AVAILABLE:
        return {"error": "Ultralytics YOLO unavailable."}
        
    model_orig = YOLO(model_orig_path)
    model_aug = YOLO(model_aug_path)
    
    valid_exts = {".jpg", ".jpeg", ".png"}
    test_images = [f for f in test_img_dir.iterdir() if f.suffix.lower() in valid_exts][:max_samples]
    
    conditions = ["clean", "rain", "fog", "night", "occlusion"]
    
    stats = {
        "conditions": conditions,
        "original_model": {c: {"detections": 0, "avg_conf": 0.0, "confs": []} for c in conditions},
        "augmented_model": {c: {"detections": 0, "avg_conf": 0.0, "confs": []} for c in conditions},
    }
    
    for img_path in test_images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        for cond in conditions:
            # Generate perturbation
            if cond == "clean":
                perturbed = img_rgb
            elif cond == "rain":
                perturbed = simulate_rain(img_rgb, preset="heavy", density=0.8, streak_intensity=0.8)
            elif cond == "fog":
                perturbed = simulate_fog(img_rgb, preset="heavy", intensity=0.75)
            elif cond == "night":
                perturbed = simulate_nighttime(img_rgb, intensity=0.75)
            elif cond == "occlusion":
                perturbed, _, _ = apply_occlusion(img_rgb, preset="high", occlusion_type="pole")
                
            # Model Original
            res_orig = model_orig.predict(source=perturbed, conf=conf_thresh, verbose=False)[0]
            boxes_orig = res_orig.boxes
            stats["original_model"][cond]["detections"] += len(boxes_orig)
            for b in boxes_orig:
                stats["original_model"][cond]["confs"].append(float(b.conf[0].item()))
                
            # Model Augmented
            res_aug = model_aug.predict(source=perturbed, conf=conf_thresh, verbose=False)[0]
            boxes_aug = res_aug.boxes
            stats["augmented_model"][cond]["detections"] += len(boxes_aug)
            for b in boxes_aug:
                stats["augmented_model"][cond]["confs"].append(float(b.conf[0].item()))
                
    # Calculate averages
    for model_key in ["original_model", "augmented_model"]:
        for cond in conditions:
            confs = stats[model_key][cond]["confs"]
            stats[model_key][cond]["avg_conf"] = round(float(np.mean(confs)), 3) if confs else 0.0
            del stats[model_key][cond]["confs"] # cleanup
            
    return stats

def plot_model_comparison_bar(eval_orig: Dict[str, Any], eval_aug: Dict[str, Any]) -> go.Figure:
    """Generates comparison bar chart of core detection metrics with high-contrast styling."""
    metrics = ["Precision", "Recall", "mAP@50", "mAP@50-95", "F1 Score"]
    orig_vals = [
        eval_orig.get("precision", 0.0),
        eval_orig.get("recall", 0.0),
        eval_orig.get("mAP50", 0.0),
        eval_orig.get("mAP50_95", 0.0),
        eval_orig.get("f1_score", 0.0)
    ]
    aug_vals = [
        eval_aug.get("precision", 0.0),
        eval_aug.get("recall", 0.0),
        eval_aug.get("mAP50", 0.0),
        eval_aug.get("mAP50_95", 0.0),
        eval_aug.get("f1_score", 0.0)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics,
        y=orig_vals,
        name="Experiment A (Clean Model)",
        marker_color="#3B82F6",
        text=[f"{v:.3f}" for v in orig_vals],
        textfont=dict(color="#FFFFFF", size=12, family="JetBrains Mono, monospace"),
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=metrics,
        y=aug_vals,
        name="Experiment B (Augmented Model)",
        marker_color="#22D3EE",
        text=[f"{v:.3f}" for v in aug_vals],
        textfont=dict(color="#0B1120", size=12, family="JetBrains Mono, monospace"),
        textposition="auto"
    ))
    
    fig.update_layout(
        title=dict(
            text="Detection Performance Benchmark: Clean vs Augmented Training",
            font=dict(color="#F8FAFC", size=16, family="Plus Jakarta Sans, sans-serif")
        ),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151F32",
        font=dict(color="#CBD5E1", family="Plus Jakarta Sans, sans-serif"),
        yaxis=dict(
            title=dict(text="Metric Score (0.00 - 1.00)", font=dict(color="#CBD5E1", size=13)),
            range=[0, 1.08],
            gridcolor="#2E4166",
            tickfont=dict(color="#CBD5E1", size=11),
            zerolinecolor="#2E4166"
        ),
        xaxis=dict(
            title=dict(text="Validation Metric", font=dict(color="#CBD5E1", size=13)),
            gridcolor="#2E4166",
            tickfont=dict(color="#F8FAFC", size=12, family="Plus Jakarta Sans, sans-serif")
        ),
        legend=dict(
            font=dict(color="#F8FAFC", size=12),
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(21, 31, 50, 0.90)",
            bordercolor="#2E4166",
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="#1D2B42",
            font_color="#F8FAFC",
            font_size=12,
            bordercolor="#22D3EE"
        ),
        margin=dict(l=40, r=40, t=55, b=85)
    )
    return fig

def plot_robustness_radar(bench_data: Dict[str, Any]) -> go.Figure:
    """Generates radar chart for environmental condition retention with high-contrast styling."""
    categories = [c.capitalize() for c in bench_data["conditions"]]
    orig_dets = [bench_data["original_model"][c]["detections"] for c in bench_data["conditions"]]
    aug_dets = [bench_data["augmented_model"][c]["detections"] for c in bench_data["conditions"]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=orig_dets + [orig_dets[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Clean Baseline Model',
        line=dict(color='#3B82F6', width=2.5),
        fillcolor='rgba(59, 130, 246, 0.25)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=aug_dets + [aug_dets[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Augmented Robust Model',
        line=dict(color='#22D3EE', width=3),
        fillcolor='rgba(34, 211, 238, 0.35)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                gridcolor="#2E4166",
                linecolor="#2E4166",
                tickfont=dict(color="#CBD5E1", size=10)
            ),
            angularaxis=dict(
                gridcolor="#2E4166",
                linecolor="#2E4166",
                tickfont=dict(color="#F8FAFC", size=12, family="Plus Jakarta Sans, sans-serif")
            ),
            bgcolor="#151F32"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family="Plus Jakarta Sans, sans-serif"),
        title=dict(
            text="Environmental Stress-Testing: Detections Under Degraded Conditions",
            font=dict(color="#F8FAFC", size=15, family="Plus Jakarta Sans, sans-serif"),
            y=0.98,
            x=0.5,
            xanchor="center"
        ),
        legend=dict(
            font=dict(color="#F8FAFC", size=12),
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(21, 31, 50, 0.90)",
            bordercolor="#2E4166",
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="#1D2B42",
            font_color="#F8FAFC",
            font_size=12,
            bordercolor="#22D3EE"
        ),
        margin=dict(l=50, r=50, t=65, b=75)
    )
    return fig
