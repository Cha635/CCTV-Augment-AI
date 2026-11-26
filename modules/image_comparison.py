"""
Image Comparison Module for CCTV Augment AI.
Provides side-by-side visual analysis, difference heatmaps,
pixel distribution metrics, and composite export.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

def compute_image_metrics(img_rgb: np.ndarray) -> Dict[str, float]:
    """Computes photometric and frequency domain metrics."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    # Laplacian variance for sharpness/edge density
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    return {
        "mean_brightness": round(mean_val, 2),
        "contrast_std": round(std_val, 2),
        "sharpness_laplacian": round(lap_var, 2),
    }

def generate_difference_heatmap(orig_rgb: np.ndarray, aug_rgb: np.ndarray) -> np.ndarray:
    """
    Computes absolute pixel difference and maps to COLORMAP_JET/INFERNO
    to highlight augmentation perturbations.
    """
    # Resize aug if dimensions differ
    if orig_rgb.shape[:2] != aug_rgb.shape[:2]:
        aug_rgb = cv2.resize(aug_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]))
        
    diff = cv2.absdiff(orig_rgb, aug_rgb)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    
    # Normalize for vivid visualization
    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_bgr = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

def generate_side_by_side_composite(
    orig_rgb: np.ndarray,
    aug_rgb: np.ndarray,
    label_left: str = "ORIGINAL CCTV",
    label_right: str = "AUGMENTED"
) -> np.ndarray:
    """Combines original and augmented images into a labeled composite image."""
    h1, w1 = orig_rgb.shape[:2]
    h2, w2 = aug_rgb.shape[:2]
    
    target_h = max(h1, h2)
    orig_res = cv2.resize(orig_rgb, (int(w1 * (target_h / h1)), target_h))
    aug_res = cv2.resize(aug_rgb, (int(w2 * (target_h / h2)), target_h))
    
    composite = np.hstack([orig_res, aug_res])
    
    # Add top banner
    banner_h = 42
    banner = np.zeros((banner_h, composite.shape[1], 3), dtype=np.uint8)
    banner[:] = (15, 23, 42) # Deep slate
    
    cv2.putText(banner, label_left, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (6, 182, 212), 2)
    w_left = orig_res.shape[1]
    cv2.putText(banner, label_right, (w_left + 20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 158, 11), 2)
    
    # Divider line
    composite = np.vstack([banner, composite])
    cv2.line(composite, (w_left, 0), (w_left, composite.shape[0]), (30, 41, 59), 3)
    
    return composite

def plot_histogram_comparison(orig_rgb: np.ndarray, aug_rgb: np.ndarray) -> go.Figure:
    """Plots luminance histogram comparison using Plotly with accessible high contrast."""
    gray_orig = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY).ravel()
    gray_aug = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2GRAY).ravel()
    
    hist_orig, bins = np.histogram(gray_orig, bins=64, range=[0, 256])
    hist_aug, _ = np.histogram(gray_aug, bins=64, range=[0, 256])
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=hist_orig,
        mode="lines",
        name="Original Frame",
        line=dict(color="#22D3EE", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(34, 211, 238, 0.18)"
    ))
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=hist_aug,
        mode="lines",
        name="Augmented Frame",
        line=dict(color="#FBBF24", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(251, 191, 36, 0.18)"
    ))
    
    fig.update_layout(
        title=dict(
            text="Luminance Histogram Shift (Original vs Augmented)",
            font=dict(color="#F8FAFC", size=15, family="Plus Jakarta Sans, sans-serif")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151F32",
        font=dict(color="#CBD5E1", family="JetBrains Mono, monospace"),
        xaxis=dict(
            title=dict(text="Pixel Intensity (0-255)", font=dict(color="#CBD5E1", size=12)),
            tickfont=dict(color="#CBD5E1", size=11),
            gridcolor="#2E4166",
            zerolinecolor="#2E4166"
        ),
        yaxis=dict(
            title=dict(text="Pixel Frequency Count", font=dict(color="#CBD5E1", size=12)),
            tickfont=dict(color="#CBD5E1", size=11),
            gridcolor="#2E4166",
            zerolinecolor="#2E4166"
        ),
        margin=dict(l=40, r=40, t=50, b=70),
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
        )
    )
    return fig
