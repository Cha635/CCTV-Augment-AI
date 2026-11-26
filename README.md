# CCTV Augment AI: Surveillance Image Augmenter

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6_CUDA_12.4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics_v8-00FFFF?style=for-the-badge&logo=opencv&logoColor=white)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## 📹 Project Overview

**CCTV Augment AI** is an advanced, AI-powered computer vision and surveillance image augmentation platform engineered to dramatically enhance the robustness of security object-detection systems under severe, real-world environmental degradation.

Standard surveillance training datasets predominantly contain clear-weather, daylight footage. When deployed in production environments, CCTV models encounter extreme challenges:
- **Low-light parking lots, deep night shadows, blinding vehicle headlights, and sensor underexposure.**
- **Adverse atmospheric phenomena including heavy rainfall streaks, dense Koschmieder fog, particulate snowfall, and haze.**
- **Physical obstructions from utility poles, security fences, moving vehicles, and pedestrians.**

**CCTV Augment AI** solves this critical domain shift by synthetically expanding training datasets using physically grounded computer vision algorithms, coupled with strict YOLO bounding-box preservation and automatic occlusion-overlap filtering.

---

## 🚀 Key Features

### 1. Photometric Lighting Engine
- **Brightness Scaling:** Linear dynamic range expansion and attenuation.
- **Contrast Adjustment:** Mean-centered contrast scaling ($\alpha$-gain).
- **Non-Linear Gamma Correction:** $I_{out} = 255 \cdot (I/255)^\gamma$ for shadow lifting and crushing.
- **Underexposure Simulation:** Shadow crush with simulated sensor gain noise.
- **Overexposure & Specular Bloom:** Highlight clipping with Gaussian specular bloom.
- **Nighttime Simulation:** Darkening, cool chromatic shift, lens vignetting, and low-light sensor noise.
- **Low-Light / Twilight:** Chromatic desaturation and sensor grain.

### 2. Environmental Weather Synthesis
- **Koschmieder Atmospheric Scattering (Fog):** Physical depth-based optical transmission decay $I_{fog} = I \cdot t(x) + L \cdot (1 - t(x))$.
- **Directional Rain Streaks:** Randomized trajectory velocity lines with motion blur kernels.
- **Atmospheric Haze:** Veiling luminance glare and high-frequency softening.
- **Multi-Scale Snowfall:** Near-lens out-of-focus flakes combined with distant particulate snow.

### 3. Physical Occlusion Simulator
- **Structural Obstructions:** Vertical utility poles, horizontal security railings, and rectangular blinds.
- **Annotation Integrity:** Calculates Intersection-over-Box-Area ($IoA$). Automatically purges or flags bounding boxes occluded beyond threshold ($> 70\%$) to prevent false negative learning.

### 4. Interactive Diagnostics & Comparison
- Side-by-side synchronized original vs. augmented inspection.
- Spatial difference heatmaps mapped to JET/Inferno spectrum ($|I_{orig} - I_{aug}|$).
- Real-time luminance histogram shift curves (Plotly).
- Quantitative metrics: Mean brightness, contrast standard deviation, and Laplacian sharpness variance.

### 5. High-Throughput Batch Processor
- Process dozens or hundreds of surveillance frames in parallel.
- Systematic descriptive naming convention (`stem_technique_intensity.jpg`).
- One-click ZIP archive packaging and download.

### 6. Dataset Partitioning & Integrity Validation
- Verifies Kaggle YOLO Person & Vehicle dataset integrity: image-label correspondence, coordinate normalization ($0 \le x, y, w, h \le 1$), and empty file detection.
- Generates reproducible, seed-locked Train / Validation / Held-out Test splits (seed 42).
- Produces training-augmented splits while keeping validation and test splits 100% untouched.

### 7. Real-Time YOLO Object Detection HUD
- Supports Ultralytics YOLO nano models (`yolov8n.pt` and fine-tuned weights `best.pt`).
- Cyber-surveillance HUD: Neon Cyan bounding boxes for **Person (Class 0)**, Amber Orange for **Vehicle (Class 1)**, with confidence badges and corner brackets.
- Real-time inference latency tracking (ms/frame).

### 8. Native YOLO Training Suite
- Genuine Ultralytics PyTorch training engine executed as a background process.
- Hardware auto-detection: CUDA GPU acceleration (NVIDIA RTX 3050) with automatic CPU fallback.
- Real-time loss convergence curves and console log streaming.
- Experiment A (Original Dataset) vs. Experiment B (Augmented Dataset).

### 9. Environmental Robustness Benchmark
- Standard COCO metrics: Precision, Recall, mAP@50, mAP@50-95, F1-Score.
- Environmental degradation stress-testing across Clean, Rain, Fog, Night, and Occlusion conditions.
- Interactive Plotly comparison bar charts and radar retention graphs.

---

## 🛠️ Technology Stack

| Layer | Component | Version / Specification |
|---|---|---|
| **Language** | Python | 3.11.9 |
| **Deep Learning** | PyTorch | 2.6.0+cu124 (NVIDIA CUDA 12.4 Acceleration) |
| **Object Detection**| Ultralytics YOLO | v8.4.216 |
| **Vision & Augment**| OpenCV / Albumentations | cv2 4.13.0, Albumentations 2.0.8 |
| **Web Dashboard** | Streamlit | 1.67.0 |
| **Data & Charts** | Plotly / NumPy / PIL | Plotly 7.1.0, NumPy 2.4.3 |

---

## 📂 System Architecture

```
CCTV_Augment_AI/
│
├── app.py                      # Main Streamlit Cyber-Surveillance Web Application
├── config.py                   # Centralized configuration, paths, hardware detection, theme
├── train_runner.py             # Standalone background YOLO training runner with callbacks
├── setup_dataset.py            # Dataset validation, split generator, and augmented set creator
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
│
├── modules/
│   ├── ui_components.py        # Dark theme CSS, metrics cards, status indicators
│   ├── image_upload.py         # File ingestion, metadata extraction, validation
│   ├── lighting_augmentation.py# 8 Photometric lighting algorithms
│   ├── weather_augmentation.py # Rain, fog (Koschmieder), haze, snow synthesis
│   ├── occlusion_augmentation.py# Structural occlusion with bounding-box overlap validation
│   ├── image_comparison.py     # Side-by-side comparison, heatmaps, histogram diffs
│   ├── batch_processing.py     # Multi-image processing & ZIP packaging
│   ├── annotation_validator.py # YOLO label integrity & bounds validator
│   ├── dataset_manager.py      # Split creator & augmented training dataset generator
│   ├── yolo_detection.py       # CCTVDetector class with tactical HUD renderer
│   ├── model_training.py       # Background training process manager & status reader
│   └── model_evaluation.py     # Validation evaluation & environmental robustness benchmark
│
├── dataset/
│   ├── splits/                 # Clean Train (1,360), Val (400), Held-out Test (240)
│   │   └── data_original.yaml
│   └── augmented/              # Augmented Train (2,260) with original Val & Test
│       └── data_augmented.yaml
│
├── models/
│   ├── pretrained/             # yolov8n.pt base model
│   └── trained/                # Trained experiment weights (best.pt, last.pt)
│
├── outputs/
│   ├── augmented_images/       # Individual augmented downloads
│   ├── batches/                # Batch archives & ZIP files
│   └── reports/                # Validation & evaluation JSON summaries
│
└── logs/                       # Training logs and JSON status telemetry
```

---

## 💻 Installation & Quickstart

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/<your-username>/CCTV_Augment_AI.git
cd CCTV_Augment_AI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize & Validate Dataset
Validate the provided Kaggle dataset and create reproducible splits:
```powershell
python setup_dataset.py
```

### 4. Launch the Web Application
```powershell
streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 📊 Model Training & Benchmarking

### Train Experiment A (Original Clean Dataset)
```powershell
python train_runner.py --data dataset/splits/data_original.yaml --epochs 15 --batch 16 --name cctv_exp_original
```

### Train Experiment B (Augmented Dataset)
```powershell
python train_runner.py --data dataset/augmented/data_augmented.yaml --epochs 15 --batch 16 --name cctv_exp_augmented
```

---

## 🐙 Pushing to GitHub

To publish this project to your GitHub account:

1. **Initialize Git (if not already initialized):**
   ```bash
   git init
   git branch -M main
   ```

2. **Stage and Commit Files:**
   ```bash
   git add .
   git commit -m "feat: complete CCTV Augment AI surveillance platform"
   ```

3. **Link to GitHub Remote and Push:**
   ```bash
   git remote add origin https://github.com/<your-username>/CCTV_Augment_AI.git
   git push -u origin main
   ```

> **Note:** The included `.gitignore` keeps large image sets, dataset raw copies, and temporary training runs out of version control while keeping fine-tuned model weights and architecture definitions intact.

---

## 🛡️ License
Distributed under the MIT License. Built for security, surveillance, and computer vision research.
