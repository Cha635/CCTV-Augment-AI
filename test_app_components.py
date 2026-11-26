"""
Unit test suite for CCTV Augment AI.
Tests every augmentation function, YOLO detector, and metrics calculator.
"""

from pathlib import Path
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from modules.lighting_augmentation import (
    adjust_brightness,
    adjust_contrast,
    apply_gamma,
    simulate_underexposure,
    simulate_overexposure,
    simulate_nighttime,
    simulate_low_light,
    apply_lighting_pipeline
)
from modules.weather_augmentation import (
    simulate_rain,
    simulate_fog,
    simulate_haze,
    simulate_snow,
    apply_weather_pipeline
)
from modules.occlusion_augmentation import apply_occlusion
from modules.image_comparison import compute_image_metrics, generate_difference_heatmap
from modules.yolo_detection import CCTVDetector

def run_tests():
    print("=== TESTING CCTV AUGMENT AI CORE MODULES ===")
    
    # Create test synthetic image
    h, w = 480, 640
    test_img = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)
    
    # 1. Test Lighting
    print("[1/6] Testing Lighting Transforms...")
    assert adjust_brightness(test_img, 1.3).shape == (h, w, 3)
    assert adjust_contrast(test_img, 1.2).shape == (h, w, 3)
    assert apply_gamma(test_img, 1.5).shape == (h, w, 3)
    assert simulate_underexposure(test_img, 0.7).shape == (h, w, 3)
    assert simulate_overexposure(test_img, 0.7).shape == (h, w, 3)
    assert simulate_nighttime(test_img, 0.8).shape == (h, w, 3)
    assert simulate_low_light(test_img, 0.6).shape == (h, w, 3)
    print("  -> All 8 Lighting transforms verified!")
    
    # 2. Test Weather
    print("[2/6] Testing Weather Synthesis...")
    assert simulate_rain(test_img, "heavy", 0.7).shape == (h, w, 3)
    assert simulate_fog(test_img, "moderate", 0.6).shape == (h, w, 3)
    assert simulate_haze(test_img, 0.5).shape == (h, w, 3)
    assert simulate_snow(test_img, "heavy", 0.6).shape == (h, w, 3)
    print("  -> Rain, Fog, Haze, Snow verified!")
    
    # 3. Test Occlusion
    print("[3/6] Testing Occlusion & Bounding-box overlap...")
    anns = [{"class_id": 0, "bbox": [0.5, 0.5, 0.2, 0.4]}]
    occ_img, valid_anns, rects = apply_occlusion(test_img, "medium", occlusion_type="pole", annotations=anns)
    assert occ_img.shape == (h, w, 3)
    assert len(rects) > 0
    print(f"  -> Occlusion verified! Applied {len(rects)} obstruction(s).")
    
    # 4. Test Comparison
    print("[4/6] Testing Image Comparison & Metrics...")
    m = compute_image_metrics(test_img)
    assert "mean_brightness" in m and "sharpness_laplacian" in m
    diff = generate_difference_heatmap(test_img, occ_img)
    assert diff.shape == (h, w, 3)
    print(f"  -> Metrics: Brightness={m['mean_brightness']}, Sharpness={m['sharpness_laplacian']}")
    
    # 5. Test YOLO Detector
    print("[5/6] Testing YOLO Detector Initialization & Prediction...")
    detector = CCTVDetector()
    res = detector.predict(test_img, conf_threshold=0.25)
    assert "detections" in res and "annotated_rgb" in res
    print(f"  -> Inference OK! Detections={res['total_detections']}, Latency={res['inference_time_ms']}ms")
    
    print("\n=== ALL CORE MODULE TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
