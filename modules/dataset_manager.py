"""
Dataset Manager Module for CCTV Augment AI.
Handles:
1. Kaggle dataset loading and discovery
2. Creation of reproducible Train/Val/Test splits (with held-out test split)
3. Augmented training dataset generation with strict annotation preservation
4. YOLO data.yaml configuration generation
5. ZIP exports
"""

from pathlib import Path
import random
import shutil
import zipfile
import os
from typing import Dict, Any, List, Optional
import cv2
import yaml
import numpy as np

from modules.lighting_augmentation import apply_lighting_pipeline
from modules.weather_augmentation import apply_weather_pipeline
from modules.occlusion_augmentation import apply_occlusion
import config

def discover_dataset(custom_path: Optional[str] = None) -> Path:
    """Discovers the root of the YOLO Person & Vehicle dataset."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    env_dataset = os.environ.get("CCTV_DATASET_DIR")
    if env_dataset:
        candidates.append(Path(env_dataset))
    candidates.extend([
        config.DEFAULT_DATASET_DIR,
        config.BASE_DIR / "yolo-person-vehicle-dataset",
        config.PROJECT_ROOT / "yolo-person-vehicle-dataset",
        config.BASE_DIR / "dataset" / "original",
        config.BASE_DIR / "dataset" / "splits",
    ])
    
    for candidate in candidates:
        if candidate.exists() and (candidate / "data.yaml").exists():
            return candidate
        if candidate.exists() and ((candidate / "dataset" / "images").exists() or (candidate / "images").exists()):
            return candidate
            
    return config.DEFAULT_DATASET_DIR

def prepare_reproducible_splits(
    original_dataset_dir: Path,
    output_splits_dir: Path = config.SPLITS_DIR,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Creates clean, reproducible train/val/test splits without altering original files.
    Extracts a dedicated held-out test split from the original training pool.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Locate original images & labels
    orig_images = original_dataset_dir / "dataset" / "images"
    orig_labels = original_dataset_dir / "dataset" / "labels"
    if not orig_images.exists():
        orig_images = original_dataset_dir / "images"
        orig_labels = original_dataset_dir / "labels"
        
    orig_train_img = orig_images / "train"
    orig_train_lbl = orig_labels / "train"
    orig_val_img = orig_images / "val"
    orig_val_lbl = orig_labels / "val"
    
    # Destination directories
    split_train_img = output_splits_dir / "images" / "train"
    split_train_lbl = output_splits_dir / "labels" / "train"
    split_val_img = output_splits_dir / "images" / "val"
    split_val_lbl = output_splits_dir / "labels" / "val"
    split_test_img = output_splits_dir / "images" / "test"
    split_test_lbl = output_splits_dir / "labels" / "test"
    
    for d in [split_train_img, split_train_lbl, split_val_img, split_val_lbl, split_test_img, split_test_lbl]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Get all training images with matching labels
    valid_extensions = {".jpg", ".jpeg", ".png"}
    train_files = [f for f in orig_train_img.iterdir() if f.suffix.lower() in valid_extensions and (orig_train_lbl / f"{f.stem}.txt").exists()]
    train_files.sort(key=lambda x: x.name)
    random.shuffle(train_files)
    
    num_test = int(len(train_files) * test_ratio)
    test_files = train_files[:num_test]
    final_train_files = train_files[num_test:]
    
    # Check if splits already populated to avoid redundant copying
    if len(list(split_train_img.iterdir())) == len(final_train_files) and len(list(split_test_img.iterdir())) == len(test_files):
        yaml_file = output_splits_dir / "data_original.yaml"
        if not yaml_file.exists():
            original_yaml_content = {
                "path": "dataset/splits",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": config.CLASS_NAMES
            }
            with open(yaml_file, "w") as yf:
                yaml.dump(original_yaml_content, yf, default_flow_style=False)
        return {
            "status": "ready",
            "train_count": len(final_train_files),
            "val_count": len(list(split_val_img.iterdir())),
            "test_count": len(test_files),
            "splits_dir": str(output_splits_dir),
            "yaml_path": str(yaml_file)
        }
        
    # Copy Clean Train Split
    for f in final_train_files:
        shutil.copy2(f, split_train_img / f.name)
        lbl_file = orig_train_lbl / f"{f.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, split_train_lbl / lbl_file.name)
        
    # Copy Held-out Test Split
    for f in test_files:
        shutil.copy2(f, split_test_img / f.name)
        lbl_file = orig_train_lbl / f"{f.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, split_test_lbl / lbl_file.name)
        
    # Copy Val Split
    if orig_val_img.exists():
        for f in orig_val_img.iterdir():
            if f.suffix.lower() in valid_extensions:
                shutil.copy2(f, split_val_img / f.name)
                lbl_file = orig_val_lbl / f"{f.stem}.txt"
                if lbl_file.exists():
                    shutil.copy2(lbl_file, split_val_lbl / lbl_file.name)
                    
    # Generate portable data_original.yaml
    original_yaml_content = {
        "path": "dataset/splits",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": config.CLASS_NAMES
    }
    with open(output_splits_dir / "data_original.yaml", "w") as yf:
        yaml.dump(original_yaml_content, yf, default_flow_style=False)
        
    return {
        "status": "created",
        "train_count": len(final_train_files),
        "val_count": len(list(split_val_img.iterdir())),
        "test_count": len(test_files),
        "splits_dir": str(output_splits_dir),
        "yaml_path": str(output_splits_dir / "data_original.yaml")
    }

def generate_augmented_training_dataset(
    splits_dir: Path = config.SPLITS_DIR,
    output_aug_dir: Path = config.AUGMENTED_DATA_DIR,
    techniques: Optional[List[str]] = None,
    samples_per_image: int = 1,
    max_images_to_augment: Optional[int] = None,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Generates an augmented training dataset from the clean training split.
    Guarantees:
    - Original training images preserved in augmented dataset.
    - Augmented images appended with distinct descriptive suffixes.
    - Photometric transforms preserve bounding box labels intact.
    - Occlusion transforms drop boxes occluded > 70%.
    - Validation and test splits are shared and kept 100% unaugmented.
    """
    if techniques is None:
        techniques = ["night", "rain", "fog", "low_light", "occlusion_pole", "underexposure"]
        
    train_img_dir = splits_dir / "images" / "train"
    train_lbl_dir = splits_dir / "labels" / "train"
    
    aug_train_img = output_aug_dir / "images" / "train"
    aug_train_lbl = output_aug_dir / "labels" / "train"
    aug_train_img.mkdir(parents=True, exist_ok=True)
    aug_train_lbl.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy original training images to augmented training dir
    orig_train_files = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
    for f in orig_train_files:
        shutil.copy2(f, aug_train_img / f.name)
        lbl = train_lbl_dir / f"{f.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, aug_train_lbl / lbl.name)
            
    # Subset if specified
    targets = orig_train_files[:max_images_to_augment] if max_images_to_augment else orig_train_files
    total_steps = len(targets)
    
    generated_count = 0
    valid_extensions = {".jpg", ".jpeg", ".png"}
    
    for idx, img_path in enumerate(targets):
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
            
        # Read image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Read labels
        annotations = []
        with open(lbl_path, "r") as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) == 5:
                    annotations.append({
                        "class_id": int(parts[0]),
                        "bbox": [float(p) for p in parts[1:]]
                    })
                    
        # Apply each requested technique
        for tech in techniques:
            aug_img_rgb = None
            aug_annotations = list(annotations)
            suffix = tech.lower()
            
            if "night" in tech:
                aug_img_rgb = apply_lighting_pipeline(img_rgb, "night", intensity=0.7)
            elif "rain" in tech:
                aug_img_rgb = apply_weather_pipeline(img_rgb, "rain", preset="heavy", intensity=0.7)
            elif "fog" in tech:
                aug_img_rgb = apply_weather_pipeline(img_rgb, "fog", preset="moderate", intensity=0.6)
            elif "haze" in tech:
                aug_img_rgb = apply_weather_pipeline(img_rgb, "haze", intensity=0.6)
            elif "snow" in tech:
                aug_img_rgb = apply_weather_pipeline(img_rgb, "snow", preset="light", intensity=0.5)
            elif "low_light" in tech:
                aug_img_rgb = apply_lighting_pipeline(img_rgb, "low_light", intensity=0.6)
            elif "under" in tech:
                aug_img_rgb = apply_lighting_pipeline(img_rgb, "underexposure", intensity=0.6)
            elif "over" in tech:
                aug_img_rgb = apply_lighting_pipeline(img_rgb, "overexposure", intensity=0.6)
            elif "contrast" in tech:
                aug_img_rgb = apply_lighting_pipeline(img_rgb, "contrast", intensity=0.7)
            elif "occlusion" in tech:
                occ_type = "pole" if "pole" in tech else ("fence" if "fence" in tech else "box")
                aug_img_rgb, valid_ann, _ = apply_occlusion(
                    img_rgb,
                    preset="medium",
                    occlusion_type=occ_type,
                    annotations=annotations,
                    max_occlusion_threshold=0.70
                )
                aug_annotations = valid_ann
                suffix = f"occ_{occ_type}"
            else:
                aug_img_rgb = apply_lighting_pipeline(img_rgb, "contrast", intensity=0.6)
                
            if aug_img_rgb is not None:
                new_stem = f"{img_path.stem}_{suffix}"
                new_img_path = aug_train_img / f"{new_stem}.jpg"
                new_lbl_path = aug_train_lbl / f"{new_stem}.txt"
                
                # Save augmented image
                cv2.imwrite(str(new_img_path), cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR))
                
                # Save preserved/updated labels
                with open(new_lbl_path, "w") as out_lf:
                    for ann in aug_annotations:
                        cls_id = ann["class_id"]
                        xc, yc, w, h = ann["bbox"]
                        out_lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                        
                generated_count += 1
                
        if progress_callback and idx % 20 == 0:
            progress_callback(min(1.0, (idx + 1) / total_steps))
            
    # Ensure shared unaugmented validation & test sets exist in output_aug_dir for full standalone portability
    aug_val_img = output_aug_dir / "images" / "val"
    aug_val_lbl = output_aug_dir / "labels" / "val"
    aug_test_img = output_aug_dir / "images" / "test"
    aug_test_lbl = output_aug_dir / "labels" / "test"
    for d in [aug_val_img, aug_val_lbl, aug_test_img, aug_test_lbl]:
        d.mkdir(parents=True, exist_ok=True)
        
    src_val_img = splits_dir / "images" / "val"
    src_val_lbl = splits_dir / "labels" / "val"
    if src_val_img.exists():
        for f in src_val_img.glob("*.jpg"):
            if not (aug_val_img / f.name).exists():
                shutil.copy2(f, aug_val_img / f.name)
            lbl = src_val_lbl / f"{f.stem}.txt"
            if lbl.exists() and not (aug_val_lbl / lbl.name).exists():
                shutil.copy2(lbl, aug_val_lbl / lbl.name)
                
    src_test_img = splits_dir / "images" / "test"
    src_test_lbl = splits_dir / "labels" / "test"
    if src_test_img.exists():
        for f in src_test_img.glob("*.jpg"):
            if not (aug_test_img / f.name).exists():
                shutil.copy2(f, aug_test_img / f.name)
            lbl = src_test_lbl / f"{f.stem}.txt"
            if lbl.exists() and not (aug_test_lbl / lbl.name).exists():
                shutil.copy2(lbl, aug_test_lbl / lbl.name)

    # Generate portable data_augmented.yaml linking augmented train with shared val & test
    aug_yaml_content = {
        "path": "dataset/augmented",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": config.CLASS_NAMES
    }
    with open(output_aug_dir / "data_augmented.yaml", "w") as yf:
        yaml.dump(aug_yaml_content, yf, default_flow_style=False)
        
    return {
        "status": "success",
        "original_train_count": len(orig_train_files),
        "generated_augmented_count": generated_count,
        "total_augmented_train": len(list(aug_train_img.iterdir())),
        "yaml_path": str(output_aug_dir / "data_augmented.yaml")
    }

def export_dataset_zip(source_dir: Path, output_zip_path: Path) -> Path:
    """Packages dataset directory into a downloadable ZIP archive."""
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in source_dir.rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(source_dir)
                zipf.write(file, arcname=str(rel_path))
    return output_zip_path
