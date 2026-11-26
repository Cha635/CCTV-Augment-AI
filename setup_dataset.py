"""
Setup and validation script for CCTV Augment AI dataset.
Runs annotation validation and creates reproducible train/val/test splits.
"""

from pathlib import Path
import sys
import json

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from modules.annotation_validator import validate_yolo_dataset, generate_validation_markdown
from modules.dataset_manager import discover_dataset, prepare_reproducible_splits, generate_augmented_training_dataset

def main():
    print("=== CCTV AUGMENT AI: DATASET INSPECTION & INITIALIZATION ===")
    
    # 1. Discover dataset
    dataset_path = discover_dataset()
    print(f"Discovered Dataset Root: {dataset_path}")
    
    # 2. Validate
    print("\n[Step 1/3] Validating Dataset Integrity...")
    rep = validate_yolo_dataset(dataset_path)
    print(f"Validation Result: {'VALID' if rep['valid'] else 'INVALID'}")
    print(f"Total Images: {rep['total_images']}, Total Labels: {rep['total_labels']}")
    print(f"Class Distribution: Person={rep['class_distribution'].get(0, 0)}, Vehicle={rep['class_distribution'].get(1, 0)}")
    
    report_file = config.REPORTS_DIR / "dataset_validation_report.json"
    with open(report_file, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"Saved validation report to: {report_file}")
    
    # 3. Create Reproducible Splits (with held-out test split)
    print("\n[Step 2/3] Creating Reproducible Splits (Train / Val / Held-out Test)...")
    split_info = prepare_reproducible_splits(dataset_path, output_splits_dir=config.SPLITS_DIR, test_ratio=0.15, seed=42)
    print(f"Splits Status: {split_info['status']}")
    print(f"Train Count: {split_info['train_count']}, Val Count: {split_info['val_count']}, Test Count: {split_info['test_count']}")
    print(f"Original YOLO Data YAML: {config.SPLITS_DIR / 'data_original.yaml'}")
    
    # 4. Generate Augmented Training Dataset (Sample 150 images to test pipeline)
    print("\n[Step 3/3] Generating Augmented Training Dataset...")
    aug_info = generate_augmented_training_dataset(
        splits_dir=config.SPLITS_DIR,
        output_aug_dir=config.AUGMENTED_DATA_DIR,
        techniques=["night", "rain", "fog", "low_light", "occlusion_pole", "underexposure"],
        max_images_to_augment=150
    )
    print(f"Augmented Dataset Generated: {aug_info['generated_augmented_count']} augmented variations created.")
    print(f"Total Augmented Train Pool: {aug_info['total_augmented_train']}")
    print(f"Augmented YOLO Data YAML: {config.AUGMENTED_DATA_DIR / 'data_augmented.yaml'}")
    
    print("\n=== DATASET INITIALIZATION COMPLETE! ===")

if __name__ == "__main__":
    main()
