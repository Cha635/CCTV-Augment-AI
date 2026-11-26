"""
Dataset Annotation Validator Module for CCTV Surveillance YOLO Datasets.
Performs thorough integrity checks on images, annotations, bounding boxes,
class distributions, and generates validation reports.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

def validate_yolo_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """
    Validates YOLO dataset structure and label integrity.
    
    Checks:
    - Train and Val image/label counts
    - Missing files (images without labels, labels without images)
    - Empty annotation files
    - Invalid bounding box coordinates (values outside [0, 1] or non-positive w/h)
    - Class ID validity (must be 0 or 1 for Person/Vehicle)
    - Class instance distributions
    """
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "dataset" / "images"
    labels_dir = dataset_path / "dataset" / "labels"
    
    # Fallback if structure is directly under dataset/ or root
    if not images_dir.exists():
        if (dataset_path / "images").exists():
            images_dir = dataset_path / "images"
            labels_dir = dataset_path / "labels"
            
    report = {
        "dataset_root": str(dataset_path),
        "valid": True,
        "splits": {},
        "total_images": 0,
        "total_labels": 0,
        "total_annotations": 0,
        "class_distribution": {0: 0, 1: 0},
        "class_names": {0: "person", 1: "vehicle"},
        "issues": [],
        "warnings": [],
    }
    
    splits = ["train", "val"]
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for split in splits:
        split_img_dir = images_dir / split
        split_lbl_dir = labels_dir / split
        
        split_report = {
            "image_count": 0,
            "label_count": 0,
            "missing_labels": [],
            "missing_images": [],
            "empty_labels": [],
            "corrupted_boxes": [],
            "invalid_classes": [],
            "class_distribution": {0: 0, 1: 0},
            "total_boxes": 0,
        }
        
        if not split_img_dir.exists() or not split_lbl_dir.exists():
            report["issues"].append(f"Missing directory for split: {split}")
            report["valid"] = False
            report["splits"][split] = split_report
            continue
            
        # Discover images and labels
        img_files = {f.stem: f for f in split_img_dir.iterdir() if f.suffix.lower() in valid_extensions}
        lbl_files = {f.stem: f for f in split_lbl_dir.iterdir() if f.suffix.lower() == ".txt"}
        
        split_report["image_count"] = len(img_files)
        split_report["label_count"] = len(lbl_files)
        report["total_images"] += len(img_files)
        report["total_labels"] += len(lbl_files)
        
        # Check correspondence
        img_stems = set(img_files.keys())
        lbl_stems = set(lbl_files.keys())
        
        unmatched_imgs = img_stems - lbl_stems
        unmatched_lbls = lbl_stems - img_stems
        
        if unmatched_imgs:
            split_report["missing_labels"] = list(unmatched_imgs)[:50]
            report["warnings"].append(f"{split} split: {len(unmatched_imgs)} images lack label files.")
            
        if unmatched_lbls:
            split_report["missing_images"] = list(unmatched_lbls)[:50]
            report["warnings"].append(f"{split} split: {len(unmatched_lbls)} labels lack image files.")
            
        # Validate label file contents
        for stem, lbl_path in lbl_files.items():
            if lbl_path.stat().st_size == 0:
                split_report["empty_labels"].append(stem)
                continue
                
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    
                if not lines:
                    split_report["empty_labels"].append(stem)
                    continue
                    
                for line_idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        split_report["corrupted_boxes"].append({
                            "file": stem,
                            "line": line_idx + 1,
                            "reason": f"Expected 5 tokens, got {len(parts)}: '{line}'"
                        })
                        report["valid"] = False
                        continue
                        
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    
                    if cls_id not in [0, 1]:
                        split_report["invalid_classes"].append({
                            "file": stem,
                            "class_id": cls_id
                        })
                        report["valid"] = False
                        
                    # Check coordinate boundaries [0.0, 1.0]
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        split_report["corrupted_boxes"].append({
                            "file": stem,
                            "box": [xc, yc, w, h],
                            "reason": "Coordinates outside [0, 1] normalized bounds"
                        })
                        report["valid"] = False
                        continue
                        
                    # Accumulate class counts
                    if cls_id in [0, 1]:
                        split_report["class_distribution"][cls_id] += 1
                        report["class_distribution"][cls_id] += 1
                        split_report["total_boxes"] += 1
                        report["total_annotations"] += 1
                        
            except Exception as e:
                split_report["corrupted_boxes"].append({"file": stem, "reason": str(e)})
                report["valid"] = False
                
        report["splits"][split] = split_report
        
    return report

def generate_validation_markdown(report: Dict[str, Any]) -> str:
    """Formats dataset validation report into markdown."""
    status_icon = "✅ VALID" if report["valid"] else "⚠️ ISSUES DETECTED"
    
    md = f"### Dataset Validation Report ({status_icon})\n\n"
    md += f"- **Dataset Path:** `{report['dataset_root']}`\n"
    md += f"- **Total Images:** {report['total_images']:,}\n"
    md += f"- **Total Labels:** {report['total_labels']:,}\n"
    md += f"- **Total Bounding Boxes:** {report['total_annotations']:,}\n"
    md += f"- **Person Instances (Class 0):** {report['class_distribution'].get(0, 0):,}\n"
    md += f"- **Vehicle Instances (Class 1):** {report['class_distribution'].get(1, 0):,}\n\n"
    
    md += "| Split | Images | Labels | Empty Labels | Person Count | Vehicle Count | Total BBoxes |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for split, data in report["splits"].items():
        md += f"| **{split.upper()}** | {data['image_count']} | {data['label_count']} | {len(data['empty_labels'])} | {data['class_distribution'].get(0, 0)} | {data['class_distribution'].get(1, 0)} | {data['total_boxes']} |\n"
        
    if report["issues"]:
        md += "\n#### Critical Issues:\n"
        for issue in report["issues"]:
            md += f"- ❌ {issue}\n"
            
    if report["warnings"]:
        md += "\n#### Advisory Warnings:\n"
        for w in report["warnings"]:
            md += f"- ⚠️ {w}\n"
            
    return md
