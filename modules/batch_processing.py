"""
Batch Processing Module for CCTV Augment AI.
Handles multi-image batch transformations, progress tracking,
systematic naming, and ZIP packaging.
"""

from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
import time
import zipfile
import cv2
import numpy as np

from modules.lighting_augmentation import apply_lighting_pipeline
from modules.weather_augmentation import apply_weather_pipeline
from modules.occlusion_augmentation import apply_occlusion
import config

def process_single_image_with_techniques(
    img_rgb: np.ndarray,
    stem: str,
    techniques: List[str],
    intensity: float = 0.5,
    output_dir: Path = config.AUGMENTED_IMAGES_DIR
) -> List[Dict[str, Any]]:
    """
    Applies requested techniques to one image and saves with meaningful names.
    Returns list of result metadata dicts.
    """
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original copy for reference
    orig_path = output_dir / f"{stem}_original.jpg"
    if not orig_path.exists():
        cv2.imwrite(str(orig_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
    for tech in techniques:
        t_lower = tech.lower()
        aug_rgb = None
        suffix = t_lower.replace(" ", "_")
        
        # Categorize
        if any(k in t_lower for k in ["night", "low_light", "under", "over", "contrast", "bright", "gamma"]):
            aug_rgb = apply_lighting_pipeline(img_rgb, t_lower, intensity=intensity)
        elif any(k in t_lower for k in ["rain", "fog", "haze", "snow"]):
            preset = "heavy" if intensity > 0.7 else ("light" if intensity < 0.35 else "moderate")
            aug_rgb = apply_weather_pipeline(img_rgb, t_lower, preset=preset, intensity=intensity)
        elif "occlusion" in t_lower or "pole" in t_lower or "fence" in t_lower:
            occ_type = "pole" if "pole" in t_lower else ("fence" if "fence" in t_lower else "box")
            aug_rgb, _, _ = apply_occlusion(img_rgb, preset="medium", size_ratio=intensity * 0.4, occlusion_type=occ_type)
            suffix = f"occlusion_{occ_type}"
        else:
            aug_rgb = apply_lighting_pipeline(img_rgb, t_lower, intensity=intensity)
            
        if aug_rgb is not None:
            filename = f"{stem}_{suffix}.jpg"
            save_path = output_dir / filename
            cv2.imwrite(str(save_path), cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR))
            results.append({
                "original_stem": stem,
                "technique": tech,
                "filename": filename,
                "path": str(save_path),
                "resolution": f"{aug_rgb.shape[1]}x{aug_rgb.shape[0]}",
                "intensity": intensity
            })
            
    return results

def run_batch_pipeline(
    images_dict: Dict[str, np.ndarray],
    techniques: List[str],
    intensity: float = 0.5,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Executes batch augmentation across multiple images.
    Returns summary statistics and output ZIP path.
    """
    if output_dir is None:
        batch_id = int(time.time())
        output_dir = config.BATCHES_DIR / f"batch_{batch_id}"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_images = len(images_dict)
    total_tasks = total_images * len(techniques)
    completed_tasks = 0
    success_count = 0
    fail_count = 0
    generated_files = []
    
    start_time = time.time()
    
    for idx, (stem, img_rgb) in enumerate(images_dict.items()):
        try:
            res = process_single_image_with_techniques(
                img_rgb=img_rgb,
                stem=stem,
                techniques=techniques,
                intensity=intensity,
                output_dir=output_dir
            )
            success_count += len(res)
            generated_files.extend(res)
        except Exception as e:
            fail_count += len(techniques)
            
        completed_tasks += len(techniques)
        if progress_callback:
            progress = min(1.0, completed_tasks / max(1, total_tasks))
            progress_callback(progress, f"Processed {idx + 1}/{total_images} images ({completed_tasks}/{total_tasks} variations)")
            
    # Create ZIP archive
    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in output_dir.glob("*.jpg"):
            zipf.write(f, arcname=f.name)
            
    duration = round(time.time() - start_time, 2)
    
    return {
        "status": "success",
        "total_source_images": total_images,
        "techniques_applied": techniques,
        "generated_count": success_count,
        "failed_count": fail_count,
        "duration_sec": duration,
        "batch_dir": str(output_dir),
        "zip_path": str(zip_path),
        "generated_files": generated_files
    }
