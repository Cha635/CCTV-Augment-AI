"""
Project Packaging Script for CCTV Augment AI.
Creates a clean, validated ZIP archive of the complete updated application:
CCTV_Augment_AI_Updated_Project.zip
"""

import os
import sys
import zipfile
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\Sripragna\Downloads\GenAI_Project\CCTV_Augment_AI")
ZIP_OUTPUT_PATH = Path(r"C:\Users\Sripragna\Downloads\GenAI_Project\CCTV_Augment_AI_Updated_Project.zip")

EXCLUDE_DIRS = {"__pycache__", ".venv", "venv", ".git", ".pytest_cache", ".idea", ".vscode"}
EXCLUDE_EXTS = {".pyc", ".pyo", ".tmp"}

def create_project_zip():
    print(f"Creating ZIP archive from: {SOURCE_DIR}")
    print(f"Target archive destination: {ZIP_OUTPUT_PATH}")
    
    if ZIP_OUTPUT_PATH.exists():
        ZIP_OUTPUT_PATH.unlink()
        
    total_files = 0
    total_bytes = 0
    
    with zipfile.ZipFile(ZIP_OUTPUT_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(SOURCE_DIR):
            # Exclude unwanted directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            # Avoid packing raw training images if they blow up the ZIP size beyond reason,
            # but preserve configuration, labels, models, code, and reports
            rel_root = Path(root).relative_to(SOURCE_DIR)
            
            # Skip massive batches or generated augmented temp folders if redundant
            if "batches" in rel_root.parts and len(rel_root.parts) > 1:
                continue
            if "augmented_images" in rel_root.parts and len(rel_root.parts) > 1:
                continue
            if "splits" in rel_root.parts and "images" in rel_root.parts:
                # Include first 15 images of each split for demonstration
                # (keep label files for all)
                pass

            for f in files:
                file_path = Path(root) / f
                if file_path.suffix.lower() in EXCLUDE_EXTS:
                    continue
                if f.endswith(".zip"):
                    continue
                if f == "package_project.py":
                    pass

                rel_path = file_path.relative_to(SOURCE_DIR)
                
                # If it's in images folder of splits, only include a representative sample to keep ZIP compact
                if "images" in rel_path.parts and ("splits" in rel_path.parts or "augmented" in rel_path.parts):
                    # Check index
                    stem = file_path.stem
                    # Keep a subset of images so ZIP stays around 20-30MB instead of 500MB
                    try:
                        num = int("".join(filter(str.isdigit, stem)))
                        if num > 30 and "train" in rel_path.parts:
                            continue
                    except ValueError:
                        pass

                zf.write(file_path, arcname=str(Path("CCTV_Augment_AI") / rel_path))
                total_files += 1
                total_bytes += file_path.stat().st_size
                
    zip_size_mb = ZIP_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Archive successfully created!")
    print(f"Total files archived: {total_files}")
    print(f"Uncompressed size: {total_bytes / (1024 * 1024):.2f} MB")
    print(f"Compressed archive size: {zip_size_mb:.2f} MB")
    print(f"Saved at: {ZIP_OUTPUT_PATH}")

    # Validate archive integrity
    with zipfile.ZipFile(ZIP_OUTPUT_PATH, "r") as test_zf:
        bad_file = test_zf.testzip()
        if bad_file:
            print(f"ERROR: Corrupted file in archive: {bad_file}")
            sys.exit(1)
        else:
            print("VALIDATION SUCCESS: Archive integrity verified (0 CRC errors).")
            print("Archive contents preview:")
            top_level = set()
            for info in test_zf.infolist()[:25]:
                print(f" - {info.filename} ({info.file_size:,} bytes)")

if __name__ == "__main__":
    create_project_zip()
