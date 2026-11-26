"""
Image Upload Module for CCTV Augment AI.
Handles single and batch uploads, metadata extraction, validation,
and session management.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import io
from PIL import Image
import numpy as np

def validate_image_file(uploaded_file) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates uploaded file format and extracts resolution & size.
    Supported: JPG, JPEG, PNG.
    """
    valid_exts = [".jpg", ".jpeg", ".png"]
    ext = Path(uploaded_file.name).suffix.lower()
    
    if ext not in valid_exts:
        return False, f"Unsupported file extension '{ext}'. Only JPG, JPEG, and PNG are allowed.", None
        
    try:
        image_bytes = uploaded_file.getvalue()
        pil_img = Image.open(io.BytesIO(image_bytes))
        width, height = pil_img.size
        mode = pil_img.mode
        
        # Convert RGBA / Grayscale to RGB if necessary
        if mode != "RGB":
            pil_img = pil_img.convert("RGB")
            
        file_size_kb = round(len(image_bytes) / 1024, 2)
        
        info = {
            "name": uploaded_file.name,
            "width": width,
            "height": height,
            "channels": 3,
            "size_kb": file_size_kb,
            "format": pil_img.format or ext.replace(".", "").upper(),
            "aspect_ratio": f"{round(width / max(1, height), 2)}:1",
            "pil_image": pil_img,
            "np_rgb": np.array(pil_img)
        }
        return True, "Valid", info
    except Exception as e:
        return False, f"Error decoding image: {str(e)}", None

def get_image_summary_table(images_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates structured metadata list for display."""
    summary = []
    for info in images_info:
        summary.append({
            "File Name": info["name"],
            "Dimensions": f"{info['width']} × {info['height']}",
            "File Size": f"{info['size_kb']} KB",
            "Format": info["format"],
            "Aspect Ratio": info["aspect_ratio"]
        })
    return summary
