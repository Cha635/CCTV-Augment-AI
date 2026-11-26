"""
YOLO Object Detection Module for CCTV Augment AI.
Provides real-time inference for Person & Vehicle surveillance detection,
custom CCTV-style bounding box visualization, and confidence analytics.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

import config

class CCTVDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initializes YOLO model.
        If model_path is None, loads default pretrained yolov8n.pt.
        """
        self.model = None
        self.model_path = model_path
        self.is_custom_trained = False
        self.load_model(model_path)
        
    def load_model(self, model_path: Optional[str] = None):
        if not ULTRALYTICS_AVAILABLE:
            return
            
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            self.model_path = model_path
            # Check model class names
            names = self.model.names
            if isinstance(names, dict) and len(names) == 2 and 0 in names and 1 in names:
                self.is_custom_trained = True
            else:
                self.is_custom_trained = False
        else:
            # Default to nano model
            nano_path = config.PRETRAINED_DIR / "yolov8n.pt"
            if not nano_path.exists():
                nano_path = "yolov8n.pt" # Ultralytics will auto-download
            self.model = YOLO(str(nano_path))
            self.model_path = str(nano_path)
            self.is_custom_trained = False

    def predict(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """
        Runs object detection on image_rgb array.
        Extracts detections mapped to Person and Vehicle classes.
        """
        if self.model is None:
            return {"error": "YOLO model not loaded or Ultralytics not available."}
            
        # Run inference
        hw_info = config.get_hardware_info()
        device = "0" if hw_info["cuda_available"] else "cpu"
        
        results = self.model.predict(
            source=image_rgb,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False
        )
        
        res = results[0]
        boxes = res.boxes
        
        detections = []
        person_count = 0
        vehicle_count = 0
        
        # COCO mapping if standard pretrained model is active
        # COCO class 0 is person; 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        coco_vehicle_classes = {1, 2, 3, 5, 7}
        
        for box in boxes:
            cls_idx = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            
            target_cls = None
            if self.is_custom_trained:
                if cls_idx == 0:
                    target_cls = "person"
                    person_count += 1
                elif cls_idx == 1:
                    target_cls = "vehicle"
                    vehicle_count += 1
            else:
                if cls_idx == 0:
                    target_cls = "person"
                    person_count += 1
                elif cls_idx in coco_vehicle_classes:
                    target_cls = "vehicle"
                    vehicle_count += 1
                    
            if target_cls is not None:
                detections.append({
                    "class": target_cls,
                    "class_id": 0 if target_cls == "person" else 1,
                    "confidence": round(conf, 3),
                    "bbox_xyxy": [int(p) for p in xyxy]
                })
                
        # Draw CCTV bounding boxes
        annotated_rgb = self.render_cctv_boxes(image_rgb, detections)
        
        return {
            "detections": detections,
            "total_detections": len(detections),
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "annotated_rgb": annotated_rgb,
            "inference_time_ms": round(res.speed.get("inference", 0.0), 2),
            "is_custom_trained": self.is_custom_trained,
            "model_path": self.model_path
        }

    def render_cctv_boxes(self, image_rgb: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draws aesthetic high-tech CCTV bounding boxes and status labels."""
        img_out = image_rgb.copy()
        h, w = img_out.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cls_name = det["class"].upper()
            conf = det["confidence"]
            
            # Palette: Person is Neon Cyan, Vehicle is Orange Amber
            if det["class"] == "person":
                box_color = (6, 182, 212)    # Cyan RGB
                glow_color = (0, 240, 255)
            else:
                box_color = (245, 158, 11)   # Amber RGB
                glow_color = (255, 180, 0)
                
            # 1. Main bounding box
            cv2.rectangle(img_out, (x1, y1), (x2, y2), box_color, 2)
            
            # 2. Corner brackets (tactical CCTV HUD look)
            corner_len = max(8, min(20, (x2 - x1) // 4, (y2 - y1) // 4))
            # Top-left
            cv2.line(img_out, (x1, y1), (x1 + corner_len, y1), glow_color, 3)
            cv2.line(img_out, (x1, y1), (x1, y1 + corner_len), glow_color, 3)
            # Top-right
            cv2.line(img_out, (x2, y1), (x2 - corner_len, y1), glow_color, 3)
            cv2.line(img_out, (x2, y1), (x2, y1 + corner_len), glow_color, 3)
            # Bottom-left
            cv2.line(img_out, (x1, y2), (x1 + corner_len, y2), glow_color, 3)
            cv2.line(img_out, (x1, y2), (x1, y2 - corner_len), glow_color, 3)
            # Bottom-right
            cv2.line(img_out, (x2, y2), (x2 - corner_len, y2), glow_color, 3)
            cv2.line(img_out, (x2, y2), (x2, y2 - corner_len), glow_color, 3)
            
            # 3. Label badge
            label_text = f"{cls_name} {int(conf * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Badge background
            by1 = max(0, y1 - th - 8)
            by2 = y1
            bx1 = x1
            bx2 = min(w, x1 + tw + 12)
            
            cv2.rectangle(img_out, (bx1, by1), (bx2, by2), (15, 23, 42), -1) # Dark backdrop
            cv2.rectangle(img_out, (bx1, by1), (bx2, by2), box_color, 1)    # Outline
            cv2.putText(img_out, label_text, (bx1 + 6, by2 - 4), font, font_scale, glow_color, thickness, cv2.LINE_AA)
            
        return img_out
