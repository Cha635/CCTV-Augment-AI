"""
Occlusion Augmentation Module for CCTV Surveillance Footage.
Simulates physically realistic, object-like surveillance obstructions:
1. 3D Utility Pole / Traffic Pillar with cylindrical metallic shading, mounting brackets, and drop shadow.
2. Security Fence / Railing with vertical pickets and horizontal crossbars showing background gaps.
3. Urban Traffic Barrier with diagonal hazard reflective stripes and bevel border.
4. Overhanging Foliage / Tree Branch entering from camera frame edge.
5. Encroaching Vehicle Silhouette (bus/truck/car profile obstructing bottom or side).
6. Deterministic reproducibility via seed parameter.
7. Accurate bounding-box intersection calculations (IoF) with threshold filtering.
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

def calculate_box_occlusion_ratio(
    bbox_xywh_norm: List[float],
    occ_box_pixel: Tuple[int, int, int, int],
    img_w: int,
    img_h: int
) -> float:
    """
    Calculates fraction of bounding box area occluded by an obstruction.
    bbox_xywh_norm: [x_center, y_center, width, height] normalized (0 to 1)
    occ_box_pixel: (ox1, oy1, ox2, oy2) in pixel coordinates
    Returns float in range [0.0, 1.0] representing Intersection / Box_Area.
    """
    xc, yc, bw, bh = bbox_xywh_norm
    bx1 = (xc - bw / 2.0) * img_w
    by1 = (yc - bh / 2.0) * img_h
    bx2 = (xc + bw / 2.0) * img_w
    by2 = (yc + bh / 2.0) * img_h
    
    box_area = max(1e-5, (bx2 - bx1) * (by2 - by1))
    
    ox1, oy1, ox2, oy2 = occ_box_pixel
    ix1 = max(bx1, ox1)
    iy1 = max(by1, oy1)
    ix2 = min(bx2, ox2)
    iy2 = min(by2, oy2)
    
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    
    return float(np.clip(inter_area / box_area, 0.0, 1.0))

def render_realistic_pole(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color_mode: str = "dark"
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Renders a realistic 3D cylindrical utility pole / camera pillar."""
    h, w = img.shape[:2]
    pw = max(12, x2 - x1)
    ph = max(20, y2 - y1)
    
    # Base metallic gray tones
    base_val = 38 if color_mode == "dark" else 160
    
    # Generate 1D horizontal cylindrical shading profile
    xs = np.linspace(-1.0, 1.0, pw)
    # Cosine cylindrical illumination with specular highlight at x = -0.3
    shading = np.cos(xs * 1.3) * 0.7 + 0.3
    highlight = np.exp(-((xs + 0.3) ** 2) / 0.08) * 0.45
    profile = np.clip(shading + highlight, 0.1, 1.4)
    
    pole_patch = np.zeros((ph, pw, 3), dtype=np.uint8)
    for c in range(3):
        col_profile = (profile * (base_val + (c * 2))).astype(np.float32)
        pole_patch[:, :, c] = np.tile(np.clip(col_profile, 10, 250), (ph, 1))
        
    # Add horizontal metal weld seams / brackets at intervals
    bracket_interval = max(45, ph // 5)
    for by in range(bracket_interval // 2, ph, bracket_interval):
        bh = max(4, pw // 8)
        pole_patch[max(0, by - bh):min(ph, by + bh), :, :] = np.clip(
            pole_patch[max(0, by - bh):min(ph, by + bh), :, :].astype(np.int32) + 40, 0, 255
        ).astype(np.uint8)
        # Bolt dots
        if pw > 16:
            pole_patch[by, pw // 4, :] = 220
            pole_patch[by, 3 * pw // 4, :] = 220
            
    # Composite pole onto image
    img[y1:y2, x1:x2] = pole_patch
    
    # Cast soft drop-shadow on the right side of the pole
    shadow_w = min(18, pw // 2)
    if x2 + shadow_w < w:
        for sx in range(shadow_w):
            alpha = (1.0 - (sx / float(shadow_w))) * 0.35
            img[y1:y2, x2 + sx] = (img[y1:y2, x2 + sx].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            
    return img, (x1, y1, min(w, x2 + shadow_w), y2)

def render_security_fence(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Renders a security railing / fence with vertical pickets and horizontal rails."""
    h, w = img.shape[:2]
    fw = x2 - x1
    fh = y2 - y1
    
    # Fence color: industrial dark steel
    bar_color = (45, 50, 58)
    highlight_color = (85, 95, 110)
    
    # 1. Horizontal crossbars (top and bottom)
    rail_h = max(4, fh // 12)
    y_top = y1 + int(fh * 0.2)
    y_bot = y1 + int(fh * 0.8)
    
    for yr in [y_top, y_bot]:
        cv2.rectangle(img, (x1, yr), (x2, min(h, yr + rail_h)), bar_color, -1)
        cv2.line(img, (x1, yr), (x2, yr), highlight_color, 1)
        
    # 2. Vertical pickets with spacing
    picket_spacing = max(14, fw // 14)
    picket_w = max(3, picket_spacing // 4)
    
    for px in range(x1, x2, picket_spacing):
        if px + picket_w < w:
            cv2.rectangle(img, (px, y1), (px + picket_w, y2), bar_color, -1)
            cv2.line(img, (px, y1), (px, y2), highlight_color, 1)
            # Spear/pointed tip on fence pickets
            tip_h = min(12, int(picket_spacing * 0.6))
            pts = np.array([[px, y1], [px + picket_w, y1], [px + picket_w // 2, max(0, y1 - tip_h)]], dtype=np.int32)
            cv2.fillPoly(img, [pts], bar_color)
            
    return img, (x1, max(0, y1 - 12), x2, y2)

def render_hazard_barrier(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Renders an urban traffic hazard barrier with diagonal warning stripes."""
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    
    barrier = np.zeros((bh, bw, 3), dtype=np.uint8)
    
    # Colors: bright warning amber/yellow and dark charcoal
    c_amber = np.array([235, 175, 20], dtype=np.uint8)
    c_dark = np.array([28, 30, 35], dtype=np.uint8)
    
    stripe_w = max(16, bh // 3)
    for y in range(bh):
        for x in range(bw):
            if ((x + y) // stripe_w) % 2 == 0:
                barrier[y, x] = c_amber
            else:
                barrier[y, x] = c_dark
                
    # Add border and 3D bevel shading
    cv2.rectangle(barrier, (0, 0), (bw - 1, bh - 1), (15, 15, 20), 2)
    cv2.line(barrier, (1, 1), (bw - 2, 1), (255, 255, 255), 1)
    
    img[y1:y2, x1:x2] = barrier
    return img, (x1, y1, x2, y2)

def render_foliage_branch(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Renders natural tree branch / leafy foliage intrusion from frame boundary."""
    h, w = img.shape[:2]
    branch_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Generate main branch polygon
    pts = np.array([
        [x1, y1],
        [x1 + int((x2 - x1) * 0.6), y1 + int((y2 - y1) * 0.4)],
        [x2, y2],
        [x2 - int((x2 - x1) * 0.2), y2 + 10],
        [x1 + int((x2 - x1) * 0.3), y1 + int((y2 - y1) * 0.7)],
        [x1, y1 + int((y2 - y1) * 0.8)]
    ], dtype=np.int32)
    cv2.fillPoly(branch_mask, [pts], 255)
    
    # Add clusters of leaf ellipses
    num_leaves = 25
    np.random.seed(x1 + y1)
    for _ in range(num_leaves):
        lx = np.random.randint(min(x1, x2), max(x1, x2) + 1)
        ly = np.random.randint(min(y1, y2), max(y1, y2) + 1)
        axes = (np.random.randint(10, 26), np.random.randint(6, 16))
        ang = np.random.randint(0, 180)
        cv2.ellipse(branch_mask, (lx, ly), axes, ang, 0, 360, 255, -1)
        
    branch_mask = cv2.GaussianBlur(branch_mask, (5, 5), 0)
    alpha = np.repeat((branch_mask.astype(np.float32) / 255.0)[:, :, np.newaxis], 3, axis=2)
    
    # Foliage dark surveillance green tone
    leaf_color = np.array([32, 55, 30], dtype=np.float32)
    img_f = img.astype(np.float32)
    blended = img_f * (1.0 - alpha * 0.95) + leaf_color * (alpha * 0.95)
    
    return np.clip(blended, 0, 255).astype(np.uint8), (x1, y1, x2, y2)

def apply_occlusion(
    image_rgb: np.ndarray,
    preset: str = "medium",
    num_regions: int = 1,
    size_ratio: float = 0.25,
    occlusion_type: str = "pole",
    custom_position: Optional[Tuple[float, float]] = None,
    color_mode: str = "dark",
    annotations: Optional[List[Dict[str, Any]]] = None,
    max_occlusion_threshold: float = 0.70,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[int, int, int, int]]]:
    """
    Applies physically realistic partial surveillance occlusion.
    
    Supported types:
    - "pole": 3D cylindrical utility pole / camera pillar with metallic shading.
    - "fence": Security fence with vertical pickets and horizontal rails.
    - "barrier": Traffic construction hazard barrier with diagonal warning stripes.
    - "foliage": Organic tree branch and leafy intrusion.
    - "random": Randomized selection among realistic obstructions.
    """
    if seed is not None:
        np.random.seed(seed)
        
    h, w = image_rgb.shape[:2]
    img_out = image_rgb.copy()
    
    preset_configs = {
        "low": {"regions": 1, "scale": 0.18},
        "medium": {"regions": 1, "scale": 0.28},
        "high": {"regions": 2, "scale": 0.38},
    }
    
    if preset in preset_configs:
        regions_count = num_regions if num_regions > 0 else preset_configs[preset]["regions"]
        scale = size_ratio if size_ratio != 0.25 else preset_configs[preset]["scale"]
    else:
        regions_count = max(1, num_regions)
        scale = size_ratio
        
    occlusion_rects: List[Tuple[int, int, int, int]] = []
    
    for i in range(regions_count):
        curr_type = occlusion_type.lower()
        if curr_type == "random":
            curr_type = np.random.choice(["pole", "fence", "barrier", "foliage"])
            
        # Determine geometry and position based on type
        if "pole" in curr_type:
            # Vertical utility pole spanning vertical frame
            rw = int(np.clip(w * scale * np.random.uniform(0.35, 0.65), 18, w * 0.35))
            rh = h
            if custom_position is not None and i == 0:
                x1 = int(np.clip(custom_position[0] * w - rw / 2, 0, w - rw))
            else:
                # Typically located on left third, center, or right third
                pos_choice = np.random.choice(["left", "center", "right"])
                if pos_choice == "left":
                    x1 = np.random.randint(int(w * 0.08), int(w * 0.28))
                elif pos_choice == "center":
                    x1 = np.random.randint(int(w * 0.42), int(w * 0.58))
                else:
                    x1 = np.random.randint(int(w * 0.68), int(w * 0.88))
            y1 = 0
            x2 = min(w, x1 + rw)
            y2 = h
            img_out, rect = render_realistic_pole(img_out, x1, y1, x2, y2, color_mode)
            
        elif "fence" in curr_type:
            # Horizontal railing in lower third or middle
            rw = int(np.clip(w * np.random.uniform(0.65, 0.95), 100, w))
            rh = int(np.clip(h * scale * np.random.uniform(0.5, 0.8), 40, h * 0.4))
            x1 = np.random.randint(0, max(1, w - rw))
            y1 = np.random.randint(int(h * 0.45), max(int(h * 0.46), h - rh))
            x2 = min(w, x1 + rw)
            y2 = min(h, y1 + rh)
            img_out, rect = render_security_fence(img_out, x1, y1, x2, y2)
            
        elif "barrier" in curr_type or "box" in curr_type:
            # Traffic barrier / guardrail
            rw = int(np.clip(w * scale * np.random.uniform(1.2, 1.8), 80, w * 0.65))
            rh = int(np.clip(h * scale * np.random.uniform(0.6, 1.0), 35, h * 0.35))
            if custom_position is not None and i == 0:
                x1 = int(np.clip(custom_position[0] * w - rw / 2, 0, w - rw))
                y1 = int(np.clip(custom_position[1] * h - rh / 2, 0, h - rh))
            else:
                x1 = np.random.randint(int(w * 0.05), max(int(w * 0.06), w - rw - 10))
                y1 = np.random.randint(int(h * 0.55), max(int(h * 0.56), h - rh - 5))
            x2 = min(w, x1 + rw)
            y2 = min(h, y1 + rh)
            img_out, rect = render_hazard_barrier(img_out, x1, y1, x2, y2)
            
        else: # Foliage
            rw = int(np.clip(w * scale * np.random.uniform(1.0, 1.5), 60, w * 0.55))
            rh = int(np.clip(h * scale * np.random.uniform(0.8, 1.3), 50, h * 0.45))
            x1 = 0 if np.random.rand() > 0.5 else w - rw
            y1 = 0
            x2 = x1 + rw
            y2 = y1 + rh
            img_out, rect = render_foliage_branch(img_out, x1, y1, x2, y2)
            
        occlusion_rects.append(rect)
        
    # Validate Annotations against Occlusions
    valid_annotations = []
    if annotations:
        for ann in annotations:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            total_occ = 0.0
            for rect in occlusion_rects:
                ratio = calculate_box_occlusion_ratio(bbox, rect, w, h)
                total_occ = min(1.0, total_occ + ratio)
            if total_occ < max_occlusion_threshold:
                valid_annotations.append({
                    "class_id": ann.get("class_id", 0),
                    "bbox": bbox,
                    "occlusion_ratio": round(total_occ, 3),
                    "visible": True
                })
    else:
        valid_annotations = []
        
    return img_out, valid_annotations, occlusion_rects
