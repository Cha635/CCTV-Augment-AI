"""
Lighting Augmentation Module for CCTV Surveillance Footage.
Implements:
1. Brightness increase
2. Brightness reduction
3. Contrast adjustment
4. Gamma correction
5. Underexposure simulation
6. Overexposure simulation
7. Nighttime simulation
8. Low-light simulation
"""

import cv2
import numpy as np
from PIL import Image

def adjust_brightness(image_rgb: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust brightness linearly.
    factor > 1.0 increases brightness; factor < 1.0 reduces brightness.
    """
    img = image_rgb.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)

def adjust_contrast(image_rgb: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust contrast centered at mid-gray (128).
    factor > 1.0 expands dynamic range; factor < 1.0 flattens contrast.
    """
    img = image_rgb.astype(np.float32)
    img = (img - 128.0) * factor + 128.0
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_gamma(image_rgb: np.ndarray, gamma: float) -> np.ndarray:
    """
    Non-linear gamma correction.
    gamma < 1.0 lifts shadows; gamma > 1.0 crushes shadows.
    """
    gamma = max(0.01, gamma)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image_rgb, table)

def simulate_underexposure(image_rgb: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    Simulates underexposed CCTV camera (narrow aperture / fast shutter speed).
    Crushes shadows, lowers dynamic range, adds mild shadow sensor noise.
    """
    # severity ranges from 0.1 to 1.0
    darken_factor = 1.0 - (0.65 * severity)
    darkened = image_rgb.astype(np.float32) * darken_factor
    
    # Non-linear toe crush
    darkened = np.power(darkened / 255.0, 1.0 + severity * 0.8) * 255.0
    
    # Mild sensor noise in underexposed CCTV
    noise = np.random.normal(0, 4.0 * severity, image_rgb.shape)
    darkened = darkened + noise
    
    return np.clip(darkened, 0, 255).astype(np.uint8)

def simulate_overexposure(image_rgb: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    Simulates overexposed CCTV camera (direct sun glare, high gain, headlights).
    Blows out highlights to pure white with blooming glow.
    """
    # severity ranges from 0.1 to 1.0
    boost_factor = 1.0 + (1.2 * severity)
    boosted = image_rgb.astype(np.float32) * boost_factor
    
    # Add highlight blooming/glow
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, int(200 - 50 * severity), 255, cv2.THRESH_BINARY)
    blur_kernel = int(15 + 20 * severity)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    bloom = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    bloom_3ch = cv2.merge([bloom, bloom, bloom]).astype(np.float32) * (0.35 * severity)
    
    overexposed = boosted + bloom_3ch
    return np.clip(overexposed, 0, 255).astype(np.uint8)

def simulate_nighttime(image_rgb: np.ndarray, intensity: float = 0.6) -> np.ndarray:
    """
    Simulates realistic night surveillance conditions:
    - Significant darkening
    - Cool blue/teal tint (IR / streetlamp atmospheric balance)
    - High-gain CCTV sensor noise
    - Peripheral lens vignetting
    """
    h, w = image_rgb.shape[:2]
    
    # 1. Darken base image
    base = image_rgb.astype(np.float32) * (1.0 - 0.75 * intensity)
    
    # 2. Cool chromatic shift (boost blue channel, reduce red channel)
    base[:, :, 0] = base[:, :, 0] * (1.0 - 0.3 * intensity)  # Red reduction
    base[:, :, 2] = base[:, :, 2] * (1.0 + 0.25 * intensity) # Blue boost
    
    # 3. Add radial CCTV lens vignette
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h / 2.0, w / 2.0
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    vignette = 1.0 - (0.45 * intensity) * (dist_from_center / max_dist) ** 2
    vignette = np.clip(vignette, 0.4, 1.0)[:, :, np.newaxis]
    base = base * vignette
    
    # 4. Sensor noise (typical of low-light CCTV gain)
    noise_sigma = 8.0 * intensity
    noise = np.random.normal(0, noise_sigma, base.shape)
    night = base + noise
    
    return np.clip(night, 0, 255).astype(np.uint8)

def simulate_low_light(image_rgb: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    Simulates twilight / dawn / dimly lit parking lot conditions:
    - Reduced brightness & contrast
    - Slight desaturation of chromatic channels
    - Moderate sensor grain
    """
    # 1. Convert to HSV for natural desaturation
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - 0.45 * intensity) # Desaturate
    hsv[:, :, 2] = hsv[:, :, 2] * (1.0 - 0.5 * intensity)  # Dim value
    dimmed_rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # 2. Add subtle sensor grain
    noise = np.random.normal(0, 4.0 * intensity, image_rgb.shape)
    result = dimmed_rgb.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_lighting_pipeline(image_rgb: np.ndarray, technique: str, intensity: float = 0.5) -> np.ndarray:
    """
    Master dispatcher for lighting augmentations.
    """
    t = technique.lower().replace(" ", "_")
    if "bright" in t and "increase" in t:
        factor = 1.0 + (1.2 * intensity)
        return adjust_brightness(image_rgb, factor)
    elif "bright" in t and ("reduc" in t or "decrease" in t):
        factor = max(0.1, 1.0 - (0.7 * intensity))
        return adjust_brightness(image_rgb, factor)
    elif "contrast" in t:
        factor = 0.4 + (1.6 * intensity)
        return adjust_contrast(image_rgb, factor)
    elif "gamma" in t:
        # intensity 0.0 -> gamma 0.4 (bright), intensity 1.0 -> gamma 2.5 (dark)
        gamma = 0.4 + (2.1 * intensity)
        return apply_gamma(image_rgb, gamma)
    elif "under" in t:
        return simulate_underexposure(image_rgb, intensity)
    elif "over" in t:
        return simulate_overexposure(image_rgb, intensity)
    elif "night" in t:
        return simulate_nighttime(image_rgb, intensity)
    elif "low" in t:
        return simulate_low_light(image_rgb, intensity)
    else:
        # Default fallback
        return image_rgb.copy()
