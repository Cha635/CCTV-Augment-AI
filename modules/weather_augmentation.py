"""
Weather Augmentation Module for CCTV Surveillance Footage.
Implements physically grounded, realistic atmospheric simulation:
1. Directional Rain: Multi-scale directional streaks using 1D motion kernels, storm light attenuation, and rain mist.
2. Atmospheric Fog: Koschmieder optical scattering model with perspective depth gradient and airlight.
3. Atmospheric Haze: Mie scattering with distance-based contrast reduction and veiling luminance.
4. Multi-Scale Snow: 3-layer depth particle synthesis (distant micro-snow, falling midground, soft bokeh foreground).
"""

from typing import Optional
import cv2
import numpy as np

def create_directional_motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Creates a normalized 1D motion blur line kernel oriented at angle_deg."""
    length = max(3, int(length))
    ksize = length if length % 2 == 1 else length + 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2
    
    rad = np.radians(angle_deg)
    dx = np.cos(rad) * (length / 2.0)
    dy = np.sin(rad) * (length / 2.0)
    
    pt1 = (int(round(center - dx)), int(round(center - dy)))
    pt2 = (int(round(center + dx)), int(round(center + dy)))
    
    cv2.line(kernel, pt1, pt2, 1.0, 1)
    k_sum = np.sum(kernel)
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, center] = 1.0
    return kernel

def simulate_rain(
    image_rgb: np.ndarray,
    preset: str = "moderate",
    density: float = 0.6,
    streak_intensity: float = 0.6,
    slant_angle: float = -20.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulates physically grounded directional rain on surveillance footage.
    
    Components:
    1. Atmospheric storm illumination attenuation (overcast dimming & cool chromatic shift).
    2. Distant rain mist (subtle depth veil).
    3. Multi-scale directional rain streaks with realistic 1D velocity kernels.
    """
    if seed is not None:
        np.random.seed(seed)
        
    h, w = image_rgb.shape[:2]
    
    preset_multipliers = {
        "light": 0.45,
        "moderate": 0.85,
        "heavy": 1.40,
    }
    mult = preset_multipliers.get(preset.lower(), 0.85) * max(0.2, (density + streak_intensity) / 2.0)
    
    # 1. Storm sky & atmospheric luminance attenuation
    # Real rainstorms darken ambient scene light and produce a slight cool overcast tone
    dimmed = image_rgb.astype(np.float32)
    darken_factor = np.clip(1.0 - 0.22 * mult, 0.65, 0.98)
    dimmed[:, :, 0] *= (darken_factor * 0.96)  # Slightly reduce red
    dimmed[:, :, 1] *= darken_factor          # Neutral green
    dimmed[:, :, 2] *= (darken_factor * 1.04)  # Subtle cool blue ambient shift
    
    # 2. Atmospheric rain haze/mist layer (stronger towards upper background)
    Y, _ = np.mgrid[0:h, 0:w]
    depth_mist = (1.0 - (Y / float(h))) * (0.28 * mult)
    mist_3ch = np.repeat(depth_mist[:, :, np.newaxis], 3, axis=2)
    mist_color = np.array([195, 205, 218], dtype=np.float32)
    scene_with_mist = dimmed * (1.0 - mist_3ch) + mist_color * mist_3ch
    
    # 3. Layer A: Distant fine rain (short, dense, thin streaks)
    num_fine = int(w * h * 0.00045 * mult)
    fine_pts = np.zeros((h, w), dtype=np.float32)
    if num_fine > 0:
        ys = np.random.randint(0, h, num_fine)
        xs = np.random.randint(0, w, num_fine)
        fine_pts[ys, xs] = np.random.uniform(140, 220, num_fine)
        fine_kernel = create_directional_motion_kernel(int(14 + 10 * mult), 90.0 + slant_angle)
        fine_streaks = cv2.filter2D(fine_pts, -1, fine_kernel)
    else:
        fine_streaks = np.zeros((h, w), dtype=np.float32)
        
    # 4. Layer B: Mid-distance rain (medium length, distinct streaks)
    num_mid = int(w * h * 0.00020 * mult)
    mid_pts = np.zeros((h, w), dtype=np.float32)
    if num_mid > 0:
        ys = np.random.randint(0, h, num_mid)
        xs = np.random.randint(0, w, num_mid)
        mid_pts[ys, xs] = np.random.uniform(180, 255, num_mid)
        mid_kernel = create_directional_motion_kernel(int(24 + 18 * mult), 90.0 + slant_angle)
        mid_streaks = cv2.filter2D(mid_pts, -1, mid_kernel)
    else:
        mid_streaks = np.zeros((h, w), dtype=np.float32)
        
    # 5. Layer C: Foreground lens streaks (longer, motion-blurred, slightly softer)
    num_fore = int(w * h * 0.00004 * mult)
    fore_pts = np.zeros((h, w), dtype=np.float32)
    if num_fore > 0:
        ys = np.random.randint(0, h, num_fore)
        xs = np.random.randint(0, w, num_fore)
        fore_pts[ys, xs] = np.random.uniform(220, 255, num_fore)
        fore_kernel = create_directional_motion_kernel(int(38 + 25 * mult), 90.0 + slant_angle + np.random.uniform(-3, 3))
        fore_streaks = cv2.filter2D(fore_pts, -1, fore_kernel)
        fore_streaks = cv2.GaussianBlur(fore_streaks, (3, 3), 0.6)
    else:
        fore_streaks = np.zeros((h, w), dtype=np.float32)
        
    # Combine streak layers
    combined_streaks = fine_streaks * 0.45 + mid_streaks * 0.70 + fore_streaks * 0.95
    rain_alpha = np.clip((combined_streaks / 255.0) * (0.65 * mult), 0.0, 0.85)
    rain_alpha_3ch = np.repeat(rain_alpha[:, :, np.newaxis], 3, axis=2)
    
    # Translucent rain streak color
    rain_color = np.array([215, 228, 242], dtype=np.float32)
    final = scene_with_mist * (1.0 - rain_alpha_3ch) + rain_color * rain_alpha_3ch
    
    return np.clip(final, 0, 255).astype(np.uint8)

def simulate_fog(
    image_rgb: np.ndarray,
    preset: str = "moderate",
    intensity: float = 0.65,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulates atmospheric fog using Koschmieder's physical optical scattering law:
    I(x) = J(x) * t(x) + L_inf * (1 - t(x))
    where t(x) = exp(-beta * d(x)).
    
    Uses perspective-aware CCTV vertical depth gradient combined with spatial turbulence.
    """
    if seed is not None:
        np.random.seed(seed)
        
    h, w = image_rgb.shape[:2]
    
    preset_scales = {
        "light": 0.40,
        "moderate": 0.75,
        "heavy": 1.25,
    }
    scale = preset_scales.get(preset.lower(), 0.75) * max(0.1, intensity)
    
    # Geometric depth model: in CCTV cameras, distance increases toward the upper horizon
    Y, _ = np.mgrid[0:h, 0:w]
    # Non-linear perspective: ground near camera has small depth, distant background has large depth
    vertical_depth = (1.0 - (Y / float(h))) ** 1.35
    
    # Spatial low-frequency atmospheric turbulence (realistic uneven fog pockets)
    grid_h = max(4, h // 32)
    grid_w = max(4, w // 32)
    noise_grid = np.random.uniform(0.75, 1.25, (grid_h, grid_w)).astype(np.float32)
    noise_map = cv2.resize(noise_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    
    effective_depth = np.clip(vertical_depth * noise_map, 0.05, 1.4)
    
    # Optical attenuation coefficient (beta)
    beta = 1.85 * scale
    transmission = np.exp(-beta * effective_depth)
    transmission_3ch = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
    
    # Atmospheric airlight (calibrated to overcast daylight surveillance fog)
    airlight = np.array([210, 220, 230], dtype=np.float32)
    
    # Scattering composition
    foggy = image_rgb.astype(np.float32) * transmission_3ch + airlight * (1.0 - transmission_3ch)
    
    # High-frequency edge softening in deeper fog regions
    blur_k = max(3, int(5 * scale))
    if blur_k % 2 == 0:
        blur_k += 1
    soft_foggy = cv2.GaussianBlur(foggy, (blur_k, blur_k), 0)
    
    # Distance-weighted blend of soft vs sharp fog
    fog_depth_weight = np.repeat((1.0 - transmission)[:, :, np.newaxis], 3, axis=2)
    final = foggy * (1.0 - 0.35 * fog_depth_weight) + soft_foggy * (0.35 * fog_depth_weight)
    
    return np.clip(final, 0, 255).astype(np.uint8)

def simulate_haze(
    image_rgb: np.ndarray,
    visibility_reduction: float = 0.55
) -> np.ndarray:
    """
    Simulates atmospheric haze:
    Applies Mie particulate scattering, dynamic range compression, and veiling luminance.
    """
    h, w = image_rgb.shape[:2]
    img = image_rgb.astype(np.float32)
    vis = np.clip(visibility_reduction, 0.1, 1.0)
    
    # 1. Mean-centered contrast compression
    mean_val = np.mean(img, axis=(0, 1), keepdims=True)
    contrast_scale = np.clip(1.0 - 0.45 * vis, 0.45, 0.95)
    hazy = (img - mean_val) * contrast_scale + mean_val
    
    # 2. Ambient veiling light gradient
    Y, _ = np.mgrid[0:h, 0:w]
    depth_grad = 1.0 - (Y / float(h)) * 0.70
    haze_alpha = np.clip(depth_grad * (0.35 * vis), 0.05, 0.55)
    haze_alpha_3ch = np.repeat(haze_alpha[:, :, np.newaxis], 3, axis=2)
    
    veiling_light = np.array([200, 208, 218], dtype=np.float32)
    result = hazy * (1.0 - haze_alpha_3ch) + veiling_light * haze_alpha_3ch
    
    return np.clip(result, 0, 255).astype(np.uint8)

def simulate_snow(
    image_rgb: np.ndarray,
    preset: str = "moderate",
    intensity: float = 0.65,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulates multi-scale realistic snowfall on CCTV footage:
    - Layer 1: Distant micro-snow particle veil (atmospheric depth).
    - Layer 2: Falling midground flakes with slight vertical motion blur.
    - Layer 3: Foreground camera-lens bokeh flakes (soft circular Gaussian discs).
    - Environmental winter cooling tone shift.
    """
    if seed is not None:
        np.random.seed(seed)
        
    h, w = image_rgb.shape[:2]
    
    preset_scales = {"light": 0.45, "moderate": 0.85, "heavy": 1.45}
    scale = preset_scales.get(preset.lower(), 0.85) * max(0.1, intensity)
    
    # 1. Winter environmental color grading (cool steel-blue temperature shift)
    scene = image_rgb.astype(np.float32)
    scene[:, :, 0] *= 0.94  # Slight red attenuation
    scene[:, :, 1] *= 0.98  # Slight green attenuation
    scene[:, :, 2] *= 1.06  # Cold blue boost
    
    # 2. Layer 1: Distant fine micro-snow (atmospheric veil)
    num_micro = int(w * h * 0.0006 * scale)
    micro_layer = np.zeros((h, w), dtype=np.float32)
    if num_micro > 0:
        ys = np.random.randint(0, h, num_micro)
        xs = np.random.randint(0, w, num_micro)
        micro_layer[ys, xs] = np.random.uniform(160, 230, num_micro)
        micro_layer = cv2.GaussianBlur(micro_layer, (3, 3), 0.5)
        
    # 3. Layer 2: Falling midground snowflakes (elongated 2-3px vertical motion)
    num_mid = int(w * h * 0.00015 * scale)
    mid_layer = np.zeros((h, w), dtype=np.float32)
    if num_mid > 0:
        for _ in range(num_mid):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            rad = np.random.randint(1, 3)
            val = np.random.uniform(200, 255)
            cv2.circle(mid_layer, (x, y), rad, val, -1)
        # Slight vertical fall streak
        snow_kernel = np.array([[0, 0.2, 0], [0, 0.6, 0], [0, 0.2, 0]], dtype=np.float32)
        mid_layer = cv2.filter2D(mid_layer, -1, snow_kernel)
        
    # 4. Layer 3: Foreground camera-lens bokeh flakes (large soft Gaussian discs)
    num_bokeh = int(max(4, 28 * scale))
    bokeh_layer = np.zeros((h, w), dtype=np.float32)
    for _ in range(num_bokeh):
        bx = np.random.randint(0, w)
        by = np.random.randint(0, h)
        radius = np.random.randint(4, 12)
        brightness = np.random.uniform(160, 245)
        cv2.circle(bokeh_layer, (bx, by), radius, brightness, -1)
    if num_bokeh > 0:
        bokeh_layer = cv2.GaussianBlur(bokeh_layer, (15, 15), 0)
        
    # Combine snow layers
    total_snow = micro_layer * 0.40 + mid_layer * 0.75 + bokeh_layer * 0.60
    snow_alpha = np.clip(total_snow / 255.0, 0.0, 0.90)
    snow_alpha_3ch = np.repeat(snow_alpha[:, :, np.newaxis], 3, axis=2)
    
    pure_snow_color = np.array([242, 248, 255], dtype=np.float32)
    result = scene * (1.0 - snow_alpha_3ch) + pure_snow_color * snow_alpha_3ch
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_weather_pipeline(
    image_rgb: np.ndarray,
    weather_type: str,
    preset: str = "moderate",
    intensity: float = 0.65,
    seed: Optional[int] = None
) -> np.ndarray:
    """Master dispatcher for weather augmentations."""
    w = weather_type.lower()
    if "rain" in w:
        return simulate_rain(image_rgb, preset=preset, density=intensity, streak_intensity=intensity, seed=seed)
    elif "fog" in w:
        return simulate_fog(image_rgb, preset=preset, intensity=intensity, seed=seed)
    elif "haze" in w:
        return simulate_haze(image_rgb, visibility_reduction=intensity)
    elif "snow" in w:
        return simulate_snow(image_rgb, preset=preset, intensity=intensity, seed=seed)
    else:
        return image_rgb.copy()
