"""
Atmospheric Weather and Illumination Disturbance Generator.
Smart India Hackathon - FSOC PAT Simulator (ISRO / DOS PS 26169)

Implements selectable, parametric atmospheric image-level transforms:
- Clear: Identity no-op transform (baseline optical propagation).
- Haze: Moderate contrast reduction, uniform gray airlight offset, and subtle diffusion.
- Fog: Heavy contrast reduction, dense gray veiling, and spatial Gaussian diffusion (strictly stronger than haze).
- Rain: Directional rain streak convolution and increased high-frequency precipitation noise.
- Low-light: Overall exposure/brightness reduction and proportional elevation of relative noise floor.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


VALID_ATMOSPHERIC_CONDITIONS: Tuple[str, ...] = (
    "clear",
    "haze",
    "fog",
    "rain",
    "low_light",
)

ATMOSPHERIC_CONDITION_ALIASES: Dict[str, str] = {
    "clear": "clear",
    "none": "clear",
    "nominal": "clear",
    "baseline": "clear",
    "identity": "clear",
    "haze": "haze",
    "mist": "haze",
    "fog": "fog",
    "heavy_fog": "fog",
    "dense_fog": "fog",
    "rain": "rain",
    "rainy": "rain",
    "precipitation": "rain",
    "low_light": "low_light",
    "lowlight": "low_light",
    "night": "low_light",
    "dark": "low_light",
    "dusk": "low_light",
}


def normalize_atmospheric_condition(condition: str) -> str:
    """Normalize atmospheric condition string or alias to canonical form."""
    key = str(condition).strip().lower().replace("-", "_").replace(" ", "_")
    canonical = ATMOSPHERIC_CONDITION_ALIASES.get(key, key)
    if canonical not in VALID_ATMOSPHERIC_CONDITIONS:
        raise ValueError(
            f"Invalid atmospheric condition '{condition}'. Valid conditions: {VALID_ATMOSPHERIC_CONDITIONS}"
        )
    return canonical


def generate_directional_rain_kernel(length: int = 15, angle_deg: float = 75.0) -> np.ndarray:
    """
    Generate a normalized linear 2D directional motion blur kernel for rain streaks.
    Angle is measured in degrees where 90 is vertical.
    """
    length = max(3, int(length))
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2

    # Calculate streak line endpoints from angle
    rad = np.deg2rad(angle_deg)
    dx = int(round((length / 2.0) * np.cos(rad)))
    dy = int(round((length / 2.0) * np.sin(rad)))

    pt1 = (max(0, min(length - 1, center - dx)), max(0, min(length - 1, center - dy)))
    pt2 = (max(0, min(length - 1, center + dx)), max(0, min(length - 1, center + dy)))

    cv2.line(kernel, pt1, pt2, 1.0, 1)
    k_sum = float(np.sum(kernel))
    if k_sum > 0:
        kernel /= k_sum
    else:
        kernel[center, center] = 1.0
    return kernel


class AtmosphericDisturbance:
    """
    Simulates optical path weather conditions and illumination regimes:
    Clear, Haze, Fog, Rain, and Low-light.
    """

    def __init__(
        self,
        condition: str = "clear",
        params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        """
        Args:
            condition: Selected atmospheric condition name.
            params: Optional parameter overrides (e.g. transmission, airlight, rain density).
            seed: PRNG seed for deterministic stochastic features (e.g. rain streaks).
        """
        self.condition = normalize_atmospheric_condition(condition)
        self.params = dict(params) if params is not None else {}
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        """Reset internal PRNG state for deterministic reproducibility."""
        self._rng = np.random.default_rng(self.seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the selected atmospheric transform to an optical sensor image.

        Args:
            image: 2D uint8 sensor frame.

        Returns:
            Processed 2D uint8 sensor frame.
        """
        if image is None or image.size == 0:
            return image

        if self.condition == "clear":
            return image

        img_float = image.astype(np.float32)

        if self.condition == "haze":
            return self._apply_haze(img_float)
        elif self.condition == "fog":
            return self._apply_fog(img_float)
        elif self.condition == "rain":
            return self._apply_rain(img_float)
        elif self.condition == "low_light":
            return self._apply_low_light(img_float)

        return image

    def _apply_haze(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate atmospheric haze via Koschmieder's scattering equation:
            I_haze = I * t + A * (1 - t)
        Moderate transmission attenuation with uniform gray airlight veil.
        """
        # Transmission factor t in (0, 1): default 0.65 (35% contrast reduction)
        t = float(self.params.get("transmission", 0.65))
        # Atmospheric airlight A: diffuse ambient light scattered into LOS (default 60 counts)
        airlight = float(self.params.get("airlight", 60.0))

        # Optical scattering transform
        hazy = image * t + airlight * (1.0 - t)

        # Mild spatial softening
        blur_sigma = float(self.params.get("blur_sigma", 0.6))
        if blur_sigma > 0.3:
            ksize = int(np.ceil(blur_sigma * 3)) * 2 + 1
            hazy = cv2.GaussianBlur(hazy, (ksize, ksize), blur_sigma)

        return np.clip(np.round(hazy), 0, 255).astype(np.uint8)

    def _apply_fog(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate dense fog via dense Mie scattering and spatial Gaussian diffusion:
            I_fog = I * t + A * (1 - t)
        Strictly stronger contrast attenuation and heavier gray lift than haze.
        """
        # Dense transmission: default 0.35 (65% contrast reduction)
        t = float(self.params.get("transmission", 0.35))
        # Dense airlight: default 95 counts
        airlight = float(self.params.get("airlight", 95.0))

        foggy = image * t + airlight * (1.0 - t)

        # Substantial Gaussian diffusion blur caused by dense water droplets
        blur_sigma = float(self.params.get("blur_sigma", 1.8))
        if blur_sigma > 0.4:
            ksize = int(np.ceil(blur_sigma * 3)) * 2 + 1
            foggy = cv2.GaussianBlur(foggy, (ksize, ksize), blur_sigma)

        return np.clip(np.round(foggy), 0, 255).astype(np.uint8)

    def _apply_rain(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate falling rain:
        1. Droplet transmission attenuation (default t = 0.82)
        2. Sparse raindrops convolved with a directional linear motion streak filter
        3. High-frequency splash noise grain
        """
        t = float(self.params.get("transmission", 0.82))
        streak_density = float(self.params.get("density", 0.0025))
        streak_length = int(self.params.get("streak_length", 15))
        streak_angle = float(self.params.get("streak_angle_deg", 75.0))
        splash_noise_std = float(self.params.get("splash_std", 3.5))

        # 1. Atmospheric droplet attenuation
        base = image * t

        # 2. Directional Rain Streaks
        h, w = image.shape
        num_drops = int(h * w * streak_density)
        if num_drops > 0:
            drop_map = np.zeros((h, w), dtype=np.float32)
            y_coords = self._rng.integers(0, h, num_drops)
            x_coords = self._rng.integers(0, w, num_drops)
            intensities = self._rng.uniform(60.0, 160.0, num_drops).astype(np.float32)
            drop_map[y_coords, x_coords] = intensities

            kernel = generate_directional_rain_kernel(length=streak_length, angle_deg=streak_angle)
            streaks = cv2.filter2D(drop_map, -1, kernel)
            base += streaks

        # 3. High-frequency splash fluctuation
        if splash_noise_std > 0:
            splash = self._rng.normal(0.0, splash_noise_std, (h, w)).astype(np.float32)
            base += splash

        return np.clip(np.round(base), 0, 255).astype(np.uint8)

    def _apply_low_light(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate low-light / nocturnal conditions:
        1. Overall exposure reduction (default 0.28x scaling)
        2. Proportional elevation of the relative noise floor
        """
        # Exposure scale factor: default 0.28 (72% reduction)
        exposure_scale = float(self.params.get("exposure_scale", 0.28))
        dark_floor_std = float(self.params.get("dark_floor_std", 2.0))

        # Attenuate optical photon flux
        dimmed = image * exposure_scale

        # Ambient dark-current / noise floor grain
        if dark_floor_std > 0:
            dark_noise = self._rng.normal(0.0, dark_floor_std, image.shape).astype(np.float32)
            dimmed += dark_noise

        return np.clip(np.round(dimmed), 0, 255).astype(np.uint8)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration parameters."""
        return {
            "atmospheric_condition": self.condition,
            "atmospheric_params": self.params,
        }
