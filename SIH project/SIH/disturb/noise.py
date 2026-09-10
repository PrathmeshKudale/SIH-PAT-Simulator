"""
Optical Sensor Noise Generator (Readout Gaussian, Poisson Shot Noise, and Salt-and-Pepper Impulse Noise).
Smart India Hackathon - FSOC PAT Simulator (ISRO / DOS PS 26169)

Deterministic, vectorized noise generator supporting selectable, stackable noise models:
- Poisson (shot) noise representing discrete photon arrival statistics
- Gaussian additive zero-mean sensor electronic readout noise
- Salt-and-pepper impulse noise representing hot/dead defective detector pixels
"""

from typing import Optional, List, Union, Dict, Any, Tuple
import numpy as np


VALID_NOISE_TYPES: Tuple[str, ...] = ("gaussian", "poisson", "salt_pepper")

NOISE_TYPE_ALIASES: Dict[str, str] = {
    "gaussian": "gaussian",
    "readout": "gaussian",
    "normal": "gaussian",
    "gauss": "gaussian",
    "poisson": "poisson",
    "shot": "poisson",
    "photon": "poisson",
    "salt_pepper": "salt_pepper",
    "salt_and_pepper": "salt_pepper",
    "impulse": "salt_pepper",
    "sp": "salt_pepper",
}


def normalize_noise_type(noise_type: str) -> str:
    """Normalize a noise type name or alias to canonical form."""
    key = str(noise_type).strip().lower().replace("-", "_").replace(" ", "_")
    norm = NOISE_TYPE_ALIASES.get(key, key)
    if norm not in VALID_NOISE_TYPES:
        raise ValueError(
            f"Invalid noise type '{noise_type}'. Valid selectable types: {VALID_NOISE_TYPES}"
        )
    return norm


def apply_poisson_noise(
    image: np.ndarray,
    scale_factor: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply Poisson (shot) noise to an optical sensor image.

    Simulates discrete photon arrival statistics where the arrival count in each pixel
    follows a Poisson distribution whose variance equals its mean:
        N_photons ~ Poisson(I * scale_factor)
        I_noisy = N_photons / scale_factor

    Args:
        image: Input 2D uint8 sensor image.
        scale_factor: Conversion factor from digital numbers (ADU) to physical photon
                      counts (photons / count). Higher values represent higher photon
                      flux (lower relative shot noise variance). Default 1.0.
        rng: Optional NumPy random Generator for reproducible sampling.

    Returns:
        2D uint8 sensor image with Poisson shot noise applied.
    """
    if image is None or image.size == 0:
        return image

    if scale_factor <= 0:
        return image.copy()

    # Convert pixel intensities to expected photon counts (lambda parameter)
    # Ensure non-negative photon expectations
    lam = np.maximum(image.astype(np.float64) * scale_factor, 0.0)

    # Sample from Poisson distribution
    if rng is not None:
        photon_counts = rng.poisson(lam)
    else:
        photon_counts = np.random.poisson(lam)

    # Rescale photon counts back to digital pixel counts (ADU)
    noisy_img = photon_counts / scale_factor

    # Round, clip to valid uint8 dynamic range [0, 255]
    return np.clip(np.round(noisy_img), 0, 255).astype(np.uint8)


class SensorNoise:
    """
    Simulates optical sensor noise artifacts with selectable, stackable models:
    - Poisson photon shot noise (incident signal-dependent arrival statistics)
    - Additive zero-mean Gaussian readout noise (sensor amplifier/ADC)
    - Salt-and-pepper impulse noise (defective hot/dead sensor pixels)
    """

    def __init__(
        self,
        gaussian_std: float = 6.0,          # Standard deviation of readout noise
        salt_pepper_prob: float = 0.0005,   # Probability of hot/dead pixels (e.g. 0.05%)
        poisson_scale: float = 1.0,         # Photon scaling factor for Poisson shot noise
        noise_types: Optional[List[str]] = None, # Selectable noise models: e.g. ["gaussian", "poisson"]
        seed: int = 42,
    ):
        self.gaussian_std = float(gaussian_std)
        self.salt_pepper_prob = float(salt_pepper_prob)
        self.poisson_scale = float(poisson_scale)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._noise_buf: Optional[np.ndarray] = None

        # Parse and normalize user-selected noise types
        if noise_types is None:
            # Default backward compatibility: active if parameters indicate presence
            types = []
            if self.gaussian_std > 0:
                types.append("gaussian")
            if self.salt_pepper_prob > 0:
                types.append("salt_pepper")
            self.noise_types = types
        else:
            normalized = []
            for nt in noise_types:
                norm = normalize_noise_type(nt)
                if norm not in normalized:
                    normalized.append(norm)
            self.noise_types = normalized

    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply selected sensor noise models sequentially/additively to sensor image.

        Pipeline ordering:
        1. Poisson photon arrival shot noise (optical signal domain)
        2. Gaussian readout noise (analog amplifier / electronic readout)
        3. Salt-and-pepper impulse noise (detector pixel defects)

        Args:
            image: Input 2D uint8 sensor image.

        Returns:
            Noisy 2D uint8 sensor image.
        """
        if image is None or image.size == 0:
            return image

        if not self.noise_types:
            return image

        out = image

        # 1. Poisson Shot Noise (signal-dependent photon statistics)
        if "poisson" in self.noise_types and self.poisson_scale > 0:
            out = apply_poisson_noise(out, scale_factor=self.poisson_scale, rng=self._rng)

        # 2. Vectorized Seeded Gaussian Readout Noise
        if "gaussian" in self.noise_types and self.gaussian_std > 0:
            if self._noise_buf is None or self._noise_buf.shape != out.shape:
                self._noise_buf = np.empty(out.shape, dtype=np.float32)

            self._rng.standard_normal(out.shape, dtype=np.float32, out=self._noise_buf)
            self._noise_buf *= self.gaussian_std
            out_float = out.astype(np.float32) + self._noise_buf
            out = np.clip(np.round(out_float), 0, 255).astype(np.uint8)

        # 3. Vectorized Seeded Salt-and-Pepper Impulse Noise
        if "salt_pepper" in self.noise_types and self.salt_pepper_prob > 0:
            num_pixels = out.size
            half_sp = int(np.ceil(self.salt_pepper_prob * num_pixels)) // 2
            if half_sp > 0:
                out = out.copy()
                # Hot pixels (salt -> 255)
                y_salt = self._rng.integers(0, out.shape[0], half_sp)
                x_salt = self._rng.integers(0, out.shape[1], half_sp)
                out[y_salt, x_salt] = 255

                # Dead pixels (pepper -> 0)
                y_pep = self._rng.integers(0, out.shape[0], half_sp)
                x_pep = self._rng.integers(0, out.shape[1], half_sp)
                out[y_pep, x_pep] = 0

        return out
