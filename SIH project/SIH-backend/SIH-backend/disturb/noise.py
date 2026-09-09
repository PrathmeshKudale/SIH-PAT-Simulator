"""
Optical Sensor Noise Generator (Readout Gaussian + Salt-and-Pepper Impulse Noise).
Smart India Hackathon - FSOC PAT Simulator (ISRO / DOS PS 26169)

Deterministic, vectorized single-precision noise generator using NumPy's seeded PRNG.
"""

from typing import Optional
import numpy as np


class SensorNoise:
    """
    Simulates optical sensor noise artifacts:
    - Additive zero-mean Gaussian readout noise
    - Salt-and-pepper impulse noise (hot and dead pixels)
    - Dark current noise
    """

    def __init__(
        self,
        gaussian_std: float = 6.0,          # Standard deviation of readout noise
        salt_pepper_prob: float = 0.0005,   # Probability of hot/dead pixels (e.g. 0.05%)
        seed: int = 42,
    ):
        self.gaussian_std = float(gaussian_std)
        self.salt_pepper_prob = float(salt_pepper_prob)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._noise_buf: Optional[np.ndarray] = None

    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian readout and salt-and-pepper impulse noise to sensor image.
        Uses vectorized NumPy single-precision Gaussian generation directly into a
        pre-allocated buffer, ensuring 100% deterministic reproducibility across runs.

        Args:
            image: Input 2D uint8 sensor image.

        Returns:
            Noisy 2D uint8 sensor image.
        """
        if image is None or image.size == 0:
            return image

        # Re-use pre-allocated single-precision buffer to eliminate per-frame allocations
        if self._noise_buf is None or self._noise_buf.shape != image.shape:
            self._noise_buf = np.empty(image.shape, dtype=np.float32)

        out = image.astype(np.float32)

        # 1. Vectorized Seeded Gaussian Readout Noise
        if self.gaussian_std > 0:
            self._rng.standard_normal(image.shape, dtype=np.float32, out=self._noise_buf)
            self._noise_buf *= self.gaussian_std
            out += self._noise_buf

        # 2. Vectorized Seeded Salt-and-Pepper Impulse Noise
        if self.salt_pepper_prob > 0:
            num_pixels = image.size
            half_sp = int(np.ceil(self.salt_pepper_prob * num_pixels)) // 2
            if half_sp > 0:
                # Hot pixels (salt -> 255)
                y_salt = self._rng.integers(0, image.shape[0], half_sp)
                x_salt = self._rng.integers(0, image.shape[1], half_sp)
                out[y_salt, x_salt] = 255.0

                # Dead pixels (pepper -> 0)
                y_pep = self._rng.integers(0, image.shape[0], half_sp)
                x_pep = self._rng.integers(0, image.shape[1], half_sp)
                out[y_pep, x_pep] = 0.0

        return np.clip(out, 0, 255).astype(np.uint8)
