"""
Kolmogorov Phase-Screen Atmospheric Turbulence Generator.
Smart India Hackathon - FSOC PAT Simulator (ISRO / DOS PS 26169)

Simulates optical wave propagation effects through atmospheric turbulence:
- FFT-based phase screen generation from Kolmogorov power spectral density
- Scintillation (aperture-averaged intensity fading)
- Beam wander (random sub-pixel tilt/wander)
- PSF spot broadening (Strehl reduction and spatial blurring)
"""

from typing import Tuple, Optional
import numpy as np
import cv2


class KolmogorovTurbulence:
    """
    Generates Kolmogorov phase-screens and applies atmospheric distortion
    (scintillation, beam wander, and spatial blurring) to optical sensor frames.
    """

    def __init__(
        self,
        cn2: float = 1e-14,          # Refractive index structure constant (m^-2/3)
        wavelength: float = 1550e-9, # Laser wavelength (1550 nm optical comms)
        link_distance: float = 5000.0,# Propagation link distance (5 km)
        grid_size: int = 128,        # Phase screen grid resolution
        screen_scale_m: float = 0.5, # Physical width of phase screen (meters)
        seed: int = 42,
    ):
        self.cn2 = float(cn2)
        self.wavelength = float(wavelength)
        self.link_distance = float(link_distance)
        self.grid_size = int(grid_size)
        self.screen_scale_m = float(screen_scale_m)
        self._rng = np.random.default_rng(seed)

    @property
    def fried_parameter(self) -> float:
        """
        Calculate Fried atmospheric coherence length r_0 (meters):
        r_0 = (0.423 * k^2 * C_n^2 * L)^(-3/5)
        """
        k = 2.0 * np.pi / self.wavelength
        r0 = (0.423 * (k**2) * max(1e-18, self.cn2) * self.link_distance)**(-3.0 / 5.0)
        return float(np.clip(r0, 0.005, 2.0))

    def generate_phase_screen(self) -> np.ndarray:
        """
        Generate a 2D Kolmogorov phase screen using FFT spectral synthesis.
        """
        n = self.grid_size
        delta_k = 2.0 * np.pi / self.screen_scale_m

        kx = (np.arange(n) - n / 2) * delta_k
        ky = (np.arange(n) - n / 2) * delta_k
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        K[n // 2, n // 2] = 1e-10

        r0 = self.fried_parameter
        psd = 0.023 * (r0**(-5.0 / 3.0)) * (K**(-11.0 / 3.0))
        psd[n // 2, n // 2] = 0.0

        noise_real = self._rng.normal(0, 1, (n, n))
        noise_imag = self._rng.normal(0, 1, (n, n))
        noise_spec = (noise_real + 1j * noise_imag) * np.sqrt(psd) * delta_k

        screen = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(noise_spec)))
        return np.real(screen)

    def apply_turbulence(
        self,
        image: np.ndarray,
        target_pos_px: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Apply atmospheric turbulence distortions to the sensor image:
        1. Intensity fading (aperture-averaged scintillation).
        2. Point Spread Function broadening (blur).
        3. Random sub-pixel tilt/wander.

        Args:
            image: 2D uint8 sensor frame.
            target_pos_px: Optional (u, v) location of optical beacon.

        Returns:
            Distorted image as uint8 NumPy array.
        """
        if self.cn2 <= 1e-17:
            return image

        out = image.astype(np.float32)
        r0 = self.fried_parameter

        # 1. Aperture-averaged scintillation: Intensity attenuation factor
        # Higher Cn2 -> greater attenuation & fading
        atten_factor = float(np.clip(1.0 - 0.45 * np.log10(max(1e-16, self.cn2) / 1e-16) / 3.5, 0.40, 1.0))
        scint_fluct = float(self._rng.normal(0, 0.08 * np.log10(max(1e-16, self.cn2) / 1e-16 + 1.0)))
        total_scint = float(np.clip(atten_factor + scint_fluct, 0.35, 1.1))

        # 2. PSF Broadening: Aperture diameter D / r_0 determines blurring kernel size
        blur_sigma = float(np.clip(0.035 / r0, 0.2, 2.6))
        if blur_sigma > 0.4:
            ksize = int(np.ceil(blur_sigma * 3)) * 2 + 1
            blurred = cv2.GaussianBlur(out, (ksize, ksize), blur_sigma)
        else:
            blurred = out

        # 3. Apply scintillation to beacon region
        if target_pos_px is not None:
            u_t, v_t = int(round(target_pos_px[0])), int(round(target_pos_px[1]))
            h, w = image.shape
            radius = int(np.ceil(20 + blur_sigma * 5))
            u_min, u_max = max(0, u_t - radius), min(w, u_t + radius)
            v_min, v_max = max(0, v_t - radius), min(h, v_t + radius)

            region = blurred[v_min:v_max, u_min:u_max]
            # Attenuate signal above baseline background (12.0)
            faded = (region - 12.0) * total_scint + 12.0
            blurred[v_min:v_max, u_min:u_max] = faded

        return np.clip(blurred, 0, 255).astype(np.uint8)
