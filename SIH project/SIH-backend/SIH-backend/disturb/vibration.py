"""
Platform Vibration and Mechanical Jitter Generator.
Smart India Hackathon - FSOC PAT Simulator (ISRO / DOS PS 26169)

Simulates satellite / airborne optical communication terminal base-motion:
- Sinusoidal structural harmonics (reaction wheels, solar panel flutter)
- Random-walk platform drift / Gauss-Markov jitter
"""

from typing import Tuple
import numpy as np


class PlatformVibration:
    """
    Simulates platform angular pointing jitter combining deterministic sinusoidal
    vibrations with stochastic random-walk drift.
    """

    def __init__(
        self,
        amplitude_rad: float = 0.0005,     # 0.5 mrad amplitude
        frequency_hz: float = 12.0,        # 12 Hz harmonic vibration
        random_walk_std: float = 0.0001,   # 0.1 mrad drift step
        phase_offset: float = 0.0,
        seed: int = 42,
    ):
        self.amplitude_rad = float(amplitude_rad)
        self.frequency_hz = float(frequency_hz)
        self.random_walk_std = float(random_walk_std)
        self.phase_offset = float(phase_offset)
        self._rng = np.random.default_rng(seed)

        # Random walk state
        self.rw_pan = 0.0
        self.rw_tilt = 0.0

    def reset(self) -> None:
        """Reset random walk accumulator."""
        self.rw_pan = 0.0
        self.rw_tilt = 0.0

    def step(self, t: float, dt: float = 0.033) -> Tuple[float, float]:
        """
        Calculate instantaneous angular jitter offset (delta_pan, delta_tilt) in radians.

        Args:
            t: Current simulation time in seconds.
            dt: Time step duration.

        Returns:
            Tuple of (jitter_pan, jitter_tilt) in radians.
        """
        # 1. Harmonic sinusoidal vibration
        omega = 2.0 * np.pi * self.frequency_hz
        sin_pan = self.amplitude_rad * np.sin(omega * t + self.phase_offset)
        sin_tilt = self.amplitude_rad * np.cos(omega * t + self.phase_offset + np.pi / 4.0)

        # 2. Gauss-Markov random-walk drift with mild mean reversion (0.95 factor)
        self.rw_pan = 0.95 * self.rw_pan + self._rng.normal(0, self.random_walk_std)
        self.rw_tilt = 0.95 * self.rw_tilt + self._rng.normal(0, self.random_walk_std)

        total_pan_jitter = float(sin_pan + self.rw_pan)
        total_tilt_jitter = float(sin_tilt + self.rw_tilt)

        return (total_pan_jitter, total_tilt_jitter)
