"""
Target kinematics and beacon optical emissions (blinking modulation and clutter).
"""

from typing import Tuple, Optional
import numpy as np
from contracts import TargetState


class Target:
    """
    Simulates an optical beacon terminal with configurable kinematics and blinking modulation.
    """

    def __init__(
        self,
        initial_pos: Tuple[float, float] = (0.030, 0.020),
        velocity: Tuple[float, float] = (0.002, -0.001),
        base_intensity: float = 220.0,
        blink_frequency: float = 4.0,   # Hz
        modulation_depth: float = 0.85, # Fractional modulation (0.0 to 1.0)
        range_km: float = 5.0,          # Optical link distance (km)
        ref_range_km: float = 5.0,      # Reference distance where nominal intensity is calibrated
    ):
        self.x = float(initial_pos[0])
        self.y = float(initial_pos[1])
        self.vx = float(velocity[0])
        self.vy = float(velocity[1])
        self.base_intensity = float(base_intensity)
        self.blink_frequency = float(blink_frequency)
        self.modulation_depth = float(modulation_depth)
        self.range_km = float(range_km)
        self.ref_range_km = float(ref_range_km)
        self.timestamp = 0.0

    @property
    def current_intensity(self) -> float:
        """
        Compute modulated beacon intensity at current timestamp.
        Accounts for:
        1. Range-dependent optical free-space path loss (inverse-square law: (R_ref / R)^2).
        2. Blinking modulation.
        """
        # Range-based optical power scaling
        r = max(0.2, self.range_km)
        r_ref = max(0.2, self.ref_range_km)
        range_factor = (r_ref / r) ** 2

        # Periodic sinusoidal blinking modulation
        if self.blink_frequency > 0:
            phase = 2.0 * np.pi * self.blink_frequency * self.timestamp
            factor = (1.0 - self.modulation_depth) + self.modulation_depth * (0.5 * (1.0 + np.sin(phase)))
        else:
            factor = 1.0

        raw_intensity = self.base_intensity * factor * range_factor
        return float(np.clip(raw_intensity, 1.0, 10000.0))

    def step(self, dt: float) -> TargetState:
        """Advance target position and time."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.timestamp += dt
        return self.get_state()

    def get_state(self) -> TargetState:
        return TargetState(
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            confidence=1.0,
            timestamp=self.timestamp,
            tracker_mode="GROUND_TRUTH",
        )


class ClutterObject:
    """
    Simulates a false target / clutter object (e.g. constant solar glint or static background reflection).
    Emits a steady, non-blinking bright optical spot.
    """

    def __init__(
        self,
        pos: Tuple[float, float] = (0.015, 0.010),
        intensity: float = 230.0,  # Bright, steady non-blinking
    ):
        self.x = float(pos[0])
        self.y = float(pos[1])
        self.intensity = float(intensity)

    @property
    def current_intensity(self) -> float:
        """Constant non-blinking intensity."""
        return self.intensity
