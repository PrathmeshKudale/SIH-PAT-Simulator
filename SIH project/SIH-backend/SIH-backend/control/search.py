"""
Systematic acquisition search patterns (Archimedean spiral search) for the coarse PAT gimbal.
"""

from typing import Tuple, Optional
import numpy as np
from contracts import CameraState


class SpiralSearchController:
    """
    Generates an expanding Archimedean spiral search trajectory to systematically scan
    the 2D angular space when the target is not locked.

    Spiral definition:
        r(theta) = b * theta
        pan(theta)  = center_pan  + r(theta) * cos(theta)
        tilt(theta) = center_tilt + r(theta) * sin(theta)

    Radial pitch (b * 2*pi) is matched to the camera FOV to avoid coverage gaps.
    """

    def __init__(
        self,
        fov_x: float = 0.10,
        fov_y: float = 0.075,
        overlap_factor: float = 0.35,     # 35% overlap between spiral arms
        linear_scan_rate: float = 0.12,   # Scan speed (rad/s in angular space)
        max_search_radius: float = 0.08,  # Max scan radius (rad)
        center_pan: float = 0.0,
        center_tilt: float = 0.0,
    ):
        self.fov_x = float(fov_x)
        self.fov_y = float(fov_y)
        self.overlap_factor = float(overlap_factor)
        self.linear_scan_rate = float(linear_scan_rate)
        self.max_search_radius = float(max_search_radius)
        self.center_pan = float(center_pan)
        self.center_tilt = float(center_tilt)

        # Minimum effective FOV dimension
        effective_fov = min(self.fov_x, self.fov_y) * (1.0 - self.overlap_factor)
        # Radial growth per radian of rotation: delta_r_per_turn = 2 * pi * b = effective_fov
        self.b = effective_fov / (2.0 * np.pi)

        self.theta = 0.0
        self.current_pan = self.center_pan
        self.current_tilt = self.center_tilt

    def reset(self, center_pan: float = 0.0, center_tilt: float = 0.0) -> None:
        """Reset spiral search to center point."""
        self.center_pan = float(center_pan)
        self.center_tilt = float(center_tilt)
        self.theta = 0.0
        self.current_pan = self.center_pan
        self.current_tilt = self.center_tilt

    def step(self, dt: float, current_camera_state: Optional[CameraState] = None) -> Tuple[float, float]:
        """
        Compute the next angular adjustment (delta_pan, delta_tilt) along the spiral.

        Args:
            dt: Time step in seconds.
            current_camera_state: Optional current CameraState.

        Returns:
            Tuple of (delta_pan, delta_tilt) in radians.
        """
        # Current radius
        r = max(0.001, self.b * self.theta)

        # Angular rate dtheta/dt chosen such that linear tangential velocity = linear_scan_rate
        # v = r * dtheta/dt -> dtheta = (v / r) * dt
        # Clamp angular velocity at small r to avoid infinite spin at origin
        max_dtheta = 15.0 * dt
        dtheta = min(max_dtheta, (self.linear_scan_rate / r) * dt)
        self.theta += dtheta

        # Compute next target gimbal coordinates
        next_r = self.b * self.theta
        if next_r > self.max_search_radius:
            # Reached max radius, loop back inwards or reset
            self.theta = 0.0
            next_r = 0.0

        target_pan = self.center_pan + next_r * np.cos(self.theta)
        target_tilt = self.center_tilt + next_r * np.sin(self.theta)

        # Delta from current tracked camera pan/tilt
        current_pan = current_camera_state.pan if current_camera_state else self.current_pan
        current_tilt = current_camera_state.tilt if current_camera_state else self.current_tilt

        delta_pan = float(target_pan - current_pan)
        delta_tilt = float(target_tilt - current_tilt)

        self.current_pan = target_pan
        self.current_tilt = target_tilt

        return (delta_pan, delta_tilt)
