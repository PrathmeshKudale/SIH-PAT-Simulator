"""
Dynamic Line-of-Sight Occluder Generator.
Smart India Hackathon - FSOC PAT Simulator (ISRO / DOS PS 26169)

Simulates physical line-of-sight blockage (e.g. clouds, plumes, satellite appendages, debris):
- Independent kinematics moving across the scene
- Configurable shape, opacity, and size
- Blocks optical beacon visibility when overlapping
"""

from typing import Tuple, Optional, List
import numpy as np
import cv2
from contracts import CameraState


class DynamicOccluder:
    """
    Simulates an opaque or semi-opaque object drifting across the field of view,
    blocking line-of-sight to the optical beacon.
    """

    def __init__(
        self,
        initial_pos: Tuple[float, float] = (0.010, 0.008),  # World angular coordinates (rad)
        velocity: Tuple[float, float] = (0.003, -0.001),    # Drift velocity (rad/s)
        radius_rad: float = 0.008,                          # Angular radius (rad, ~8 mrad)
        opacity: float = 1.0,                               # 1.0 = fully opaque (complete blackout)
        attenuation_floor: float = 12.0,                    # Background intensity floor behind occluder
    ):
        self.x = float(initial_pos[0])
        self.y = float(initial_pos[1])
        self.vx = float(velocity[0])
        self.vy = float(velocity[1])
        self.radius_rad = float(radius_rad)
        self.opacity = float(np.clip(opacity, 0.0, 1.0))
        self.attenuation_floor = float(attenuation_floor)

    def step(self, dt: float) -> None:
        """Advance occluder kinematics."""
        self.x += self.vx * dt
        self.y += self.vy * dt

    def is_target_occluded(self, target_pos: Tuple[float, float]) -> bool:
        """Check whether target at (tgt_x, tgt_y) overlaps this occluder."""
        dist = np.hypot(target_pos[0] - self.x, target_pos[1] - self.y)
        return bool(dist <= self.radius_rad)

    def apply_to_frame(
        self,
        image: np.ndarray,
        camera_state: CameraState,
    ) -> Tuple[np.ndarray, bool]:
        """
        Render the occluder onto the sensor frame.

        Args:
            image: 2D uint8 sensor frame.
            camera_state: Current CameraState to project occluder into focal plane pixels.

        Returns:
            Tuple of (occluded_image, is_visible_in_fov).
        """
        w, h = camera_state.resolution
        cx = w / 2.0
        cy = h / 2.0

        # Project occluder center into focal plane
        delta_pan = self.x - camera_state.pan
        delta_tilt = self.y - camera_state.tilt

        # Check if anywhere near FOV
        if abs(delta_pan) > (camera_state.fov_x / 2.0 + self.radius_rad) or \
           abs(delta_tilt) > (camera_state.fov_y / 2.0 + self.radius_rad):
            return image, False

        u_c = int(round(cx + delta_pan * (w / camera_state.fov_x)))
        v_c = int(round(cy + delta_tilt * (h / camera_state.fov_y)))
        radius_px = max(5, int(round(self.radius_rad * (w / camera_state.fov_x))))

        out = image.copy().astype(np.float32)

        # Create smooth boundary mask for realism
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (u_c, v_c), radius_px, 1.0, -1)
        # Soft boundary edges
        mask = cv2.GaussianBlur(mask, (15, 15), 3.0) * self.opacity

        # In occluded region: attenuate transmission toward background floor
        out = out * (1.0 - mask) + (self.attenuation_floor * mask)

        return np.clip(out, 0, 255).astype(np.uint8), True
