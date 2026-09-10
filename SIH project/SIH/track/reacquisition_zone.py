"""
Predictive Search Zone Computation for Re-Acquisition After Track Loss.
Smart India Hackathon - Problem Statement 26169 (ISRO / Department of Space)

Projects target kinematic position and positional uncertainty forward along
last validated velocity vector (vx, vy) to define an optimized localized search zone.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from contracts import TargetState, ReacquisitionZone


class ReacquisitionZonePredictor:
    """
    Computes a kinematically projected search zone based on target's last known state
    and elapsed time since track loss.
    """

    def __init__(
        self,
        default_accel_uncertainty: float = 0.05,  # Estimated maximum target maneuver acceleration (rad/s^2)
        initial_pos_uncertainty: float = 0.002,   # Initial position covariance standard deviation (rad, ~2 mrad)
        max_zone_radius: float = 0.060,          # Maximum capped search zone radius (rad, ~60 mrad)
    ):
        self.accel_uncertainty = float(default_accel_uncertainty)
        self.initial_pos_uncertainty = float(initial_pos_uncertainty)
        self.max_zone_radius = float(max_zone_radius)

    def compute_zone(
        self,
        last_known_state: TargetState,
        current_time: float,
        camera_fov: Tuple[float, float] = (0.10, 0.075),
    ) -> ReacquisitionZone:
        """
        Compute predicted re-acquisition zone.

        Args:
            last_known_state: Last validated TargetState before dropout.
            current_time: Current simulation timestamp in seconds.
            camera_fov: (fov_x, fov_y) in radians.

        Returns:
            ReacquisitionZone containing forward-projected center and uncertainty bounds.
        """
        if last_known_state is None:
            center_pan = 0.0
            center_tilt = 0.0
            dt_elapsed = 0.0
            search_radius = float(min(camera_fov[0], camera_fov[1]))
            return ReacquisitionZone(
                center_pan=center_pan,
                center_tilt=center_tilt,
                search_radius=search_radius,
                predicted_velocity=(0.0, 0.0),
                projection_time_s=dt_elapsed,
                pan_bounds=(center_pan - search_radius, center_pan + search_radius),
                tilt_bounds=(center_tilt - search_radius, center_tilt + search_radius),
            )

        dt_elapsed = max(0.01, float(current_time - last_known_state.timestamp))

        # Kinematic forward projection along estimated velocity vector
        center_pan = float(last_known_state.x + last_known_state.vx * dt_elapsed)
        center_tilt = float(last_known_state.y + last_known_state.vy * dt_elapsed)

        # Dynamic uncertainty expansion: sigma(dt) = sigma_0 + 0.5 * a_max * dt^2
        maneuver_growth = 0.5 * self.accel_uncertainty * (dt_elapsed ** 2)
        vel_uncertainty_growth = 0.010 * dt_elapsed  # 10 mrad/s velocity error accumulation
        computed_radius = self.initial_pos_uncertainty + maneuver_growth + vel_uncertainty_growth

        # Ensure search zone is at least matched to half the sensor FOV
        min_zone_radius = 0.5 * min(camera_fov[0], camera_fov[1])
        search_radius = float(np.clip(
            max(min_zone_radius, computed_radius),
            min_zone_radius,
            self.max_zone_radius,
        ))

        pan_bounds = (center_pan - search_radius, center_pan + search_radius)
        tilt_bounds = (center_tilt - search_radius, center_tilt + search_radius)

        return ReacquisitionZone(
            center_pan=center_pan,
            center_tilt=center_tilt,
            search_radius=search_radius,
            predicted_velocity=(float(last_known_state.vx), float(last_known_state.vy)),
            projection_time_s=dt_elapsed,
            pan_bounds=pan_bounds,
            tilt_bounds=tilt_bounds,
        )
