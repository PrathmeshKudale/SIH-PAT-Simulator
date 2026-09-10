"""
Hierarchical Re-Acquisition Search Controller.
Smart India Hackathon - Problem Statement 26169 (ISRO / Department of Space)

Implements two-tier re-acquisition after full track loss:
- Tier 1: Localized Predictive Search centered on forward-projected target coordinates (ReacquisitionZone).
- Tier 2: Wide-Area Global Archimedean Spiral Fallback scanning full field of regard if Tier 1 times out.
- Includes toggle flag `enable_predictive_search` for direct A/B comparative benchmarking.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from contracts import CameraState, ReacquisitionZone
from control.search import SpiralSearchController


class HierarchicalReacquisitionController:
    """
    Coordinates two-tier re-acquisition search strategy upon track loss:
    1. Localized scan around predicted target intercept coordinates.
    2. Global wide-area Archimedean spiral if localized search does not recover target.
    """

    def __init__(
        self,
        fov_x: float = 0.10,
        fov_y: float = 0.075,
        tier1_budget_frames: int = 15,          # Number of frames allocated to localized search
        enable_predictive_search: bool = True,   # Toggle between localized predictive search and blind global spiral
        scan_rate: float = 0.12,                # Gimbal scan speed (rad/s)
    ):
        self.fov_x = float(fov_x)
        self.fov_y = float(fov_y)
        self.tier1_budget = int(tier1_budget_frames)
        self.enable_predictive_search = bool(enable_predictive_search)
        self.scan_rate = float(scan_rate)

        # Sub-controllers for localized and global spiral search
        self.localized_search = SpiralSearchController(
            fov_x=fov_x,
            fov_y=fov_y,
            linear_scan_rate=scan_rate,
            max_search_radius=0.045,
        )
        self.global_search = SpiralSearchController(
            fov_x=fov_x,
            fov_y=fov_y,
            linear_scan_rate=scan_rate,
            max_search_radius=0.080,
            center_pan=0.0,
            center_tilt=0.0,
        )

        # State machine
        self.is_active = False
        self.current_tier = "INACTIVE"  # "INACTIVE", "TIER1_PREDICTIVE", "TIER2_GLOBAL"
        self.tier1_frame_counter = 0
        self.current_zone: Optional[ReacquisitionZone] = None

    def reset(self) -> None:
        """Reset re-acquisition state machine to inactive."""
        self.is_active = False
        self.current_tier = "INACTIVE"
        self.tier1_frame_counter = 0
        self.current_zone = None
        self.localized_search.reset()
        self.global_search.reset()

    def start_reacquisition(
        self,
        zone: ReacquisitionZone,
        current_camera_state: CameraState,
    ) -> str:
        """
        Initiate re-acquisition search upon confirmed track loss.

        Args:
            zone: ReacquisitionZone computed from last known state and elapsed time.
            current_camera_state: Current CameraState of the gimbal.

        Returns:
            Active search tier string ('TIER1_PREDICTIVE' or 'TIER2_GLOBAL').
        """
        self.is_active = True
        self.current_zone = zone
        self.tier1_frame_counter = 0

        if self.enable_predictive_search:
            # Tier 1: Initialize localized search centered at predicted forward-projected position
            self.current_tier = "TIER1_PREDICTIVE"
            self.localized_search.reset(
                center_pan=zone.center_pan,
                center_tilt=zone.center_tilt,
            )
            self.localized_search.max_search_radius = max(0.025, zone.search_radius)
        else:
            # Bypass Tier 1: Fall back immediately to global Archimedean spiral search
            self.current_tier = "TIER2_GLOBAL"
            self.global_search.reset(
                center_pan=0.0,
                center_tilt=0.0,
            )

        return self.current_tier

    def step(
        self,
        dt: float,
        current_camera_state: CameraState,
    ) -> Tuple[float, float]:
        """
        Compute gimbal control step along the active re-acquisition search pattern.

        Args:
            dt: Time step duration in seconds.
            current_camera_state: Current CameraState.

        Returns:
            Tuple of (delta_pan, delta_tilt) angular adjustment in radians.
        """
        if not self.is_active:
            return (0.0, 0.0)

        if self.current_tier == "TIER1_PREDICTIVE":
            self.tier1_frame_counter += 1

            # Check if Tier 1 budget has expired
            if self.tier1_frame_counter > self.tier1_budget:
                # Fall back to Tier 2 Global Spiral Search
                self.current_tier = "TIER2_GLOBAL"
                self.global_search.reset(
                    center_pan=0.0,
                    center_tilt=0.0,
                )
                return self.global_search.step(dt, current_camera_state)

            # Advance localized search center dynamically along target predicted velocity
            if self.current_zone is not None and self.current_zone.predicted_velocity is not None:
                vx, vy = self.current_zone.predicted_velocity
                self.localized_search.center_pan += vx * dt
                self.localized_search.center_tilt += vy * dt

            # Step localized search
            return self.localized_search.step(dt, current_camera_state)

        elif self.current_tier == "TIER2_GLOBAL":
            # Step global wide-area spiral search
            return self.global_search.step(dt, current_camera_state)

        return (0.0, 0.0)
