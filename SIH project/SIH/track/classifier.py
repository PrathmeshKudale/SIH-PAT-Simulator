"""
Track Loss Root Cause Classifier for FSOC Coarse PAT Simulator.
Smart India Hackathon - Problem Statement 26169 (ISRO / Department of Space)

Diagnoses and classifies full track loss events into physical root causes:
1. OCCLUSION: Line-of-sight blockage by opaque/semi-opaque dynamic obstacles.
2. TURBULENCE: Deep scintillation fading under severe atmospheric turbulence (Cn^2 >= 1e-13).
3. FAST_MOTION: Target kinematic speed or acceleration exceeding gimbal physical slew limits.
4. DETECTION_DROPOUT: Benign/transient sensor readout or threshold anomalies.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from contracts import TargetState, CameraState


class TrackLossClassifier:
    """
    Evaluates kinematic state, environmental disturbance levels, and physical gimbal
    constraints to classify the root cause of a track-loss event.
    """

    def __init__(
        self,
        turbulence_cn2_threshold: float = 1.0e-13,  # Cn^2 threshold for severe turbulence fade
        max_slew_velocity: float = 0.50,            # Gimbal max velocity cap (rad/s)
        max_slew_acceleration: float = 1.00,        # Gimbal max acceleration cap (rad/s^2)
    ):
        self.cn2_threshold = float(turbulence_cn2_threshold)
        self.max_slew_vel = float(max_slew_velocity)
        self.max_slew_accel = float(max_slew_acceleration)

    def classify(
        self,
        consecutive_misses: int,
        last_known_state: Optional[TargetState],
        camera_state: CameraState,
        active_occluder: Optional[Any] = None,
        active_cn2: Optional[float] = None,
        target_true_pos: Optional[Tuple[float, float]] = None,
    ) -> Tuple[str, float, str]:
        """
        Classify the root cause of track loss.

        Args:
            consecutive_misses: Number of consecutive frames without valid detection.
            last_known_state: Last validated TargetState before dropout.
            camera_state: Current CameraState of the gimbal.
            active_occluder: Optional active DynamicOccluder instance in the scene.
            active_cn2: Active atmospheric turbulence parameter Cn^2 (m^-2/3).
            target_true_pos: Optional ground-truth target angular position (x, y) in rad.

        Returns:
            Tuple of:
                - cause: Root cause identifier string
                - confidence: Classification confidence [0.0, 1.0]
                - rationale: Detailed technical diagnostic rationale string
        """
        # 1. Check for physical Line-of-Sight Occlusion
        if active_occluder is not None:
            # Check if target true position or last known position intersects occluder
            eval_pos = target_true_pos if target_true_pos is not None else (
                (last_known_state.x, last_known_state.y) if last_known_state else None
            )
            if eval_pos is not None and hasattr(active_occluder, "is_target_occluded"):
                if active_occluder.is_target_occluded(eval_pos):
                    dist = np.hypot(eval_pos[0] - active_occluder.x, eval_pos[1] - active_occluder.y)
                    rationale = (
                        f"Target position [{eval_pos[0]*1e3:+.1f}, {eval_pos[1]*1e3:+.1f}] mrad "
                        f"intersected by dynamic occluder at [{active_occluder.x*1e3:+.1f}, {active_occluder.y*1e3:+.1f}] mrad "
                        f"(dist = {dist*1e3:.2f} mrad <= radius {active_occluder.radius_rad*1e3:.2f} mrad)"
                    )
                    return ("OCCLUSION", 0.98, rationale)

        # 2. Check for Severe Atmospheric Turbulence Scintillation Fade
        if active_cn2 is not None and active_cn2 >= self.cn2_threshold:
            rationale = (
                f"Severe atmospheric turbulence Cn^2 = {active_cn2:.1e} m^-2/3 >= {self.cn2_threshold:.1e} "
                f"causing deep scintillation intensity fade below detection threshold"
            )
            return ("TURBULENCE", 0.92, rationale)

        # 3. Check for Fast Motion Exceeding Physical Gimbal Slew Limits
        if last_known_state is not None:
            target_speed = np.hypot(last_known_state.vx, last_known_state.vy)
            # Check if target speed exceeds gimbal capability or if target escaped FOV
            if target_speed > self.max_slew_vel:
                rationale = (
                    f"Target angular speed {target_speed*1e3:.1f} mrad/s exceeds "
                    f"maximum gimbal slew velocity cap {self.max_slew_vel*1e3:.1f} mrad/s"
                )
                return ("FAST_MOTION", 0.95, rationale)

            # Check if target escaped camera FOV while gimbal was slew-saturated
            delta_pan = abs(last_known_state.x - camera_state.pan)
            delta_tilt = abs(last_known_state.y - camera_state.tilt)
            half_fov_x = camera_state.fov_x / 2.0
            half_fov_y = camera_state.fov_y / 2.0

            if delta_pan > half_fov_x or delta_tilt > half_fov_y:
                rationale = (
                    f"Target angular separation [{delta_pan*1e3:.1f}, {delta_tilt*1e3:.1f}] mrad "
                    f"exceeds camera half-FOV [{half_fov_x*1e3:.1f}, {half_fov_y*1e3:.1f}] mrad "
                    f"due to high-rate target maneuver"
                )
                return ("FAST_MOTION", 0.88, rationale)

        # 4. Default: Transient Detection Dropout
        rationale = (
            f"Transient optical detection dropout sustained across {consecutive_misses} consecutive frames "
            f"under benign atmospheric conditions (Cn^2 < {self.cn2_threshold:.1e}) and moderate kinematics"
        )
        return ("DETECTION_DROPOUT", 0.75, rationale)
