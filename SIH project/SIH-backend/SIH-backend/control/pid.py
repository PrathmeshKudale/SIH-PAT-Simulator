"""
Gimbal PID Controller with Integral Anti-Windup, Rate Damping, and Velocity Feedforward.
Smart India Hackathon - FSOC PAT Simulator (ISRO / DOS PS 26169)
"""

from typing import Any, Optional, Tuple
import numpy as np
from contracts import CameraState, TargetState


class PIDController:
    """
    Closed-loop PID controller with velocity feedforward, rate damping, and integral anti-windup,
    driving camera boresight (pan, tilt) toward the Kalman Filter's estimated target position.
    """

    def __init__(
        self,
        kp: float = 0.75,
        ki: float = 0.15,
        kd: float = 0.04,                # Rate damping gain (derivative on measurement)
        k_ff: float = 0.95,              # Velocity feedforward gain
        enable_feedforward: bool = True, # Toggle velocity feedforward on/off
        latency_compensation_frames: int = 0, # Lead factor for latency compensation
        max_integral: float = 0.005,     # Max integral windup clamp (5 mrad)
        max_angular_step: Optional[float] = None,
        slew_limiter: Optional[Any] = None,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.k_ff = float(k_ff)
        self.enable_feedforward = bool(enable_feedforward)
        self.latency_compensation_frames = max(0, int(latency_compensation_frames))
        self.max_integral = float(max_integral)
        self.max_angular_step = float(max_angular_step) if max_angular_step is not None else None
        self.slew_limiter = slew_limiter

        # Controller state
        self.integral_pan = 0.0
        self.integral_tilt = 0.0
        self.prev_cam_pan: Optional[float] = None
        self.prev_cam_tilt: Optional[float] = None

    def reset(self) -> None:
        """Reset internal integrator, rate history, and slew limiter."""
        self.integral_pan = 0.0
        self.integral_tilt = 0.0
        self.prev_cam_pan = None
        self.prev_cam_tilt = None
        if self.slew_limiter is not None:
            self.slew_limiter.reset()

    def compute_command(
        self,
        target_state: Optional[TargetState],
        camera_state: CameraState,
        dt: float = 0.033,
    ) -> Tuple[float, float]:
        """
        Compute commanded gimbal angular adjustments (delta_pan, delta_tilt) in radians.

        Args:
            target_state: TargetState estimated by Kalman Filter (in world angular coordinates),
                          or None if completely lost.
            camera_state: Current CameraState.
            dt: Control loop sample period in seconds.

        Returns:
            Tuple of (delta_pan, delta_tilt) in radians.
        """
        if target_state is None or dt <= 0:
            return (0.0, 0.0)

        # 1. Position error (target estimate vs current camera boresight)
        error_pan = target_state.x - camera_state.pan
        error_tilt = target_state.y - camera_state.tilt

        # 2. Proportional action
        p_pan = self.kp * error_pan
        p_tilt = self.kp * error_tilt

        # 3. Integral action with anti-windup clamping
        self.integral_pan += error_pan * dt
        self.integral_tilt += error_tilt * dt
        self.integral_pan = float(np.clip(self.integral_pan, -self.max_integral, self.max_integral))
        self.integral_tilt = float(np.clip(self.integral_tilt, -self.max_integral, self.max_integral))

        i_pan = self.ki * self.integral_pan
        i_tilt = self.ki * self.integral_tilt

        # 4. Derivative action on measurement (rate damping to prevent derivative kick)
        if self.prev_cam_pan is not None and self.prev_cam_tilt is not None:
            pan_vel = (camera_state.pan - self.prev_cam_pan) / dt
            tilt_vel = (camera_state.tilt - self.prev_cam_tilt) / dt
        else:
            pan_vel = camera_state.pan_rate
            tilt_vel = camera_state.tilt_rate

        self.prev_cam_pan = camera_state.pan
        self.prev_cam_tilt = camera_state.tilt

        d_pan = -self.kd * pan_vel * dt
        d_tilt = -self.kd * tilt_vel * dt

        # 5. Velocity Feedforward action (cancels lag behind moving target)
        if self.enable_feedforward:
            lead_factor = 1.0 + float(self.latency_compensation_frames)
            ff_pan = self.k_ff * target_state.vx * dt * lead_factor
            ff_tilt = self.k_ff * target_state.vy * dt * lead_factor
        else:
            ff_pan = 0.0
            ff_tilt = 0.0

        # Total commanded angular adjustment
        cmd_pan = p_pan + i_pan + d_pan + ff_pan
        cmd_tilt = p_tilt + i_tilt + d_tilt + ff_tilt

        # 6. Slew-rate limiting (acceleration and velocity clamping)
        if self.slew_limiter is not None:
            cmd_pan, cmd_tilt = self.slew_limiter.apply_limit(cmd_pan, cmd_tilt, dt=dt)
        elif self.max_angular_step is not None:
            cmd_pan = float(np.clip(cmd_pan, -self.max_angular_step, self.max_angular_step))
            cmd_tilt = float(np.clip(cmd_tilt, -self.max_angular_step, self.max_angular_step))

        return (cmd_pan, cmd_tilt)


class ProportionalController:
    """Legacy proportional controller retained for backward compatibility."""

    def __init__(self, kp: float = 0.8, max_angular_step: Optional[float] = None):
        self.kp = float(kp)
        self.max_angular_step = float(max_angular_step) if max_angular_step is not None else None

    def compute_command(
        self,
        detected_target: Optional[TargetState],
        camera_state: CameraState,
    ) -> Tuple[float, float]:
        if detected_target is None:
            return (0.0, 0.0)

        w, h = camera_state.resolution
        cx = w / 2.0
        cy = h / 2.0

        error_u = detected_target.x - cx
        error_v = detected_target.y - cy

        angular_err_pan = error_u * (camera_state.fov_x / w)
        angular_err_tilt = error_v * (camera_state.fov_y / h)

        delta_pan = self.kp * angular_err_pan
        delta_tilt = self.kp * angular_err_tilt

        if self.max_angular_step is not None:
            delta_pan = float(np.clip(delta_pan, -self.max_angular_step, self.max_angular_step))
            delta_tilt = float(np.clip(delta_tilt, -self.max_angular_step, self.max_angular_step))

        return (delta_pan, delta_tilt)
