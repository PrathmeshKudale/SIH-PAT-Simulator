"""
Slew-Rate Limiter capping maximum gimbal angular velocity and acceleration.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Enforces physical gimbal actuator dynamics:
- Maximum angular velocity cap (rad/s)
- Maximum angular acceleration cap (rad/s^2)
Prevents instantaneous or unphysical gimbal jumps even if commanded by the controller.
"""

from typing import Tuple
import numpy as np


class SlewRateLimiter:
    """
    Limits the rate of change of gimbal angles by enforcing maximum velocity
    and maximum acceleration constraints on both pan and tilt axes.
    """

    def __init__(
        self,
        max_velocity: float = 0.5,      # Maximum angular rate in rad/s (default 500 mrad/s)
        max_acceleration: float = 1.0,  # Maximum angular acceleration in rad/s^2 (default 1000 mrad/s^2)
    ):
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)

        # Internal state: current angular velocities (rad/s)
        self.current_pan_rate = 0.0
        self.current_tilt_rate = 0.0

    def reset(self) -> None:
        """Reset internal rate states to zero."""
        self.current_pan_rate = 0.0
        self.current_tilt_rate = 0.0

    def apply_limit(
        self,
        delta_pan_cmd: float,
        delta_tilt_cmd: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Apply velocity and acceleration caps to commanded angular steps.

        Args:
            delta_pan_cmd: Commanded azimuth angle change (rad).
            delta_tilt_cmd: Commanded elevation angle change (rad).
            dt: Time step duration in seconds.

        Returns:
            Tuple of (delta_pan_actual, delta_tilt_actual) strictly bounded by kinematics.
        """
        if dt <= 0:
            return (0.0, 0.0)

        # Commanded velocities (rad/s)
        target_pan_rate = delta_pan_cmd / dt
        target_tilt_rate = delta_tilt_cmd / dt

        # Maximum permissible rate change this time step (acceleration limit)
        max_rate_step = self.max_acceleration * dt

        # 1. Pan acceleration clamp
        pan_rate_delta = target_pan_rate - self.current_pan_rate
        pan_rate_delta_clamped = float(np.clip(pan_rate_delta, -max_rate_step, max_rate_step))
        new_pan_rate = self.current_pan_rate + pan_rate_delta_clamped

        # 2. Pan velocity clamp
        new_pan_rate = float(np.clip(new_pan_rate, -self.max_velocity, self.max_velocity))

        # 3. Tilt acceleration clamp
        tilt_rate_delta = target_tilt_rate - self.current_tilt_rate
        tilt_rate_delta_clamped = float(np.clip(tilt_rate_delta, -max_rate_step, max_rate_step))
        new_tilt_rate = self.current_tilt_rate + tilt_rate_delta_clamped

        # 4. Tilt velocity clamp
        new_tilt_rate = float(np.clip(new_tilt_rate, -self.max_velocity, self.max_velocity))

        # Update internal state
        self.current_pan_rate = new_pan_rate
        self.current_tilt_rate = new_tilt_rate

        # Delivered angular displacement over dt
        delta_pan_actual = new_pan_rate * dt
        delta_tilt_actual = new_tilt_rate * dt

        return (delta_pan_actual, delta_tilt_actual)
