"""
Target kinematics and beacon optical emissions (blinking modulation and clutter).
"""

from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
from contracts import TargetState, TargetConfig
from sim.motion_profiles import MotionProfile, create_motion_profile


class Target:
    """
    Simulates an optical beacon terminal with configurable kinematics and blinking modulation.
    Supports multi-target discrimination via target_id and unique blink frequencies.
    Kinematic trajectories are governed by selectable MotionProfile strategies.
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
        target_id: Union[str, int] = "target_0",
        motion_type: str = "straight_line",
        motion_params: Optional[Dict[str, Any]] = None,
        motion_profile: Optional[MotionProfile] = None,
    ):
        self.target_id = target_id
        self.base_intensity = float(base_intensity)
        self.blink_frequency = float(blink_frequency)
        self.modulation_depth = float(modulation_depth)
        self.range_km = float(range_km)
        self.ref_range_km = float(ref_range_km)
        self.timestamp = 0.0

        # Kinematics motion profile
        self.motion_type = str(motion_type)
        self.motion_params = dict(motion_params) if motion_params is not None else {}
        if motion_profile is not None:
            self.motion_profile = motion_profile
        else:
            self.motion_profile = create_motion_profile(
                motion_type=self.motion_type,
                initial_pos=initial_pos,
                velocity=velocity,
                **self.motion_params
            )

        self.x, self.y = self.motion_profile.get_position(0.0)
        self.vx, self.vy = self.motion_profile.get_velocity(0.0)

    @classmethod
    def from_config(cls, cfg: Union[Dict[str, Any], TargetConfig]) -> "Target":
        """Factory method to instantiate a Target from a config dict or TargetConfig."""
        if isinstance(cfg, TargetConfig):
            return cls(
                initial_pos=cfg.initial_pos,
                velocity=cfg.velocity,
                base_intensity=cfg.base_intensity,
                blink_frequency=cfg.blink_frequency,
                modulation_depth=cfg.modulation_depth,
                range_km=cfg.range_km,
                ref_range_km=cfg.ref_range_km,
                target_id=cfg.target_id,
                motion_type=cfg.motion_type,
                motion_params=cfg.motion_params,
            )
        pos = cfg.get("initial_pos", cfg.get("initial_position", (0.030, 0.020)))
        vel = cfg.get("velocity", cfg.get("initial_velocity", (0.002, -0.001)))
        motion_type = cfg.get("motion_type", "straight_line")
        motion_params = dict(cfg.get("motion_params", {}))
        for k in [
            "center", "radius", "angular_velocity", "amplitude_x", "amplitude_y",
            "frequency", "phase_offset", "step_variance", "bounds", "seed",
            "initial_radius", "radial_velocity", "min_radius", "max_radius",
            "amplitude", "axis"
        ]:
            if k in cfg and k not in motion_params:
                motion_params[k] = cfg[k]

        return cls(
            initial_pos=tuple(pos) if isinstance(pos, (list, tuple)) else pos,
            velocity=tuple(vel) if isinstance(vel, (list, tuple)) else vel,
            base_intensity=float(cfg.get("base_intensity", cfg.get("beacon_power", 220.0))),
            blink_frequency=float(cfg.get("blink_frequency", 4.0)),
            modulation_depth=float(cfg.get("modulation_depth", 0.5)),
            range_km=float(cfg.get("range_km", cfg.get("distance_km", 5.0))),
            ref_range_km=float(cfg.get("ref_range_km", 5.0)),
            target_id=cfg.get("target_id", "target_0"),
            motion_type=motion_type,
            motion_params=motion_params,
        )

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
        """Advance target position and time according to active motion profile."""
        self.timestamp += dt
        self.x, self.y = self.motion_profile.get_position(self.timestamp)
        self.vx, self.vy = self.motion_profile.get_velocity(self.timestamp, dt=dt)
        return self.get_state()

    def reset(self) -> None:
        """Reset target kinematics to initial timestamp 0.0."""
        self.timestamp = 0.0
        self.motion_profile.reset()
        self.x, self.y = self.motion_profile.get_position(0.0)
        self.vx, self.vy = self.motion_profile.get_velocity(0.0)

    def get_state(self) -> TargetState:
        return TargetState(
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            confidence=1.0,
            timestamp=self.timestamp,
            tracker_mode="GROUND_TRUTH",
            target_id=self.target_id,
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
