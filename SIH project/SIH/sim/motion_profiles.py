"""
Parametric Motion Profiles for FSOC Target and Platform Kinematics.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Implements selectable parametric trajectory generators:
- Straight line (linear, baseline)
- Circular (constant radius and angular velocity)
- Figure-eight (Lissajous 1:2 frequency figure-8 lemniscate)
- Random walk (bounded stochastic drift with boundary reflections)
- Spiral (expanding/contracting radius with angular velocity)
- Sinusoidal (linear drift with transverse harmonic oscillation)

All profiles implement the unified MotionProfile interface and can be applied
identically to both optical targets and platform base motion.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np


class MotionProfile(ABC):
    """
    Abstract interface for kinematic trajectory generators.
    Returns 2D position (x, y) and velocity (vx, vy) at simulation time t.
    """

    def __init__(self):
        self.t: float = 0.0

    @abstractmethod
    def get_position(self, t: float) -> Tuple[float, float]:
        """
        Calculate instantaneous 2D position (x, y) in radians or meters at time t.

        Args:
            t: Simulation time in seconds.

        Returns:
            Tuple of (x, y) coordinates.
        """
        pass

    def get_velocity(self, t: float, dt: float = 1e-4) -> Tuple[float, float]:
        """
        Calculate instantaneous 2D velocity (vx, vy) at time t.
        Default implementation uses central finite differences if analytical is not overridden.

        Args:
            t: Simulation time in seconds.
            dt: Small time delta for numerical differentiation.

        Returns:
            Tuple of (vx, vy) velocity components.
        """
        p_plus = self.get_position(t + dt)
        p_minus = self.get_position(max(0.0, t - dt))
        span = (t + dt) - max(0.0, t - dt)
        if span <= 0:
            span = dt
        vx = (p_plus[0] - p_minus[0]) / span
        vy = (p_plus[1] - p_minus[1]) / span
        return float(vx), float(vy)

    def step(self, dt: float) -> Tuple[float, float]:
        """Advance internal clock by dt and return position."""
        self.t += dt
        return self.get_position(self.t)

    def reset(self) -> None:
        """Reset internal trajectory clock and state."""
        self.t = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile parameters to dictionary."""
        return {"motion_type": self.__class__.__name__}


class StraightLineProfile(MotionProfile):
    """
    Constant velocity rectilinear motion:
        x(t) = x0 + vx * t
        y(t) = y0 + vy * t
    """

    def __init__(
        self,
        initial_pos: Tuple[float, float] = (0.0, 0.0),
        velocity: Tuple[float, float] = (0.002, 0.001),
    ):
        super().__init__()
        self.x0 = float(initial_pos[0])
        self.y0 = float(initial_pos[1])
        self.vx = float(velocity[0])
        self.vy = float(velocity[1])

    def get_position(self, t: float) -> Tuple[float, float]:
        return (self.x0 + self.vx * t, self.y0 + self.vy * t)

    def get_velocity(self, t: float, dt: float = 1e-4) -> Tuple[float, float]:
        return (self.vx, self.vy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motion_type": "straight_line",
            "initial_pos": (self.x0, self.y0),
            "velocity": (self.vx, self.vy),
        }


class CircularProfile(MotionProfile):
    """
    Constant radius circular orbit:
        x(t) = xc + R * cos(omega * t + phi)
        y(t) = yc + R * sin(omega * t + phi)
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        radius: float = 0.010,
        angular_velocity: float = 0.5,
        phase_offset: float = 0.0,
    ):
        super().__init__()
        self.xc = float(center[0])
        self.yc = float(center[1])
        self.radius = float(abs(radius))
        self.omega = float(angular_velocity)
        self.phi = float(phase_offset)

    def get_position(self, t: float) -> Tuple[float, float]:
        theta = self.omega * t + self.phi
        x = self.xc + self.radius * np.cos(theta)
        y = self.yc + self.radius * np.sin(theta)
        return (float(x), float(y))

    def get_velocity(self, t: float, dt: float = 1e-4) -> Tuple[float, float]:
        theta = self.omega * t + self.phi
        vx = -self.radius * self.omega * np.sin(theta)
        vy = self.radius * self.omega * np.cos(theta)
        return (float(vx), float(vy))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motion_type": "circular",
            "center": (self.xc, self.yc),
            "radius": self.radius,
            "angular_velocity": self.omega,
            "phase_offset": self.phi,
        }


class FigureEightProfile(MotionProfile):
    """
    Lissajous 1:2 frequency figure-8 trajectory (Lemniscate):
        x(t) = xc + Ax * sin(omega * t + phi)
        y(t) = yc + Ay * sin(2 * omega * t + 2 * phi)
    At t = 0 (and multiples of half-period), the trajectory crosses the center (xc, yc).
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        amplitude_x: float = 0.010,
        amplitude_y: float = 0.005,
        frequency: float = 0.2,
        phase_offset: float = 0.0,
    ):
        super().__init__()
        self.xc = float(center[0])
        self.yc = float(center[1])
        self.ax = float(amplitude_x)
        self.ay = float(amplitude_y)
        self.freq = float(frequency)
        self.omega = 2.0 * np.pi * self.freq
        self.phi = float(phase_offset)

    def get_position(self, t: float) -> Tuple[float, float]:
        theta = self.omega * t + self.phi
        x = self.xc + self.ax * np.sin(theta)
        y = self.yc + self.ay * np.sin(2.0 * theta)
        return (float(x), float(y))

    def get_velocity(self, t: float, dt: float = 1e-4) -> Tuple[float, float]:
        theta = self.omega * t + self.phi
        vx = self.ax * self.omega * np.cos(theta)
        vy = 2.0 * self.ay * self.omega * np.cos(2.0 * theta)
        return (float(vx), float(vy))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motion_type": "figure_eight",
            "center": (self.xc, self.yc),
            "amplitude_x": self.ax,
            "amplitude_y": self.ay,
            "frequency": self.freq,
            "phase_offset": self.phi,
        }


class RandomWalkProfile(MotionProfile):
    """
    Bounded 2D stochastic random walk with specular boundary reflection:
    Advances via Gaussian random increments per time step and reflects back
    into specified [min_x, max_x] x [min_y, max_y] bounds.
    """

    def __init__(
        self,
        initial_pos: Tuple[float, float] = (0.0, 0.0),
        step_variance: float = 0.0003,
        bounds: Tuple[float, float, float, float] = (-0.025, 0.025, -0.020, 0.020),
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.x0 = float(initial_pos[0])
        self.y0 = float(initial_pos[1])
        self.step_std = float(np.sqrt(max(1e-12, step_variance)))
        self.bounds = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Discrete simulation history for deterministic query
        self.curr_x = self.x0
        self.curr_y = self.y0
        self.curr_vx = 0.0
        self.curr_vy = 0.0
        self.history: Dict[float, Tuple[float, float, float, float]] = {
            0.0: (self.x0, self.y0, 0.0, 0.0)
        }
        self.sorted_times: list[float] = [0.0]

    def _reflect(self, val: float, v_min: float, v_max: float) -> Tuple[float, float]:
        """Reflect a coordinate back into bounds [v_min, v_max], returning (clamped_val, sign_flip)."""
        sign = 1.0
        while val < v_min or val > v_max:
            if val > v_max:
                val = 2.0 * v_max - val
                sign *= -1.0
            elif val < v_min:
                val = 2.0 * v_min - val
                sign *= -1.0
        return float(np.clip(val, v_min, v_max)), sign

    def get_position(self, t: float) -> Tuple[float, float]:
        t = max(0.0, float(t))
        if t in self.history:
            return (self.history[t][0], self.history[t][1])

        # If t is beyond last generated time, simulate up to t in fixed increments
        last_t = self.sorted_times[-1]
        if t > last_t:
            dt_step = 0.010  # 10ms sub-stepping for smooth random walk
            sim_t = last_t
            x, y, vx, vy = self.history[last_t]
            min_x, max_x, min_y, max_y = self.bounds

            while sim_t < t:
                step_dt = min(dt_step, t - sim_t)
                sim_t += step_dt
                dx = self._rng.normal(0, self.step_std * np.sqrt(step_dt))
                dy = self._rng.normal(0, self.step_std * np.sqrt(step_dt))

                cand_x = x + dx
                cand_y = y + dy
                x, flip_x = self._reflect(cand_x, min_x, max_x)
                y, flip_y = self._reflect(cand_y, min_y, max_y)
                vx = (dx * flip_x) / step_dt
                vy = (dy * flip_y) / step_dt

                self.history[sim_t] = (x, y, vx, vy)
                self.sorted_times.append(sim_t)

            self.curr_x = x
            self.curr_y = y
            self.curr_vx = vx
            self.curr_vy = vy
            return (x, y)

        # Interpolate between known history keys
        idx = int(np.searchsorted(self.sorted_times, t))
        if idx == 0:
            return (self.history[0.0][0], self.history[0.0][1])
        t_prev = self.sorted_times[idx - 1]
        t_next = self.sorted_times[idx]
        alpha = (t - t_prev) / (t_next - t_prev) if t_next > t_prev else 0.0
        x1, y1, _, _ = self.history[t_prev]
        x2, y2, _, _ = self.history[t_next]
        return (x1 + alpha * (x2 - x1), y1 + alpha * (y2 - y1))

    def get_velocity(self, t: float, dt: float = 1e-4) -> Tuple[float, float]:
        self.get_position(t)
        if t in self.history:
            return (self.history[t][2], self.history[t][3])
        return super().get_velocity(t, dt=dt)

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)
        self.curr_x = self.x0
        self.curr_y = self.y0
        self.curr_vx = 0.0
        self.curr_vy = 0.0
        self.history = {0.0: (self.x0, self.y0, 0.0, 0.0)}
        self.sorted_times = [0.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motion_type": "random",
            "initial_pos": (self.x0, self.y0),
            "step_variance": self.step_std ** 2,
            "bounds": self.bounds,
            "seed": self.seed,
        }


class SpiralProfile(MotionProfile):
    """
    Spiral trajectory with radially expanding or contracting radius:
        R(t) = R0 + vr * t
        x(t) = xc + R(t) * cos(omega * t + phi)
        y(t) = yc + R(t) * sin(omega * t + phi)
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0.0, 0.0),
        initial_radius: float = 0.003,
        radial_velocity: float = 0.001,
        angular_velocity: float = 0.8,
        phase_offset: float = 0.0,
        min_radius: float = 0.0,
        max_radius: Optional[float] = None,
    ):
        super().__init__()
        self.xc = float(center[0])
        self.yc = float(center[1])
        self.r0 = float(initial_radius)
        self.vr = float(radial_velocity)
        self.omega = float(angular_velocity)
        self.phi = float(phase_offset)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius) if max_radius is not None else None

    def _get_radius(self, t: float) -> float:
        r = self.r0 + self.vr * t
        if self.max_radius is not None and self.max_radius > self.min_radius:
            span = self.max_radius - self.min_radius
            # Triangle wave oscillation between min_radius and max_radius
            folded = (r - self.min_radius) % (2.0 * span)
            if folded > span:
                folded = 2.0 * span - folded
            return self.min_radius + folded
        return max(self.min_radius, r)

    def get_position(self, t: float) -> Tuple[float, float]:
        r = self._get_radius(t)
        theta = self.omega * t + self.phi
        x = self.xc + r * np.cos(theta)
        y = self.yc + r * np.sin(theta)
        return (float(x), float(y))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motion_type": "spiral",
            "center": (self.xc, self.yc),
            "initial_radius": self.r0,
            "radial_velocity": self.vr,
            "angular_velocity": self.omega,
            "phase_offset": self.phi,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
        }


class SinusoidalProfile(MotionProfile):
    """
    Sinusoidal / wavy trajectory:
        Linear motion in primary direction with harmonic oscillation in orthogonal axis.
        If axis == 'y': x(t) = x0 + vx*t,  y(t) = y0 + vy*t + A * sin(2*pi*f*t + phi)
        If axis == 'x': x(t) = x0 + vx*t + A * sin(2*pi*f*t + phi),  y(t) = y0 + vy*t
    """

    def __init__(
        self,
        initial_pos: Tuple[float, float] = (0.0, 0.0),
        velocity: Tuple[float, float] = (0.002, 0.0),
        amplitude: float = 0.004,
        frequency: float = 0.5,
        phase_offset: float = 0.0,
        axis: str = "y",
    ):
        super().__init__()
        self.x0 = float(initial_pos[0])
        self.y0 = float(initial_pos[1])
        self.vx = float(velocity[0])
        self.vy = float(velocity[1])
        self.amp = float(amplitude)
        self.freq = float(frequency)
        self.omega = 2.0 * np.pi * self.freq
        self.phi = float(phase_offset)
        self.axis = str(axis).lower()

    def get_position(self, t: float) -> Tuple[float, float]:
        wave = self.amp * np.sin(self.omega * t + self.phi)
        if self.axis == "x":
            x = self.x0 + self.vx * t + wave
            y = self.y0 + self.vy * t
        else:
            x = self.x0 + self.vx * t
            y = self.y0 + self.vy * t + wave
        return (float(x), float(y))

    def get_velocity(self, t: float, dt: float = 1e-4) -> Tuple[float, float]:
        wave_v = self.amp * self.omega * np.cos(self.omega * t + self.phi)
        if self.axis == "x":
            return (float(self.vx + wave_v), float(self.vy))
        else:
            return (float(self.vx), float(self.vy + wave_v))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "motion_type": "sinusoidal",
            "initial_pos": (self.x0, self.y0),
            "velocity": (self.vx, self.vy),
            "amplitude": self.amp,
            "frequency": self.freq,
            "phase_offset": self.phi,
            "axis": self.axis,
        }


# Registry of supported motion types and alias normalizations
MOTION_PROFILE_REGISTRY = {
    "straight_line": StraightLineProfile,
    "linear": StraightLineProfile,
    "circular": CircularProfile,
    "circle": CircularProfile,
    "figure_eight": FigureEightProfile,
    "figure_8": FigureEightProfile,
    "figure8": FigureEightProfile,
    "lissajous": FigureEightProfile,
    "random": RandomWalkProfile,
    "random_walk": RandomWalkProfile,
    "spiral": SpiralProfile,
    "sinusoidal": SinusoidalProfile,
    "sine": SinusoidalProfile,
    "wave": SinusoidalProfile,
}


def normalize_motion_type(motion_type: Optional[str]) -> str:
    """Normalize user or config motion string to canonical identifier."""
    if not motion_type:
        return "straight_line"
    norm = str(motion_type).strip().lower().replace("-", "_").replace(" ", "_")
    alias_map = {
        "linear": "straight_line",
        "circle": "circular",
        "figure_8": "figure_eight",
        "figure8": "figure_eight",
        "lissajous": "figure_eight",
        "random_walk": "random",
        "sine": "sinusoidal",
        "wave": "sinusoidal",
    }
    return alias_map.get(norm, norm)


def create_motion_profile(
    motion_type: str = "straight_line",
    initial_pos: Optional[Tuple[float, float]] = None,
    velocity: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> MotionProfile:
    """
    Factory function to instantiate a MotionProfile by type name with parameter injection.

    Args:
        motion_type: Name of motion profile ('straight_line', 'circular', 'figure_eight', etc.)
        initial_pos: Optional default position (x0, y0).
        velocity: Optional default velocity (vx, vy).
        **kwargs: Motion-specific parameters.

    Returns:
        Configured MotionProfile instance.
    """
    canonical_type = normalize_motion_type(motion_type)
    if canonical_type not in MOTION_PROFILE_REGISTRY:
        valid_options = sorted(list(set(
            ["straight_line", "circular", "figure_eight", "random", "spiral", "sinusoidal"]
        )))
        raise ValueError(
            f"Invalid motion_type: '{motion_type}'. Supported motion patterns: {valid_options}"
        )

    cls = MOTION_PROFILE_REGISTRY[canonical_type]

    # Combine kwargs with initial_pos / velocity if accepted by constructor
    init_kwargs = dict(kwargs)
    if canonical_type in ["straight_line", "linear"]:
        if initial_pos is not None and "initial_pos" not in init_kwargs:
            init_kwargs["initial_pos"] = initial_pos
        if velocity is not None and "velocity" not in init_kwargs:
            init_kwargs["velocity"] = velocity
        return cls(**init_kwargs)

    if canonical_type in ["circular", "circle"]:
        if "center" not in init_kwargs and initial_pos is not None:
            init_kwargs["center"] = initial_pos
        return cls(**init_kwargs)

    if canonical_type in ["figure_eight", "figure_8", "figure8", "lissajous"]:
        if "center" not in init_kwargs and initial_pos is not None:
            init_kwargs["center"] = initial_pos
        return cls(**init_kwargs)

    if canonical_type in ["random", "random_walk"]:
        if "initial_pos" not in init_kwargs and initial_pos is not None:
            init_kwargs["initial_pos"] = initial_pos
        return cls(**init_kwargs)

    if canonical_type in ["spiral"]:
        if "center" not in init_kwargs and initial_pos is not None:
            init_kwargs["center"] = initial_pos
        return cls(**init_kwargs)

    if canonical_type in ["sinusoidal", "sine", "wave"]:
        if "initial_pos" not in init_kwargs and initial_pos is not None:
            init_kwargs["initial_pos"] = initial_pos
        if velocity is not None and "velocity" not in init_kwargs:
            init_kwargs["velocity"] = velocity
        return cls(**init_kwargs)

    return cls(**init_kwargs)
