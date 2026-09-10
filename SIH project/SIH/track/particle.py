"""
Sequential Importance Resampling (SIR) Particle Filter for Non-Gaussian Tracking.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Estimates target kinematic state [x, y, vx, vy]^T using a non-parametric particle distribution.
Specifically engineered for:
- Non-Gaussian and heavy-tailed measurement noise
- Deep scintillation fading and multimodal clutter hypotheses
- Graceful dispersion during extended dropouts and occlusions
- Seamless bidirectional state handoff with Kalman Filter
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from contracts import TargetState, CameraState


class ParticleFilter:
    """
    Bootstrap / Sequential Importance Resampling (SIR) Particle Filter.
    Maintains N_p weighted particles exploring kinematic hypotheses in 2D angular space.
    """

    def __init__(
        self,
        num_particles: int = 300,
        process_noise_pos: float = 0.0001,  # rad/s^2 jitter
        process_noise_vel: float = 0.0004,  # rad/s^2 jitter
        meas_noise_std: float = 0.0015,     # rad measurement likelihood scale
        min_valid_confidence: float = 0.40, # Explicit confidence gate
        seed: Optional[int] = 42,
    ):
        self.num_particles = int(num_particles)
        self.process_noise_pos = float(process_noise_pos)
        self.process_noise_vel = float(process_noise_vel)
        self.meas_noise_std = float(meas_noise_std)
        self.min_valid_confidence = float(min_valid_confidence)
        self._rng = np.random.default_rng(seed)

        # Particles array of shape (N_p, 4): columns are [x, y, vx, vy]
        self.particles = np.zeros((self.num_particles, 4), dtype=np.float64)
        # Normalized weights of shape (N_p,)
        self.weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles

        self.is_initialized = False
        self.timestamp = 0.0
        self.consecutive_coasts = 0
        self.mode = "PF"

    def init_state(
        self,
        pos_x: float,
        pos_y: float,
        vel_x: float = 0.0,
        vel_y: float = 0.0,
        pos_std: float = 0.001,
        vel_std: float = 0.005,
        timestamp: float = 0.0,
    ) -> None:
        """Initialize particle cloud around an initial state with specified variance."""
        self.particles[:, 0] = self._rng.normal(pos_x, pos_std, self.num_particles)
        self.particles[:, 1] = self._rng.normal(pos_y, pos_std, self.num_particles)
        self.particles[:, 2] = self._rng.normal(vel_x, vel_std, self.num_particles)
        self.particles[:, 3] = self._rng.normal(vel_y, vel_std, self.num_particles)
        self.weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles
        self.is_initialized = True
        self.timestamp = timestamp
        self.consecutive_coasts = 0
        self.mode = "PF"

    def init_from_distribution(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        timestamp: float = 0.0,
    ) -> None:
        """
        Initialize particle distribution from a Gaussian mean vector (4, 1) and covariance (4, 4).
        Enables seamless zero-jump handoff from Kalman Filter to Particle Filter.
        """
        mu = mean.flatten()
        # Symmetrize and regularize covariance for positive-definiteness
        cov_sym = 0.5 * (cov + cov.T) + 1e-9 * np.eye(4)
        self.particles = self._rng.multivariate_normal(mu, cov_sym, size=self.num_particles)
        self.weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles
        self.is_initialized = True
        self.timestamp = timestamp
        self.consecutive_coasts = 0
        self.mode = "PF"

    def predict(self, dt: float, severity: Optional[float] = None) -> TargetState:
        """
        Propagate particles through constant-velocity kinematic model with additive process noise.
        Severity widens process noise to accommodate high turbulence/scintillation jumps.
        """
        if dt <= 0:
            dt = 0.033

        self.timestamp += dt

        # Severity-adaptive process noise scaling
        q_scale = 1.0
        if severity is not None:
            s_clamped = float(np.clip(severity, 0.0, 1.0))
            q_scale = 1.0 + 2.0 * s_clamped

        dt_pos = self.process_noise_pos * np.sqrt(dt) * q_scale
        dt_vel = self.process_noise_vel * np.sqrt(dt) * q_scale

        # Position updates: x + vx * dt + noise
        self.particles[:, 0] += self.particles[:, 2] * dt + self._rng.normal(0, dt_pos, self.num_particles)
        self.particles[:, 1] += self.particles[:, 3] * dt + self._rng.normal(0, dt_pos, self.num_particles)
        # Velocity updates: vx + noise
        self.particles[:, 2] += self._rng.normal(0, dt_vel, self.num_particles)
        self.particles[:, 3] += self._rng.normal(0, dt_vel, self.num_particles)

        return self.get_state()

    def update(
        self,
        meas_x: float,
        meas_y: float,
        confidence: float = 1.0,
        severity: Optional[float] = None,
    ) -> TargetState:
        """
        Evaluate particle likelihood against measurement with heavy-tail mixture model.
        Updates weights and performs systematic resampling when effective sample size drops.
        """
        if not self.is_initialized:
            self.init_state(meas_x, meas_y, timestamp=self.timestamp)
            return self.get_state()

        # Scale measurement likelihood width with confidence / severity
        if severity is not None:
            s_clamped = float(np.clip(severity, 0.0, 1.0))
            sigma_meas = self.meas_noise_std * (1.0 + 3.0 * s_clamped)
        else:
            conf_clamped = max(0.15, min(1.0, confidence))
            sigma_meas = self.meas_noise_std / np.sqrt(conf_clamped)

        # Compute Euclidean distance from each particle to measurement
        dx = self.particles[:, 0] - meas_x
        dy = self.particles[:, 1] - meas_y
        dist_sq = dx**2 + dy**2

        # Mixture likelihood: 90% Gaussian core + 10% uniform heavy tail for non-Gaussian clutter
        gauss_like = np.exp(-0.5 * dist_sq / (sigma_meas**2)) / (2.0 * np.pi * (sigma_meas**2))
        uniform_tail = 1e-4
        likelihood = 0.90 * gauss_like + 0.10 * uniform_tail

        # Weight update
        self.weights *= likelihood
        total_w = np.sum(self.weights)

        if total_w > 1e-12:
            self.weights /= total_w
        else:
            # Re-initialize weights to uniform if all likelihoods numerically underflow
            self.weights = np.ones(self.num_particles, dtype=np.float64) / self.num_particles

        # Effective sample size: N_eff = 1.0 / sum(w_i^2)
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff < self.num_particles / 2.0:
            self._systematic_resample()

        self.consecutive_coasts = 0
        self.mode = "PF"

        return self.get_state()

    def _systematic_resample(self) -> None:
        """
        Low-variance Systematic Resampling algorithm.
        Avoids particle degeneracy while preserving distribution diversity.
        """
        n = self.num_particles
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1.0  # Avoid roundoff issues

        # Single random offset
        u0 = self._rng.uniform(0.0, 1.0 / n)
        u = u0 + np.arange(n) / n

        # Indices from inverse CDF
        indices = np.searchsorted(cdf, u)
        indices = np.clip(indices, 0, n - 1)

        self.particles = self.particles[indices].copy()
        self.weights = np.ones(n, dtype=np.float64) / n

        # Apply slight roughening jitter (Gaussian dispersion) to prevent identical duplicate collapse
        pos_rough = self.meas_noise_std * 0.05
        vel_rough = self.process_noise_vel * 0.05
        self.particles[:, 0] += self._rng.normal(0, pos_rough, n)
        self.particles[:, 1] += self._rng.normal(0, pos_rough, n)
        self.particles[:, 2] += self._rng.normal(0, vel_rough, n)
        self.particles[:, 3] += self._rng.normal(0, vel_rough, n)

    def step(
        self,
        dt: float,
        measurement: Optional[Tuple[float, float]] = None,
        confidence: float = 1.0,
        severity: Optional[float] = None,
    ) -> TargetState:
        """
        Unified predict-update cycle for Particle Filter.
        Runs predict, then measurement update if valid and above confidence gate.
        If measurement is None or confidence < min_valid_confidence, enters COAST mode.
        """
        self.predict(dt, severity=severity)

        if measurement is not None and confidence >= self.min_valid_confidence:
            return self.update(measurement[0], measurement[1], confidence=confidence, severity=severity)
        else:
            self.consecutive_coasts += 1
            self.mode = "COAST"
            degraded_conf = max(0.1, 1.0 - 0.15 * self.consecutive_coasts)
            mean_state, _ = self.get_mean_and_cov()
            return TargetState(
                x=float(mean_state[0, 0]),
                y=float(mean_state[1, 0]),
                vx=float(mean_state[2, 0]),
                vy=float(mean_state[3, 0]),
                confidence=degraded_conf,
                timestamp=self.timestamp,
                tracker_mode="COAST",
            )

    def get_mean_and_cov(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute empirical weighted mean (4, 1) and covariance (4, 4) of particle distribution.
        Used for state estimation and seamless handoff back to Kalman Filter.
        """
        mean = np.sum(self.particles * self.weights[:, np.newaxis], axis=0).reshape(4, 1)
        diff = self.particles - mean.flatten()
        cov = (diff.T * self.weights) @ diff
        # Regularize
        cov += 1e-9 * np.eye(4)
        return mean, cov

    def get_state(self) -> TargetState:
        """Return current estimated state as TargetState dataclass."""
        mean_state, _ = self.get_mean_and_cov()
        conf = 0.95 if self.mode == "PF" else max(0.1, 1.0 - 0.15 * self.consecutive_coasts)
        return TargetState(
            x=float(mean_state[0, 0]),
            y=float(mean_state[1, 0]),
            vx=float(mean_state[2, 0]),
            vy=float(mean_state[3, 0]),
            confidence=conf,
            timestamp=self.timestamp,
            tracker_mode=self.mode,
        )

    def to_pixel(self, camera_state: CameraState) -> Optional[Tuple[float, float]]:
        """Project angular state estimate into focal plane pixel coordinates."""
        mean_state, _ = self.get_mean_and_cov()
        delta_pan = mean_state[0, 0] - camera_state.pan
        delta_tilt = mean_state[1, 0] - camera_state.tilt

        w, h = camera_state.resolution
        cx = w / 2.0
        cy = h / 2.0

        u = cx + delta_pan * (w / camera_state.fov_x)
        v = cy + delta_tilt * (h / camera_state.fov_y)

        return (float(u), float(v))
