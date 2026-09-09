"""
Constant-Velocity Kalman Filter with Coast Mode and Adaptive Measurement Covariance.
Smart India Hackathon - FSOC PAT Simulator (ISRO / DOS PS 26169)
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from contracts import TargetState, CameraState


class ConstantVelocityKalmanFilter:
    """
    Linear Discrete Kalman Filter tracking target kinematics in 2D space.
    State vector: x = [pos_x, pos_y, vel_x, vel_y]^T
    Supports coast mode (predict-only) during detection dropouts.
    """

    def __init__(
        self,
        q_noise_std: float = 0.05,        # Process noise spectral density (acceleration jitter)
        r_noise_std: float = 0.001,       # Nominal measurement noise std (rad)
        min_valid_confidence: float = 0.40,# Explicit confidence threshold gate for measurement acceptance
        initial_pos: Optional[Tuple[float, float]] = None,
        initial_vel: Tuple[float, float] = (0.0, 0.0),
    ):
        self.q_noise_std = float(q_noise_std)
        self.r_noise_std = float(r_noise_std)
        self.min_valid_confidence = float(min_valid_confidence)

        # State vector: [x, y, vx, vy]^T
        self.x = np.zeros((4, 1), dtype=np.float64)
        if initial_pos is not None:
            self.x[0, 0] = initial_pos[0]
            self.x[1, 0] = initial_pos[1]
        self.x[2, 0] = initial_vel[0]
        self.x[3, 0] = initial_vel[1]

        # State covariance matrix P
        self.P = np.diag([1e-4, 1e-4, 1e-3, 1e-3]).astype(np.float64)

        # Measurement matrix H: maps state to measurement [x, y]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Measurement noise covariance R
        self.R_nominal = np.diag([self.r_noise_std**2, self.r_noise_std**2]).astype(np.float64)
        self.R = self.R_nominal.copy()

        # Tracking state
        self.is_initialized = (initial_pos is not None)
        self.consecutive_coasts = 0
        self.mode = "KF"  # 'KF' or 'COAST'
        self.timestamp = 0.0

    def init_state(self, pos_x: float, pos_y: float, vel_x: float = 0.0, vel_y: float = 0.0, timestamp: float = 0.0) -> None:
        """Initialize filter state on first detection."""
        self.x[0, 0] = pos_x
        self.x[1, 0] = pos_y
        self.x[2, 0] = vel_x
        self.x[3, 0] = vel_y
        self.P = np.diag([1e-4, 1e-4, 1e-3, 1e-3]).astype(np.float64)
        self.is_initialized = True
        self.consecutive_coasts = 0
        self.mode = "KF"
        self.timestamp = timestamp

    def predict(self, dt: float, severity: Optional[float] = None) -> TargetState:
        """
        State propagation step:
            x_pred = F * x
            P_pred = F * P * F^T + Q(severity)
        Higher severity increases process noise Q to allow faster adaptation to apparent jumps.
        """
        if dt <= 0:
            dt = 0.033

        self.timestamp += dt

        # State transition matrix F
        F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Scale process noise Q with severity if provided
        q_scale = 1.0
        if severity is not None:
            s_clamped = float(np.clip(severity, 0.0, 1.0))
            q_scale = 1.0 + 8.0 * s_clamped

        # Continuous white noise acceleration process covariance Q
        q = (self.q_noise_std**2) * q_scale
        dt2 = (dt**2) / 2.0
        dt3 = (dt**3) / 3.0
        self.Q = q * np.array([
            [dt3, 0.0, dt2, 0.0],
            [0.0, dt3, 0.0, dt2],
            [dt2, 0.0,  dt, 0.0],
            [0.0, dt2, 0.0,  dt],
        ], dtype=np.float64)

        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        return self.get_state()

    def update(
        self,
        meas_x: float,
        meas_y: float,
        confidence: float = 1.0,
        severity: Optional[float] = None,
    ) -> TargetState:
        """
        Measurement correction step:
            y = z - H * x
            S = H * P * H^T + R(severity)
            K = P * H^T * S^(-1)
            x = x + K * y
            P = (I - K * H) * P
        Higher severity increases measurement covariance R, trusting measurements less.
        """
        if not (np.isfinite(meas_x) and np.isfinite(meas_y)):
            return self.get_state()

        if not self.is_initialized:
            self.init_state(meas_x, meas_y, timestamp=self.timestamp)
            return self.get_state()

        z = np.array([[meas_x], [meas_y]], dtype=np.float64)

        # Adaptive measurement noise covariance R
        if severity is not None:
            s_clamped = float(np.clip(severity, 0.0, 1.0))
            r_scale = (1.0 + 25.0 * (s_clamped**2)) / (max(0.10, 1.0 - s_clamped)**2)
            R_adaptive = self.R_nominal * r_scale
        else:
            conf_clamped = max(0.15, min(1.0, confidence))
            R_adaptive = self.R_nominal / (conf_clamped**2)

        self.R = R_adaptive

        # Innovation (residual)
        y = z - (self.H @ self.x)

        # Innovation covariance S
        S = self.H @ self.P @ self.H.T + R_adaptive

        # Near-optimal Kalman Gain K with singular matrix protection
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
            new_x = self.x + K @ y
            if np.all(np.isfinite(new_x)):
                self.x = new_x
                # Joseph form covariance update for numerical stability
                I = np.eye(4, dtype=np.float64)
                IKH = I - K @ self.H
                self.P = IKH @ self.P @ IKH.T + K @ R_adaptive @ K.T
                self.consecutive_coasts = 0
                self.mode = "KF"
                return self.get_state()
            else:
                raise np.linalg.LinAlgError("Non-finite state encountered in update")
        except np.linalg.LinAlgError:
            self.mode = "COAST"
            self.consecutive_coasts += 1
            return TargetState(
                x=float(self.x[0, 0]),
                y=float(self.x[1, 0]),
                vx=float(self.x[2, 0]),
                vy=float(self.x[3, 0]),
                confidence=0.2,
                timestamp=self.timestamp,
                tracker_mode="COAST",
            )

    def step(
        self,
        dt: float,
        measurement: Optional[Tuple[float, float]] = None,
        confidence: float = 1.0,
        severity: Optional[float] = None,
    ) -> TargetState:
        """
        Unified predict-update cycle with explicit confidence threshold gating and real-time noise adaptation.
        If measurement is None, non-finite, or confidence < min_valid_confidence, runs in COAST mode (predict-only).
        """
        # Always run predict (with severity adaptation for process noise Q)
        pred_state = self.predict(dt, severity=severity)

        # Explicit finite measurement and confidence threshold gate
        is_finite_meas = (
            measurement is not None
            and len(measurement) >= 2
            and np.isfinite(measurement[0])
            and np.isfinite(measurement[1])
        )

        if is_finite_meas and confidence >= self.min_valid_confidence:
            # Measurement valid and above confidence gate: correct state with severity-adapted R
            return self.update(float(measurement[0]), float(measurement[1]), confidence=confidence, severity=severity)
        else:
            # No measurement, non-finite, or below confidence gate: enter/continue coast mode
            self.consecutive_coasts += 1
            self.mode = "COAST"
            # State confidence degrades with coasting duration
            degraded_conf = max(0.1, 1.0 - 0.15 * self.consecutive_coasts)
            return TargetState(
                x=float(self.x[0, 0]),
                y=float(self.x[1, 0]),
                vx=float(self.x[2, 0]),
                vy=float(self.x[3, 0]),
                confidence=degraded_conf,
                timestamp=self.timestamp,
                tracker_mode="COAST",
            )

    def get_state(self) -> TargetState:
        """Return current estimated state as TargetState dataclass."""
        conf = 1.0 if self.mode == "KF" else max(0.1, 1.0 - 0.15 * self.consecutive_coasts)
        return TargetState(
            x=float(self.x[0, 0]),
            y=float(self.x[1, 0]),
            vx=float(self.x[2, 0]),
            vy=float(self.x[3, 0]),
            confidence=conf,
            timestamp=self.timestamp,
            tracker_mode=self.mode,
        )

    def to_pixel(self, camera_state: CameraState) -> Optional[Tuple[float, float]]:
        """Project world angular state estimate into focal plane pixel coordinates."""
        delta_pan = self.x[0, 0] - camera_state.pan
        delta_tilt = self.x[1, 0] - camera_state.tilt

        w, h = camera_state.resolution
        cx = w / 2.0
        cy = h / 2.0

        u = cx + delta_pan * (w / camera_state.fov_x)
        v = cy + delta_tilt * (h / camera_state.fov_y)

        if 0 <= u < w and 0 <= v < h:
            return (float(u), float(v))
        return (float(u), float(v))  # Return coordinates even if slightly outside focal plane
