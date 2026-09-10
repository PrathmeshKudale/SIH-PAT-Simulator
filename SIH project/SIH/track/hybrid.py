"""
Adaptive Hybrid Tracker managing Kalman <-> Particle Filter transitions.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Implements dynamic estimator switching with hysteresis:
- Low severity / nominal tracking: Runs optimal, low-complexity Kalman Filter (KF)
- High severity / scintillation fades / consecutive dropouts: Transitions to Particle Filter (PF)
- Subsiding disturbances: Requires M consecutive calm frames before switching back to KF
- Seamless state handoff preserving continuous state and covariance across transitions
- Logs all transition events with detailed technical triggers to MetricsLogger
"""

from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
import numpy as np
from contracts import TargetState, CameraState
from track.kalman import ConstantVelocityKalmanFilter
from track.particle import ParticleFilter
from track.severity import SeverityCalculator

if TYPE_CHECKING:
    from metrics.logger import MetricsLogger, SwitchEvent


class HybridTracker:
    """
    Unified hybrid estimator coordinating Kalman Filter and Particle Filter execution.
    """

    def __init__(
        self,
        kf: Optional[ConstantVelocityKalmanFilter] = None,
        pf: Optional[ParticleFilter] = None,
        severity_calculator: Optional[SeverityCalculator] = None,
        logger: Optional[MetricsLogger] = None,
        initial_mode: str = "KF",
        severity_switch_to_pf: float = 0.55,
        severity_switch_to_kf: float = 0.30,
        consecutive_miss_to_pf: int = 3,
        consecutive_calm_to_kf: int = 5,
        min_valid_confidence: float = 0.40,
    ):
        self.kf = kf if kf is not None else ConstantVelocityKalmanFilter(min_valid_confidence=min_valid_confidence)
        self.pf = pf if pf is not None else ParticleFilter(min_valid_confidence=min_valid_confidence)
        self.severity_calc = severity_calculator if severity_calculator is not None else SeverityCalculator()
        self.logger = logger
        self.active_mode = initial_mode.upper()
        self.severity_switch_to_pf = float(severity_switch_to_pf)
        self.severity_switch_to_kf = float(severity_switch_to_kf)
        self.consecutive_miss_to_pf = int(consecutive_miss_to_pf)
        self.consecutive_calm_to_kf = int(consecutive_calm_to_kf)
        self.min_valid_confidence = float(min_valid_confidence)

        # Hysteresis counters
        self.consecutive_misses = 0
        self.consecutive_calm_frames = 0
        self.frame_idx = 0
        self.last_severity = 0.0
        self.timestamp = 0.0

    def init_state(
        self,
        pos_x: float,
        pos_y: float,
        vel_x: float = 0.0,
        vel_y: float = 0.0,
        timestamp: float = 0.0,
    ) -> None:
        """Initialize both estimators on initial target detection."""
        self.kf.init_state(pos_x, pos_y, vel_x, vel_y, timestamp=timestamp)
        self.pf.init_state(pos_x, pos_y, vel_x, vel_y, timestamp=timestamp)
        self.timestamp = timestamp
        self.consecutive_misses = 0
        self.consecutive_calm_frames = 0

    def _switch_to_pf(self, rationale: str, confidence: float) -> None:
        """
        Execute handoff from Kalman Filter to Particle Filter.
        Samples particle cloud from KF Gaussian distribution N(x_hat, P).
        """
        from_mode = self.active_mode
        self.active_mode = "PF"

        # Seamless distribution handoff with regularized velocity variance
        if self.kf.is_initialized:
            cov = self.kf.P.copy()
            # Bound velocity variance to prevent particle cloud dispersion (max std = 1.0 mrad/s)
            max_vel_var = (0.001)**2
            cov[2, 2] = min(cov[2, 2], max_vel_var)
            cov[3, 3] = min(cov[3, 3], max_vel_var)
            cov[0, 2] = cov[2, 0] = 0.0
            cov[1, 3] = cov[3, 1] = 0.0
            self.pf.init_from_distribution(self.kf.x, cov, timestamp=self.timestamp)

        self.consecutive_calm_frames = 0

        if self.logger is not None:
            self.logger.log_switch_event(
                frame_id=self.frame_idx,
                timestamp=self.timestamp,
                from_mode=from_mode,
                to_mode="PF",
                severity=self.last_severity,
                confidence=confidence,
                rationale=rationale,
            )

    def _switch_to_kf(self, rationale: str, confidence: float) -> None:
        """
        Execute handoff from Particle Filter to Kalman Filter.
        Initializes KF mean and covariance from particle empirical distribution.
        """
        from_mode = self.active_mode
        self.active_mode = "KF"

        # Seamless distribution handoff
        if self.pf.is_initialized:
            mean_state, cov_state = self.pf.get_mean_and_cov()
            self.kf.x = mean_state.copy()
            self.kf.P = cov_state.copy()
            self.kf.consecutive_coasts = 0
            self.kf.mode = "KF"
            self.kf.is_initialized = True
            self.kf.timestamp = self.timestamp

        self.consecutive_calm_frames = 0

        if self.logger is not None:
            self.logger.log_switch_event(
                frame_id=self.frame_idx,
                timestamp=self.timestamp,
                from_mode=from_mode,
                to_mode="KF",
                severity=self.last_severity,
                confidence=confidence,
                rationale=rationale,
            )

    def step(
        self,
        dt: float,
        measurement: Optional[Tuple[float, float]] = None,
        confidence: float = 1.0,
        cn2: Optional[float] = None,
        frame_id: Optional[int] = None,
    ) -> TargetState:
        """
        Unified hybrid estimation step:
        1. Evaluate detection quality against min_valid_confidence gate.
        2. Compute composite operational severity score S using live detection confidence.
        3. Evaluate hysteresis transition logic and execute handoff if triggered.
        4. Step active estimator with strictly gated measurement and real-time severity adaptation.
        """
        if frame_id is not None:
            self.frame_idx = int(frame_id)
        else:
            self.frame_idx += 1

        self.timestamp += dt

        # 1. Apply explicit confidence gate to incoming measurement
        is_valid_detection = (measurement is not None) and (confidence >= self.min_valid_confidence)
        meas_gated = measurement if is_valid_detection else None

        if not is_valid_detection:
            self.consecutive_misses += 1
        else:
            self.consecutive_misses = 0

        # 2. Compute operational severity score S using live measured confidence
        raw_conf = float(confidence) if measurement is not None else 0.0
        severity = self.severity_calc.compute(
            confidence=raw_conf,
            consecutive_misses=self.consecutive_misses,
            cn2=cn2,
        )
        self.last_severity = severity

        # 3. Hybrid Switch Logic with Hysteresis
        if self.active_mode == "KF":
            # Condition A: Operational severity crosses high threshold
            if severity >= self.severity_switch_to_pf:
                rationale = f"Severity surge ({severity:.3f} >= {self.severity_switch_to_pf:.2f})"
                self._switch_to_pf(rationale=rationale, confidence=raw_conf)
            # Condition B: Track lost for consecutive frames
            elif self.consecutive_misses >= self.consecutive_miss_to_pf:
                rationale = f"Track loss ({self.consecutive_misses} misses >= {self.consecutive_miss_to_pf})"
                self._switch_to_pf(rationale=rationale, confidence=raw_conf)

        elif self.active_mode == "PF":
            # Check if current frame satisfies calm threshold
            is_calm = (severity <= self.severity_switch_to_kf) and is_valid_detection
            if is_calm:
                self.consecutive_calm_frames += 1
                if self.consecutive_calm_frames >= self.consecutive_calm_to_kf:
                    rationale = f"Calm conditions ({severity:.3f} <= {self.severity_switch_to_kf:.2f}) sustained for {self.consecutive_calm_frames} frames"
                    self._switch_to_kf(rationale=rationale, confidence=raw_conf)
            else:
                # Reset calm counter on any non-calm frame
                self.consecutive_calm_frames = 0

        # 4. Step the active estimator with gated measurement
        if self.active_mode == "KF":
            state = self.kf.step(
                dt=dt,
                measurement=meas_gated,
                confidence=confidence,
                severity=severity,
            )
        else:
            state = self.pf.step(
                dt=dt,
                measurement=meas_gated,
                confidence=confidence,
                severity=severity,
            )

        self.current_mode = state.tracker_mode
        return state

    def get_state(self) -> TargetState:
        """Return state from active estimator."""
        if self.active_mode == "KF":
            return self.kf.get_state()
        return self.pf.get_state()

    def to_pixel(self, camera_state: CameraState) -> Optional[Tuple[float, float]]:
        """Project current active estimate to pixel coordinates."""
        if self.active_mode == "KF":
            return self.kf.to_pixel(camera_state)
        return self.pf.to_pixel(camera_state)
