"""
Severity Score Engine for FSOC PAT Adaptive State Estimation.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Computes a real-time tracking severity metric S in [0.0, 1.0] from:
1. Optical detection confidence (signal quality / contrast / scintillation fades)
2. Consecutive miss count (occlusions / track loss)
3. Atmospheric turbulence Cn^2 level (if available from environmental sensors)
"""

from typing import Optional, Dict, Any
import numpy as np


class SeverityCalculator:
    """
    Calculates operational tracking severity score to dynamically scale estimator
    noise covariances and govern Kalman <-> Particle Filter transitions.
    """

    def __init__(
        self,
        w_confidence: float = 0.50,
        w_miss: float = 0.30,
        w_turb: float = 0.20,
        miss_scale: float = 0.25, # 4 consecutive misses reaches 1.0 miss severity
        cn2_min_log: float = -16.0, # 1e-16 (clear sky) -> 0.0
        cn2_max_log: float = -12.0, # 1e-12 (extreme turbulence) -> 1.0
    ):
        self.w_confidence = float(w_confidence)
        self.w_miss = float(w_miss)
        self.w_turb = float(w_turb)
        self.miss_scale = float(miss_scale)
        self.cn2_min_log = float(cn2_min_log)
        self.cn2_max_log = float(cn2_max_log)

    def compute(
        self,
        confidence: float,
        consecutive_misses: int = 0,
        cn2: Optional[float] = None,
    ) -> float:
        """
        Compute scalar severity score S in [0.0, 1.0].
        Higher score indicates severe tracking degradation / non-Gaussian disturbance.
        """
        breakdown = self.compute_breakdown(
            confidence=confidence,
            consecutive_misses=consecutive_misses,
            cn2=cn2,
        )
        return breakdown["severity"]

    def compute_breakdown(
        self,
        confidence: float,
        consecutive_misses: int = 0,
        cn2: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute severity score along with individual component contributions.
        """
        # 1. Detection confidence severity: 1.0 - confidence
        conf_clamped = float(np.clip(confidence, 0.0, 1.0))
        s_det = 1.0 - conf_clamped

        # 2. Miss count severity
        s_miss = float(np.clip(consecutive_misses * self.miss_scale, 0.0, 1.0))

        # 3. Turbulence severity (if available)
        if cn2 is not None and cn2 > 0:
            log_cn2 = np.log10(max(1e-18, cn2))
            s_turb = float(np.clip(
                (log_cn2 - self.cn2_min_log) / (self.cn2_max_log - self.cn2_min_log),
                0.0,
                1.0,
            ))
            # Normalize weights
            total_w = self.w_confidence + self.w_miss + self.w_turb
            severity = (
                self.w_confidence * s_det +
                self.w_miss * s_miss +
                self.w_turb * s_turb
            ) / total_w
        else:
            s_turb = 0.0
            total_w = self.w_confidence + self.w_miss
            severity = (
                self.w_confidence * s_det +
                self.w_miss * s_miss
            ) / total_w

        severity = float(np.clip(severity, 0.0, 1.0))

        return {
            "severity": severity,
            "det_component": s_det,
            "miss_component": s_miss,
            "turb_component": s_turb,
        }
