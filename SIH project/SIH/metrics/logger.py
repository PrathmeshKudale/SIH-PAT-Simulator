"""
Structured Metric and Event Logger for FSOC Coarse PAT Simulator.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Records frame telemetry, estimator transitions (KF <-> PF), tracking errors,
and disturbance statistics. Exports structured event logs to JSON, CSV, and formatted tables.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import json
import csv
import numpy as np


@dataclass
class SwitchEvent:
    """
    Represents a discrete estimator handoff event between Kalman and Particle Filter.

    Attributes:
        frame_id: Frame index at which switch occurred.
        timestamp: Simulation timestamp (seconds).
        from_mode: Previous estimator mode ('KF' or 'PF').
        to_mode: New active estimator mode ('PF' or 'KF').
        severity: Operational tracking severity score at time of switch [0.0, 1.0].
        confidence: Optical detection confidence score at time of switch [0.0, 1.0].
        rationale: Explicit technical trigger condition explaining the transition.
    """
    frame_id: int
    timestamp: float
    from_mode: str
    to_mode: str
    severity: float
    confidence: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackLossEvent:
    """
    Represents a full track loss event with root cause classification.

    Attributes:
        frame_id: Frame index at which full track loss was declared.
        timestamp: Simulation timestamp in seconds.
        cause: Root cause ('OCCLUSION', 'TURBULENCE', 'FAST_MOTION', 'DETECTION_DROPOUT').
        confidence: Optical detection confidence score at loss.
        severity: Operational tracking severity score at loss.
        last_known_pos: Last known target position (pan, tilt) in radians.
        last_known_vel: Last known target velocity (vx, vy) in rad/s.
        predicted_zone: Optional dictionary representation of ReacquisitionZone.
        rationale: Explanatory diagnosis of root cause.
    """
    frame_id: int
    timestamp: float
    cause: str
    confidence: float
    severity: float
    last_known_pos: Tuple[float, float]
    last_known_vel: Tuple[float, float]
    predicted_zone: Optional[Dict[str, Any]] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsLogger:
    """
    Central repository for simulation telemetry, tracking errors, estimator switches,
    and track loss classification events.
    """

    def __init__(self):
        self.switch_events: List[SwitchEvent] = []
        self.track_loss_events: List[TrackLossEvent] = []
        self.frame_records: List[Dict[str, Any]] = []

    def log_track_loss_event(
        self,
        frame_id: int,
        timestamp: float,
        cause: str,
        confidence: float,
        severity: float,
        last_known_pos: Tuple[float, float],
        last_known_vel: Tuple[float, float],
        predicted_zone: Optional[Dict[str, Any]] = None,
        rationale: str = "",
    ) -> TrackLossEvent:
        """Record a classified full track loss event."""
        event = TrackLossEvent(
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            cause=str(cause),
            confidence=float(confidence),
            severity=float(severity),
            last_known_pos=(float(last_known_pos[0]), float(last_known_pos[1])),
            last_known_vel=(float(last_known_vel[0]), float(last_known_vel[1])),
            predicted_zone=predicted_zone,
            rationale=str(rationale),
        )
        self.track_loss_events.append(event)
        return event

    def log_switch_event(
        self,
        frame_id: int,
        timestamp: float,
        from_mode: str,
        to_mode: str,
        severity: float,
        confidence: float,
        rationale: str,
    ) -> SwitchEvent:
        """
        Record a Kalman <-> Particle Filter transition event.
        """
        event = SwitchEvent(
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            from_mode=str(from_mode),
            to_mode=str(to_mode),
            severity=float(severity),
            confidence=float(confidence),
            rationale=str(rationale),
        )
        self.switch_events.append(event)
        return event

    def log_frame(self, record: Dict[str, Any]) -> None:
        """Record per-frame telemetry dictionary."""
        self.frame_records.append(record)

    def get_switch_events(self) -> List[SwitchEvent]:
        """Return full list of recorded switch events."""
        return list(self.switch_events)

    def get_track_loss_events(self) -> List[TrackLossEvent]:
        """Return full list of recorded track loss events."""
        return list(self.track_loss_events)

    def format_switch_table(self) -> str:
        """Generate formatted ASCII table of all estimator switch events."""
        if not self.switch_events:
            return "No estimator switch events recorded."

        lines = [
            "=" * 95,
            f"{'Frame':^7} | {'Time(s)':^8} | {'Transition':^13} | {'Severity':^10} | {'Confidence':^12} | {'Trigger Rationale':^34}",
            "-" * 95,
        ]

        for ev in self.switch_events:
            trans_str = f"{ev.from_mode} -> {ev.to_mode}"
            lines.append(
                f"{ev.frame_id:^7d} | {ev.timestamp:^8.3f} | {trans_str:^13} | {ev.severity:^10.3f} | {ev.confidence:^12.3f} | {ev.rationale:<34}"
            )

        lines.append("=" * 95)
        return "\n".join(lines)

    def format_track_loss_table(self) -> str:
        """Generate formatted ASCII table of all track loss classification events."""
        if not self.track_loss_events:
            return "No track loss events recorded."

        lines = [
            "=" * 115,
            f"{'Frame':^7} | {'Time(s)':^8} | {'Cause':^18} | {'Severity':^10} | {'Last Pos [mrad]':^18} | {'Last Vel [mrad/s]':^19} | {'Rationale':<25}",
            "-" * 115,
        ]

        for ev in self.track_loss_events:
            pos_str = f"[{ev.last_known_pos[0]*1e3:+5.1f}, {ev.last_known_pos[1]*1e3:+5.1f}]"
            vel_str = f"[{ev.last_known_vel[0]*1e3:+5.1f}, {ev.last_known_vel[1]*1e3:+5.1f}]"
            lines.append(
                f"{ev.frame_id:^7d} | {ev.timestamp:^8.3f} | {ev.cause:^18} | {ev.severity:^10.3f} | {pos_str:^18} | {vel_str:^19} | {ev.rationale:<25}"
            )

        lines.append("=" * 115)
        return "\n".join(lines)

    def compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate performance and mode breakdown statistics."""
        total_frames = len(self.frame_records)
        mode_counts = {"KF": 0, "PF": 0, "COAST": 0, "PF_COAST": 0}
        tracking_errors = []
        active_locked_frames = 0
        coasting_frames = 0

        for rec in self.frame_records:
            mode = rec.get("tracker_mode", "UNKNOWN")
            # Differentiate active tracking modes from coasting modes
            if mode == "KF":
                mode_counts["KF"] = mode_counts.get("KF", 0) + 1
                active_locked_frames += 1
            elif mode == "PF":
                mode_counts["PF"] = mode_counts.get("PF", 0) + 1
                active_locked_frames += 1
            elif mode == "COAST":
                mode_counts["COAST"] = mode_counts.get("COAST", 0) + 1
                coasting_frames += 1
            elif mode == "PF_COAST":
                mode_counts["PF_COAST"] = mode_counts.get("PF_COAST", 0) + 1
                coasting_frames += 1
            else:
                if "COAST" in mode:
                    mode_counts["COAST"] = mode_counts.get("COAST", 0) + 1
                    coasting_frames += 1
                elif "KF" in mode:
                    mode_counts["KF"] = mode_counts.get("KF", 0) + 1
                    active_locked_frames += 1
                elif "PF" in mode:
                    mode_counts["PF"] = mode_counts.get("PF", 0) + 1
                    active_locked_frames += 1

            if "radial_error_mrad" in rec:
                tracking_errors.append(rec["radial_error_mrad"])

        total_tracked = max(1, total_frames)
        breakdown = {k: v / total_tracked for k, v in mode_counts.items()}
        lock_retention_rate = float(active_locked_frames / total_tracked)

        summary = {
            "total_frames": total_frames,
            "active_locked_frames": active_locked_frames,
            "coasting_frames": coasting_frames,
            "lock_retention_rate": lock_retention_rate,
            "total_switch_events": len(self.switch_events),
            "switch_events": [e.to_dict() for e in self.switch_events],
            "total_track_loss_events": len(self.track_loss_events),
            "track_loss_events": [e.to_dict() for e in self.track_loss_events],
            "mode_breakdown": breakdown,
            "mean_tracking_error_mrad": float(np.mean(tracking_errors)) if tracking_errors else 0.0,
            "max_tracking_error_mrad": float(np.max(tracking_errors)) if tracking_errors else 0.0,
        }
        return summary

    def export_csv(self, filepath: str) -> None:
        """Export per-frame telemetry records to a CSV file."""
        if not self.frame_records:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                f.write("frame_id,timestamp,tracker_mode,radial_error_mrad\n")
            return

        # Collect union of all keys across frame records preserving predictable ordering
        fieldnames = []
        for r in self.frame_records:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in self.frame_records:
                writer.writerow(r)

    def export_json(self, filepath: str) -> None:
        """Export all telemetry and switch events to a JSON file."""
        data = {
            "summary": self.compute_summary(),
            "switch_events": [e.to_dict() for e in self.switch_events],
            "track_loss_events": [e.to_dict() for e in self.track_loss_events],
            "frame_records": self.frame_records,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
