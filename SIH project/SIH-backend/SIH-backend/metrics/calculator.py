"""
Quantitative Metric Calculator for FSOC Coarse PAT Simulator.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Computes simulation duration, FPS, acquisition time, tracking error statistics
(Mean, Max, RMSE), lock retention rates (strictly separating active lock from coasting),
per-frame processing latencies, and stage-by-stage profiling breakdowns.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from contracts import MetricsRecord
from metrics.logger import MetricsLogger


class MetricsCalculator:
    """
    Computes rigorous cumulative and summary performance metrics for a simulation run.
    """

    def __init__(self, dt: float = 0.033, lock_error_threshold_mrad: float = 2.0):
        """
        Args:
            dt: Simulation timestep in seconds (default 0.033s ~ 30 Hz).
            lock_error_threshold_mrad: Radial error threshold (mrad) within which target is considered locked.
        """
        self.dt = dt
        self.lock_error_threshold_mrad = lock_error_threshold_mrad

    def compute_from_logger(
        self,
        logger: MetricsLogger,
        start_wall_time: float,
        end_wall_time: float,
        scenario_name: str = "",
        is_held_out: bool = False,
        stage_timings: Optional[Dict[str, List[float]]] = None,
        pipeline_errors: int = 0,
    ) -> MetricsRecord:
        """
        Compute a complete MetricsRecord from a MetricsLogger instance and wall-clock times.

        Args:
            logger: MetricsLogger holding frame records and track loss events.
            start_wall_time: perf_counter start timestamp.
            end_wall_time: perf_counter end timestamp.
            scenario_name: Identifier name of scenario.
            is_held_out: Flag indicating whether scenario is a held-out test case.
            stage_timings: Dictionary of stage execution times list in milliseconds.
                           e.g. {'detection_ms': [...], 'tracking_ms': [...], 'control_ms': [...]}
            pipeline_errors: Number of caught pipeline exceptions during run.

        Returns:
            MetricsRecord populated with all verified statistics.
        """
        return self.compute_metrics(
            frame_records=logger.frame_records,
            track_loss_count=len(logger.track_loss_events),
            start_wall_time=start_wall_time,
            end_wall_time=end_wall_time,
            scenario_name=scenario_name,
            is_held_out=is_held_out,
            stage_timings=stage_timings,
            pipeline_errors=pipeline_errors,
        )

    def compute_metrics(
        self,
        frame_records: List[Dict[str, Any]],
        track_loss_count: int = 0,
        start_wall_time: float = 0.0,
        end_wall_time: float = 0.0,
        scenario_name: str = "",
        is_held_out: bool = False,
        stage_timings: Optional[Dict[str, List[float]]] = None,
        pipeline_errors: int = 0,
    ) -> MetricsRecord:
        """
        Compute MetricsRecord from frame records list and profiler timestamps.
        """
        total_frames = len(frame_records)
        wall_time_s = max(1e-9, end_wall_time - start_wall_time)
        fps = float(total_frames / wall_time_s) if total_frames > 0 else 0.0
        per_frame_ms = float((wall_time_s * 1000.0) / total_frames) if total_frames > 0 else 0.0

        simulation_duration = float(total_frames * self.dt)

        mode_counts = {"KF": 0, "PF": 0, "COAST": 0, "PF_COAST": 0}
        active_locked_frames = 0
        coasting_frames = 0
        tracking_errors = []
        acquisition_frame: Optional[int] = None

        for idx, rec in enumerate(frame_records):
            mode = rec.get("tracker_mode", "UNKNOWN")
            is_active_mode = False

            if mode == "KF":
                mode_counts["KF"] = mode_counts.get("KF", 0) + 1
                active_locked_frames += 1
                is_active_mode = True
            elif mode == "PF":
                mode_counts["PF"] = mode_counts.get("PF", 0) + 1
                active_locked_frames += 1
                is_active_mode = True
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
                    is_active_mode = True
                elif "PF" in mode:
                    mode_counts["PF"] = mode_counts.get("PF", 0) + 1
                    active_locked_frames += 1
                    is_active_mode = True

            # Extract tracking error
            err = rec.get("radial_error_mrad")
            if err is None:
                err = rec.get("tracking_error")
            if err is not None:
                tracking_errors.append(float(err))

            # Acquisition detection: first frame where active lock is achieved
            # (active estimator mode and either valid detection or within error threshold)
            if acquisition_frame is None and is_active_mode:
                if rec.get("detected", True) or (err is not None and err <= self.lock_error_threshold_mrad):
                    acquisition_frame = idx

        total_tracked = max(1, total_frames)
        active_tracker_breakdown = {k: float(v / total_tracked) for k, v in mode_counts.items()}
        lock_retention_rate = float(active_locked_frames / total_tracked)

        if tracking_errors:
            err_arr = np.array(tracking_errors, dtype=np.float64)
            avg_tracking_error = float(np.mean(err_arr))
            max_tracking_error = float(np.max(err_arr))
            rmse_tracking_error = float(np.sqrt(np.mean(err_arr**2)))
        else:
            avg_tracking_error = 0.0
            max_tracking_error = 0.0
            rmse_tracking_error = 0.0

        if acquisition_frame is not None:
            acquisition_time = float(acquisition_frame * self.dt)
        else:
            acquisition_time = simulation_duration  # Never acquired

        # Process stage timings
        stage_breakdown = {}
        if stage_timings:
            for stage_name, times in stage_timings.items():
                if times:
                    stage_breakdown[stage_name] = float(np.mean(times))
                else:
                    stage_breakdown[stage_name] = 0.0

        return MetricsRecord(
            simulation_duration=simulation_duration,
            fps=fps,
            acquisition_time=acquisition_time,
            avg_tracking_error=avg_tracking_error,
            max_tracking_error=max_tracking_error,
            rmse_tracking_error=rmse_tracking_error,
            lock_retention_rate=lock_retention_rate,
            per_frame_processing_time_ms=per_frame_ms,
            total_frames=total_frames,
            active_locked_frames=active_locked_frames,
            coasting_frames=coasting_frames,
            track_loss_count=track_loss_count,
            active_tracker_breakdown=active_tracker_breakdown,
            stage_timing_breakdown=stage_breakdown,
            scenario_name=scenario_name,
            is_held_out=is_held_out,
            pipeline_errors=pipeline_errors,
        )
