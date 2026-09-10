"""
Quantitative Metric Calculator for FSOC Coarse PAT Simulator.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Computes simulation duration, FPS, acquisition time, tracking error statistics
(Mean, Max, RMSE in both mrad and pixel-space), lock retention rates,
PS-exact target loss percentage, centroiding error logs, per-frame processing
latencies, and stage-by-stage profiling breakdowns.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from contracts import MetricsRecord
from metrics.logger import MetricsLogger


def angular_to_pixel_error(
    angular_error: float,
    fov_deg: Optional[float] = None,
    fov_rad: Optional[float] = None,
    resolution_px: int = 640,
    angular_is_mrad: bool = True,
) -> float:
    """
    Convert angular tracking error to focal plane pixel tracking error.

    Formula (per SIH Problem Statement specification):
        pixel_error = angular_error_rad * (resolution_px / fov_rad)

    Args:
        angular_error: Angular error magnitude (mrad if angular_is_mrad=True, else radians).
        fov_deg: Camera horizontal field of view in degrees (e.g. 2.29183 deg).
        fov_rad: Camera horizontal field of view in radians (e.g. 0.040 rad).
        resolution_px: Sensor resolution along the corresponding axis in pixels (default 640).
        angular_is_mrad: True if angular_error is in milliradians, False if in radians.

    Returns:
        Tracking error in pixels.
    """
    if fov_rad is None:
        if fov_deg is not None:
            fov_rad = float(np.deg2rad(fov_deg))
        else:
            fov_rad = 0.040  # Default 40 mrad

    if fov_rad <= 0:
        return 0.0

    angular_error_rad = float(angular_error) * 1e-3 if angular_is_mrad else float(angular_error)
    return float(angular_error_rad * (resolution_px / fov_rad))


class MetricsCalculator:
    """
    Computes rigorous cumulative and summary performance metrics for a simulation run
    in both angular (mrad) and pixel-space (px) coordinates.
    """

    def __init__(
        self,
        dt: float = 0.033,
        lock_error_threshold_mrad: float = 2.0,
        fov_deg: Optional[float] = None,
        fov_rad: float = 0.040,
        resolution_px: int = 640,
    ):
        """
        Args:
            dt: Simulation timestep in seconds (default 0.033s ~ 30 Hz).
            lock_error_threshold_mrad: Radial error threshold (mrad) within which target is considered locked.
            fov_deg: Optional camera horizontal FOV in degrees.
            fov_rad: Camera horizontal FOV in radians (default 0.040 rad).
            resolution_px: Camera focal plane resolution (default 640 pixels).
        """
        self.dt = dt
        self.lock_error_threshold_mrad = lock_error_threshold_mrad
        self.fov_deg = fov_deg
        self.fov_rad = float(np.deg2rad(fov_deg)) if fov_deg is not None else float(fov_rad)
        self.resolution_px = int(resolution_px)

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
        Calculates both mrad and pixel-space tracking metrics, exact PS target loss percentage,
        and centroiding error logs.
        """
        total_frames = len(frame_records)
        wall_time_s = max(1e-9, end_wall_time - start_wall_time)
        fps = float(total_frames / wall_time_s) if total_frames > 0 else 0.0
        per_frame_ms = float((wall_time_s * 1000.0) / total_frames) if total_frames > 0 else 0.0

        simulation_duration = float(total_frames * self.dt)

        mode_counts = {"KF": 0, "PF": 0, "COAST": 0, "PF_COAST": 0}
        active_locked_frames = 0
        coasting_frames = 0
        tracking_errors_mrad = []
        tracking_errors_px = []
        centroiding_errors_px = []
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

            # Extract tracking error (mrad and pixel space)
            err = rec.get("radial_error_mrad")
            if err is None:
                err = rec.get("tracking_error")
            if err is not None:
                err_val = float(err)
                tracking_errors_mrad.append(err_val)

                # Compute pixel-space tracking error
                px_err = rec.get("tracking_error_px")
                if px_err is None:
                    px_err = angular_to_pixel_error(
                        angular_error=err_val,
                        fov_rad=self.fov_rad,
                        resolution_px=self.resolution_px,
                        angular_is_mrad=True,
                    )
                tracking_errors_px.append(float(px_err))

            # Centroiding error per frame (for Benchmark Performance-1 log requirement)
            cent_err = rec.get("centroiding_error_px")
            if cent_err is not None:
                centroiding_errors_px.append(float(cent_err))

            # Acquisition detection: first frame where active lock is achieved
            if acquisition_frame is None and is_active_mode:
                if err is not None:
                    if err <= self.lock_error_threshold_mrad:
                        acquisition_frame = idx
                elif rec.get("locked", False) or rec.get("is_locked", False):
                    acquisition_frame = idx

        total_tracked = max(1, total_frames)
        active_tracker_breakdown = {k: float(v / total_tracked) for k, v in mode_counts.items()}

        # Exact PS Item 18 Target Loss phrasing: (frames with no valid lock / total frames) * 100
        unlocked_frames = total_frames - active_locked_frames
        target_loss_percent = float((unlocked_frames / total_tracked) * 100.0) if total_frames > 0 else 0.0

        # Calculate Re-acquisition time (s) across loss intervals
        reacq_durations = []
        loss_epoch_start: Optional[float] = None
        for idx, rec in enumerate(frame_records):
            m = rec.get("tracker_mode", "UNKNOWN")
            is_lk = (m in ["KF", "PF"] or "KF" in m or "PF" in m) and "COAST" not in m
            t_now = float(rec.get("timestamp", idx * self.dt))
            if not is_lk:
                if loss_epoch_start is None and idx > (acquisition_frame or 0):
                    loss_epoch_start = t_now
            else:
                if loss_epoch_start is not None:
                    reacq_durations.append(max(0.0, t_now - loss_epoch_start))
                    loss_epoch_start = None
        reacquisition_time_s = float(np.mean(reacq_durations)) if reacq_durations else 0.0

        has_ground_truth = (len(tracking_errors_mrad) > 0)

        if has_ground_truth:
            err_arr_mrad = np.array(tracking_errors_mrad, dtype=np.float64)
            avg_tracking_error = float(np.mean(err_arr_mrad))
            max_tracking_error = float(np.max(err_arr_mrad))
            rmse_tracking_error = float(np.sqrt(np.mean(err_arr_mrad**2)))

            err_arr_px = np.array(tracking_errors_px, dtype=np.float64)
            avg_tracking_error_px = float(np.mean(err_arr_px))
            max_tracking_error_px = float(np.max(err_arr_px))
            rmse_px = float(np.sqrt(np.mean(err_arr_px**2)))
            tracking_error_px = avg_tracking_error_px
            lock_retention_rate = float(active_locked_frames / total_tracked)

            # Compute steady-state tracking metrics (post-acquisition locked portion)
            start_steady = acquisition_frame if (acquisition_frame is not None and acquisition_frame < len(err_arr_px)) else 0
            steady_err_arr_px = err_arr_px[start_steady:] if len(err_arr_px) > start_steady else err_arr_px
            steady_tracking_error_px = float(np.mean(steady_err_arr_px)) if len(steady_err_arr_px) > 0 else avg_tracking_error_px
            steady_rmse_px = float(np.sqrt(np.mean(steady_err_arr_px**2))) if len(steady_err_arr_px) > 0 else rmse_px

            steady_err_arr_mrad = err_arr_mrad[start_steady:] if len(err_arr_mrad) > start_steady else err_arr_mrad
            steady_avg_tracking_error_mrad = float(np.mean(steady_err_arr_mrad)) if len(steady_err_arr_mrad) > 0 else avg_tracking_error
            steady_rmse_tracking_error_mrad = float(np.sqrt(np.mean(steady_err_arr_mrad**2))) if len(steady_err_arr_mrad) > 0 else rmse_tracking_error
        else:
            avg_tracking_error = "N/A - no ground truth available"
            max_tracking_error = "N/A - no ground truth available"
            rmse_tracking_error = "N/A - no ground truth available"
            avg_tracking_error_px = "N/A - no ground truth available"
            max_tracking_error_px = "N/A - no ground truth available"
            rmse_px = "N/A - no ground truth available"
            tracking_error_px = "N/A - no ground truth available"
            steady_tracking_error_px = "N/A - no ground truth available"
            steady_rmse_px = "N/A - no ground truth available"
            steady_avg_tracking_error_mrad = "N/A - no ground truth available"
            steady_rmse_tracking_error_mrad = "N/A - no ground truth available"

            # Without ground truth, compute lock retention as fraction of frames with a valid detection
            valid_det_count = sum(1 for r in frame_records if r.get("detected", False))
            lock_retention_rate = float(valid_det_count / total_tracked)
            target_loss_percent = float((1.0 - lock_retention_rate) * 100.0)

        if centroiding_errors_px:
            c_arr = np.array(centroiding_errors_px, dtype=np.float64)
            avg_centroiding_error_px = float(np.mean(c_arr))
            max_centroiding_error_px = float(np.max(c_arr))
            centroiding_error_px = avg_centroiding_error_px
            centroiding_error_log_px = [float(c) for c in centroiding_errors_px]
        else:
            avg_centroiding_error_px = "N/A - no centroiding log"
            max_centroiding_error_px = "N/A - no centroiding log"
            centroiding_error_px = "N/A - no centroiding log"
            centroiding_error_log_px = []

        if acquisition_frame is not None:
            acquisition_time = float(acquisition_frame * self.dt)
        else:
            acquisition_time = simulation_duration  # Never acquired
        acquisition_time_s = acquisition_time

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
            has_ground_truth=has_ground_truth,
            tracking_error_px=tracking_error_px,
            avg_tracking_error_px=avg_tracking_error_px,
            max_tracking_error_px=max_tracking_error_px,
            rmse_px=rmse_px,
            centroiding_error_px=centroiding_error_px,
            avg_centroiding_error_px=avg_centroiding_error_px,
            max_centroiding_error_px=max_centroiding_error_px,
            target_loss_percent=target_loss_percent,
            acquisition_time_s=acquisition_time_s,
            reacquisition_time_s=reacquisition_time_s,
            centroiding_error_log_px=centroiding_error_log_px,
            steady_tracking_error_px=steady_tracking_error_px,
            steady_rmse_px=steady_rmse_px,
            steady_avg_tracking_error_mrad=steady_avg_tracking_error_mrad,
            steady_rmse_tracking_error_mrad=steady_rmse_tracking_error_mrad,
        )
