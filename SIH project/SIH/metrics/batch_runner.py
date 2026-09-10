"""
Batch Scenario Runner and Benchmarking Suite for FSOC Coarse PAT Simulator.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Runs the complete closed-loop PAT pipeline across a scenario disturbance matrix
(atmospheric turbulence, platform vibration, sensor readout noise, dynamic occlusions,
and link distances). Computes end-to-end metrics, per-stage profiling breakdowns,
and exports combined JSON and CSV results.
"""

from __future__ import annotations
import os
import json
import csv
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from contracts import MetricsRecord, TargetState, normalize_scenario_targets, get_resource_path
from sim.target import Target
from sim.camera import Camera
from sim.environment import Environment
from detect.detector import AdaptiveOpticalDetector
from track.kalman import ConstantVelocityKalmanFilter
from control.pid import PIDController
from control.slew import SlewRateLimiter
from control.latency import ControlDelayQueue
from control.reacquisition import HierarchicalReacquisitionController
from track.classifier import TrackLossClassifier
from track.reacquisition_zone import ReacquisitionZonePredictor
from disturb.turbulence import KolmogorovTurbulence
from disturb.vibration import PlatformVibration
from disturb.noise import SensorNoise
from disturb.occluder import DynamicOccluder
from metrics.logger import MetricsLogger
from metrics.calculator import MetricsCalculator


class BatchScenarioRunner:
    """
    Automated batch executor running the complete PAT simulation across
    a configurable matrix of environmental and kinematic disturbance scenarios.
    """

    def __init__(self, config_path: Optional[str] = None, use_hybrid_tracker: bool = False):
        """
        Initialize the runner with a configuration file path or auto-detect default.
        
        Args:
            config_path: Path to scenarios JSON configuration.
            use_hybrid_tracker: If True, uses HybridTracker (adaptive KF/PF switching);
                                If False (default), uses standalone ConstantVelocityKalmanFilter.
        """
        if config_path is None:
            # Auto-detect default config path
            if os.path.exists("config/scenarios.json"):
                self.config_path = "config/scenarios.json"
            elif os.path.exists("configs/scenarios.json"):
                self.config_path = "configs/scenarios.json"
            else:
                self.config_path = "config/scenarios.json"
        else:
            self.config_path = config_path

        self.use_hybrid_tracker = use_hybrid_tracker
        self.calculator = MetricsCalculator()
        self.results: List[MetricsRecord] = []
        self.last_logger: Optional[MetricsLogger] = None

    def load_scenarios(self) -> List[Dict[str, Any]]:
        """Load scenario specifications from JSON config file."""
        resolved = get_resource_path(self.config_path)
        actual_path = resolved if os.path.exists(resolved) else self.config_path
        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"Scenario configuration file not found at: {self.config_path}")
        with open(actual_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_scenario(self, scenario_cfg: Dict[str, Any], use_hybrid_tracker: Optional[bool] = None) -> MetricsRecord:
        """
        Execute a single scenario through the full closed-loop PAT pipeline with per-stage timing.

        Args:
            scenario_cfg: Scenario configuration dictionary.
            use_hybrid_tracker: Optional override for tracker mode (Hybrid vs standalone KF).

        Returns:
            Populated MetricsRecord with quantitative error, retention, and latency metrics.
        """
        enable_hybrid = self.use_hybrid_tracker if use_hybrid_tracker is None else use_hybrid_tracker
        name = scenario_cfg.get("scenario_name", "UNKNOWN_SCENARIO")
        is_held_out = scenario_cfg.get("is_held_out", False)
        num_frames = int(scenario_cfg.get("num_frames", 50))
        dt = float(scenario_cfg.get("dt", 0.033))

        targets_cfg, primary_target_id = normalize_scenario_targets(scenario_cfg)
        primary_cfg = next((t for t in targets_cfg if t.get("target_id") == primary_target_id), targets_cfg[0])
        init_pos = tuple(primary_cfg.get("initial_pos", [0.010, -0.005]))
        vel = tuple(primary_cfg.get("velocity", [0.003, -0.002]))
        base_intensity = float(primary_cfg.get("base_intensity", 220.0))
        dist_km = float(primary_cfg.get("distance_km", primary_cfg.get("range_km", 5.0)))

        dist_cfg = scenario_cfg.get("disturbances", {})
        cn2 = float(dist_cfg.get("cn2", 1.0e-14))
        vib_amp = float(dist_cfg.get("vibration_amplitude", 0.0003))
        vib_freq = float(dist_cfg.get("vibration_frequency", 10.0))
        noise_std = float(dist_cfg.get("noise_level", 4.0))
        seed = int(scenario_cfg.get("seed", 42))

        ctrl_cfg = scenario_cfg.get("control", {})
        kp = float(ctrl_cfg.get("kp", 0.35))
        ki = float(ctrl_cfg.get("ki", 0.0))
        kd = float(ctrl_cfg.get("kd", 0.15))
        k_ff = float(ctrl_cfg.get("k_ff", 1.0))
        latency_frames = int(ctrl_cfg.get("latency_frames", 2))
        max_vel = float(ctrl_cfg.get("max_velocity", 0.5))
        max_acc = float(ctrl_cfg.get("max_acceleration", 1.0))

        reacq_cfg = scenario_cfg.get("reacquisition", {})
        enable_pred = bool(reacq_cfg.get("enable_predictive_search", True))
        tier1_budget = int(reacq_cfg.get("tier1_budget_frames", 25))

        # Instantiate simulation objects
        targets = [Target.from_config(t) for t in targets_cfg]
        target = next((t for t in targets if t.target_id == primary_target_id), targets[0])
        camera = Camera(
            pan=0.0,
            tilt=0.0,
            fov_x=0.040,
            fov_y=0.030,
            resolution=(640, 480),
            enable_auto_exposure=True,
        )

        turb = KolmogorovTurbulence(cn2=cn2, seed=seed) if cn2 > 0 else None
        vib = PlatformVibration(amplitude_rad=vib_amp, frequency_hz=vib_freq, random_walk_std=vib_amp * 0.1, seed=seed) if vib_amp > 0 else None
        noise_types = dist_cfg.get("noise_types", None)
        poisson_scale = float(dist_cfg.get("poisson_scale", 1.0))
        sp_prob = float(dist_cfg.get("salt_pepper_prob", 0.0005 if (noise_types and "salt_pepper" in str(noise_types)) else 0.0))
        has_noise = (noise_types is not None and len(noise_types) > 0) or noise_std > 0
        noise = SensorNoise(
            gaussian_std=noise_std,
            salt_pepper_prob=sp_prob,
            poisson_scale=poisson_scale,
            noise_types=noise_types,
            seed=seed,
        ) if has_noise else None

        # Check frame source mode
        frame_source = scenario_cfg.get("frame_source", "synthetic")
        video_path = scenario_cfg.get("video_path", None)
        video_source = None
        if frame_source == "video_file":
            from sim.video_source import VideoFrameSource
            video_source = VideoFrameSource(video_path)
            if "dt" not in scenario_cfg and video_source.fps > 0:
                dt = 1.0 / video_source.fps
            if "num_frames" not in scenario_cfg and video_source.total_frames > 0:
                num_frames = video_source.total_frames

        # Build list of scheduled dynamic occluders (supporting both single and multiple occlusions)
        scheduled_occluders: List[Tuple[int, int, DynamicOccluder]] = []
        raw_occs: List[Dict[str, Any]] = []
        if dist_cfg.get("occlusion"):
            raw_occs.append(dist_cfg["occlusion"])
        if dist_cfg.get("occlusions"):
            raw_occs.extend(dist_cfg["occlusions"])

        for occ_item in raw_occs:
            if isinstance(occ_item, dict):
                occ_obj = DynamicOccluder(
                    initial_pos=init_pos,
                    velocity=vel,
                    radius_rad=float(occ_item.get("radius_rad", 0.008)),
                    opacity=float(occ_item.get("opacity", 1.0)),
                )
                start_f = int(occ_item.get("start_frame", 10))
                end_f = int(occ_item.get("end_frame", 25))
                scheduled_occluders.append((start_f, end_f, occ_obj))

        # Platform base motion profile (optional)
        plat_motion_cfg = scenario_cfg.get("platform_motion", dist_cfg.get("platform_motion", None))

        env = Environment(
            target=target,
            targets=targets,
            primary_target_id=primary_target_id,
            camera=camera,
            turbulence=turb,
            vibration=vib,
            sensor_noise=noise,
            occluders=[],
            frame_source=frame_source,
            video_source=video_source,
            platform_motion=plat_motion_cfg,
        )

        # Core pipeline components
        logger = MetricsLogger()
        detector = AdaptiveOpticalDetector(
            enable_signature_verification=False,
            targets=targets,
            primary_target_id=primary_target_id,
        )
        if enable_hybrid:
            from track.hybrid import HybridTracker
            tracker = HybridTracker(logger=logger, min_valid_confidence=0.40)
            kf = None
        else:
            kf = ConstantVelocityKalmanFilter(min_valid_confidence=0.40)
            tracker = None

        slew = SlewRateLimiter(max_velocity=max_vel, max_acceleration=max_acc)
        delay_queue = ControlDelayQueue(delay_frames=latency_frames)
        pid = PIDController(kp=kp, ki=ki, kd=kd, k_ff=k_ff, enable_feedforward=True)
        reacq_ctrl = HierarchicalReacquisitionController(
            fov_x=0.040,
            fov_y=0.030,
            tier1_budget_frames=tier1_budget,
            enable_predictive_search=enable_pred,
            scan_rate=0.12,
        )
        classifier = TrackLossClassifier()
        zone_predictor = ReacquisitionZonePredictor()

        stage_timings: Dict[str, List[float]] = {
            "rendering_ms": [],
            "disturbances_ms": [],
            "detection_ms": [],
            "tracking_ms": [],
            "control_ms": [],
            "logging_ms": [],
            "disturb_turbulence_ms": [],
            "disturb_noise_ms": [],
            "disturb_occlusion_ms": [],
            "disturb_vibration_ms": [],
        }

        state = "TRACKING"
        last_known_state = None
        consecutive_misses = 0
        delayed_p, delayed_t = 0.0, 0.0
        kf_init = False
        pipeline_errors = 0

        wall_start = time.perf_counter()

        for f in range(num_frames):
            # Dynamic occlusion injection
            active_occluders = [occ for (s, e, occ) in scheduled_occluders if s <= f <= e]
            env.occluders = active_occluders
            active_occ = active_occluders[0] if active_occluders else None

            # 1. Physical Camera Step & Optical Frame Rendering + Disturbances
            try:
                act_p, act_t = slew.apply_limit(delayed_p, delayed_t, dt)
                frame, tgt_state, cam_state = env.step(dt, act_p, act_t)
            except Exception:
                pipeline_errors += 1
                from contracts import FrameData
                frame = FrameData(image=np.zeros((480, 640), dtype=np.uint8), timestamp=f * dt, frame_id=f)
                tgt_state = TargetState(x=init_pos[0], y=init_pos[1]) if frame_source != "video_file" else None
                cam_state = camera.state

            prof = getattr(env, "last_step_profile", {})
            stage_timings["rendering_ms"].append(prof.get("rendering_ms", 0.0))
            stage_timings["disturbances_ms"].append(prof.get("disturbances_total_ms", 0.0))
            stage_timings["disturb_turbulence_ms"].append(prof.get("disturb_turbulence_ms", 0.0))
            stage_timings["disturb_noise_ms"].append(prof.get("disturb_noise_ms", 0.0))
            stage_timings["disturb_occlusion_ms"].append(prof.get("disturb_occlusion_ms", 0.0))
            stage_timings["disturb_vibration_ms"].append(prof.get("disturb_vibration_ms", 0.0))

            # 2. Optical Detection Stage (Timed)
            t_det0 = time.perf_counter_ns()
            try:
                det, _ = detector.detect(frame)
            except Exception:
                pipeline_errors += 1
                det = None
            t_det1 = time.perf_counter_ns()
            stage_timings["detection_ms"].append((t_det1 - t_det0) * 1e-6)

            # 3. Tracking & State Estimation Stage (Timed)
            t_trk0 = time.perf_counter_ns()
            try:
                # Confidence gating (reject below 0.40)
                is_valid_det = (det is not None and getattr(det, "confidence", 0.0) >= 0.40)

                if is_valid_det:
                    consecutive_misses = 0
                    err_p, err_t = camera.pixel_to_angular_error(det.x, det.y)
                    wx, wy = cam_state.pan + err_p, cam_state.tilt + err_t
                    if not (np.isfinite(wx) and np.isfinite(wy)):
                        raise ValueError(f"Non-finite coordinates: ({wx}, {wy})")

                    if enable_hybrid:
                        est = tracker.step(dt, (wx, wy), det.confidence, cn2=cn2, frame_id=f)
                    else:
                        if not kf_init:
                            kf.init_state(wx, wy, 0.0, 0.0, timestamp=frame.timestamp)
                            kf_init = True
                            est = kf.get_state()
                        else:
                            est = kf.step(dt, (wx, wy), det.confidence)

                    if est is not None and not (np.isfinite(est.x) and np.isfinite(est.y)):
                        raise ValueError(f"Non-finite estimate: ({est.x}, {est.y})")

                    last_known_state = est

                    if state == "SEARCHING":
                        state = "REACQUIRED"
                        reacq_ctrl.reset()
                else:
                    consecutive_misses += 1
                    if enable_hybrid:
                        est = tracker.step(dt, None, 0.0, cn2=cn2, frame_id=f)
                    else:
                        if kf_init:
                            est = kf.step(dt, None, 0.0)
                        else:
                            est = None

                    # Declare full track loss upon 8 consecutive misses after lock was established
                    if state == "TRACKING" and consecutive_misses >= 8 and last_known_state is not None:
                        state = "SEARCHING"
                        computed_zone = zone_predictor.compute_zone(
                            last_known_state,
                            frame.timestamp,
                            (camera.state.fov_x, camera.state.fov_y),
                        )
                        cause, conf, rat = classifier.classify(
                            consecutive_misses,
                            last_known_state,
                            cam_state,
                            active_occluder=active_occ,
                            active_cn2=cn2,
                            target_true_pos=(tgt_state.x, tgt_state.y),
                        )
                        reacq_ctrl.start_reacquisition(computed_zone, cam_state)
                        logger.log_track_loss_event(
                            frame_id=f,
                            timestamp=frame.timestamp,
                            cause=cause,
                            confidence=conf,
                            severity=1.0,
                            last_known_pos=(last_known_state.x, last_known_state.y),
                            last_known_vel=(last_known_state.vx, last_known_state.vy),
                            predicted_zone=computed_zone.to_dict(),
                            rationale=rat,
                        )
            except Exception:
                pipeline_errors += 1
                est = last_known_state
                consecutive_misses += 1
            t_trk1 = time.perf_counter_ns()
            stage_timings["tracking_ms"].append((t_trk1 - t_trk0) * 1e-6)

            # 4. Control Command Computation Stage (Timed)
            t_ctl0 = time.perf_counter_ns()
            try:
                if state in ["TRACKING", "REACQUIRED"] and est is not None:
                    c_p, c_t = pid.compute_command(est, cam_state, dt)
                elif state == "SEARCHING":
                    c_p, c_t = reacq_ctrl.step(dt, cam_state)
                else:
                    c_p, c_t = 0.0, 0.0

                if not (np.isfinite(c_p) and np.isfinite(c_t)):
                    c_p, c_t = 0.0, 0.0

                delayed_p, delayed_t = delay_queue.step(c_p, c_t)
            except Exception:
                pipeline_errors += 1
                delayed_p, delayed_t = 0.0, 0.0
            t_ctl1 = time.perf_counter_ns()
            stage_timings["control_ms"].append((t_ctl1 - t_ctl0) * 1e-6)

            # 5. Telemetry & Metrics Overhead (Timed)
            t_log0 = time.perf_counter_ns()
            try:
                _, _, rad_err_rad = env.get_angular_tracking_error()
                rad_err_mrad = (rad_err_rad * 1e3) if rad_err_rad is not None else None

                # Pixel-space tracking error and centroiding error calculation
                rad_err_px = None
                if rad_err_rad is not None:
                    rad_err_px = float(rad_err_rad * (camera.state.resolution[0] / camera.state.fov_x))

                cent_err_px = None
                if det is not None and frame.ground_truth_target_pos is not None:
                    cent_err_px = float(np.hypot(det.x - frame.ground_truth_target_pos[0], det.y - frame.ground_truth_target_pos[1]))

                logger.log_frame({
                    "frame_id": f,
                    "timestamp": frame.timestamp,
                    "state": state,
                    "tracker_mode": est.tracker_mode if est is not None else "LOST",
                    "detected": is_valid_det,
                    "confidence": det.confidence if det is not None else 0.0,
                    "radial_error_mrad": rad_err_mrad,
                    "tracking_error_px": rad_err_px,
                    "centroiding_error_px": cent_err_px,
                    "cam_pan_mrad": cam_state.pan * 1e3 if cam_state is not None else 0.0,
                    "cam_tilt_mrad": cam_state.tilt * 1e3 if cam_state is not None else 0.0,
                    "tgt_pan_mrad": (tgt_state.x * 1e3) if tgt_state is not None else None,
                    "tgt_tilt_mrad": (tgt_state.y * 1e3) if tgt_state is not None else None,
                })
            except Exception:
                pipeline_errors += 1
            t_log1 = time.perf_counter_ns()
            stage_timings["logging_ms"].append((t_log1 - t_log0) * 1e-6)

        wall_end = time.perf_counter()

        self.last_logger = logger
        self.calculator.fov_rad = camera.state.fov_x
        self.calculator.resolution_px = camera.state.resolution[0]
        record = self.calculator.compute_from_logger(
            logger=logger,
            start_wall_time=wall_start,
            end_wall_time=wall_end,
            scenario_name=name,
            is_held_out=is_held_out,
            stage_timings=stage_timings,
            pipeline_errors=pipeline_errors,
        )

        return record

    def run_all(self, scenarios: Optional[List[Dict[str, Any]]] = None) -> List[MetricsRecord]:
        """
        Execute the full scenario matrix, collecting MetricsRecord per run.

        Args:
            scenarios: Optional custom scenario matrix. If None, loaded from self.config_path.

        Returns:
            List of MetricsRecord objects.
        """
        matrix = scenarios if scenarios is not None else self.load_scenarios()
        self.results = []

        for scenario in matrix:
            record = self.run_scenario(scenario)
            self.results.append(record)

        return self.results

    def export_batch_json(self, filepath: str = "results/batch_results.json") -> None:
        """Export all run metrics to a single combined JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = [r.to_dict() for r in self.results]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_batch_csv(self, filepath: str = "results/batch_results.csv") -> None:
        """Export all run metrics to a combined CSV file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        if not self.results:
            return

        fieldnames = [
            "scenario_name",
            "is_held_out",
            "total_frames",
            "simulation_duration",
            "fps",
            "per_frame_processing_time_ms",
            "acquisition_time",
            "acquisition_time_s",
            "reacquisition_time_s",
            "avg_tracking_error",
            "max_tracking_error",
            "rmse_tracking_error",
            "tracking_error_px",
            "max_tracking_error_px",
            "rmse_px",
            "centroiding_error_px",
            "target_loss_percent",
            "lock_retention_rate",
            "active_locked_frames",
            "coasting_frames",
            "track_loss_count",
            "rendering_ms",
            "disturbances_ms",
            "detection_ms",
            "tracking_ms",
            "control_ms",
            "logging_ms",
            "pipeline_errors",
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in self.results:
                row = r.to_dict()
                st = row.get("stage_timing_breakdown", {})
                row["rendering_ms"] = f"{st.get('rendering_ms', 0.0):.3f}"
                row["disturbances_ms"] = f"{st.get('disturbances_ms', 0.0):.3f}"
                row["detection_ms"] = f"{st.get('detection_ms', 0.0):.3f}"
                row["tracking_ms"] = f"{st.get('tracking_ms', 0.0):.3f}"
                row["control_ms"] = f"{st.get('control_ms', 0.0):.3f}"
                row["logging_ms"] = f"{st.get('logging_ms', 0.0):.3f}"
                row["pipeline_errors"] = r.pipeline_errors
                writer.writerow(row)

    def format_results_table(self) -> str:
        """
        Generate a comprehensive ASCII table showing all performance metrics and
        complete per-stage latency breakdown across all scenarios.
        """
        if not self.results:
            return "No batch results available. Run scenarios first."

        col_w = [32, 10, 6, 9, 8, 8, 9, 8, 6, 7, 5, 8]
        total_w = sum(col_w) + (len(col_w) - 1) * 3

        lines = [
            "=" * total_w,
            f"{'Scenario Name':^32} | {'Type':^10} | {'FPS':^6} | {'Lock Ret%':^9} | {'Proc(ms)':^8} | {'Render':^8} | {'Disturb':^9} | {'Detect':^8} | {'Track':^6} | {'Control':^7} | {'Log':^5} | {'Sum(ms)':^8}",
            "-" * total_w,
        ]

        for r in self.results:
            name_str = r.scenario_name[:32]
            type_str = "[HELD-OUT]" if r.is_held_out else "Standard"
            lock_pct = f"{r.lock_retention_rate * 100.0:5.1f}%"
            st = r.stage_timing_breakdown
            rnd_ms = st.get("rendering_ms", 0.0)
            dst_ms = st.get("disturbances_ms", 0.0)
            det_ms = st.get("detection_ms", 0.0)
            trk_ms = st.get("tracking_ms", 0.0)
            ctl_ms = st.get("control_ms", 0.0)
            log_ms = st.get("logging_ms", 0.0)
            stage_sum = rnd_ms + dst_ms + det_ms + trk_ms + ctl_ms + log_ms

            lines.append(
                f"{name_str:<32} | {type_str:^10} | {r.fps:^6.1f} | {lock_pct:^9} | "
                f"{r.per_frame_processing_time_ms:^8.2f} | {rnd_ms:^8.2f} | {dst_ms:^9.2f} | {det_ms:^8.2f} | "
                f"{trk_ms:^6.2f} | {ctl_ms:^7.2f} | {log_ms:^5.2f} | {stage_sum:^8.2f}"
            )

        lines.append("=" * total_w)
        return "\n".join(lines)

    def format_scenario_stage_breakdown(self, scenario_name: str) -> str:
        """
        Format a detailed per-stage timing breakdown for a specific scenario,
        calculating each stage's % relative to the ACTUAL total per-frame processing time.
        """
        target_rec = next((r for r in self.results if r.scenario_name == scenario_name), None)
        if target_rec is None:
            return f"Scenario '{scenario_name}' not found in results."

        total_ms = target_rec.per_frame_processing_time_ms
        st = target_rec.stage_timing_breakdown

        rnd = st.get("rendering_ms", 0.0)
        dst = st.get("disturbances_ms", 0.0)
        turb = st.get("disturb_turbulence_ms", 0.0)
        noise = st.get("disturb_noise_ms", 0.0)
        occ = st.get("disturb_occlusion_ms", 0.0)
        vib = st.get("disturb_vibration_ms", 0.0)
        det = st.get("detection_ms", 0.0)
        trk = st.get("tracking_ms", 0.0)
        ctl = st.get("control_ms", 0.0)
        log = st.get("logging_ms", 0.0)

        sum_profiled = rnd + dst + det + trk + ctl + log
        other_overhead = max(0.0, total_ms - sum_profiled)

        lines = [
            "=" * 105,
            f"  STAGE PROFILING BREAKDOWN: {scenario_name} (Total Proc = {total_ms:.2f} ms/frame)",
            "=" * 105,
            f"{'Pipeline Stage / Operation':<42} | {'Time (ms)':^11} | {'% Actual Frame Time':^21} | {'Category':<22}",
            "-" * 105,
            f"{'1. Camera Frame Rendering (PSF & Optics)':<42} | {rnd:^11.2f} | {rnd/total_ms*100.0:^21.1f}% | {'Synthetic Image Render':<22}",
            f"{'2. Disturbance Simulation (Total)':<42} | {dst:^11.2f} | {dst/total_ms*100.0:^21.1f}% | {'Environmental Sim':<22}",
            f"{'   - Sensor Noise (Gaussian & Salt/Pepper)':<42} | {noise:^11.2f} | {noise/total_ms*100.0:^21.1f}% | {'Readout Noise':<22}",
            f"{'   - Kolmogorov Turbulence (Phase Screen)':<42} | {turb:^11.2f} | {turb/total_ms*100.0:^21.1f}% | {'Atmospheric Blur':<22}",
            f"{'   - Dynamic Occluder (LOS Blockage)':<42} | {occ:^11.2f} | {occ/total_ms*100.0:^21.1f}% | {'Obstacle Geometry':<22}",
            f"{'   - Platform Vibration (Jitter)':<42} | {vib:^11.2f} | {vib/total_ms*100.0:^21.1f}% | {'Mechanical Dynamics':<22}",
            f"{'3. Optical Detection (Top-hat, Centroid)':<42} | {det:^11.2f} | {det/total_ms*100.0:^21.1f}% | {'Computer Vision':<22}",
            f"{'4. State Estimation (KF/PF, Gating, Loss)':<42} | {trk:^11.2f} | {trk/total_ms*100.0:^21.1f}% | {'Estimation / Tracking':<22}",
            f"{'5. Gimbal Control (PID, Slew, Latency)':<42} | {ctl:^11.2f} | {ctl/total_ms*100.0:^21.1f}% | {'Control Law':<22}",
            f"{'6. Telemetry Logging & Error Computation':<42} | {log:^11.2f} | {log/total_ms*100.0:^21.1f}% | {'Instrumentation':<22}",
            f"{'7. Uninstrumented Loop Overhead':<42} | {other_overhead:^11.2f} | {other_overhead/total_ms*100.0:^21.1f}% | {'Python Loop / Function':<22}",
            "-" * 105,
            f"{'TOTAL ACCOUNTED FRAME TIME':<42} | {total_ms:^11.2f} | {100.0:^21.1f}% | {'100% Accounted':<22}",
            "=" * 105,
        ]
        return "\n".join(lines)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Batch Scenario Runner for FSOC Coarse PAT Simulator")
    parser.add_argument("--scenarios", type=str, default="config/scenarios.json", help="Path to scenario JSON config file")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid Kalman/Particle filter tracker")
    args = parser.parse_args()

    runner = BatchScenarioRunner(config_path=args.scenarios, use_hybrid_tracker=args.hybrid)
    scenarios = runner.load_scenarios()
    print(f"[*] Loaded {len(scenarios)} scenarios from {args.scenarios}. Executing batch run...")
    runner.run_all(scenarios)

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "batch_results.json")
    csv_path = os.path.join(args.output_dir, "batch_results.csv")
    runner.export_batch_json(json_path)
    runner.export_batch_csv(csv_path)

    print("\n" + runner.format_results_table())
    print(f"\n[OK] Batch evaluation complete! Results saved to:\n  - {json_path}\n  - {csv_path}")


if __name__ == "__main__":
    main()

