"""
Standalone Visual Simulation UI for FSOC Coarse PAT Simulator.
Smart India Hackathon 2024 | Problem Statement 26169 (ISRO / DOS)

Interactive Tkinter application providing real-time 2D graphical visualization
of the closed-loop PAT pipeline:
- Main 2D Canvas: Camera FOV boundary, target marker (with detection status),
  camera boresight reticle (with lock/coast dimming), dynamic occluders,
  Tier 1 predictive re-acquisition zone, and motion history trails.
- Live Telemetry Sidebar: Tracker mode (KF/PF/COAST), supervisor state
  (TRACKING/SEARCHING/REACQUIRED), confidence and severity meters, tracking error,
  and a scrolling real-time event log.
- Bottom Controls: Scenario selector (from config/flight_scenarios.json),
  Play/Pause, Step-Frame, Reset, and adjustable simulation speed.
"""

from __future__ import annotations
import os
import json
import time
import math
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import numpy as np

import tkinter as tk
from tkinter import ttk

from contracts import FrameData, TargetState, CameraState, MetricsRecord
from sim.target import Target
from sim.camera import Camera
from sim.environment import Environment
from detect.detector import AdaptiveOpticalDetector
from track.kalman import ConstantVelocityKalmanFilter
from track.particle import ParticleFilter
from track.severity import SeverityCalculator
from track.hybrid import HybridTracker
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


class SimulationSession:
    """
    Closed-loop PAT simulation session wrapper for interactive per-frame stepping.
    Directly executes real backend algorithms without mocking.
    """

    def __init__(self, scenario_cfg: Dict[str, Any]):
        self.cfg = scenario_cfg
        self.scenario_name = scenario_cfg.get("scenario_name", "UNKNOWN")
        self.num_frames = int(scenario_cfg.get("num_frames", 100))
        self.dt = float(scenario_cfg.get("dt", 0.033))
        self.seed = int(scenario_cfg.get("seed", 42))

        tgt_cfg = scenario_cfg.get("target", {})
        self.init_pos = tuple(tgt_cfg.get("initial_pos", [0.010, -0.005]))
        self.vel = tuple(tgt_cfg.get("velocity", [0.003, -0.002]))
        self.base_intensity = float(tgt_cfg.get("base_intensity", 220.0))
        self.dist_km = float(tgt_cfg.get("distance_km", 5.0))

        dist_cfg = scenario_cfg.get("disturbances", {})
        self.cn2 = float(dist_cfg.get("cn2", 1.0e-14))
        self.vib_amp = float(dist_cfg.get("vibration_amplitude", 0.0003))
        self.vib_freq = float(dist_cfg.get("vibration_frequency", 10.0))
        self.noise_std = float(dist_cfg.get("noise_level", 4.0))

        # Occlusions (single or list)
        self.scheduled_occluders: List[Tuple[int, int, DynamicOccluder]] = []
        raw_occs = []
        if dist_cfg.get("occlusion"):
            raw_occs.append(dist_cfg["occlusion"])
        if dist_cfg.get("occlusions"):
            raw_occs.extend(dist_cfg["occlusions"])

        for item in raw_occs:
            if isinstance(item, dict):
                occ = DynamicOccluder(
                    initial_pos=self.init_pos,
                    velocity=self.vel,
                    radius_rad=float(item.get("radius_rad", 0.008)),
                    opacity=float(item.get("opacity", 1.0)),
                )
                s_f = int(item.get("start_frame", 10))
                e_f = int(item.get("end_frame", 25))
                self.scheduled_occluders.append((s_f, e_f, occ))

        ctrl_cfg = scenario_cfg.get("control", {})
        self.kp = float(ctrl_cfg.get("kp", 0.35))
        self.ki = float(ctrl_cfg.get("ki", 0.0))
        self.kd = float(ctrl_cfg.get("kd", 0.15))
        self.k_ff = float(ctrl_cfg.get("k_ff", 1.0))
        self.latency_frames = int(ctrl_cfg.get("latency_frames", 2))
        self.max_vel = float(ctrl_cfg.get("max_velocity", 0.5))
        self.max_acc = float(ctrl_cfg.get("max_acceleration", 1.0))

        reacq_cfg = scenario_cfg.get("reacquisition", {})
        self.enable_pred = bool(reacq_cfg.get("enable_predictive_search", True))
        self.tier1_budget = int(reacq_cfg.get("tier1_budget_frames", 25))

        self.reset()

    def reset(self) -> None:
        """Reset simulation state to initial frame 0."""
        self.frame_id = 0
        self.sim_time = 0.0

        # Physical simulator objects
        self.target = Target(
            initial_pos=self.init_pos,
            velocity=self.vel,
            base_intensity=self.base_intensity,
            blink_frequency=4.0,
            modulation_depth=0.5,
            range_km=self.dist_km,
            ref_range_km=5.0,
        )
        self.camera = Camera(
            pan=0.0,
            tilt=0.0,
            fov_x=0.040,
            fov_y=0.030,
            resolution=(640, 480),
            enable_auto_exposure=True,
        )

        turb = KolmogorovTurbulence(cn2=self.cn2, seed=self.seed) if self.cn2 > 0 else None
        vib = PlatformVibration(amplitude_rad=self.vib_amp, frequency_hz=self.vib_freq, seed=self.seed) if self.vib_amp > 0 else None
        noise = SensorNoise(gaussian_std=self.noise_std, seed=self.seed) if self.noise_std > 0 else None

        self.env = Environment(
            target=self.target,
            camera=self.camera,
            turbulence=turb,
            vibration=vib,
            sensor_noise=noise,
            occluders=[],
        )

        # Algorithmic modules
        self.detector = AdaptiveOpticalDetector(enable_signature_verification=False)
        self.hybrid_tracker = HybridTracker(min_valid_confidence=0.40)
        self.slew = SlewRateLimiter(max_velocity=self.max_vel, max_acceleration=self.max_acc)
        self.delay_queue = ControlDelayQueue(delay_frames=self.latency_frames)
        self.pid = PIDController(kp=self.kp, ki=self.ki, kd=self.kd, k_ff=self.k_ff, enable_feedforward=True)
        self.reacq_ctrl = HierarchicalReacquisitionController(
            fov_x=0.040,
            fov_y=0.030,
            tier1_budget_frames=self.tier1_budget,
            enable_predictive_search=self.enable_pred,
            scan_rate=0.12,
        )
        self.classifier = TrackLossClassifier()
        self.zone_predictor = ReacquisitionZonePredictor()

        self.state = "TRACKING"
        self.last_known_state = None
        self.consecutive_misses = 0
        self.delayed_p = 0.0
        self.delayed_t = 0.0
        self.computed_zone = None
        self.prev_tracker_mode = "KF"

    def step(self) -> Dict[str, Any]:
        """Advance simulation by one dt time step and return frame telemetry."""
        f = self.frame_id
        dt = self.dt

        # Dynamic occlusion injection
        active_occluders = [occ for (s, e, occ) in self.scheduled_occluders if s <= f <= e]
        self.env.occluders = active_occluders
        active_occ = active_occluders[0] if active_occluders else None

        event_msg: Optional[str] = None

        # 1. Camera step & frame rendering
        act_p, act_t = self.slew.apply_limit(self.delayed_p, self.delayed_t, dt)
        frame, tgt_state, cam_state = self.env.step(dt, act_p, act_t)

        # 2. Optical Detection
        det, _ = self.detector.detect(frame)
        is_valid_det = (det is not None and getattr(det, "confidence", 0.0) >= 0.40)
        conf_val = float(det.confidence) if det is not None else 0.0

        # Check if target is inside camera FOV geometry
        dx_cam = tgt_state.x - cam_state.pan
        dy_cam = tgt_state.y - cam_state.tilt
        target_in_fov = (abs(dx_cam) <= cam_state.fov_x / 2.0) and (abs(dy_cam) <= cam_state.fov_y / 2.0)

        # 3. State Estimation & Tracking
        if is_valid_det:
            self.consecutive_misses = 0
            err_p, err_t = self.camera.pixel_to_angular_error(det.x, det.y)
            wx, wy = cam_state.pan + err_p, cam_state.tilt + err_t
            est = self.hybrid_tracker.step(dt, (wx, wy), det.confidence, cn2=self.cn2, frame_id=f)
            self.last_known_state = est

            if self.state == "SEARCHING":
                self.state = "REACQUIRED"
                self.reacq_ctrl.reset()
                event_msg = f"Frame {f}: Target REACQUIRED at ({tgt_state.x*1e3:.1f}, {tgt_state.y*1e3:.1f}) mrad"
        else:
            self.consecutive_misses += 1
            est = self.hybrid_tracker.step(dt, None, 0.0, cn2=self.cn2, frame_id=f)

            if self.state == "TRACKING" and self.consecutive_misses >= 8 and self.last_known_state is not None:
                self.state = "SEARCHING"
                self.computed_zone = self.zone_predictor.compute_zone(
                    self.last_known_state,
                    frame.timestamp,
                    (self.camera.state.fov_x, self.camera.state.fov_y),
                )
                cause, conf, rat = self.classifier.classify(
                    self.consecutive_misses,
                    self.last_known_state,
                    cam_state,
                    active_occluder=active_occ,
                    active_cn2=self.cn2,
                    target_true_pos=(tgt_state.x, tgt_state.y),
                )
                self.reacq_ctrl.start_reacquisition(self.computed_zone, cam_state)
                event_msg = f"Frame {f}: TRACK LOST ({cause}) -> Predictive Zone Search"

        # Check for tracker mode switches (KF <-> PF)
        cur_mode = est.tracker_mode if est is not None else "LOST"
        if cur_mode != self.prev_tracker_mode and cur_mode in ["KF", "PF"] and self.prev_tracker_mode in ["KF", "PF"]:
            event_msg = f"Frame {f}: Tracker switched {self.prev_tracker_mode} -> {cur_mode} (Sev={self.hybrid_tracker.last_severity:.2f})"
        self.prev_tracker_mode = cur_mode

        # 4. Control Command Computation
        if self.state in ["TRACKING", "REACQUIRED"] and est is not None:
            c_p, c_t = self.pid.compute_command(est, cam_state, dt)
        elif self.state == "SEARCHING":
            c_p, c_t = self.reacq_ctrl.step(dt, cam_state)
        else:
            c_p, c_t = 0.0, 0.0

        if not (np.isfinite(c_p) and np.isfinite(c_t)):
            c_p, c_t = 0.0, 0.0

        self.delayed_p, self.delayed_t = self.delay_queue.step(c_p, c_t)

        # 5. Tracking Error Calculation
        _, _, rad_err_rad = self.env.get_angular_tracking_error()
        rad_err_mrad = float(rad_err_rad * 1e3)

        occ_info = None
        if active_occ:
            occ_info = (active_occ.x, active_occ.y, active_occ.radius_rad)

        telemetry = {
            "frame_id": f,
            "sim_time": f * dt,
            "target_pos": (tgt_state.x, tgt_state.y),
            "camera_pan_tilt": (cam_state.pan, cam_state.tilt),
            "fov_bounds": cam_state.fov_bounds,
            "is_valid_det": is_valid_det,
            "target_in_fov": target_in_fov,
            "confidence": conf_val,
            "severity": float(getattr(self.hybrid_tracker, "last_severity", 0.0)),
            "tracker_mode": cur_mode,
            "supervisor_state": self.state,
            "tracking_error_mrad": rad_err_mrad,
            "active_occluder": occ_info,
            "reacq_zone": self.computed_zone if self.state == "SEARCHING" else None,
            "event_message": event_msg,
            "detector_pixel": (float(det.x), float(det.y)) if det is not None else None,
            "ground_truth_pixel": (float(frame.ground_truth_target_pos[0]), float(frame.ground_truth_target_pos[1])) if getattr(frame, "ground_truth_target_pos", None) is not None else None,
            "estimated_pos": (float(est.x), float(est.y)) if est is not None else None,
            "control_command": (float(c_p), float(c_t)),
            "sensor_resolution": list(cam_state.resolution),
            "range_km": float(self.dist_km),
        }

        self.frame_id += 1
        self.sim_time += dt
        return telemetry

    def inject_disturbance(self, disturbance_type: str, value: Any = None) -> str:
        """Dynamically inject or modify disturbance parameters during live simulation."""
        if disturbance_type == "occlusion":
            radius = float(value.get("radius_rad", 0.010)) if isinstance(value, dict) else 0.010
            duration = int(value.get("duration_frames", 20)) if isinstance(value, dict) else 20
            occ = DynamicOccluder(
                initial_pos=(self.target.x, self.target.y),
                velocity=self.vel,
                radius_rad=radius,
                opacity=1.0,
            )
            self.scheduled_occluders.append((self.frame_id, self.frame_id + duration, occ))
            return f"Injected dynamic occlusion at frame {self.frame_id} for {duration} frames"
        elif disturbance_type == "turbulence":
            new_cn2 = float(value) if value is not None else (self.cn2 * 10.0 if self.cn2 > 0 else 2.0e-14)
            self.cn2 = new_cn2
            self.env.turbulence = KolmogorovTurbulence(cn2=self.cn2, seed=self.seed)
            return f"Turbulence Cn2 updated to {self.cn2:.1e}"
        elif disturbance_type == "vibration":
            new_amp = float(value) if value is not None else (self.vib_amp * 2.5 if self.vib_amp > 0 else 0.0008)
            self.vib_amp = new_amp
            self.env.vibration = PlatformVibration(amplitude_rad=self.vib_amp, frequency_hz=self.vib_freq, seed=self.seed)
            return f"Platform vibration amplitude updated to {self.vib_amp*1e3:.2f} mrad"
        elif disturbance_type == "noise":
            new_std = float(value) if value is not None else (self.noise_std + 3.0)
            self.noise_std = new_std
            self.env.sensor_noise = SensorNoise(gaussian_std=self.noise_std, seed=self.seed)
            return f"Sensor noise updated to sigma={self.noise_std:.1f}"
        return f"Unknown disturbance: {disturbance_type}"


class VisualSimulatorUI:
    """
    Tkinter visualization GUI for FSOC coarse PAT simulator.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FSOC Coarse PAT Visual Simulator | SIH 2024 PS-26169 (ISRO / DOS)")
        self.root.geometry("1180x820")
        self.root.configure(bg="#0B0F17")
        self.root.minsize(1000, 720)

        # Application state
        self.is_running = False
        self.playback_speed = 1.0
        self.scenario_list = self.load_scenarios()
        self.current_session: Optional[SimulationSession] = None

        # Visual history queues (last 35 points)
        self.target_history: deque = deque(maxlen=35)
        self.camera_history: deque = deque(maxlen=35)

        # Canvas projection parameters (radians -> canvas pixels)
        self.canvas_w = 740
        self.canvas_h = 560
        # World view center and angular span (radians)
        self.view_center_pan = 0.015
        self.view_center_tilt = -0.005
        self.view_span_pan = 0.065   # 65 mrad width
        self.view_span_tilt = 0.050  # 50 mrad height

        self.setup_ui()
        self.select_scenario(0)

    def load_scenarios(self) -> List[Dict[str, Any]]:
        """Load flight scenarios from config file."""
        path = "config/flight_scenarios.json"
        if not os.path.exists(path):
            path = "configs/flight_scenarios.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def setup_ui(self) -> None:
        """Construct Tkinter layout: Header, Main Canvas, Side Telemetry, and Controls."""
        # 1. Header Banner
        header = tk.Frame(self.root, bg="#121824", height=50)
        header.pack(fill=tk.X, side=tk.TOP, padx=0, pady=0)

        lbl_title = tk.Label(
            header,
            text="FSOC COARSE PAT SIMULATOR — VISUAL TRACKING TELEMETRY HUD",
            font=("Segoe UI", 12, "bold"),
            fg="#00F0FF",
            bg="#121824",
        )
        lbl_title.pack(side=tk.LEFT, padx=16, pady=10)

        lbl_sub = tk.Label(
            header,
            text="ISRO / DOS PS-26169 | Closed-Loop Optical PAT Verification",
            font=("Segoe UI", 9),
            fg="#8A99AD",
            bg="#121824",
        )
        lbl_sub.pack(side=tk.RIGHT, padx=16, pady=10)

        # 2. Main Content Container (Canvas + Sidebar)
        main_body = tk.Frame(self.root, bg="#0B0F17")
        main_body.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Left Column: Canvas + Reticle Legends
        canvas_box = tk.Frame(main_body, bg="#101622", relief=tk.RIDGE, bd=1)
        canvas_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(
            canvas_box,
            width=self.canvas_w,
            height=self.canvas_h,
            bg="#080C14",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Canvas Legend bar
        legend_bar = tk.Frame(canvas_box, bg="#101622")
        legend_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=4)
        legends = [
            ("■ Camera FOV", "#00C8FF"),
            ("● Target (Locked)", "#00FF88"),
            ("● Target (Occluded)", "#667788"),
            ("✚ Camera Boresight", "#00F0FF"),
            ("◌ Occluder", "#556677"),
            ("⬡ Reacq Zone", "#FFAA00"),
        ]
        for txt, col in legends:
            lbl = tk.Label(legend_bar, text=txt, font=("Segoe UI", 8, "bold"), fg=col, bg="#101622")
            lbl.pack(side=tk.LEFT, padx=8)

        # Right Column: Live Telemetry Sidebar
        sidebar = tk.Frame(main_body, bg="#121824", width=380, relief=tk.RIDGE, bd=1)
        sidebar.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(0, 0))
        sidebar.pack_propagate(False)

        self.setup_sidebar(sidebar)

        # 3. Bottom Controls Panel
        ctrl_bar = tk.Frame(self.root, bg="#121824", height=68, relief=tk.RIDGE, bd=1)
        ctrl_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=12, pady=(0, 10))

        self.setup_controls(ctrl_bar)

    def setup_sidebar(self, parent: tk.Frame) -> None:
        """Construct telemetry cards and event log in sidebar."""
        pad_x, pad_y = 12, 6

        # Section Header
        lbl_head = tk.Label(parent, text="FLIGHT TELEMETRY HUD", font=("Segoe UI", 10, "bold"), fg="#00E5FF", bg="#121824")
        lbl_head.pack(anchor=tk.W, padx=pad_x, pady=(10, 4))

        # Status Cards Grid
        cards_frame = tk.Frame(parent, bg="#121824")
        cards_frame.pack(fill=tk.X, padx=pad_x, pady=4)

        # Card 1: Tracker Mode
        c1 = tk.Frame(cards_frame, bg="#182232", relief=tk.GROOVE, bd=1)
        c1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4), pady=2)
        tk.Label(c1, text="TRACKER MODE", font=("Segoe UI", 7, "bold"), fg="#7A8CA3", bg="#182232").pack(pady=(4, 0))
        self.lbl_tracker_mode = tk.Label(c1, text="KF (ACTIVE)", font=("Segoe UI", 12, "bold"), fg="#00FF88", bg="#182232")
        self.lbl_tracker_mode.pack(pady=(2, 6))

        # Card 2: Supervisor State
        c2 = tk.Frame(cards_frame, bg="#182232", relief=tk.GROOVE, bd=1)
        c2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=2)
        tk.Label(c2, text="SUPERVISOR STATE", font=("Segoe UI", 7, "bold"), fg="#7A8CA3", bg="#182232").pack(pady=(4, 0))
        self.lbl_supervisor_state = tk.Label(c2, text="TRACKING", font=("Segoe UI", 12, "bold"), fg="#00FF88", bg="#182232")
        self.lbl_supervisor_state.pack(pady=(2, 6))

        # Numeric Indicators Frame
        info_frame = tk.Frame(parent, bg="#121824")
        info_frame.pack(fill=tk.X, padx=pad_x, pady=6)

        # Frame counter & time
        f_row = tk.Frame(info_frame, bg="#121824")
        f_row.pack(fill=tk.X, pady=2)
        tk.Label(f_row, text="Frame / Sim Time:", font=("Segoe UI", 9), fg="#A0B0C4", bg="#121824").pack(side=tk.LEFT)
        self.lbl_frame_time = tk.Label(f_row, text="0 / 100  (0.00 s)", font=("Segoe UI", 9, "bold"), fg="#FFFFFF", bg="#121824")
        self.lbl_frame_time.pack(side=tk.RIGHT)

        # Tracking Error
        err_row = tk.Frame(info_frame, bg="#121824")
        err_row.pack(fill=tk.X, pady=2)
        tk.Label(err_row, text="Radial Error:", font=("Segoe UI", 9), fg="#A0B0C4", bg="#121824").pack(side=tk.LEFT)
        self.lbl_error = tk.Label(err_row, text="0.000 mrad", font=("Segoe UI", 10, "bold"), fg="#00FF88", bg="#121824")
        self.lbl_error.pack(side=tk.RIGHT)

        # Gauges (Confidence & Severity)
        gauge_box = tk.Frame(parent, bg="#182232", relief=tk.GROOVE, bd=1)
        gauge_box.pack(fill=tk.X, padx=pad_x, pady=6)

        # Confidence Bar
        conf_hdr = tk.Frame(gauge_box, bg="#182232")
        conf_hdr.pack(fill=tk.X, padx=8, pady=(6, 2))
        tk.Label(conf_hdr, text="Detection Confidence", font=("Segoe UI", 8, "bold"), fg="#C5D3E3", bg="#182232").pack(side=tk.LEFT)
        self.lbl_conf_val = tk.Label(conf_hdr, text="1.000", font=("Segoe UI", 8, "bold"), fg="#00FF88", bg="#182232")
        self.lbl_conf_val.pack(side=tk.RIGHT)

        self.canv_conf_bar = tk.Canvas(gauge_box, height=10, bg="#0E141E", highlightthickness=0)
        self.canv_conf_bar.pack(fill=tk.X, padx=8, pady=(0, 6))

        # Severity Bar
        sev_hdr = tk.Frame(gauge_box, bg="#182232")
        sev_hdr.pack(fill=tk.X, padx=8, pady=(4, 2))
        tk.Label(sev_hdr, text="Disturbance Severity", font=("Segoe UI", 8, "bold"), fg="#C5D3E3", bg="#182232").pack(side=tk.LEFT)
        self.lbl_sev_val = tk.Label(sev_hdr, text="0.000", font=("Segoe UI", 8, "bold"), fg="#00F0FF", bg="#182232")
        self.lbl_sev_val.pack(side=tk.RIGHT)

        self.canv_sev_bar = tk.Canvas(gauge_box, height=10, bg="#0E141E", highlightthickness=0)
        self.canv_sev_bar.pack(fill=tk.X, padx=8, pady=(0, 8))

        # Active Disturbance Badges
        dist_box = tk.Frame(parent, bg="#121824")
        dist_box.pack(fill=tk.X, padx=pad_x, pady=4)
        tk.Label(dist_box, text="ACTIVE DISTURBANCES", font=("Segoe UI", 7, "bold"), fg="#7A8CA3", bg="#121824").pack(anchor=tk.W)

        self.lbl_dist_info = tk.Label(
            dist_box,
            text="Cn2=1e-15 | Vib=10Hz | Noise=std3",
            font=("Segoe UI", 8),
            fg="#6DE3B5",
            bg="#182232",
            padx=6,
            pady=4,
            relief=tk.FLAT,
        )
        self.lbl_dist_info.pack(fill=tk.X, pady=2)

        # Real-time Event Log
        tk.Label(parent, text="LIVE EVENT LOG", font=("Segoe UI", 8, "bold"), fg="#7A8CA3", bg="#121824").pack(anchor=tk.W, padx=pad_x, pady=(8, 2))

        self.txt_log = tk.Text(
            parent,
            height=10,
            bg="#080C14",
            fg="#78E6B8",
            font=("Consolas", 8),
            relief=tk.FLAT,
            bd=0,
            padx=6,
            pady=6,
        )
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=pad_x, pady=(0, 10))
        self.txt_log.insert(tk.END, "PAT Simulator initialized.\nReady for playback.\n")
        self.txt_log.config(state=tk.DISABLED)

    def setup_controls(self, parent: tk.Frame) -> None:
        """Construct playback control buttons, speed slider, and scenario dropdown."""
        # Left: Scenario dropdown
        scen_frame = tk.Frame(parent, bg="#121824")
        scen_frame.pack(side=tk.LEFT, padx=12, pady=10)

        tk.Label(scen_frame, text="Scenario:", font=("Segoe UI", 9, "bold"), fg="#A0B0C4", bg="#121824").pack(side=tk.LEFT, padx=(0, 6))
        names = [s.get("scenario_name", f"Scenario {i}") for i, s in enumerate(self.scenario_list)]
        self.cb_scenario = ttk.Combobox(scen_frame, values=names, state="readonly", width=34)
        if names:
            self.cb_scenario.current(0)
        self.cb_scenario.pack(side=tk.LEFT)
        self.cb_scenario.bind("<<ComboboxSelected>>", self.on_scenario_changed)

        # Center: Playback Buttons
        btn_frame = tk.Frame(parent, bg="#121824")
        btn_frame.pack(side=tk.LEFT, padx=16, pady=10)

        self.btn_play = tk.Button(
            btn_frame,
            text="▶ Play",
            font=("Segoe UI", 9, "bold"),
            bg="#00B894",
            fg="#FFFFFF",
            width=8,
            relief=tk.FLAT,
            command=self.toggle_play,
        )
        self.btn_play.pack(side=tk.LEFT, padx=4)

        self.btn_step = tk.Button(
            btn_frame,
            text="⏭ Step",
            font=("Segoe UI", 9, "bold"),
            bg="#2D3A4F",
            fg="#FFFFFF",
            width=8,
            relief=tk.FLAT,
            command=self.step_one_frame,
        )
        self.btn_step.pack(side=tk.LEFT, padx=4)

        self.btn_reset = tk.Button(
            btn_frame,
            text="↺ Reset",
            font=("Segoe UI", 9, "bold"),
            bg="#3F4D63",
            fg="#FFFFFF",
            width=8,
            relief=tk.FLAT,
            command=self.reset_session,
        )
        self.btn_reset.pack(side=tk.LEFT, padx=4)

        # Right: Speed Slider
        speed_frame = tk.Frame(parent, bg="#121824")
        speed_frame.pack(side=tk.RIGHT, padx=16, pady=10)

        self.lbl_speed = tk.Label(speed_frame, text="Speed: 1.0x", font=("Segoe UI", 9, "bold"), fg="#A0B0C4", bg="#121824")
        self.lbl_speed.pack(side=tk.LEFT, padx=(0, 6))

        self.slider_speed = tk.Scale(
            speed_frame,
            from_=0.2,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=130,
            bg="#121824",
            fg="#FFFFFF",
            highlightthickness=0,
            command=self.on_speed_changed,
        )
        self.slider_speed.set(1.0)
        self.slider_speed.pack(side=tk.LEFT)

    def on_scenario_changed(self, event=None) -> None:
        idx = self.cb_scenario.current()
        self.select_scenario(idx)

    def select_scenario(self, idx: int) -> None:
        if 0 <= idx < len(self.scenario_list):
            cfg = self.scenario_list[idx]
            self.current_session = SimulationSession(cfg)
            self.target_history.clear()
            self.camera_history.clear()
            self.is_running = False
            self.btn_play.config(text="▶ Play", bg="#00B894")

            # Update disturbance badge
            dist = cfg.get("disturbances", {})
            cn2 = dist.get("cn2", 0.0)
            vib = dist.get("vibration_frequency", 0.0)
            noise = dist.get("noise_level", 0.0)
            occ_count = 1 if dist.get("occlusion") else len(dist.get("occlusions", []))
            txt = f"Cn2={cn2:.1e} | Vib={vib:.0f}Hz | Noise=std{noise:.0f} | Occluder={'YES' if occ_count>0 else 'NONE'}"
            self.lbl_dist_info.config(text=txt)

            self.log_event(f"Loaded scenario '{cfg.get('scenario_name')}' ({cfg.get('num_frames')} frames)")
            self.draw_canvas_placeholder()

    def on_speed_changed(self, val: str) -> None:
        self.playback_speed = float(val)
        self.lbl_speed.config(text=f"Speed: {self.playback_speed:.1f}x")

    def toggle_play(self) -> None:
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_play.config(text="⏸ Pause", bg="#E17055")
            self.animation_loop()
        else:
            self.btn_play.config(text="▶ Play", bg="#00B894")

    def step_one_frame(self) -> None:
        if self.is_running:
            self.toggle_play()
        self.advance_frame()

    def reset_session(self) -> None:
        if self.current_session:
            self.current_session.reset()
            self.target_history.clear()
            self.camera_history.clear()
            self.is_running = False
            self.btn_play.config(text="▶ Play", bg="#00B894")
            self.log_event("Simulation reset to frame 0")
            self.draw_canvas_placeholder()

    def advance_frame(self) -> None:
        if not self.current_session:
            return

        if self.current_session.frame_id >= self.current_session.num_frames:
            self.is_running = False
            self.btn_play.config(text="▶ Play", bg="#00B894")
            self.log_event("Scenario completed.")
            return

        telemetry = self.current_session.step()
        self.update_telemetry_ui(telemetry)
        self.render_canvas(telemetry)

    def animation_loop(self) -> None:
        if not self.is_running:
            return

        self.advance_frame()

        # Target dt nominal is 33ms (30 FPS); scale by playback_speed
        base_ms = int(self.current_session.dt * 1000.0) if self.current_session else 33
        delay_ms = max(5, int(base_ms / max(0.1, self.playback_speed)))
        self.root.after(delay_ms, self.animation_loop)

    def log_event(self, msg: str) -> None:
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, f"{msg}\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    # -------------------------------------------------------------------------
    # Canvas Coordinate Projection & Rendering
    # -------------------------------------------------------------------------

    def on_canvas_resize(self, event) -> None:
        self.canvas_w = event.width
        self.canvas_h = event.height

    def world_to_screen(self, pan_rad: float, tilt_rad: float) -> Tuple[float, float]:
        """Map angular radian coordinates to canvas pixel coordinates."""
        cx = self.canvas_w / 2.0
        cy = self.canvas_h / 2.0
        scale_x = self.canvas_w / self.view_span_pan
        scale_y = self.canvas_h / self.view_span_tilt

        dx = pan_rad - self.view_center_pan
        dy = tilt_rad - self.view_center_tilt

        # Invert tilt so positive tilt is up
        sx = cx + dx * scale_x
        sy = cy - dy * scale_y
        return sx, sy

    def draw_canvas_placeholder(self) -> None:
        """Render initial background grid before simulation start."""
        self.canvas.delete("all")
        self.draw_grid()

    def draw_grid(self) -> None:
        """Draw background coordinate grid with mrad tick marks."""
        w, h = self.canvas_w, self.canvas_h
        cx, cy = w / 2.0, h / 2.0

        # Subdued grid lines every 10 mrad
        grid_step_rad = 0.010  # 10 mrad
        scale_x = w / self.view_span_pan
        scale_y = h / self.view_span_tilt

        # Vertical gridlines
        p_min = self.view_center_pan - self.view_span_pan / 2.0
        p_max = self.view_center_pan + self.view_span_pan / 2.0
        start_p = math.floor(p_min / grid_step_rad) * grid_step_rad
        curr_p = start_p
        while curr_p <= p_max:
            sx, _ = self.world_to_screen(curr_p, 0.0)
            self.canvas.create_line(sx, 0, sx, h, fill="#121B2A", width=1, dash=(2, 4))
            self.canvas.create_text(sx + 2, h - 12, text=f"{curr_p*1e3:.0f}mrad", fill="#2C3D55", font=("Consolas", 7), anchor=tk.W)
            curr_p += grid_step_rad

        # Horizontal gridlines
        t_min = self.view_center_tilt - self.view_span_tilt / 2.0
        t_max = self.view_center_tilt + self.view_span_tilt / 2.0
        start_t = math.floor(t_min / grid_step_rad) * grid_step_rad
        curr_t = start_t
        while curr_t <= t_max:
            _, sy = self.world_to_screen(0.0, curr_t)
            self.canvas.create_line(0, sy, w, sy, fill="#121B2A", width=1, dash=(2, 4))
            self.canvas.create_text(8, sy - 6, text=f"{curr_t*1e3:.0f}mrad", fill="#2C3D55", font=("Consolas", 7), anchor=tk.W)
            curr_t += grid_step_rad

    def render_canvas(self, data: Dict[str, Any]) -> None:
        """Render complete visual frame on main canvas."""
        self.canvas.delete("all")
        self.draw_grid()

        tgt_pan, tgt_tilt = data["target_pos"]
        cam_pan, cam_tilt = data["camera_pan_tilt"]
        p_min, p_max, t_min, t_max = data["fov_bounds"]
        is_valid = data["is_valid_det"]
        in_fov = data["target_in_fov"]
        mode = data["tracker_mode"]
        state = data["supervisor_state"]
        occ = data["active_occluder"]
        zone = data["reacq_zone"]

        # 1. Update Motion Trajectory History Trails
        tgt_sx, tgt_sy = self.world_to_screen(tgt_pan, tgt_tilt)
        cam_sx, cam_sy = self.world_to_screen(cam_pan, cam_tilt)
        self.target_history.append((tgt_sx, tgt_sy))
        self.camera_history.append((cam_sx, cam_sy))

        # Draw Target Breadcrumb Trail (Dotted Green)
        if len(self.target_history) >= 2:
            pts = list(self.target_history)
            for i in range(len(pts) - 1):
                alpha = int(255 * (i + 1) / len(pts))
                # Fade color towards head
                col = "#155C3A" if i < len(pts) // 2 else "#00BA66"
                self.canvas.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill=col, width=1, dash=(2, 2))

        # Draw Camera Boresight Trail (Solid Cyan)
        if len(self.camera_history) >= 2:
            pts = list(self.camera_history)
            for i in range(len(pts) - 1):
                col = "#0B4C63" if i < len(pts) // 2 else "#00C8FF"
                self.canvas.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], fill=col, width=2)

        # 2. Draw Dynamic Occluder (if active)
        if occ:
            ox, oy, orad = occ
            osx, osy = self.world_to_screen(ox, oy)
            scale_r = (self.canvas_w / self.view_span_pan) * orad
            self.canvas.create_oval(
                osx - scale_r,
                osy - scale_r,
                osx + scale_r,
                osy + scale_r,
                fill="#253245",
                outline="#4F637D",
                width=2,
                stipple="gray50",
            )
            self.canvas.create_text(osx, osy, text="OCCLUDER", fill="#8EA2BD", font=("Segoe UI", 7, "bold"))

        # 3. Draw Camera FOV Rectangle
        fov_x1, fov_y1 = self.world_to_screen(p_min, t_max)  # Top-left
        fov_x2, fov_y2 = self.world_to_screen(p_max, t_min)  # Bottom-right
        fov_col = "#00D4FF" if mode in ["KF", "PF"] else "#667788"
        self.canvas.create_rectangle(fov_x1, fov_y1, fov_x2, fov_y2, outline=fov_col, width=2, dash=(4, 3))
        self.canvas.create_text(fov_x1 + 6, fov_y1 + 10, text="CAM FOV (40x30 mrad)", fill=fov_col, font=("Segoe UI", 7), anchor=tk.W)

        # 4. Draw Predictive Re-acquisition Search Zone (when SEARCHING)
        if zone is not None:
            zx, zy = self.world_to_screen(zone.center_pan, zone.center_tilt)
            zr = (self.canvas_w / self.view_span_pan) * zone.search_radius
            self.canvas.create_oval(zx - zr, zy - zr, zx + zr, zy + zr, outline="#FFAA00", width=2, dash=(3, 2))
            self.canvas.create_text(zx, zy - zr - 8, text="TIER 1 PREDICTED ZONE", fill="#FFAA00", font=("Segoe UI", 7, "bold"))

        # 5. Draw Target Marker
        # Green if in FOV & verified, gray if occluded/lost, amber if unverified
        if is_valid and in_fov:
            tgt_fill = "#00FF88"
            tgt_outline = "#FFFFFF"
            tgt_label = "TARGET (LOCKED)"
            r = 7
        elif in_fov and not occ:
            tgt_fill = "#FFAA00"
            tgt_outline = "#FFDD66"
            tgt_label = "TARGET (LOW CONF)"
            r = 6
        else:
            tgt_fill = "#505C6C"
            tgt_outline = "#7D8C9E"
            tgt_label = "TARGET (OCCLUDED/LOST)"
            r = 5

        # Target circle & halo
        if is_valid:
            self.canvas.create_oval(tgt_sx - r - 4, tgt_sy - r - 4, tgt_sx + r + 4, tgt_sy + r + 4, outline="#00FF88", width=1)
        self.canvas.create_oval(tgt_sx - r, tgt_sy - r, tgt_sx + r, tgt_sy + r, fill=tgt_fill, outline=tgt_outline, width=1.5)
        self.canvas.create_text(tgt_sx + 12, tgt_sy, text=tgt_label, fill=tgt_fill, font=("Segoe UI", 7, "bold"), anchor=tk.W)

        # 6. Draw Camera Boresight Reticle
        # CRITICAL: Visibly dim/gray out reticle when coasting blind!
        if mode in ["KF", "PF"]:
            ret_col = "#00F0FF"
            ret_width = 2
            ret_status = f"LOCKED ({mode})"
        elif mode in ["COAST", "PF_COAST"]:
            ret_col = "#5A6878"  # Visibly grayed out
            ret_width = 1.5
            ret_status = "COASTING (BLIND)"
        else:
            ret_col = "#FFAA00" if state == "SEARCHING" else "#4A5568"
            ret_width = 1.5
            ret_status = "SEARCHING"

        # Reticle Crosshairs
        cross_len = 16
        ring_r = 9
        self.canvas.create_oval(cam_sx - ring_r, cam_sy - ring_r, cam_sx + ring_r, cam_sy + ring_r, outline=ret_col, width=ret_width)
        self.canvas.create_line(cam_sx - cross_len, cam_sy, cam_sx - ring_r, cam_sy, fill=ret_col, width=ret_width)
        self.canvas.create_line(cam_sx + ring_r, cam_sy, cam_sx + cross_len, cam_sy, fill=ret_col, width=ret_width)
        self.canvas.create_line(cam_sx, cam_sy - cross_len, cam_sx, cam_sy - ring_r, fill=ret_col, width=ret_width)
        self.canvas.create_line(cam_sx, cam_sy + ring_r, cam_sx, cam_sy + cross_len, fill=ret_col, width=ret_width)
        self.canvas.create_text(cam_sx + 14, cam_sy - 12, text=f"BORESIGHT: {ret_status}", fill=ret_col, font=("Segoe UI", 7, "bold"), anchor=tk.W)

    def update_telemetry_ui(self, data: Dict[str, Any]) -> None:
        """Update sidebar badges, meters, and event log."""
        f_id = data["frame_id"]
        total_f = self.current_session.num_frames if self.current_session else 100
        sim_t = data["sim_time"]
        mode = data["tracker_mode"]
        state = data["supervisor_state"]
        conf = data["confidence"]
        sev = data["severity"]
        err_mrad = data["tracking_error_mrad"]
        event_msg = data.get("event_message")

        # 1. Update Tracker Mode Badge
        if mode == "KF":
            self.lbl_tracker_mode.config(text="KF (KALMAN)", fg="#00FF88")
        elif mode == "PF":
            self.lbl_tracker_mode.config(text="PF (PARTICLE)", fg="#00D4FF")
        elif "COAST" in mode:
            self.lbl_tracker_mode.config(text="COAST (BLIND)", fg="#FFAA00")
        else:
            self.lbl_tracker_mode.config(text="LOST", fg="#FF4757")

        # 2. Update Supervisor State Badge
        if state == "TRACKING":
            self.lbl_supervisor_state.config(text="TRACKING", fg="#00FF88")
        elif state == "SEARCHING":
            self.lbl_supervisor_state.config(text="SEARCHING", fg="#FFAA00")
        elif state == "REACQUIRED":
            self.lbl_supervisor_state.config(text="REACQUIRED", fg="#A29BFE")

        # 3. Numeric counters
        self.lbl_frame_time.config(text=f"{f_id} / {total_f}  ({sim_t:.2f} s)")

        err_col = "#00FF88" if err_mrad < 2.0 else ("#FFAA00" if err_mrad < 6.0 else "#FF4757")
        self.lbl_error.config(text=f"{err_mrad:.3f} mrad", fg=err_col)

        # 4. Confidence Meter
        self.lbl_conf_val.config(text=f"{conf:.3f}")
        self.canv_conf_bar.delete("all")
        bw = self.canv_conf_bar.winfo_width() or 200
        bar_fill_w = max(0, min(bw, int(bw * conf)))
        conf_bar_col = "#00FF88" if conf >= 0.50 else ("#FFAA00" if conf >= 0.40 else "#FF4757")
        self.canv_conf_bar.create_rectangle(0, 0, bar_fill_w, 10, fill=conf_bar_col, width=0)

        # 5. Severity Meter
        self.lbl_sev_val.config(text=f"{sev:.3f}")
        self.canv_sev_bar.delete("all")
        sev_fill_w = max(0, min(bw, int(bw * sev)))
        sev_bar_col = "#00FF88" if sev < 0.35 else ("#FFAA00" if sev < 0.65 else "#FF4757")
        self.canv_sev_bar.create_rectangle(0, 0, sev_fill_w, 10, fill=sev_bar_col, width=0)

        # 6. Event log entry
        if event_msg:
            self.log_event(event_msg)


def main():
    root = tk.Tk()
    app = VisualSimulatorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
