"""
Core Data Contracts & Types for FSOC Coarse PAT Simulator
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

This module defines standard data exchange contracts across all backend modules
(sim, detect, track, control, disturb, metrics) and provides a clean boundary
for consumption by frontend/GUI dashboards.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any, Union, List
import numpy as np


def get_resource_path(relative_path: str) -> str:
    """
    Resolves the absolute path to a resource file, compatible with development,
    package installation, and PyInstaller bundled distribution.
    """
    if not relative_path:
        return ""

    rel_path = os.path.normpath(str(relative_path))

    # 1. Already existing absolute path
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path

    # 2. PyInstaller temporary extraction directory (sys._MEIPASS for --onefile / --onedir)
    if hasattr(sys, "_MEIPASS"):
        meipass_candidate = os.path.join(sys._MEIPASS, rel_path)
        if os.path.exists(meipass_candidate):
            return os.path.abspath(meipass_candidate)

    # 3. PyInstaller frozen application directory (alongside executable in --onedir)
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        exe_candidate = os.path.join(exe_dir, rel_path)
        if os.path.exists(exe_candidate):
            return os.path.abspath(exe_candidate)

    # 4. Project root directory (where contracts.py resides)
    project_root = os.path.dirname(os.path.abspath(__file__))
    proj_candidate = os.path.join(project_root, rel_path)
    if os.path.exists(proj_candidate):
        return os.path.abspath(proj_candidate)

    # 5. Current working directory
    cwd_candidate = os.path.join(os.getcwd(), rel_path)
    if os.path.exists(cwd_candidate):
        return os.path.abspath(cwd_candidate)

    # Fallback: return path anchored to project root or executable dir
    if getattr(sys, "frozen", False):
        return os.path.abspath(os.path.join(os.path.dirname(sys.executable), rel_path))
    if hasattr(sys, "_MEIPASS"):
        return os.path.abspath(os.path.join(sys._MEIPASS, rel_path))
    return os.path.abspath(proj_candidate)


@dataclass
class FrameData:
    """
    Represents a single optical sensor frame and its associated metadata.

    Attributes:
        image: 2D or 3D NumPy array representing the optical sensor readout.
        timestamp: Simulation time at frame exposure in seconds.
        frame_id: Monotonically increasing frame index.
        ground_truth_target_pos: Optional (x, y) ground-truth target centroid in pixels for primary target.
        ground_truth_targets: Optional mapping of target_id to (x, y) ground-truth centroid for all visible targets.
    """
    image: np.ndarray
    timestamp: float
    frame_id: int = 0
    ground_truth_target_pos: Optional[Tuple[float, float]] = None
    ground_truth_targets: Optional[Dict[Union[str, int], Tuple[float, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dict (excludes large raw array for JSON safety)."""
        d = {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "image_shape": list(self.image.shape) if self.image is not None else None,
            "ground_truth_target_pos": self.ground_truth_target_pos,
        }
        if self.ground_truth_targets is not None:
            d["ground_truth_targets"] = self.ground_truth_targets
        return d


@dataclass
class TargetState:
    """
    Estimated or ground-truth kinematic state of an optical beacon.

    Attributes:
        x: Position along horizontal axis (pixels in focal plane or angular coordinate).
        y: Position along vertical axis (pixels in focal plane or angular coordinate).
        vx: Velocity along horizontal axis (pixels/s or rad/s).
        vy: Velocity along vertical axis (pixels/s or rad/s).
        confidence: Confidence score in [0.0, 1.0] indicating estimation quality.
        timestamp: Timestamp of the state estimate in seconds.
        tracker_mode: Active tracker mode identifier (e.g., 'KF', 'PF', 'COAST', 'LOST').
        target_id: Target identifier (str or int) for multi-target tracking.
    """
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 1.0
    timestamp: float = 0.0
    tracker_mode: str = "KF"
    target_id: Union[str, int] = "target_0"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to standard dictionary."""
        return asdict(self)


VALID_MOTION_TYPES = [
    "straight_line",
    "circular",
    "figure_eight",
    "random",
    "spiral",
    "sinusoidal",
]


@dataclass
class TargetConfig:
    """
    Configuration specification for a single moving target beacon.
    """
    target_id: Union[str, int] = "target_0"
    initial_pos: Tuple[float, float] = (0.010, -0.005)
    velocity: Tuple[float, float] = (0.002, 0.001)
    blink_frequency: float = 4.0
    base_intensity: float = 220.0
    modulation_depth: float = 0.5
    range_km: float = 5.0
    ref_range_km: float = 5.0
    motion_type: str = "straight_line"
    motion_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], default_id: Union[str, int] = "target_0") -> TargetConfig:
        pos = d.get("initial_pos", d.get("initial_position", [0.010, -0.005]))
        vel = d.get("velocity", d.get("initial_velocity", [0.002, 0.001]))

        motion_type_raw = d.get("motion_type", "straight_line")
        motion_type = str(motion_type_raw).strip().lower().replace("-", "_").replace(" ", "_")
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
        motion_type = alias_map.get(motion_type, motion_type)
        if motion_type not in VALID_MOTION_TYPES:
            raise ValueError(
                f"Invalid motion_type: '{motion_type_raw}'. Supported types: {VALID_MOTION_TYPES}"
            )

        motion_params = dict(d.get("motion_params", {}))
        # Also capture any inline motion-specific parameters if passed directly in d
        motion_specific_keys = [
            "center", "radius", "angular_velocity", "amplitude_x", "amplitude_y",
            "frequency", "phase_offset", "step_variance", "bounds", "seed",
            "initial_radius", "radial_velocity", "min_radius", "max_radius",
            "amplitude", "axis"
        ]
        for k in motion_specific_keys:
            if k in d and k not in motion_params:
                motion_params[k] = d[k]

        return cls(
            target_id=d.get("target_id", default_id),
            initial_pos=tuple(pos) if isinstance(pos, (list, tuple)) else pos,
            velocity=tuple(vel) if isinstance(vel, (list, tuple)) else vel,
            blink_frequency=float(d.get("blink_frequency", 4.0)),
            base_intensity=float(d.get("base_intensity", d.get("beacon_power", 220.0))),
            modulation_depth=float(d.get("modulation_depth", 0.5)),
            range_km=float(d.get("range_km", d.get("distance_km", 5.0))),
            ref_range_km=float(d.get("ref_range_km", 5.0)),
            motion_type=motion_type,
            motion_params=motion_params,
        )


def normalize_scenario_targets(
    scenario_cfg: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Union[str, int]]:
    """
    Normalize scenario configuration to support multi-target schema while maintaining
    complete backward compatibility with legacy single-target scenarios.

    Args:
        scenario_cfg: Scenario configuration dictionary.

    Returns:
        (targets_list, primary_target_id)
        - targets_list: List of target configuration dicts, each with target_id.
        - primary_target_id: Identifier of the primary target camera should point at.
    """
    primary_target_id = scenario_cfg.get("primary_target_id", None)

    # Multi-target schema: "targets" array
    if "targets" in scenario_cfg and isinstance(scenario_cfg["targets"], list) and len(scenario_cfg["targets"]) > 0:
        raw_targets = scenario_cfg["targets"]
        targets_list: List[Dict[str, Any]] = []
        for i, t in enumerate(raw_targets):
            t_copy = dict(t)
            if "target_id" not in t_copy:
                t_copy["target_id"] = f"target_{i}"
            if "motion_type" not in t_copy:
                t_copy["motion_type"] = "straight_line"
            targets_list.append(t_copy)
        if primary_target_id is None:
            primary_target_id = targets_list[0]["target_id"]
        return targets_list, primary_target_id

    # Backward compatibility: legacy single "target" object
    raw_target = scenario_cfg.get("target", {})
    t_copy = dict(raw_target)
    if "target_id" not in t_copy:
        t_copy["target_id"] = "target_0"
    if "motion_type" not in t_copy:
        t_copy["motion_type"] = "straight_line"
    if primary_target_id is None:
        primary_target_id = t_copy["target_id"]

    return [t_copy], primary_target_id



@dataclass
class CameraState:
    """
    State and optical parameters of the virtual Pan-Tilt Gimbal Camera.

    Attributes:
        pan: Azimuth gimbal angle in radians (or degrees as per config).
        tilt: Elevation gimbal angle in radians (or degrees as per config).
        pan_rate: Azimuth angular velocity (rad/s).
        tilt_rate: Elevation angular velocity (rad/s).
        fov_x: Horizontal field of view in radians (or degrees).
        fov_y: Vertical field of view in radians (or degrees).
        focal_length: Optical focal length (mm or meters).
        resolution: Sensor resolution as (width, height) in pixels.
    """
    pan: float = 0.0
    tilt: float = 0.0
    pan_rate: float = 0.0
    tilt_rate: float = 0.0
    fov_x: float = 0.1  # ~5.7 degrees default coarse FOV
    fov_y: float = 0.1
    focal_length: float = 100.0  # mm
    resolution: Tuple[int, int] = (640, 480)

    @property
    def fov_bounds(self) -> Tuple[float, float, float, float]:
        """
        Compute angular bounds (pan_min, pan_max, tilt_min, tilt_max)
        centered at the current gimbal boresight.
        """
        half_x = self.fov_x / 2.0
        half_y = self.fov_y / 2.0
        return (
            self.pan - half_x,
            self.pan + half_x,
            self.tilt - half_y,
            self.tilt + half_y,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize camera state including computed FOV bounds."""
        d = asdict(self)
        d["fov_bounds"] = self.fov_bounds
        return d


@dataclass
class DisturbanceConfig:
    """
    Configuration parameters for environmental and physical disturbances.

    Attributes:
        cn2: Refractive index structure constant C_n^2 for Kolmogorov atmospheric
             turbulence (m^(-2/3)), e.g., 1e-14 (moderate) to 1e-12 (severe).
        vibration_amplitude: Platform jitter amplitude (radians or pixels).
        vibration_frequency: Dominant platform vibration frequency (Hz).
        noise_level: Standard deviation of additive Gaussian sensor noise.
        noise_types: Selectable noise models (e.g. ['gaussian', 'poisson', 'salt_pepper']).
        poisson_scale: Conversion factor for Poisson shot noise (photons per gray level).
        salt_pepper_prob: Probability of hot/dead pixel impulse noise.
        occluder_frequency: Probability or rate of dynamic occluder appearance.
        occluder_size: Size / radius of occluders (pixels or angular span).
        random_walk_jitter: Drift rate for platform random-walk jitter.
    """
    cn2: float = 1e-14
    vibration_amplitude: float = 0.5  # pixels or mrad
    vibration_frequency: float = 10.0  # Hz
    noise_level: float = 5.0  # sensor readout noise std dev
    noise_types: List[str] = field(default_factory=lambda: ["gaussian"])
    poisson_scale: float = 1.0  # photon conversion scale factor for Poisson shot noise
    salt_pepper_prob: float = 0.0005  # impulse noise probability
    atmospheric_condition: str = "clear"  # "clear", "haze", "fog", "rain", "low_light"
    atmospheric_params: Dict[str, Any] = field(default_factory=dict)
    occluder_frequency: float = 0.05
    occluder_size: float = 30.0  # pixels
    random_walk_jitter: float = 0.01

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DisturbanceConfig":
        """Construct DisturbanceConfig from a configuration dictionary with fallback defaults."""
        noise_types_raw = d.get("noise_types", None)
        if noise_types_raw is not None:
            if isinstance(noise_types_raw, str):
                noise_types = [noise_types_raw]
            else:
                noise_types = list(noise_types_raw)
        else:
            noise_types = ["gaussian"]

        return cls(
            cn2=float(d.get("cn2", 1e-14)),
            vibration_amplitude=float(d.get("vibration_amplitude", 0.5)),
            vibration_frequency=float(d.get("vibration_frequency", 10.0)),
            noise_level=float(d.get("noise_level", 5.0)),
            noise_types=noise_types,
            poisson_scale=float(d.get("poisson_scale", 1.0)),
            salt_pepper_prob=float(d.get("salt_pepper_prob", 0.0005)),
            atmospheric_condition=str(d.get("atmospheric_condition", "clear")).lower().strip(),
            atmospheric_params=dict(d.get("atmospheric_params", {})),
            occluder_frequency=float(d.get("occluder_frequency", 0.05)),
            occluder_size=float(d.get("occluder_size", 30.0)),
            random_walk_jitter=float(d.get("random_walk_jitter", 0.01)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class MetricsRecord:
    """
    Cumulative and summary performance metrics for the PAT simulation run.

    Attributes:
        simulation_duration: Total elapsed simulation time in seconds.
        fps: Average frame rate processed by the simulation pipeline.
        acquisition_time: Time taken from start until initial target lock (seconds).
        avg_tracking_error: Mean radial error between boresight/lock point and target (pixels or rad).
        max_tracking_error: Peak radial tracking error observed during lock (pixels or rad).
        lock_retention_rate: Proportion of time target was retained within tracking threshold [0.0, 1.0].
        per_frame_processing_time_ms: Average processing time per frame in milliseconds.
        total_frames: Total number of frames processed.
        track_loss_count: Number of times tracking lock was broken.
        active_tracker_breakdown: Distribution of tracker modes (e.g. {'KF': 0.85, 'PF': 0.15}).
    """
    simulation_duration: float = 0.0
    fps: float = 0.0
    acquisition_time: float = 0.0
    avg_tracking_error: Union[float, str] = 0.0
    max_tracking_error: Union[float, str] = 0.0
    rmse_tracking_error: Union[float, str] = 0.0
    lock_retention_rate: float = 0.0
    per_frame_processing_time_ms: float = 0.0
    total_frames: int = 0
    active_locked_frames: int = 0
    coasting_frames: int = 0
    track_loss_count: int = 0
    active_tracker_breakdown: Dict[str, float] = field(default_factory=dict)
    stage_timing_breakdown: Dict[str, float] = field(default_factory=dict)
    scenario_name: str = ""
    is_held_out: bool = False
    pipeline_errors: int = 0
    has_ground_truth: bool = True

    tracking_error_px: Union[float, str] = 0.0
    avg_tracking_error_px: Union[float, str] = 0.0
    max_tracking_error_px: Union[float, str] = 0.0
    rmse_px: Union[float, str] = 0.0
    centroiding_error_px: Union[float, str] = 0.0
    avg_centroiding_error_px: Union[float, str] = 0.0
    max_centroiding_error_px: Union[float, str] = 0.0
    target_loss_percent: float = 0.0
    acquisition_time_s: float = 0.0
    reacquisition_time_s: float = 0.0
    centroiding_error_log_px: List[float] = field(default_factory=list)

    steady_tracking_error_px: Union[float, str] = 0.0
    steady_rmse_px: Union[float, str] = 0.0
    steady_avg_tracking_error_mrad: Union[float, str] = 0.0
    steady_rmse_tracking_error_mrad: Union[float, str] = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary for easy JSON/CSV export or GUI consumption."""
        d = asdict(self)
        d["tracking_error_px"] = self.tracking_error_px or self.avg_tracking_error_px
        d["tracking_error_px_avg"] = self.avg_tracking_error_px or self.tracking_error_px
        d["tracking_error_px_max"] = self.max_tracking_error_px
        d["rmse_px"] = self.rmse_px
        d["steady_tracking_error_px"] = self.steady_tracking_error_px
        d["steady_rmse_px"] = self.steady_rmse_px
        d["steady_avg_tracking_error_mrad"] = self.steady_avg_tracking_error_mrad
        d["steady_rmse_tracking_error_mrad"] = self.steady_rmse_tracking_error_mrad
        d["target_loss_percent"] = self.target_loss_percent
        d["acquisition_time_s"] = self.acquisition_time_s if self.acquisition_time_s > 0 else self.acquisition_time
        d["reacquisition_time_s"] = self.reacquisition_time_s
        d["centroiding_error_px"] = self.centroiding_error_px or self.avg_centroiding_error_px

        # Human-readable exact PS Terminology mapping (for evaluation rubric / QA)
        d["ps_terminology"] = {
            "Tracking Error": self.tracking_error_px or self.avg_tracking_error_px,
            "Tracking Error (Steady-State)": self.steady_tracking_error_px or self.tracking_error_px,
            "Target Loss": self.target_loss_percent,
            "Centroiding error": self.centroiding_error_px or self.avg_centroiding_error_px,
            "RMSE": self.rmse_px,
            "RMSE (Steady-State)": self.steady_rmse_px or self.rmse_px,
            "Acquisition Time": self.acquisition_time_s if self.acquisition_time_s > 0 else self.acquisition_time,
            "Re-acquisition time": self.reacquisition_time_s,
            "Lock retention rate": self.lock_retention_rate,
            "FPS": self.fps,
        }
        return d


@dataclass
class ReacquisitionZone:
    """
    Kinematically forward-projected search zone for predictive target re-acquisition.

    Attributes:
        center_pan: Forward-projected horizontal center in radians.
        center_tilt: Forward-projected vertical center in radians.
        search_radius: Uncertainty radius of the predicted search zone in radians.
        predicted_velocity: Estimated velocity (vx, vy) in rad/s used for projection.
        projection_time_s: Total time delta (seconds) elapsed since last validated track.
        pan_bounds: Minimum and maximum pan boundaries (pan_min, pan_max) in radians.
        tilt_bounds: Minimum and maximum tilt boundaries (tilt_min, tilt_max) in radians.
    """
    center_pan: float
    center_tilt: float
    search_radius: float
    predicted_velocity: Tuple[float, float]
    projection_time_s: float
    pan_bounds: Tuple[float, float]
    tilt_bounds: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize reacquisition zone to dictionary."""
        return asdict(self)

