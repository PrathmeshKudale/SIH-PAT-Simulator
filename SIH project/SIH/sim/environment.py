"""
Virtual 2D Simulation Environment coordinating target kinematics, camera optics,
clutter objects, and realistic physical disturbances.
"""

import time
import time
from typing import Tuple, Optional, List, Any, Dict, Union
import numpy as np
from contracts import FrameData, CameraState, TargetState, DisturbanceConfig, normalize_scenario_targets
from sim.target import Target, ClutterObject
from sim.camera import Camera
from sim.motion_profiles import MotionProfile, create_motion_profile
from disturb.turbulence import KolmogorovTurbulence
from disturb.vibration import PlatformVibration
from disturb.noise import SensorNoise
from disturb.occluder import DynamicOccluder
from disturb.atmospheric import AtmosphericDisturbance
from sim.video_source import VideoFrameSource


class Environment:
    """
    Coordinates target kinematics, clutter, camera gimbal dynamics, and
    physical disturbances (turbulence, vibration, sensor noise, occluders).
    Supports multiple simultaneous moving targets while designating one as primary for tracking.
    Also supports consuming real pre-recorded video files via VideoFrameSource.
    Supports selectable parametric motion profiles for both targets and platform motion.
    """

    def __init__(
        self,
        target: Optional[Target] = None,
        camera: Optional[Camera] = None,
        clutter_objects: Optional[List[ClutterObject]] = None,
        turbulence: Optional[KolmogorovTurbulence] = None,
        atmospheric_dist: Optional[AtmosphericDisturbance] = None,
        vibration: Optional[PlatformVibration] = None,
        sensor_noise: Optional[SensorNoise] = None,
        occluders: Optional[List[DynamicOccluder]] = None,
        targets: Optional[List[Target]] = None,
        primary_target_id: Optional[Union[str, int]] = None,
        frame_source: str = "synthetic",
        video_source: Optional[VideoFrameSource] = None,
        video_path: Optional[str] = None,
        platform_motion: Optional[Union[MotionProfile, Dict[str, Any], str]] = None,
    ):
        self.frame_source = str(frame_source)
        if video_source is not None:
            self.video_source = video_source
            self.frame_source = "video_file"
        elif video_path is not None:
            self.video_source = VideoFrameSource(video_path)
            self.frame_source = "video_file"
        else:
            self.video_source = None

        if targets is not None and len(targets) > 0:
            self.targets = list(targets)
            if primary_target_id is not None:
                self.primary_target_id = primary_target_id
                matched = next((t for t in self.targets if t.target_id == primary_target_id), None)
                self.target = matched if matched is not None else self.targets[0]
            else:
                self.primary_target_id = self.targets[0].target_id
                self.target = self.targets[0]
        elif target is not None:
            self.targets = [target]
            self.primary_target_id = target.target_id
            self.target = target
        else:
            default_tgt = Target()
            self.targets = [default_tgt]
            self.primary_target_id = default_tgt.target_id
            self.target = default_tgt

        self.camera = camera if camera is not None else Camera()
        self.clutter_objects = clutter_objects if clutter_objects is not None else []
        self.turbulence = turbulence
        self.atmospheric_dist = atmospheric_dist
        self.vibration = vibration
        self.sensor_noise = sensor_noise
        self.occluders = occluders if occluders is not None else []

        # Platform base motion profile (default: None / zero motion)
        if isinstance(platform_motion, MotionProfile):
            self.platform_motion = platform_motion
        elif isinstance(platform_motion, dict):
            self.platform_motion = create_motion_profile(**platform_motion)
        elif isinstance(platform_motion, str):
            self.platform_motion = create_motion_profile(motion_type=platform_motion)
        else:
            self.platform_motion = None

        self.frame_count = 0
        self.sim_time = 0.0
        self.last_step_profile: Dict[str, float] = {}

    def reset(self) -> None:
        """Reset environment simulation clock, targets, and disturbance generators."""
        self.frame_count = 0
        self.sim_time = 0.0
        for tgt in self.targets:
            tgt.reset()
        if self.platform_motion is not None:
            self.platform_motion.reset()
        if self.vibration is not None:
            self.vibration.reset()
        if self.atmospheric_dist is not None:
            self.atmospheric_dist.reset()

    @classmethod
    def from_config(cls, scenario_cfg: Dict[str, Any], camera: Optional[Camera] = None) -> "Environment":
        """
        Factory method to instantiate Environment directly from scenario config dictionary.
        Reads motion_type and parameters for targets and platform motion.
        """
        targets_cfg, primary_target_id = normalize_scenario_targets(scenario_cfg)
        targets = [Target.from_config(t) for t in targets_cfg]

        dist_cfg = scenario_cfg.get("disturbances", {})
        cn2 = float(dist_cfg.get("cn2", 0.0))
        seed = int(scenario_cfg.get("seed", 42))
        turb = KolmogorovTurbulence(cn2=cn2, seed=seed) if cn2 > 0 else None

        vib_amp = float(dist_cfg.get("vibration_amplitude", 0.0))
        vib_freq = float(dist_cfg.get("vibration_frequency", 10.0))
        vib = PlatformVibration(amplitude_rad=vib_amp, frequency_hz=vib_freq, seed=seed) if vib_amp > 0 else None

        noise_std = float(dist_cfg.get("noise_level", 0.0))
        noise_types = dist_cfg.get("noise_types", None)
        poisson_scale = float(dist_cfg.get("poisson_scale", 1.0))
        sp_prob = float(dist_cfg.get("salt_pepper_prob", 0.0005 if (noise_types and "salt_pepper" in str(noise_types)) else 0.0))

        has_noise = (
            (noise_types is not None and len(noise_types) > 0)
            or noise_std > 0
        )
        if has_noise:
            noise = SensorNoise(
                gaussian_std=noise_std,
                salt_pepper_prob=sp_prob,
                poisson_scale=poisson_scale,
                noise_types=noise_types,
                seed=seed,
            )
        else:
            noise = None

        # Atmospheric weather condition & illumination
        atm_cond = dist_cfg.get("atmospheric_condition", "clear")
        atm_params = dist_cfg.get("atmospheric_params", {})
        if (atm_cond and str(atm_cond).strip().lower() not in ("clear", "none")) or atm_params:
            atm_dist = AtmosphericDisturbance(condition=atm_cond, params=atm_params, seed=seed)
        else:
            atm_dist = None

        # Platform motion (mandatory default linear, optional circular/random/spiral/figure-8)
        plat_cfg = scenario_cfg.get("platform_motion", dist_cfg.get("platform_motion", None))

        frame_source = scenario_cfg.get("frame_source", "synthetic")
        video_path = scenario_cfg.get("video_path", None)

        return cls(
            targets=targets,
            primary_target_id=primary_target_id,
            camera=camera if camera is not None else Camera(),
            turbulence=turb,
            atmospheric_dist=atm_dist,
            vibration=vib,
            sensor_noise=noise,
            platform_motion=plat_cfg,
            frame_source=frame_source,
            video_path=video_path,
        )

    def add_target(self, target: Target) -> None:
        """Add another moving target to the simulation environment."""
        self.targets.append(target)

    def set_primary_target(self, target_id: Union[str, int]) -> bool:
        """Designate which target the camera actively tracks."""
        for t in self.targets:
            if t.target_id == target_id:
                self.primary_target_id = target_id
                self.target = t
                return True
        return False

    def get_all_target_states(self) -> List[TargetState]:
        """Return the current ground-truth kinematic states for all targets."""
        return [t.get_state() for t in self.targets]

    def add_clutter(self, clutter: ClutterObject) -> None:
        """Add a clutter object to the environment."""
        self.clutter_objects.append(clutter)

    def add_occluder(self, occluder: DynamicOccluder) -> None:
        """Add an occluder object to the environment."""
        self.occluders.append(occluder)

    def step(
        self,
        dt: float,
        control_pan_delta: float = 0.0,
        control_tilt_delta: float = 0.0,
    ) -> Tuple[FrameData, TargetState, CameraState]:
        """
        Advance simulation by one time step:
        1. Apply commanded gimbal adjustments.
        2. Advance target kinematics and blinking modulation.
        3. Apply platform vibration to apparent camera boresight.
        4. Render optical sensor frame with beacon and clutter.
        5. Apply atmospheric turbulence (scintillation + PSF blur).
        6. Apply dynamic occluders (line-of-sight blockage).
        7. Apply sensor readout and impulse noise.
        """
        self.sim_time += dt

        # 1. Update true camera gimbal pose
        self.camera.apply_control(control_pan_delta, control_tilt_delta, dt=dt)

        if self.frame_source == "video_file" and self.video_source is not None:
            t_render0 = time.perf_counter_ns()
            vframe = self.video_source.get_frame(self.frame_count)
            t_render1 = time.perf_counter_ns()
            if vframe is not None:
                frame = vframe
                self.sim_time = frame.timestamp
            else:
                self.sim_time += dt
                w, h = self.camera.state.resolution
                frame = FrameData(
                    image=np.zeros((h, w), dtype=np.uint8),
                    timestamp=self.sim_time,
                    frame_id=self.frame_count,
                    ground_truth_target_pos=None,
                    ground_truth_targets=None,
                )
            self.frame_count += 1
            target_state = TargetState(
                x=0.0,
                y=0.0,
                vx=0.0,
                vy=0.0,
                confidence=0.0,
                timestamp=self.sim_time,
                tracker_mode="NO_GROUND_TRUTH",
                target_id="video_target",
            )
            self.last_step_profile = {
                "rendering_ms": (t_render1 - t_render0) * 1e-6,
                "disturb_turbulence_ms": 0.0,
                "disturb_atmospheric_ms": 0.0,
                "disturb_noise_ms": 0.0,
                "disturb_occlusion_ms": 0.0,
                "disturb_vibration_ms": 0.0,
                "disturbances_total_ms": 0.0,
            }
            return frame, None, self.camera.state

        # 2. Advance target kinematics for all targets
        for tgt in self.targets:
            tgt.step(dt)
        primary_target_state = self.target.get_state()

        # 3. Platform base motion & high-frequency vibration offset
        plat_pan, plat_tilt = 0.0, 0.0
        if self.platform_motion is not None:
            plat_pan, plat_tilt = self.platform_motion.get_position(self.sim_time)

        vib_pan, vib_tilt = 0.0, 0.0
        t_vib0 = time.perf_counter_ns()
        if self.vibration is not None:
            vib_pan, vib_tilt = self.vibration.step(self.sim_time, dt=dt)
        t_vib1 = time.perf_counter_ns()

        total_pan_jitter = vib_pan + plat_pan
        total_tilt_jitter = vib_tilt + plat_tilt

        # Temporarily offset camera pointing by vibration & platform motion for optical rendering
        actual_pan = self.camera.state.pan
        actual_tilt = self.camera.state.tilt
        self.camera.state.pan += total_pan_jitter
        self.camera.state.tilt += total_tilt_jitter

        # 4. Render raw optical frame with all targets
        t_render0 = time.perf_counter_ns()
        frame = self.camera.render_frame(
            target_state=primary_target_state,
            beacon_intensity=self.target.current_intensity,
            clutter_objects=self.clutter_objects,
            frame_id=self.frame_count,
            targets=self.targets,
        )
        t_render1 = time.perf_counter_ns()

        # Restore camera true gimbal angles (vibration & platform motion are apparent pointing offsets)
        self.camera.state.pan = actual_pan
        self.camera.state.tilt = actual_tilt

        raw_img = frame.image

        # 5. Atmospheric Turbulence (Kolmogorov phase screen, scintillation, PSF blur)
        t_turb0 = time.perf_counter_ns()
        if self.turbulence is not None:
            raw_img = self.turbulence.apply_turbulence(
                raw_img, target_pos_px=frame.ground_truth_target_pos
            )
        t_turb1 = time.perf_counter_ns()

        # 5b. Atmospheric Weather Condition (Haze, Fog, Rain, Low-light contrast/brightness)
        t_atm0 = time.perf_counter_ns()
        if self.atmospheric_dist is not None:
            raw_img = self.atmospheric_dist.apply(raw_img)
        t_atm1 = time.perf_counter_ns()

        # 6. Dynamic Occluders (block line-of-sight)
        t_occ0 = time.perf_counter_ns()
        if self.occluders:
            for occ in self.occluders:
                occ.step(dt)
                raw_img, _ = occ.apply_to_frame(raw_img, self.camera.state)
        t_occ1 = time.perf_counter_ns()

        # 7. Sensor Readout & Shot Noise
        t_noise0 = time.perf_counter_ns()
        if self.sensor_noise is not None:
            raw_img = self.sensor_noise.apply_noise(raw_img)
        t_noise1 = time.perf_counter_ns()

        frame.image = raw_img
        self.frame_count += 1

        self.last_step_profile = {
            "rendering_ms": (t_render1 - t_render0) * 1e-6,
            "disturb_turbulence_ms": (t_turb1 - t_turb0) * 1e-6,
            "disturb_atmospheric_ms": (t_atm1 - t_atm0) * 1e-6,
            "disturb_noise_ms": (t_noise1 - t_noise0) * 1e-6,
            "disturb_occlusion_ms": (t_occ1 - t_occ0) * 1e-6,
            "disturb_vibration_ms": (t_vib1 - t_vib0) * 1e-6,
            "disturbances_total_ms": (
                (t_turb1 - t_turb0)
                + (t_atm1 - t_atm0)
                + (t_noise1 - t_noise0)
                + (t_occ1 - t_occ0)
                + (t_vib1 - t_vib0)
            ) * 1e-6,
        }

        return frame, primary_target_state, self.camera.state

    def get_angular_tracking_error(
        self,
        target_id: Optional[Union[str, int]] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate angular error between target world position and camera boresight.
        Returns (None, None, None) in video_file mode or when no ground truth target is present.
        """
        if self.frame_source == "video_file" or not self.targets:
            return None, None, None

        if target_id is not None:
            matched = next((t for t in self.targets if t.target_id == target_id), self.target)
            tgt = matched.get_state()
        else:
            tgt = self.target.get_state()

        cam = self.camera.state
        err_pan = tgt.x - cam.pan
        err_tilt = tgt.y - cam.tilt
        radial_err = float(np.hypot(err_pan, err_tilt))
        return err_pan, err_tilt, radial_err
