"""
Virtual 2D Simulation Environment coordinating target kinematics, camera optics,
clutter objects, and realistic physical disturbances.
"""

import time
from typing import Tuple, Optional, List, Any, Dict
import numpy as np
from contracts import FrameData, CameraState, TargetState, DisturbanceConfig
from sim.target import Target, ClutterObject
from sim.camera import Camera
from disturb.turbulence import KolmogorovTurbulence
from disturb.vibration import PlatformVibration
from disturb.noise import SensorNoise
from disturb.occluder import DynamicOccluder


class Environment:
    """
    Coordinates target kinematics, clutter, camera gimbal dynamics, and
    physical disturbances (turbulence, vibration, sensor noise, occluders).
    """

    def __init__(
        self,
        target: Optional[Target] = None,
        camera: Optional[Camera] = None,
        clutter_objects: Optional[List[ClutterObject]] = None,
        turbulence: Optional[KolmogorovTurbulence] = None,
        vibration: Optional[PlatformVibration] = None,
        sensor_noise: Optional[SensorNoise] = None,
        occluders: Optional[List[DynamicOccluder]] = None,
    ):
        self.target = target if target is not None else Target()
        self.camera = camera if camera is not None else Camera()
        self.clutter_objects = clutter_objects if clutter_objects is not None else []
        self.turbulence = turbulence
        self.vibration = vibration
        self.sensor_noise = sensor_noise
        self.occluders = occluders if occluders is not None else []

        self.frame_count = 0
        self.sim_time = 0.0
        self.last_step_profile: Dict[str, float] = {}

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

        # 2. Advance target kinematics
        target_state = self.target.step(dt)

        # 3. Platform vibration offset
        vib_pan, vib_tilt = 0.0, 0.0
        t_vib0 = time.perf_counter_ns()
        if self.vibration is not None:
            vib_pan, vib_tilt = self.vibration.step(self.sim_time, dt=dt)
        t_vib1 = time.perf_counter_ns()

        # Temporarily offset camera pointing by vibration jitter for optical rendering
        actual_pan = self.camera.state.pan
        actual_tilt = self.camera.state.tilt
        self.camera.state.pan += vib_pan
        self.camera.state.tilt += vib_tilt

        # 4. Render raw optical frame
        t_render0 = time.perf_counter_ns()
        frame = self.camera.render_frame(
            target_state=target_state,
            beacon_intensity=self.target.current_intensity,
            clutter_objects=self.clutter_objects,
            frame_id=self.frame_count,
        )
        t_render1 = time.perf_counter_ns()

        # Restore camera true gimbal angles (vibration is high-frequency apparent jitter)
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
            "disturb_noise_ms": (t_noise1 - t_noise0) * 1e-6,
            "disturb_occlusion_ms": (t_occ1 - t_occ0) * 1e-6,
            "disturb_vibration_ms": (t_vib1 - t_vib0) * 1e-6,
            "disturbances_total_ms": (
                (t_turb1 - t_turb0)
                + (t_noise1 - t_noise0)
                + (t_occ1 - t_occ0)
                + (t_vib1 - t_vib0)
            ) * 1e-6,
        }

        return frame, target_state, self.camera.state

    def get_angular_tracking_error(self) -> Tuple[float, float, float]:
        """Calculate angular error between target world position and camera boresight."""
        tgt = self.target.get_state()
        cam = self.camera.state
        err_pan = tgt.x - cam.pan
        err_tilt = tgt.y - cam.tilt
        radial_err = float(np.hypot(err_pan, err_tilt))
        return err_pan, err_tilt, radial_err
