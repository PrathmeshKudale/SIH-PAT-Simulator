"""
Virtual Pan-Tilt Gimbal Camera model rendering synthetic sensor frames with target and clutter spots.
"""

from typing import Tuple, Optional, List, Any, Dict, Union
import numpy as np
from contracts import CameraState, FrameData, TargetState


class AutoExposureController:
    """
    Simulates optical sensor Auto-Exposure / Auto-Gain Control (AEC/AGC).
    Dynamically adjusts sensor exposure gain to keep optical beacon spots within
    an optimal dynamic range (setpoint ~180-220 counts) across large range variations.
    """

    def __init__(
        self,
        target_peak: float = 200.0,      # Desired target spot peak count [0, 255]
        min_gain: float = 0.02,          # Minimum exposure gain (for very close/bright targets)
        max_gain: float = 50.0,          # Maximum exposure gain (for very far/dim targets)
        adaptation_rate: float = 0.35,   # Proportional adaptation step per frame
        initial_gain: float = 1.0,
    ):
        self.target_peak = float(target_peak)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.adaptation_rate = float(adaptation_rate)
        self.gain = float(initial_gain)

    def reset(self, initial_gain: float = 1.0) -> None:
        """Reset sensor gain."""
        self.gain = float(initial_gain)

    def update(self, measured_peak: float) -> float:
        """
        Adjust sensor gain based on the peak intensity observed in the frame.
        Uses exponential proportional adjustment to smoothly converge without overshooting:
        gain_new = gain * (target_peak / measured_peak) ** adaptation_rate
        """
        if measured_peak > 1.0:
            target_ratio = self.target_peak / measured_peak
            self.gain = float(np.clip(self.gain * (target_ratio ** self.adaptation_rate), self.min_gain, self.max_gain))
        return self.gain


class Camera:
    """
    Simulates an optical sensor mounted on a 2-axis gimbal.
    Maps 2D world angular coordinates to focal plane pixel coordinates and
    synthesizes image frames with Gaussian PSF rendering for both beacon and clutter.
    """

    def __init__(
        self,
        pan: float = 0.0,
        tilt: float = 0.0,
        fov_x: float = 0.10,   # 100 mrad (~5.7 deg)
        fov_y: float = 0.075,  # 75 mrad (~4.3 deg)
        resolution: Tuple[int, int] = (640, 480),
        psf_sigma: float = 2.5,
        bg_noise_mean: float = 12.0,
        bg_noise_std: float = 2.0,
        enable_auto_exposure: bool = False,
        aec_target_peak: float = 200.0,
        target_shape: str = "gaussian",  # "gaussian", "square", "circle" (PS default: square/circle 10x10, range 5-20 px)
        target_size_px: float = 10.0,    # Target beacon spot size in pixels (PS default: 10, range: 5 to 20)
    ):
        self.state = CameraState(
            pan=float(pan),
            tilt=float(tilt),
            fov_x=float(fov_x),
            fov_y=float(fov_y),
            resolution=resolution,
        )
        self.psf_sigma = float(psf_sigma)
        self.bg_noise_mean = float(bg_noise_mean)
        self.bg_noise_std = float(bg_noise_std)
        self.enable_auto_exposure = bool(enable_auto_exposure)
        self.aec = AutoExposureController(target_peak=aec_target_peak) if enable_auto_exposure else None
        self.target_shape = str(target_shape)
        self.target_size_px = float(target_size_px)
        self._rng = np.random.default_rng(seed=42)

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.state.resolution

    def world_to_pixel(self, target_x: float, target_y: float) -> Optional[Tuple[float, float]]:
        """
        Convert target world angular coordinates (rad) to focal plane pixel coordinates (u, v).
        Returns None if target is outside the camera's FOV.
        """
        delta_pan = target_x - self.state.pan
        delta_tilt = target_y - self.state.tilt

        # Check FOV limits
        if abs(delta_pan) > (self.state.fov_x / 2.0) or abs(delta_tilt) > (self.state.fov_y / 2.0):
            return None

        w, h = self.state.resolution
        cx = w / 2.0
        cy = h / 2.0

        u = cx + delta_pan * (w / self.state.fov_x)
        v = cy + delta_tilt * (h / self.state.fov_y)

        # Check boundary edge
        if 0 <= u < w and 0 <= v < h:
            return (u, v)
        return None

    def pixel_to_angular_error(self, u: float, v: float) -> Tuple[float, float]:
        """
        Convert pixel coordinate (u, v) on sensor to angular error relative to camera boresight.

        Returns:
            (error_pan, error_tilt) in radians.
        """
        w, h = self.state.resolution
        cx = w / 2.0
        cy = h / 2.0
        error_pan = (u - cx) * (self.state.fov_x / w)
        error_tilt = (v - cy) * (self.state.fov_y / h)
        return (error_pan, error_tilt)

    def _render_spot(
        self,
        image: np.ndarray,
        u_t: float,
        v_t: float,
        intensity: float,
        shape: Optional[str] = None,
        size_px: Optional[float] = None,
    ) -> None:
        """
        Render an optical beacon or clutter spot centered at (u_t, v_t) onto the image.
        Supports Gaussian PSF, square, or circular spots (PS default: 10x10, range 5-20 px).
        """
        w, h = self.state.resolution
        spot_shape = (shape or self.target_shape).lower()
        spot_size = float(size_px if size_px is not None else self.target_size_px)
        spot_size = float(np.clip(spot_size, 5.0, 20.0))

        if spot_shape == "square":
            half_s = spot_size / 2.0
            r_int = int(np.ceil(half_s + 2))
            u_min = max(0, int(np.floor(u_t - r_int)))
            u_max = min(w, int(np.ceil(u_t + r_int + 1)))
            v_min = max(0, int(np.floor(v_t - r_int)))
            v_max = min(h, int(np.ceil(v_t + r_int + 1)))
            if u_max > u_min and v_max > v_min:
                grid_x, grid_y = np.meshgrid(np.arange(u_min, u_max), np.arange(v_min, v_max))
                dx = np.abs(grid_x - u_t)
                dy = np.abs(grid_y - v_t)
                dist = np.maximum(dx, dy)
                profile = np.exp(-0.5 * (np.maximum(0.0, dist - half_s * 0.4) / (spot_size * 0.25)) ** 2) * intensity
                image[v_min:v_max, u_min:u_max] += profile
        elif spot_shape == "circle":
            sigma = spot_size / 4.0
            radius = int(np.ceil(4 * sigma))
            u_min = max(0, int(np.floor(u_t - radius)))
            u_max = min(w, int(np.ceil(u_t + radius + 1)))
            v_min = max(0, int(np.floor(v_t - radius)))
            v_max = min(h, int(np.ceil(v_t + radius + 1)))
            if u_max > u_min and v_max > v_min:
                grid_x, grid_y = np.meshgrid(np.arange(u_min, u_max), np.arange(v_min, v_max))
                dist_sq = (grid_x - u_t) ** 2 + (grid_y - v_t) ** 2
                profile = intensity * np.exp(-dist_sq / (2.0 * sigma ** 2))
                image[v_min:v_max, u_min:u_max] += profile
        else:
            # Gaussian PSF: equivalent radius based on spot_size (sigma = spot_size / 4.0)
            sigma = (spot_size / 4.0) if spot_size > 0 else self.psf_sigma
            radius = int(np.ceil(4 * sigma))
            u_min = max(0, int(np.floor(u_t - radius)))
            u_max = min(w, int(np.ceil(u_t + radius + 1)))
            v_min = max(0, int(np.floor(v_t - radius)))
            v_max = min(h, int(np.ceil(v_t + radius + 1)))
            if u_max > u_min and v_max > v_min:
                grid_x, grid_y = np.meshgrid(np.arange(u_min, u_max), np.arange(v_min, v_max))
                dist_sq = (grid_x - u_t) ** 2 + (grid_y - v_t) ** 2
                blob = intensity * np.exp(-dist_sq / (2.0 * (sigma ** 2)))
                image[v_min:v_max, u_min:u_max] += blob

    def render_frame(
        self,
        target_state: Optional[TargetState] = None,
        beacon_intensity: float = 220.0,
        clutter_objects: Optional[List[Any]] = None,
        frame_id: int = 0,
        targets: Optional[List[Any]] = None,
    ) -> FrameData:
        """
        Synthesize an optical sensor image frame with target beacon(s) and optional clutter objects.

        Args:
            target_state: True kinematic state of the primary target in world coordinates.
            beacon_intensity: Modulated intensity of the optical beacon (used if single target_state).
            clutter_objects: Optional list of ClutterObject instances in the scene.
            frame_id: Monotonically increasing frame index.
            targets: Optional list of Target instances or (TargetState, intensity) tuples for multi-target rendering.

        Returns:
            FrameData containing synthesized image and metadata.
        """
        w, h = self.state.resolution

        # Baseline dark sensor frame with background noise
        noise = self._rng.normal(self.bg_noise_mean, self.bg_noise_std, (h, w))
        image = np.clip(noise, 0, 255).astype(np.float32)

        # Apply auto-exposure / auto-gain if enabled
        gain = self.aec.gain if (self.enable_auto_exposure and self.aec) else 1.0

        # Render clutter objects if visible
        if clutter_objects:
            for clutter in clutter_objects:
                c_pixel = self.world_to_pixel(clutter.x, clutter.y)
                if c_pixel is not None:
                    self._render_spot(image, c_pixel[0], c_pixel[1], clutter.current_intensity * gain)

        ground_truth_targets: Dict[Union[str, int], Tuple[float, float]] = {}
        primary_pixel: Optional[Tuple[float, float]] = None
        frame_timestamp = 0.0

        # Render multi-target list if provided
        if targets:
            for tgt in targets:
                if hasattr(tgt, "get_state") and hasattr(tgt, "current_intensity"):
                    t_state = tgt.get_state()
                    t_intensity = tgt.current_intensity
                elif isinstance(tgt, tuple) and len(tgt) == 2:
                    t_state, t_intensity = tgt
                elif isinstance(tgt, TargetState):
                    t_state = tgt
                    t_intensity = beacon_intensity
                else:
                    continue

                eff_intensity = t_intensity * gain
                t_pixel = self.world_to_pixel(t_state.x, t_state.y)
                if t_pixel is not None:
                    ground_truth_targets[t_state.target_id] = t_pixel
                    if eff_intensity > 0:
                        self._render_spot(image, t_pixel[0], t_pixel[1], eff_intensity)

            # Designate primary target pixel
            if target_state is not None:
                primary_pixel = self.world_to_pixel(target_state.x, target_state.y)
                frame_timestamp = target_state.timestamp
            elif ground_truth_targets:
                first_tgt = targets[0]
                first_id = getattr(first_tgt, "target_id", None)
                if first_id is None and isinstance(first_tgt, tuple):
                    first_id = getattr(first_tgt[0], "target_id", "target_0")
                primary_pixel = ground_truth_targets.get(first_id or "target_0")
                frame_timestamp = getattr(targets[0], "timestamp", 0.0)

        elif target_state is not None:
            # Single-target fallback
            eff_beacon_intensity = beacon_intensity * gain
            target_pixel = self.world_to_pixel(target_state.x, target_state.y)
            if target_pixel is not None and eff_beacon_intensity > 0:
                self._render_spot(image, target_pixel[0], target_pixel[1], eff_beacon_intensity)
                ground_truth_targets[target_state.target_id] = target_pixel
            primary_pixel = target_pixel
            frame_timestamp = target_state.timestamp

        # Update auto-exposure based on frame peak intensity (measured before clipping)
        if self.enable_auto_exposure and self.aec:
            measured_peak = float(np.max(image))
            # Adapt only if peak is meaningfully above dark sensor noise floor
            if measured_peak > (self.bg_noise_mean + 3.0 * self.bg_noise_std):
                self.aec.update(measured_peak)

        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

        return FrameData(
            image=image_uint8,
            timestamp=frame_timestamp,
            frame_id=frame_id,
            ground_truth_target_pos=primary_pixel,
            ground_truth_targets=ground_truth_targets if ground_truth_targets else None,
        )

    def apply_control(self, delta_pan: float, delta_tilt: float, dt: float = 0.033) -> None:
        """Update camera pan and tilt based on control command."""
        self.state.pan += delta_pan
        self.state.tilt += delta_tilt
        if dt > 0:
            self.state.pan_rate = delta_pan / dt
            self.state.tilt_rate = delta_tilt / dt
