"""
Comprehensive PS Specification Verification Tests.
Smart India Hackathon - Problem Statement 26169 (ISRO / Department of Space)

Validates all PS-mandated physical and kinematic parameters:
1. Screen size >= 2000x2000 px support (e.g., 2000x2000, 2048x2048).
2. Initial camera position: boresight (0, 0) rad corresponds identically to screen center (W/2, H/2).
3. Target shape/size defaults: square/circle/Gaussian, 10x10 px, and range 5 to 20 px.
4. Max pan/tilt speed: slew rate limiter bounds speed to 5 to 10 deg/s.
5. Re-acquisition time: measured recovery under dynamic line-of-sight occlusion.
6. Acquisition time: measured initial convergence time to boresight lock (<= 2.0s).
7. Steady-state tracking error and RMSE <= 10 pixels.
"""

import unittest
import numpy as np
from sim.camera import Camera
from contracts import TargetState, CameraState
from detect.detector import AdaptiveOpticalDetector
from control.slew import SlewRateLimiter
from metrics.batch_runner import BatchScenarioRunner
from metrics.calculator import MetricsCalculator


class TestPSSpecVerification(unittest.TestCase):
    """Rigorous verification test suite for PS-specific operational criteria."""

    def test_screen_size_ge_2000x2000(self):
        """Verify screen size >= 2000x2000 px renders and detects properly."""
        resolutions = [(2000, 2000), (2048, 2048)]
        for res in resolutions:
            cam = Camera(pan=0.0, tilt=0.0, resolution=res, fov_x=0.10, fov_y=0.10)
            self.assertEqual(cam.resolution, res)

            # Render frame with target at boresight
            tgt = TargetState(x=0.0, y=0.0, vx=0.0, vy=0.0, confidence=1.0, timestamp=0.0)
            frame = cam.render_frame(target_state=tgt, beacon_intensity=220.0, frame_id=0)
            self.assertEqual(frame.image.shape, (res[1], res[0]))

            # Run detection
            detector = AdaptiveOpticalDetector(enable_signature_verification=False)
            det_tgt, report = detector.detect(frame, camera_state=cam.state)
            self.assertIsNotNone(det_tgt, f"Detection failed on resolution {res}")
            # Detected centroid should be near screen center
            self.assertAlmostEqual(det_tgt.x, res[0] / 2.0, delta=2.0)
            self.assertAlmostEqual(det_tgt.y, res[1] / 2.0, delta=2.0)

    def test_initial_camera_position_screen_center(self):
        """Verify initial boresight (pan=0, tilt=0) maps identically to center of screen."""
        for res in [(640, 480), (1920, 1080), (2000, 2000), (2048, 2048)]:
            cam = Camera(pan=0.0, tilt=0.0, resolution=res)
            center_px = cam.world_to_pixel(0.0, 0.0)
            expected_center = (res[0] / 2.0, res[1] / 2.0)
            self.assertEqual(center_px, expected_center,
                             f"Boresight at res {res} did not map to center {expected_center}")

    def test_target_shape_and_size_defaults(self):
        """Verify target shape (square, circle, gaussian) and sizes (5, 10, 20 px)."""
        detector = AdaptiveOpticalDetector(enable_signature_verification=False)
        for shape in ["square", "circle", "gaussian"]:
            for size in [5.0, 10.0, 20.0]:
                cam = Camera(
                    pan=0.0,
                    tilt=0.0,
                    resolution=(640, 480),
                    target_shape=shape,
                    target_size_px=size,
                )
                tgt = TargetState(x=0.001, y=0.001, vx=0.0, vy=0.0, confidence=1.0, timestamp=0.0)
                frame = cam.render_frame(target_state=tgt, beacon_intensity=220.0, frame_id=0)
                det_tgt, _ = detector.detect(frame, camera_state=cam.state)
                self.assertIsNotNone(det_tgt, f"Detection failed for shape={shape}, size={size}")

    def test_slew_rate_limiter_ps_defaults_5_to_10_deg_s(self):
        """Verify slew rate limiter strictly bounds speed to 5 to 10 deg/s."""
        # 1. Test 5 deg/s
        limiter_5 = SlewRateLimiter(max_velocity_deg_s=5.0, max_acceleration_deg_s2=20.0)
        self.assertAlmostEqual(limiter_5.max_velocity_deg_s, 5.0, places=4)
        # Command an abrupt 45 deg step over 0.1s (requested rate 450 deg/s)
        dt = 0.1
        cmd_rad = np.deg2rad(45.0)
        actual_pan, _ = limiter_5.apply_limit(cmd_rad, 0.0, dt=dt)
        actual_speed_deg_s = np.rad2deg(actual_pan) / dt
        self.assertLessEqual(actual_speed_deg_s, 5.0001, "Slew rate exceeded 5.0 deg/s")

        # 2. Test 10 deg/s
        limiter_10 = SlewRateLimiter(max_velocity_deg_s=10.0, max_acceleration_deg_s2=50.0)
        self.assertAlmostEqual(limiter_10.max_velocity_deg_s, 10.0, places=4)
        actual_pan_10, _ = limiter_10.apply_limit(cmd_rad, 0.0, dt=dt)
        actual_speed_10_deg_s = np.rad2deg(actual_pan_10) / dt
        self.assertLessEqual(actual_speed_10_deg_s, 10.0001, "Slew rate exceeded 10.0 deg/s")

    def test_acquisition_time_under_2_seconds(self):
        """Verify acquisition time <= 2.0s on an off-boresight target scenario."""
        runner = BatchScenarioRunner(config_path="config/demo_scenarios.json")
        scenarios = runner.load_scenarios()
        res = runner.run_scenario(scenarios[0])  # DEMO_ACQUISITION_AND_TRACK
        self.assertGreater(res.acquisition_time_s, 0.0, "Acquisition time must be non-zero for off-boresight target")
        self.assertLessEqual(res.acquisition_time_s, 2.0, f"Acquisition time {res.acquisition_time_s}s exceeded 2.0s")

    def test_reacquisition_time_measured(self):
        """Verify re-acquisition time is measured and logged on dynamic occlusion recovery."""
        runner = BatchScenarioRunner(config_path="config/demo_scenarios.json")
        scenarios = runner.load_scenarios()
        res = runner.run_scenario(scenarios[2])  # DEMO_DYNAMIC_OCCLUSION_RECOVERY
        self.assertGreater(res.reacquisition_time_s, 0.0, "Reacquisition time must be measured on occlusion scenario")
        self.assertLessEqual(res.reacquisition_time_s, 1.0, f"Reacquisition time {res.reacquisition_time_s}s took too long")

    def test_steady_state_tracking_error_and_rmse_under_10_pixels(self):
        """Verify steady-state tracking error and steady-state RMSE are <= 10 pixels."""
        runner = BatchScenarioRunner(config_path="config/demo_scenarios.json")
        scenarios = runner.load_scenarios()
        res = runner.run_scenario(scenarios[0])  # DEMO_ACQUISITION_AND_TRACK
        self.assertLessEqual(res.steady_tracking_error_px, 10.0,
                             f"Steady tracking error {res.steady_tracking_error_px} px exceeded 10 px threshold")
        self.assertLessEqual(res.steady_rmse_px, 10.0,
                             f"Steady RMSE {res.steady_rmse_px} px exceeded 10 px threshold")


if __name__ == "__main__":
    unittest.main()
