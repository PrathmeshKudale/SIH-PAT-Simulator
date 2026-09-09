"""
Unit tests for sim module: Target, Camera, and Environment.
"""

import unittest
import numpy as np
from sim.target import Target
from sim.camera import Camera
from sim.environment import Environment
from contracts import TargetState, CameraState, FrameData


class TestSim(unittest.TestCase):

    def test_target_kinematics_linear(self):
        target = Target(initial_pos=(0.010, -0.005), velocity=(0.002, 0.001))
        self.assertAlmostEqual(target.x, 0.010)
        self.assertAlmostEqual(target.y, -0.005)

        state = target.step(dt=0.5)
        self.assertAlmostEqual(state.x, 0.011)  # 0.010 + 0.002 * 0.5
        self.assertAlmostEqual(state.y, -0.0045) # -0.005 + 0.001 * 0.5
        self.assertAlmostEqual(state.timestamp, 0.5)

    def test_camera_world_to_pixel_center(self):
        camera = Camera(pan=0.0, tilt=0.0, fov_x=0.10, fov_y=0.075, resolution=(640, 480))
        # Target directly on boresight
        px = camera.world_to_pixel(0.0, 0.0)
        self.assertIsNotNone(px)
        self.assertAlmostEqual(px[0], 320.0)
        self.assertAlmostEqual(px[1], 240.0)

    def test_camera_outside_fov(self):
        camera = Camera(pan=0.0, tilt=0.0, fov_x=0.10, fov_y=0.075, resolution=(640, 480))
        # Point far outside FOV (0.2 rad >> 0.05 rad half-FOV)
        px = camera.world_to_pixel(0.20, 0.0)
        self.assertIsNone(px)

    def test_camera_frame_rendering(self):
        camera = Camera(pan=0.0, tilt=0.0, resolution=(640, 480))
        target_state = TargetState(x=0.0, y=0.0, vx=0.0, vy=0.0, timestamp=0.1)
        frame = camera.render_frame(target_state, beacon_intensity=200.0, frame_id=1)

        self.assertIsInstance(frame, FrameData)
        self.assertEqual(frame.image.shape, (480, 640))
        self.assertEqual(frame.image.dtype, np.uint8)
        self.assertIsNotNone(frame.ground_truth_target_pos)
        # Center should be bright (higher than dark background)
        self.assertGreater(frame.image[240, 320], 100)

    def test_environment_step(self):
        env = Environment(
            target=Target(initial_pos=(0.01, 0.01), velocity=(0.001, 0.0)),
            camera=Camera(pan=0.0, tilt=0.0),
        )
        frame, tgt_state, cam_state = env.step(dt=0.1, control_pan_delta=0.005, control_tilt_delta=0.0)
        self.assertAlmostEqual(cam_state.pan, 0.005)
        self.assertAlmostEqual(tgt_state.x, 0.0101)

    def test_target_range_intensity_scaling(self):
        """Verify target apparent intensity scales inversely with square of range (R_ref / R)^2."""
        # Reference range is 5 km, base intensity is 200.0
        target_ref = Target(base_intensity=200.0, range_km=5.0, ref_range_km=5.0, blink_frequency=0.0)
        self.assertAlmostEqual(target_ref.current_intensity, 200.0)

        # At 10 km (2x distance), intensity should be 1/4 = 50.0
        target_far = Target(base_intensity=200.0, range_km=10.0, ref_range_km=5.0, blink_frequency=0.0)
        self.assertAlmostEqual(target_far.current_intensity, 50.0)

        # At 2.5 km (0.5x distance), intensity should be 4x = 800.0
        target_near = Target(base_intensity=200.0, range_km=2.5, ref_range_km=5.0, blink_frequency=0.0)
        self.assertAlmostEqual(target_near.current_intensity, 800.0)

    def test_auto_exposure_controller_convergence(self):
        """Verify AutoExposureController adapts gain dynamically to achieve target spot peak."""
        from sim.camera import AutoExposureController
        aec = AutoExposureController(target_peak=200.0, initial_gain=1.0, adaptation_rate=0.4)

        # Dim spot (measured peak = 20.0, 10x too dim)
        for _ in range(15):
            aec.update(20.0 * aec.gain)
        # Gain should have scaled up close to 10.0 so effective peak is ~200
        self.assertAlmostEqual(aec.gain, 10.0, delta=0.5)

        # Reset and test bright spot (measured peak = 1000.0, 5x too bright)
        aec.reset(initial_gain=1.0)
        for _ in range(15):
            aec.update(1000.0 * aec.gain)
        # Gain should have scaled down close to 0.20
        self.assertAlmostEqual(aec.gain, 0.20, delta=0.05)


if __name__ == "__main__":
    unittest.main()
