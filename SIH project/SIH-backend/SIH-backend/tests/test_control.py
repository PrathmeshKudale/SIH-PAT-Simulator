"""
Unit tests for control module: ProportionalController and SpiralSearchController.
"""

import unittest
import numpy as np
from control.pid import ProportionalController
from control.search import SpiralSearchController
from contracts import CameraState, TargetState
from sim.target import Target
from sim.camera import Camera
from sim.environment import Environment
from detect.detector import AdaptiveOpticalDetector


class TestControl(unittest.TestCase):

    def test_proportional_controller_calculation(self):
        cam_state = CameraState(
            pan=0.0,
            tilt=0.0,
            fov_x=0.10,
            fov_y=0.10,
            resolution=(640, 480),
        )
        detected = TargetState(x=320 + 64, y=240, confidence=1.0)
        controller = ProportionalController(kp=0.8)

        d_pan, d_tilt = controller.compute_command(detected, cam_state)
        self.assertAlmostEqual(d_pan, 0.008, places=5)
        self.assertAlmostEqual(d_tilt, 0.0, places=5)

    def test_spiral_search_expansion(self):
        """Spiral search pattern should expand outward over time."""
        search = SpiralSearchController(
            fov_x=0.10,
            fov_y=0.075,
            linear_scan_rate=0.06,
            center_pan=0.0,
            center_tilt=0.0,
        )
        radii = []
        cam = CameraState(pan=0.0, tilt=0.0)
        for _ in range(20):
            d_pan, d_tilt = search.step(dt=0.033, current_camera_state=cam)
            cam.pan += d_pan
            cam.tilt += d_tilt
            r = np.hypot(cam.pan, cam.tilt)
            radii.append(r)

        # Radius should monotonically expand outward initially
        self.assertGreater(radii[-1], radii[0])
        self.assertGreater(radii[-1], 0.010)

    def test_tracking_error_stays_bounded(self):
        """
        Asserts that the camera's tracking error stays strictly bounded and does not diverge.
        """
        target = Target(
            initial_pos=(0.012, -0.008),
            velocity=(0.004, -0.002),
            base_intensity=220.0,
            blink_frequency=0.0,
            modulation_depth=0.0,
        )
        camera = Camera(
            pan=0.0,
            tilt=0.0,
            fov_x=0.10,
            fov_y=0.075,
            resolution=(640, 480),
        )
        env = Environment(target=target, camera=camera)
        detector = AdaptiveOpticalDetector(enable_signature_verification=False)
        controller = ProportionalController(kp=0.85)

        num_frames = 60
        dt = 0.033
        control_pan_delta = 0.0
        control_tilt_delta = 0.0
        radial_errors = []

        for _ in range(num_frames):
            frame, true_tgt, cam_state = env.step(
                dt=dt,
                control_pan_delta=control_pan_delta,
                control_tilt_delta=control_tilt_delta,
            )
            detected, _ = detector.detect(frame)
            self.assertIsNotNone(detected, "Optical beacon was lost from FOV during tracking run!")

            _, _, rad_err = env.get_angular_tracking_error()
            radial_errors.append(rad_err)

            control_pan_delta, control_tilt_delta = controller.compute_command(detected, cam_state)

        half_fov = camera.state.fov_x / 2.0
        max_error = max(radial_errors)
        self.assertLess(max_error, half_fov)

        initial_error = radial_errors[0]
        final_error = radial_errors[-1]
        self.assertLess(final_error, initial_error * 0.1)

    def test_slew_rate_limiter_velocity_and_accel_caps(self):
        """Verify SlewRateLimiter enforces both max velocity and max acceleration limits."""
        from control.slew import SlewRateLimiter
        limiter = SlewRateLimiter(
            max_velocity=0.5,      # 0.5 rad/s
            max_acceleration=1.0,  # 1.0 rad/s^2
        )
        dt = 0.033  # 33 ms

        # Send an enormous instantaneous step command (1.0 rad)
        d_pan, d_tilt = limiter.apply_limit(1.0, -1.0, dt=dt)

        # In one frame (dt = 0.033), max change in velocity from 0 is a_max * dt = 0.033 rad/s
        # Max angle delta is therefore <= 0.033 * dt = 0.001089 rad
        self.assertLessEqual(abs(d_pan / dt), limiter.max_velocity)
        self.assertLessEqual(abs(d_tilt / dt), limiter.max_velocity)
        self.assertLessEqual(abs(d_pan), 0.033 * dt + 1e-6)

        # After several steps, velocity can ramp up toward max_velocity, but never exceed it
        for _ in range(50):
            d_pan, d_tilt = limiter.apply_limit(1.0, -1.0, dt=dt)
            self.assertLessEqual(abs(d_pan / dt), limiter.max_velocity + 1e-6)
            self.assertLessEqual(abs(d_tilt / dt), limiter.max_velocity + 1e-6)

    def test_control_delay_queue(self):
        """Verify ControlDelayQueue delays commands by exactly N frames."""
        from control.latency import ControlDelayQueue
        queue = ControlDelayQueue(delay_frames=2)
        self.assertEqual(queue.delay_frames, 2)

        # Frame 0: push (0.01, 0.02), should return (0.0, 0.0)
        c0 = queue.step(0.01, 0.02)
        self.assertEqual(c0, (0.0, 0.0))

        # Frame 1: push (0.03, 0.04), should return (0.0, 0.0)
        c1 = queue.step(0.03, 0.04)
        self.assertEqual(c1, (0.0, 0.0))

        # Frame 2: push (0.05, 0.06), should return frame 0's command (0.01, 0.02)
        c2 = queue.step(0.05, 0.06)
        self.assertAlmostEqual(c2[0], 0.01)
        self.assertAlmostEqual(c2[1], 0.02)

        # Frame 3: push (0.07, 0.08), should return frame 1's command (0.03, 0.04)
        c3 = queue.step(0.07, 0.08)
        self.assertAlmostEqual(c3[0], 0.03)
        self.assertAlmostEqual(c3[1], 0.04)

    def test_pid_controller_feedforward(self):
        """Verify that velocity feedforward produces additional lead in the direction of motion."""
        from control.pid import PIDController
        pid_no_ff = PIDController(kp=0.8, ki=0.0, kd=0.0, enable_feedforward=False)
        pid_with_ff = PIDController(
            kp=0.8, ki=0.0, kd=0.0,
            enable_feedforward=True,
            k_ff=1.0,
            latency_compensation_frames=2,
        )

        cam_state = CameraState(pan=0.0, tilt=0.0)
        tgt = TargetState(x=0.005, y=0.0, vx=0.015, vy=0.0)

        d_pan_no_ff, _ = pid_no_ff.compute_command(
            target_state=tgt,
            camera_state=cam_state,
            dt=0.033,
        )
        d_pan_with_ff, _ = pid_with_ff.compute_command(
            target_state=tgt,
            camera_state=cam_state,
            dt=0.033,
        )

        # Feedforward command should be strictly larger in the direction of motion
        self.assertGreater(d_pan_with_ff, d_pan_no_ff)
        # Difference should match lead: v * dt * (1 + latency_compensation_frames)
        expected_ff = 1.0 * 0.015 * 0.033 * (1.0 + 2)
        self.assertAlmostEqual(d_pan_with_ff - d_pan_no_ff, expected_ff, places=5)


if __name__ == "__main__":
    unittest.main()
