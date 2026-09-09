"""
Unit tests for data contracts (contracts.py).
Verifies instantiation, field types, default values, and serialization.
"""

import unittest
import numpy as np
from contracts import (
    FrameData,
    TargetState,
    CameraState,
    DisturbanceConfig,
    MetricsRecord,
)


class TestContracts(unittest.TestCase):

    def test_frame_data_creation_and_to_dict(self):
        img = np.zeros((480, 640), dtype=np.uint8)
        frame = FrameData(
            image=img,
            timestamp=0.033,
            frame_id=1,
            ground_truth_target_pos=(320.0, 240.0),
        )
        self.assertEqual(frame.frame_id, 1)
        self.assertAlmostEqual(frame.timestamp, 0.033)
        self.assertEqual(frame.ground_truth_target_pos, (320.0, 240.0))
        self.assertEqual(frame.image.shape, (480, 640))

        d = frame.to_dict()
        self.assertEqual(d["frame_id"], 1)
        self.assertEqual(d["image_shape"], [480, 640])
        self.assertEqual(d["ground_truth_target_pos"], (320.0, 240.0))

    def test_target_state_creation_and_to_dict(self):
        state = TargetState(
            x=150.5,
            y=200.25,
            vx=12.0,
            vy=-4.5,
            confidence=0.95,
            timestamp=1.5,
            tracker_mode="KF",
        )
        self.assertAlmostEqual(state.x, 150.5)
        self.assertAlmostEqual(state.y, 200.25)
        self.assertAlmostEqual(state.vx, 12.0)
        self.assertAlmostEqual(state.vy, -4.5)
        self.assertAlmostEqual(state.confidence, 0.95)
        self.assertEqual(state.tracker_mode, "KF")

        d = state.to_dict()
        self.assertEqual(d["x"], 150.5)
        self.assertEqual(d["confidence"], 0.95)
        self.assertEqual(d["tracker_mode"], "KF")

    def test_camera_state_creation_and_fov(self):
        cam = CameraState(
            pan=0.5,
            tilt=-0.2,
            pan_rate=0.01,
            tilt_rate=0.0,
            fov_x=0.1,
            fov_y=0.1,
            focal_length=100.0,
            resolution=(640, 480),
        )
        self.assertAlmostEqual(cam.pan, 0.5)
        self.assertAlmostEqual(cam.tilt, -0.2)

        bounds = cam.fov_bounds
        # pan +- 0.05 -> (0.45, 0.55), tilt +- 0.05 -> (-0.25, -0.15)
        self.assertAlmostEqual(bounds[0], 0.45)
        self.assertAlmostEqual(bounds[1], 0.55)
        self.assertAlmostEqual(bounds[2], -0.25)
        self.assertAlmostEqual(bounds[3], -0.15)

        d = cam.to_dict()
        self.assertIn("fov_bounds", d)
        self.assertEqual(d["resolution"], (640, 480))

    def test_disturbance_config(self):
        cfg = DisturbanceConfig(
            cn2=1e-13,
            vibration_amplitude=1.2,
            vibration_frequency=25.0,
            noise_level=8.0,
            occluder_frequency=0.1,
            occluder_size=45.0,
            random_walk_jitter=0.02,
        )
        self.assertEqual(cfg.cn2, 1e-13)
        self.assertEqual(cfg.vibration_frequency, 25.0)

        d = cfg.to_dict()
        self.assertEqual(d["noise_level"], 8.0)
        self.assertEqual(d["occluder_size"], 45.0)

    def test_metrics_record(self):
        metrics = MetricsRecord(
            simulation_duration=30.0,
            fps=29.8,
            acquisition_time=0.45,
            avg_tracking_error=1.35,
            max_tracking_error=4.2,
            lock_retention_rate=0.98,
            per_frame_processing_time_ms=2.1,
            total_frames=900,
            track_loss_count=1,
            active_tracker_breakdown={"KF": 0.9, "PF": 0.1},
        )
        self.assertAlmostEqual(metrics.simulation_duration, 30.0)
        self.assertAlmostEqual(metrics.fps, 29.8)
        self.assertAlmostEqual(metrics.avg_tracking_error, 1.35)
        self.assertAlmostEqual(metrics.lock_retention_rate, 0.98)

        d = metrics.to_dict()
        self.assertEqual(d["total_frames"], 900)
        self.assertEqual(d["active_tracker_breakdown"]["KF"], 0.9)


if __name__ == "__main__":
    unittest.main()
