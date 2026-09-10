"""
Unit and Integration Tests for FSOC PAT Video Frame Source (Phase V0 / V1).
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Tests:
1. VideoFrameSource decoding, FPS detection, dimension verification, and EOF behavior.
2. Conversion of multi-channel color video to detector-compatible 2D uint8 monochrome frames.
3. Environment stepping in 'video_file' mode with bypass of synthetic target rendering.
4. End-to-end closed-loop pipeline execution without ground truth (reporting N/A for error metrics).
"""

import os
import tempfile
import unittest
import numpy as np
import cv2

from contracts import FrameData, MetricsRecord
from sim.video_source import VideoFrameSource
from sim.camera import Camera
from sim.environment import Environment
from metrics.batch_runner import BatchScenarioRunner


def create_test_video(
    filepath: str,
    width: int = 640,
    height: int = 480,
    num_frames: int = 15,
    fps: float = 25.0,
    is_color: bool = True,
) -> str:
    """Helper to generate a short synthetic .mp4 video with a moving bright blob."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height), isColor=is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open cv2.VideoWriter for {filepath}")

    for i in range(num_frames):
        if is_color:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            # Add dark background tint
            img[:, :, 0] = 20
            # Moving bright green/white circle
            cx = int(width * 0.4 + i * (width * 0.2 / num_frames))
            cy = int(height * 0.5 + 20 * np.sin(i * 0.5))
            cv2.circle(img, (cx, cy), 12, (255, 255, 255), -1)
        else:
            img = np.zeros((height, width), dtype=np.uint8)
            cx = int(width * 0.4 + i * (width * 0.2 / num_frames))
            cy = int(height * 0.5 + 20 * np.sin(i * 0.5))
            cv2.circle(img, (cx, cy), 12, 255, -1)

        writer.write(img)

    writer.release()
    return filepath


class TestVideoFrameSource(unittest.TestCase):
    """Test suite for VideoFrameSource reading, formatting, and iterator compliance."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "test_target_blob.mp4")
        create_test_video(self.video_path, width=640, height=480, num_frames=15, fps=25.0, is_color=True)

    def tearDown(self):
        if os.path.exists(self.video_path):
            try:
                os.remove(self.video_path)
            except OSError:
                pass
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except OSError:
                pass

    def test_video_metadata_and_dimensions(self):
        """Confirm FPS, resolution, frame count, and non-assumed FPS detection."""
        source = VideoFrameSource(self.video_path)
        self.assertAlmostEqual(source.fps, 25.0, delta=1.0)
        self.assertEqual(source.resolution, (640, 480))
        self.assertEqual(source.total_frames, 15)
        source.close()

    def test_frame_decoding_and_format(self):
        """Confirm decoded frames are FrameData objects with 2D uint8 monochrome arrays."""
        source = VideoFrameSource(self.video_path)
        frame = source.get_frame(0)
        self.assertIsInstance(frame, FrameData)
        self.assertEqual(frame.frame_id, 0)
        self.assertEqual(frame.timestamp, 0.0)
        self.assertIsInstance(frame.image, np.ndarray)
        self.assertEqual(frame.image.ndim, 2)
        self.assertEqual(frame.image.dtype, np.uint8)
        self.assertEqual(frame.image.shape, (480, 640))

        # Check frame 1 timestamp derived from container FPS
        frame1 = source.get_frame(1)
        self.assertIsNotNone(frame1)
        self.assertEqual(frame1.frame_id, 1)
        expected_ts = 1.0 / source.fps
        self.assertAlmostEqual(frame1.timestamp, expected_ts, delta=1e-3)
        source.close()

    def test_end_of_video_handling(self):
        """Confirm get_frame returns None and iterator raises StopIteration past EOF."""
        source = VideoFrameSource(self.video_path)
        frames_read = 0
        for frame in source:
            self.assertIsInstance(frame, FrameData)
            frames_read += 1
        self.assertEqual(frames_read, 15)

        # Calling get_frame after EOF returns None cleanly without crashing
        eof_frame = source.get_frame(16)
        self.assertIsNone(eof_frame)

        # Calling next() raises StopIteration cleanly
        with self.assertRaises(StopIteration):
            next(source)

        # Resetting allows reading again from frame 0
        source.reset()
        rewound_frame = source.get_frame(0)
        self.assertIsNotNone(rewound_frame)
        self.assertEqual(rewound_frame.frame_id, 0)
        source.close()

    def test_missing_file_raises_error(self):
        """Confirm opening a non-existent video path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            VideoFrameSource("non_existent_file_xyz_123.mp4")


class TestVideoPipelineIntegration(unittest.TestCase):
    """Integration test suite for Environment and end-to-end pipeline in video_file mode."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "test_eval_run.mp4")
        create_test_video(self.video_path, width=640, height=480, num_frames=20, fps=30.0, is_color=True)

    def tearDown(self):
        if os.path.exists(self.video_path):
            try:
                os.remove(self.video_path)
            except OSError:
                pass
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except OSError:
                pass

    def test_environment_video_file_mode(self):
        """Verify Environment steps in video_file mode without Target or ground truth position."""
        vsource = VideoFrameSource(self.video_path)
        cam = Camera(resolution=(640, 480))
        env = Environment(
            camera=cam,
            frame_source="video_file",
            video_source=vsource,
        )

        for f in range(5):
            frame, tgt_state, cam_state = env.step(0.033, 0.0, 0.0)
            self.assertIsInstance(frame, FrameData)
            self.assertIsNone(tgt_state, "tgt_state must be None in video_file mode (no ground truth).")
            self.assertIsNotNone(cam_state)
            self.assertEqual(frame.frame_id, f)

            pan_err, tilt_err, rad_err = env.get_angular_tracking_error()
            self.assertIsNone(pan_err)
            self.assertIsNone(tilt_err)
            self.assertIsNone(rad_err)

        vsource.close()

    def test_full_pipeline_end_to_end_video_file(self):
        """
        Verify BatchScenarioRunner runs completely through a video_file scenario,
        producing 0 errors and reporting 'N/A - no ground truth available' for GT metrics
        while computing valid non-GT metrics (lock retention, FPS, timing).
        """
        runner = BatchScenarioRunner()
        scenario_cfg = {
            "scenario_name": "BENCHMARK_PERFORMANCE_2_EVAL",
            "frame_source": "video_file",
            "video_path": self.video_path,
            "num_frames": 20,
            "disturbances": {
                "cn2": 0.0,
                "vibration_amplitude": 0.0,
                "noise_level": 0.0,
            },
            "control": {
                "kp": 0.35,
                "ki": 0.0,
                "kd": 0.15,
                "k_ff": 1.0,
                "latency_frames": 1,
            },
        }

        record = runner.run_scenario(scenario_cfg)

        self.assertEqual(record.pipeline_errors, 0, "Pipeline must execute with zero unhandled exceptions.")
        self.assertFalse(record.has_ground_truth)
        self.assertEqual(record.avg_tracking_error, "N/A - no ground truth available")
        self.assertEqual(record.max_tracking_error, "N/A - no ground truth available")
        self.assertEqual(record.rmse_tracking_error, "N/A - no ground truth available")

        # Non-ground-truth metrics must be computed normally
        self.assertGreater(record.fps, 0.0)
        self.assertGreater(record.per_frame_processing_time_ms, 0.0)
        self.assertGreaterEqual(record.lock_retention_rate, 0.0)
        self.assertLessEqual(record.lock_retention_rate, 1.0)
        self.assertEqual(record.total_frames, 20)

        # Verify serialization
        d = record.to_dict()
        self.assertEqual(d["avg_tracking_error"], "N/A - no ground truth available")
        self.assertFalse(d["has_ground_truth"])

    def test_real_video_generalization_and_stress(self):
        """
        Verify detector and tracker generalization against real-world camera footage
        and adversarial non-circular compressed video content.
        """
        real_video_path = "data/videos/real_laser_pointer.mp4"
        if not os.path.exists(real_video_path):
            self.skipTest(f"Real video test file not found at {real_video_path}")

        runner = BatchScenarioRunner(use_hybrid_tracker=False)
        cfg = {
            "scenario_name": "BENCHMARK_2_REAL_LASER",
            "frame_source": "video_file",
            "video_path": real_video_path,
            "num_frames": 30,
        }
        record = runner.run_scenario(cfg)
        self.assertEqual(record.pipeline_errors, 0)
        self.assertEqual(record.avg_tracking_error, "N/A - no ground truth available")
        # In real laser pointer footage, detector should maintain high lock retention (>90%)
        self.assertGreaterEqual(record.lock_retention_rate, 0.90)

        # Inspect detector confidences on real footage
        confs = [f.get("confidence", 0.0) for f in runner.last_logger.frame_records]
        self.assertGreater(np.mean(confs), 0.75, "Real laser pointer spot should achieve mean confidence > 0.75")


if __name__ == "__main__":
    unittest.main()
