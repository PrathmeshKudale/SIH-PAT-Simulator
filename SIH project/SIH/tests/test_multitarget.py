"""
Unit tests for multi-target PAT extension:
- Phase M0: Multi-target contracts, TargetID tagging, and scenario schema normalization.
- Phase M1: Multi-target synthetic rendering in Camera/Environment and per-target signature
  discrimination with clutter/unmatched frequency rejection.
"""

import unittest
import numpy as np
from contracts import (
    TargetState,
    FrameData,
    TargetConfig,
    CameraState,
    normalize_scenario_targets,
)
from sim.target import Target, ClutterObject
from sim.camera import Camera
from sim.environment import Environment
from detect.detector import (
    BlobCandidate,
    BeaconVerifier,
    AdaptiveOpticalDetector,
    VerifiedBlobList,
    VerifiedTargetList,
)


class TestMultiTargetContracts(unittest.TestCase):
    """Test Phase M0 contracts and scenario config normalization."""

    def test_target_state_target_id(self):
        """TargetState must support target_id field (default 'target_0') and serialize it."""
        ts_default = TargetState(x=10.0, y=20.0)
        self.assertEqual(ts_default.target_id, "target_0")
        d_default = ts_default.to_dict()
        self.assertIn("target_id", d_default)
        self.assertEqual(d_default["target_id"], "target_0")

        ts_custom = TargetState(x=15.0, y=25.0, target_id="leo_sat_42")
        self.assertEqual(ts_custom.target_id, "leo_sat_42")
        d_custom = ts_custom.to_dict()
        self.assertEqual(d_custom["target_id"], "leo_sat_42")

    def test_frame_data_ground_truth_targets(self):
        """FrameData must support ground_truth_targets mapping all visible targets."""
        img = np.zeros((100, 100), dtype=np.uint8)
        frame = FrameData(
            image=img,
            timestamp=0.1,
            frame_id=1,
            ground_truth_target_pos=(50.0, 50.0),
            ground_truth_targets={"sat_alpha": (50.0, 50.0), "sat_beta": (20.0, 80.0)},
        )
        self.assertEqual(frame.ground_truth_target_pos, (50.0, 50.0))
        self.assertIn("sat_alpha", frame.ground_truth_targets)
        self.assertIn("sat_beta", frame.ground_truth_targets)

        d = frame.to_dict()
        self.assertIn("ground_truth_targets", d)
        self.assertEqual(d["ground_truth_targets"]["sat_beta"], (20.0, 80.0))

    def test_normalize_scenario_targets_backward_compatible(self):
        """Legacy single-target scenario dict must be normalized to 1-target list automatically."""
        legacy_cfg = {
            "scenario_name": "LEGACY_SCENARIO",
            "target": {
                "initial_pos": [0.010, -0.005],
                "velocity": [0.002, 0.001],
                "base_intensity": 200.0,
            },
        }
        targets_list, primary_id = normalize_scenario_targets(legacy_cfg)
        self.assertEqual(len(targets_list), 1)
        self.assertEqual(primary_id, "target_0")
        self.assertEqual(targets_list[0]["target_id"], "target_0")
        self.assertEqual(targets_list[0]["initial_pos"], [0.010, -0.005])

    def test_normalize_scenario_targets_multi(self):
        """Multi-target scenario dict with 'targets' array must preserve all targets and primary ID."""
        multi_cfg = {
            "scenario_name": "MULTI_TARGET_SCENARIO",
            "primary_target_id": "tgt_2",
            "targets": [
                {"target_id": "tgt_1", "initial_pos": [0.01, 0.01], "blink_frequency": 4.0},
                {"target_id": "tgt_2", "initial_pos": [-0.01, -0.01], "blink_frequency": 8.0},
                {"target_id": "tgt_3", "initial_pos": [0.02, -0.01], "blink_frequency": 6.0},
            ],
        }
        targets_list, primary_id = normalize_scenario_targets(multi_cfg)
        self.assertEqual(len(targets_list), 3)
        self.assertEqual(primary_id, "tgt_2")
        self.assertEqual(targets_list[1]["target_id"], "tgt_2")
        self.assertEqual(targets_list[1]["blink_frequency"], 8.0)


class TestMultiTargetRendering(unittest.TestCase):
    """Test Phase M1 multi-target kinematics and optical frame rendering."""

    def test_multi_target_kinematics_and_rendering(self):
        """Environment must advance and render multiple targets simultaneously into one frame."""
        t1 = Target(
            target_id="alpha",
            initial_pos=(0.005, 0.005),
            velocity=(0.001, 0.000),
            base_intensity=220.0,
            blink_frequency=4.0,
            modulation_depth=0.5,
        )
        t2 = Target(
            target_id="beta",
            initial_pos=(-0.008, -0.004),
            velocity=(-0.001, 0.001),
            base_intensity=220.0,
            blink_frequency=8.0,
            modulation_depth=0.5,
        )
        cam = Camera(pan=0.0, tilt=0.0, fov_x=0.040, fov_y=0.030, resolution=(640, 480))
        env = Environment(targets=[t1, t2], primary_target_id="alpha", camera=cam)

        self.assertEqual(env.primary_target_id, "alpha")
        self.assertEqual(env.target.target_id, "alpha")
        self.assertEqual(len(env.targets), 2)

        # Step environment by 0.1s
        frame, primary_state, cam_state = env.step(dt=0.1)

        # 1. Kinematics validation
        self.assertAlmostEqual(t1.x, 0.005 + 0.001 * 0.1)
        self.assertAlmostEqual(t1.y, 0.005)
        self.assertAlmostEqual(t2.x, -0.008 - 0.001 * 0.1)
        self.assertAlmostEqual(t2.y, -0.004 + 0.001 * 0.1)

        # 2. Frame metadata validation
        self.assertIsNotNone(frame.ground_truth_target_pos)
        self.assertIsNotNone(frame.ground_truth_targets)
        self.assertIn("alpha", frame.ground_truth_targets)
        self.assertIn("beta", frame.ground_truth_targets)

        # Primary position corresponds to target 'alpha'
        px_alpha = frame.ground_truth_targets["alpha"]
        px_beta = frame.ground_truth_targets["beta"]
        self.assertEqual(frame.ground_truth_target_pos, px_alpha)
        self.assertNotEqual(px_alpha, px_beta)

        # 3. Image spot intensity check: both spots must be significantly brighter than dark noise
        u_a, v_a = int(round(px_alpha[0])), int(round(px_alpha[1]))
        u_b, v_b = int(round(px_beta[0])), int(round(px_beta[1]))
        self.assertGreater(frame.image[v_a, u_a], 50)
        self.assertGreater(frame.image[v_b, u_b], 50)

    def test_environment_primary_target_switching(self):
        """Switching primary target updates boresight tracking error calculation."""
        t1 = Target(target_id="tgt_1", initial_pos=(0.010, 0.000))
        t2 = Target(target_id="tgt_2", initial_pos=(-0.015, 0.000))
        cam = Camera(pan=0.0, tilt=0.0)
        env = Environment(targets=[t1, t2], primary_target_id="tgt_1", camera=cam)

        err_pan, err_tilt, r_err = env.get_angular_tracking_error()
        self.assertAlmostEqual(err_pan, 0.010)

        # Switch primary target to tgt_2
        success = env.set_primary_target("tgt_2")
        self.assertTrue(success)
        self.assertEqual(env.primary_target_id, "tgt_2")

        err_pan2, _, _ = env.get_angular_tracking_error()
        self.assertAlmostEqual(err_pan2, -0.015)


class TestMultiTargetDiscrimination(unittest.TestCase):
    """Test Phase M1 signature-based discrimination and clutter rejection."""

    def test_per_target_signature_discrimination_two_targets(self):
        """
        Verifier must correctly discriminate 2 simultaneous targets with distinct frequencies (4 Hz vs 8 Hz)
        and tag each candidate with its matching target_id.
        """
        verifier = BeaconVerifier(
            min_history_len=6,
            known_signatures={"sat_A": 4.0, "sat_B": 8.0},
            primary_target_id="sat_A",
        )

        dt = 0.0333
        pos_A = (200.0, 180.0)
        pos_B = (420.0, 300.0)

        verified_blobs = None
        report = {}

        for i in range(12):
            t = i * dt
            int_A = 120.0 + 90.0 * np.sin(2.0 * np.pi * 4.0 * t)
            int_B = 120.0 + 90.0 * np.sin(2.0 * np.pi * 8.0 * t)

            cand_A = BlobCandidate(
                x=pos_A[0], y=pos_A[1],
                peak_intensity=int_A, flux=800.0, area=20,
                contrast=0.8, size_consistency=0.9, confidence=0.85
            )
            cand_B = BlobCandidate(
                x=pos_B[0], y=pos_B[1],
                peak_intensity=int_B, flux=800.0, area=20,
                contrast=0.8, size_consistency=0.9, confidence=0.85
            )

            verified_blobs, report = verifier.update([cand_A, cand_B], timestamp=t)

        self.assertIsNotNone(verified_blobs)
        self.assertIsInstance(verified_blobs, list)
        self.assertEqual(len(verified_blobs), 2, f"Expected 2 verified targets, got {len(verified_blobs)}")

        # Check that both targets are detected and tagged with correct target_id
        target_ids = {b.target_id for b in verified_blobs}
        self.assertIn("sat_A", target_ids)
        self.assertIn("sat_B", target_ids)

        # Primary target sat_A should be first
        self.assertEqual(verified_blobs[0].target_id, "sat_A")
        self.assertAlmostEqual(verified_blobs[0].x, pos_A[0], delta=1.0)
        self.assertAlmostEqual(verified_blobs[0].y, pos_A[1], delta=1.0)

        # Target sat_B is second
        self.assertEqual(verified_blobs[1].target_id, "sat_B")
        self.assertAlmostEqual(verified_blobs[1].x, pos_B[0], delta=1.0)
        self.assertAlmostEqual(verified_blobs[1].y, pos_B[1], delta=1.0)

    def test_negative_rejection_of_unmatched_signature_and_static_clutter(self):
        """
        Negative test: When candidate blobs with unconfigured frequencies (15 Hz) or static clutter
        are present alongside valid targets, they MUST be rejected as clutter.
        """
        verifier = BeaconVerifier(
            min_history_len=6,
            known_signatures={"valid_target": 4.0},
        )

        dt = 0.0333
        rng = np.random.default_rng(999)

        valid_pos = (250.0, 200.0)
        static_pos = (120.0, 100.0)
        unmatched_pos = (450.0, 350.0)

        verified_blobs = None
        report = {}

        for i in range(12):
            t = i * dt
            # 1. Valid target at 4.0 Hz
            int_valid = 120.0 + 90.0 * np.sin(2.0 * np.pi * 4.0 * t)
            cand_valid = BlobCandidate(
                x=valid_pos[0], y=valid_pos[1],
                peak_intensity=int_valid, flux=800.0, area=20,
                contrast=0.8, size_consistency=0.9, confidence=0.85
            )

            # 2. Static clutter (non-blinking glint)
            int_static = 230.0 + rng.normal(0, 0.5)
            cand_static = BlobCandidate(
                x=static_pos[0], y=static_pos[1],
                peak_intensity=int_static, flux=1000.0, area=25,
                contrast=0.9, size_consistency=0.9, confidence=0.90
            )

            # 3. Unknown frequency emitter (15.0 Hz - not registered)
            int_unknown = 120.0 + 90.0 * np.sin(2.0 * np.pi * 15.0 * t)
            cand_unknown = BlobCandidate(
                x=unmatched_pos[0], y=unmatched_pos[1],
                peak_intensity=int_unknown, flux=800.0, area=20,
                contrast=0.8, size_consistency=0.9, confidence=0.85
            )

            verified_blobs, report = verifier.update(
                [cand_valid, cand_static, cand_unknown], timestamp=t
            )

        # Must acquire ONLY the valid target (4 Hz), rejecting both static clutter and 15 Hz emitter
        self.assertIsNotNone(verified_blobs)
        self.assertEqual(len(verified_blobs), 1, f"Expected exactly 1 verified target, got {len(verified_blobs)}")
        self.assertEqual(verified_blobs[0].target_id, "valid_target")
        self.assertAlmostEqual(verified_blobs[0].x, valid_pos[0], delta=1.0)
        self.assertAlmostEqual(verified_blobs[0].y, valid_pos[1], delta=1.0)

        # Clutter rejection count must account for rejected candidates
        self.assertGreaterEqual(report["clutter_rejected"], 2)

    def test_adaptive_detector_detect_multi(self):
        """AdaptiveOpticalDetector.detect_multi returns full list of TargetState instances."""
        detector = AdaptiveOpticalDetector(
            known_signatures={"sat_1": 4.0, "sat_2": 8.0},
            primary_target_id="sat_1",
        )

        dt = 0.0333
        for i in range(12):
            t = i * dt
            int_1 = 120.0 + 90.0 * np.sin(2.0 * np.pi * 4.0 * t)
            int_2 = 120.0 + 90.0 * np.sin(2.0 * np.pi * 8.0 * t)

            # Build synthetic frame
            img = np.zeros((480, 640), dtype=np.uint8)
            # Spot 1 at (200, 200)
            img[198:203, 198:203] = int(np.clip(int_1, 0, 255))
            # Spot 2 at (400, 300)
            img[298:303, 398:403] = int(np.clip(int_2, 0, 255))

            frame = FrameData(image=img, timestamp=t, frame_id=i)
            detections, report = detector.detect_multi(frame)

        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 2)
        target_ids = [d.target_id for d in detections]
        self.assertIn("sat_1", target_ids)
        self.assertIn("sat_2", target_ids)


if __name__ == "__main__":
    unittest.main()
