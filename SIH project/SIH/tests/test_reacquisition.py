"""
Unit tests for Phase 7: Predictive Re-Acquisition and Track Loss Classification.
Smart India Hackathon - Problem Statement 26169 (ISRO / Department of Space)
"""

import unittest
import numpy as np
from contracts import TargetState, CameraState, ReacquisitionZone
from track.classifier import TrackLossClassifier
from track.reacquisition_zone import ReacquisitionZonePredictor
from control.reacquisition import HierarchicalReacquisitionController
from disturb.occluder import DynamicOccluder
from metrics.logger import MetricsLogger, TrackLossEvent


class TestReacquisition(unittest.TestCase):

    def setUp(self):
        self.camera_state = CameraState(pan=0.0, tilt=0.0, fov_x=0.10, fov_y=0.075)

    def test_track_loss_classifier_occlusion(self):
        """Classifier must identify occlusion when an active occluder intersects target."""
        classifier = TrackLossClassifier()
        target_state = TargetState(x=0.010, y=0.008, vx=0.005, vy=0.0, timestamp=1.0)
        occluder = DynamicOccluder(initial_pos=(0.010, 0.008), radius_rad=0.010)

        cause, conf, rationale = classifier.classify(
            consecutive_misses=10,
            last_known_state=target_state,
            camera_state=self.camera_state,
            active_occluder=occluder,
            active_cn2=1e-15,
            target_true_pos=(0.010, 0.008),
        )

        self.assertEqual(cause, "OCCLUSION")
        self.assertGreaterEqual(conf, 0.90)
        self.assertIn("intersected by dynamic occluder", rationale)

    def test_track_loss_classifier_turbulence(self):
        """Classifier must identify severe turbulence when Cn^2 exceeds threshold."""
        classifier = TrackLossClassifier(turbulence_cn2_threshold=1.0e-13)
        target_state = TargetState(x=0.005, y=0.005, vx=0.001, vy=0.0, timestamp=1.0)

        cause, conf, rationale = classifier.classify(
            consecutive_misses=10,
            last_known_state=target_state,
            camera_state=self.camera_state,
            active_occluder=None,
            active_cn2=5.0e-13,
            target_true_pos=(0.005, 0.005),
        )

        self.assertEqual(cause, "TURBULENCE")
        self.assertGreaterEqual(conf, 0.90)
        self.assertIn("Severe atmospheric turbulence", rationale)

    def test_track_loss_classifier_fast_motion(self):
        """Classifier must identify fast motion when target speed exceeds slew limit."""
        classifier = TrackLossClassifier(max_slew_velocity=0.50)
        # Target moving at 0.60 rad/s (600 mrad/s), exceeding 500 mrad/s limit
        target_state = TargetState(x=0.010, y=0.0, vx=0.60, vy=0.0, timestamp=1.0)

        cause, conf, rationale = classifier.classify(
            consecutive_misses=10,
            last_known_state=target_state,
            camera_state=self.camera_state,
            active_occluder=None,
            active_cn2=1e-15,
            target_true_pos=(0.010, 0.0),
        )

        self.assertEqual(cause, "FAST_MOTION")
        self.assertGreaterEqual(conf, 0.90)
        self.assertIn("exceeds maximum gimbal slew velocity", rationale)

    def test_track_loss_classifier_detection_dropout(self):
        """Classifier must default to detection dropout under benign conditions."""
        classifier = TrackLossClassifier()
        target_state = TargetState(x=0.005, y=0.002, vx=0.010, vy=0.0, timestamp=1.0)

        cause, conf, rationale = classifier.classify(
            consecutive_misses=10,
            last_known_state=target_state,
            camera_state=self.camera_state,
            active_occluder=None,
            active_cn2=1e-16,
            target_true_pos=(0.005, 0.002),
        )

        self.assertEqual(cause, "DETECTION_DROPOUT")
        self.assertIn("Transient optical detection dropout", rationale)

    def test_reacquisition_zone_forward_projection(self):
        """ReacquisitionZonePredictor must forward-project position along velocity vector."""
        predictor = ReacquisitionZonePredictor(default_accel_uncertainty=0.05)
        last_state = TargetState(x=0.010, y=-0.005, vx=0.012, vy=-0.004, timestamp=1.0)

        # Elapsed time = 0.5 s (t = 1.5 s)
        zone = predictor.compute_zone(last_state, current_time=1.5, camera_fov=(0.10, 0.075))

        expected_pan = 0.010 + 0.012 * 0.5  # 0.016 rad
        expected_tilt = -0.005 - 0.004 * 0.5 # -0.007 rad

        self.assertAlmostEqual(zone.center_pan, expected_pan, places=5)
        self.assertAlmostEqual(zone.center_tilt, expected_tilt, places=5)
        self.assertAlmostEqual(zone.projection_time_s, 0.5, places=5)
        self.assertGreater(zone.search_radius, 0.030)
        self.assertLess(zone.pan_bounds[0], zone.center_pan)
        self.assertGreater(zone.pan_bounds[1], zone.center_pan)

    def test_hierarchical_controller_tier1_to_tier2_transition(self):
        """Hierarchical controller must search Tier 1 localized zone before falling back to Tier 2."""
        controller = HierarchicalReacquisitionController(
            fov_x=0.10,
            fov_y=0.075,
            tier1_budget_frames=3,
            enable_predictive_search=True,
        )
        zone = ReacquisitionZone(
            center_pan=0.020,
            center_tilt=-0.010,
            search_radius=0.035,
            predicted_velocity=(0.010, -0.005),
            projection_time_s=0.4,
            pan_bounds=(-0.015, 0.055),
            tilt_bounds=(-0.045, 0.025),
        )

        initial_tier = controller.start_reacquisition(zone, self.camera_state)
        self.assertEqual(initial_tier, "TIER1_PREDICTIVE")
        self.assertEqual(controller.current_tier, "TIER1_PREDICTIVE")

        # Step 3 frames (within Tier 1 budget)
        for _ in range(3):
            controller.step(0.033, self.camera_state)
            self.assertEqual(controller.current_tier, "TIER1_PREDICTIVE")

        # Step 4th frame (exceeds Tier 1 budget) -> Must fall back to Tier 2 Global Spiral
        controller.step(0.033, self.camera_state)
        self.assertEqual(controller.current_tier, "TIER2_GLOBAL")

    def test_hierarchical_controller_predictive_bypass_flag(self):
        """When enable_predictive_search is False, controller must immediately jump to Tier 2."""
        controller = HierarchicalReacquisitionController(
            tier1_budget_frames=5,
            enable_predictive_search=False,
        )
        zone = ReacquisitionZone(
            center_pan=0.020,
            center_tilt=-0.010,
            search_radius=0.035,
            predicted_velocity=(0.010, -0.005),
            projection_time_s=0.4,
            pan_bounds=(-0.015, 0.055),
            tilt_bounds=(-0.045, 0.025),
        )

        initial_tier = controller.start_reacquisition(zone, self.camera_state)
        self.assertEqual(initial_tier, "TIER2_GLOBAL")
        self.assertEqual(controller.current_tier, "TIER2_GLOBAL")

    def test_metrics_logger_track_loss_event(self):
        """MetricsLogger must record TrackLossEvent and format ASCII table."""
        logger = MetricsLogger()
        ev = logger.log_track_loss_event(
            frame_id=25,
            timestamp=0.825,
            cause="OCCLUSION",
            confidence=0.0,
            severity=0.98,
            last_known_pos=(0.015, -0.008),
            last_known_vel=(0.012, -0.004),
            rationale="Occluder intersection",
        )

        self.assertIsInstance(ev, TrackLossEvent)
        self.assertEqual(len(logger.get_track_loss_events()), 1)
        table_str = logger.format_track_loss_table()
        self.assertIn("OCCLUSION", table_str)
        self.assertIn("25", table_str)


if __name__ == "__main__":
    unittest.main()
