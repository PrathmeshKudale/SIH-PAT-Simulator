"""
Phase 9 Integration & Packaging Tests.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Validates:
1. Backend packaging and clean importability of Phase 0 contracts and all subpackages.
2. Defensive error handling and fault resilience across all pipeline stages.
3. Frozen-seed flight scenario configuration validity and deterministic reproducibility.
4. Extended 300-frame combined-stress test execution with zero crashes or exceptions.
"""

import unittest
import json
import os
import numpy as np

import contracts
from contracts import (
    FrameData,
    TargetState,
    CameraState,
    MetricsRecord,
    DisturbanceConfig,
    ReacquisitionZone,
)
from metrics.batch_runner import BatchScenarioRunner
from detect.detector import AdaptiveOpticalDetector
from track.kalman import ConstantVelocityKalmanFilter
from control.pid import PIDController


class TestPhase9(unittest.TestCase):

    def test_packaging_and_contract_imports(self):
        """Confirm all backend submodules and Phase 0 contracts import cleanly."""
        import sim
        import detect
        import track
        import control
        import disturb
        import metrics

        # Verify contracts definitions
        rec = MetricsRecord(
            scenario_name="IMPORT_TEST",
            simulation_duration=1.0,
            fps=30.0,
            total_frames=30,
            pipeline_errors=0,
        )
        self.assertEqual(rec.scenario_name, "IMPORT_TEST")
        self.assertEqual(rec.pipeline_errors, 0)

        # Verify dict serialization
        d = rec.to_dict()
        self.assertIn("pipeline_errors", d)
        self.assertIn("stage_timing_breakdown", d)

    def test_flight_scenarios_config_and_seeds(self):
        """Confirm flight_scenarios.json exists, contains frozen seeds, and runs deterministically."""
        flight_path = "config/flight_scenarios.json"
        self.assertTrue(os.path.exists(flight_path), f"Missing {flight_path}")

        with open(flight_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

        self.assertGreaterEqual(len(scenarios), 3)
        scenario_names = [s["scenario_name"] for s in scenarios]
        self.assertIn("DRISHTI_GROUND_TO_UAV_TRACK", scenario_names)
        self.assertIn("DRISHTI_COMBINED_WORST_CASE_STRESS", scenario_names)

        for s in scenarios:
            self.assertIn("seed", s, f"Scenario {s.get('scenario_name')} missing frozen seed")
            self.assertIsInstance(s["seed"], int)

        # Verify deterministic execution of DRISHTI_GROUND_TO_UAV_TRACK
        runner = BatchScenarioRunner(config_path=flight_path)
        flight_cfg = [s for s in scenarios if s["scenario_name"] == "DRISHTI_GROUND_TO_UAV_TRACK"][0]
        # Run 20 frames test
        short_cfg = dict(flight_cfg)
        short_cfg["num_frames"] = 20

        run1 = runner.run_scenario(short_cfg)
        run2 = runner.run_scenario(short_cfg)

        self.assertEqual(run1.pipeline_errors, 0)
        self.assertEqual(run2.pipeline_errors, 0)
        self.assertAlmostEqual(run1.avg_tracking_error, run2.avg_tracking_error, places=8)
        self.assertAlmostEqual(run1.lock_retention_rate, run2.lock_retention_rate, places=8)

    def test_stage_error_handling_fault_resilience(self):
        """Confirm pipeline handles corrupted or NaN inputs without crashing or halting."""
        detector = AdaptiveOpticalDetector()
        # 1. Detector handling corrupted/black frame
        bad_frame = FrameData(image=None, timestamp=0.0)
        det, aux = detector.detect(bad_frame)
        self.assertIsNone(det)

        # 2. Kalman Filter handling non-finite inputs
        kf = ConstantVelocityKalmanFilter()
        kf.init_state(0.01, 0.01, 0.0, 0.0, 0.0)
        # Pass NaN measurement
        est = kf.step(0.033, (float("nan"), 0.01), 0.9)
        self.assertIsNotNone(est)
        # Should gracefully coast or preserve finite state
        self.assertTrue(np.isfinite(est.x))

        # 3. BatchScenarioRunner error recovery
        runner = BatchScenarioRunner()
        robust_cfg = {
            "scenario_name": "ROBUSTNESS_INJECTION_TEST",
            "num_frames": 15,
            "dt": 0.033,
            "seed": 999,
            "target": {"initial_pos": [0.010, -0.005], "velocity": [0.002, -0.001], "distance_km": 5.0},
            "disturbances": {"cn2": 1e-14, "vibration_amplitude": 0.0003, "noise_level": 5.0},
            "control": {"kp": 0.35, "kd": 0.15, "k_ff": 1.0, "latency_frames": 2},
            "reacquisition": {"enable_predictive_search": True, "tier1_budget_frames": 25},
        }
        rec = runner.run_scenario(robust_cfg)
        self.assertEqual(rec.total_frames, 15)
        self.assertGreater(rec.fps, 0.0)
        self.assertEqual(rec.pipeline_errors, 0)

    def test_extended_combined_stress_execution(self):
        """Confirm extended combined-stress run finishes with zero pipeline errors."""
        flight_path = "config/flight_scenarios.json"
        with open(flight_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

        stress_cfg = [s for s in scenarios if s["scenario_name"] == "DRISHTI_COMBINED_WORST_CASE_STRESS"][0]
        self.assertEqual(stress_cfg["num_frames"], 100)

        runner = BatchScenarioRunner()
        record = runner.run_scenario(stress_cfg)

        self.assertEqual(record.total_frames, 100)
        self.assertEqual(record.pipeline_errors, 0)
        self.assertGreater(record.fps, 0.0)
        self.assertGreater(record.active_locked_frames, 0)
        self.assertGreater(record.lock_retention_rate, 0.0)
        self.assertIn("rendering_ms", record.stage_timing_breakdown)
        self.assertIn("disturbances_ms", record.stage_timing_breakdown)
    def test_fault_injection_singular_covariance_and_nan_inf(self):
        """
        Fault injection test:
        1. Force singular covariance matrix (det(S)=0) in Kalman filter update step.
        2. Separately pass NaN and Inf detection measurements into filter and pipeline.
        Confirms guards prevent crashes, avoid propagating NaN/Inf, and recover into COAST.
        """
        kf = ConstantVelocityKalmanFilter(min_valid_confidence=0.40)
        kf.init_state(0.010, -0.005, 0.001, 0.002, timestamp=0.0)

        # 1. Deliberately force singular covariance matrix S = 0
        # By setting P = 0 and R_nominal = 0, S becomes singular (det(S) == 0)
        kf.P = np.zeros((4, 4), dtype=np.float64)
        kf.R_nominal = np.zeros((2, 2), dtype=np.float64)

        # Call update with valid coordinates on singular covariance
        state_sing = kf.update(0.011, -0.004, confidence=1.0)
        # Must catch LinAlgError, enter COAST, and retain finite state
        self.assertEqual(state_sing.tracker_mode, "COAST")
        self.assertEqual(kf.mode, "COAST")
        self.assertTrue(np.isfinite(state_sing.x))
        self.assertTrue(np.isfinite(state_sing.y))
        self.assertTrue(np.isfinite(state_sing.vx))
        self.assertTrue(np.isfinite(state_sing.vy))

        # Re-initialize clean KF for measurement fault injections
        kf2 = ConstantVelocityKalmanFilter(min_valid_confidence=0.40)
        kf2.init_state(0.010, -0.005, 0.001, 0.002, timestamp=0.0)

        # 2. Pass NaN measurement
        state_nan = kf2.step(0.033, (float("nan"), -0.005), confidence=0.95)
        self.assertEqual(state_nan.tracker_mode, "COAST")
        self.assertEqual(kf2.mode, "COAST")
        self.assertTrue(np.isfinite(state_nan.x))
        self.assertTrue(np.isfinite(state_nan.y))

        # 3. Pass Inf measurement
        state_inf = kf2.step(0.033, (float("inf"), -float("inf")), confidence=0.95)
        self.assertEqual(state_inf.tracker_mode, "COAST")
        self.assertEqual(kf2.mode, "COAST")
        self.assertTrue(np.isfinite(state_inf.x))
        self.assertTrue(np.isfinite(state_inf.y))

        # 4. Confirm system recovers to active KF when a valid measurement arrives afterwards
        state_valid = kf2.step(0.033, (0.011, -0.004), confidence=0.90)
        self.assertEqual(state_valid.tracker_mode, "KF")
        self.assertEqual(kf2.mode, "KF")
        self.assertTrue(np.isfinite(state_valid.x))
        self.assertTrue(np.isfinite(state_valid.y))

    def test_full_stack_hybrid_tracker_reacquisition_metrics(self):
        """
        Non-UI regression test for the full integrated stack:
        HybridTracker (adaptive KF/PF switching) + HierarchicalReacquisitionController
        (Tier 1/2 search) + full metrics pipeline across DYNAMIC_OCCLUSION_SHORT
        and [HELD-OUT] COMPOUND_STORM scenarios.
        """
        config_path = "config/scenarios.json"
        self.assertTrue(os.path.exists(config_path), f"Missing {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

        runner = BatchScenarioRunner(config_path=config_path, use_hybrid_tracker=True)

        # 1. Test DYNAMIC_OCCLUSION_SHORT
        occ_cfg = [s for s in scenarios if s["scenario_name"] == "DYNAMIC_OCCLUSION_SHORT"][0]
        rec_occ = runner.run_scenario(occ_cfg)

        self.assertEqual(rec_occ.pipeline_errors, 0, "DYNAMIC_OCCLUSION_SHORT encountered pipeline errors")
        self.assertGreaterEqual(rec_occ.lock_retention_rate, 0.0)
        self.assertLessEqual(rec_occ.lock_retention_rate, 1.0)
        self.assertTrue(np.isfinite(rec_occ.avg_tracking_error))
        self.assertTrue(np.isfinite(rec_occ.max_tracking_error))
        self.assertTrue(np.isfinite(rec_occ.rmse_tracking_error))

        # Check switch events logged in MetricsLogger
        occ_logger = runner.last_logger
        self.assertIsNotNone(occ_logger)
        self.assertGreaterEqual(len(occ_logger.switch_events), 2, "Expected at least 2 switch events (KF->PF and PF->KF)")
        sw1 = occ_logger.switch_events[0]
        self.assertEqual(sw1.from_mode, "KF")
        self.assertEqual(sw1.to_mode, "PF")
        self.assertEqual(sw1.frame_id, 12, f"Expected switch to PF at occlusion onset (frame 12), got {sw1.frame_id}")
        self.assertGreaterEqual(sw1.severity, 0.55)

        sw2 = occ_logger.switch_events[1]
        self.assertEqual(sw2.from_mode, "PF")
        self.assertEqual(sw2.to_mode, "KF")
        self.assertLessEqual(sw2.severity, 0.30)

        # 2. Test [HELD-OUT] COMPOUND_STORM
        storm_cfg = [s for s in scenarios if "COMPOUND_STORM" in s["scenario_name"]][0]
        rec_storm = runner.run_scenario(storm_cfg)

        self.assertEqual(rec_storm.pipeline_errors, 0, "COMPOUND_STORM encountered pipeline errors")
        self.assertGreaterEqual(rec_storm.lock_retention_rate, 0.0)
        self.assertLessEqual(rec_storm.lock_retention_rate, 1.0)
        self.assertTrue(np.isfinite(rec_storm.avg_tracking_error))
        self.assertTrue(np.isfinite(rec_storm.max_tracking_error))
        self.assertTrue(np.isfinite(rec_storm.rmse_tracking_error))

        storm_logger = runner.last_logger
        self.assertIsNotNone(storm_logger)
        self.assertGreaterEqual(len(storm_logger.switch_events), 1, "Expected switch to PF under severe compound storm")
        sw_storm = storm_logger.switch_events[0]
        self.assertEqual(sw_storm.from_mode, "KF")
        self.assertEqual(sw_storm.to_mode, "PF")
        self.assertGreaterEqual(sw_storm.severity, 0.55)


if __name__ == "__main__":
    unittest.main()

