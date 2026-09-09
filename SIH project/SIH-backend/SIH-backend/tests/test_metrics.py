"""
Unit tests for metrics module: MetricsLogger, SwitchEvent, MetricsCalculator,
and BatchScenarioRunner.
"""

import unittest
import tempfile
import os
import csv
import json
from metrics.logger import MetricsLogger, SwitchEvent
from metrics.calculator import MetricsCalculator
from metrics.batch_runner import BatchScenarioRunner
from contracts import MetricsRecord


class TestMetrics(unittest.TestCase):

    def test_switch_event_logging_and_table_format(self):
        logger = MetricsLogger()

        # Log KF -> PF
        ev1 = logger.log_switch_event(
            frame_id=12,
            timestamp=0.396,
            from_mode="KF",
            to_mode="PF",
            severity=0.682,
            confidence=0.180,
            rationale="Severity surge (0.682 >= 0.55)",
        )
        self.assertEqual(ev1.from_mode, "KF")
        self.assertEqual(ev1.to_mode, "PF")
        self.assertAlmostEqual(ev1.severity, 0.682)

        # Log PF -> KF
        ev2 = logger.log_switch_event(
            frame_id=35,
            timestamp=1.155,
            from_mode="PF",
            to_mode="KF",
            severity=0.125,
            confidence=0.910,
            rationale="Calm conditions sustained for 5 frames",
        )
        self.assertEqual(ev2.to_mode, "KF")

        events = logger.get_switch_events()
        self.assertEqual(len(events), 2)

        # Format switch table
        table_str = logger.format_switch_table()
        self.assertIn("KF -> PF", table_str)
        self.assertIn("PF -> KF", table_str)
        self.assertIn("Severity surge", table_str)

    def test_metrics_summary_computation(self):
        logger = MetricsLogger()

        for i in range(20):
            mode = "KF" if i < 10 else "PF"
            logger.log_frame({
                "frame_id": i,
                "tracker_mode": mode,
                "radial_error_mrad": 0.5 + 0.1 * i,
            })

        summary = logger.compute_summary()
        self.assertEqual(summary["total_frames"], 20)
        self.assertAlmostEqual(summary["mode_breakdown"]["KF"], 0.5)
        self.assertAlmostEqual(summary["mode_breakdown"]["PF"], 0.5)
        self.assertGreater(summary["mean_tracking_error_mrad"], 0.0)
        self.assertEqual(summary["active_locked_frames"], 20)
        self.assertEqual(summary["coasting_frames"], 0)
        self.assertAlmostEqual(summary["lock_retention_rate"], 1.0)

    def test_lock_retention_counts_coasting_separately(self):
        """Confirm that COAST and PF_COAST frames are counted as coasting, not active lock."""
        logger = MetricsLogger()
        # 10 frames active KF, 5 frames COAST, 5 frames PF_COAST
        for i in range(20):
            if i < 10:
                mode = "KF"
            elif i < 15:
                mode = "COAST"
            else:
                mode = "PF_COAST"
            logger.log_frame({
                "frame_id": i,
                "tracker_mode": mode,
                "radial_error_mrad": 0.5,
            })

        summary = logger.compute_summary()
        self.assertEqual(summary["total_frames"], 20)
        self.assertEqual(summary["active_locked_frames"], 10)
        self.assertEqual(summary["coasting_frames"], 10)
        self.assertAlmostEqual(summary["lock_retention_rate"], 0.50)
        self.assertAlmostEqual(summary["mode_breakdown"]["COAST"], 0.25)
        self.assertAlmostEqual(summary["mode_breakdown"]["PF_COAST"], 0.25)

    def test_logger_export_csv_and_json(self):
        """Verify that MetricsLogger exports valid CSV and JSON files."""
        logger = MetricsLogger()
        for i in range(5):
            logger.log_frame({
                "frame_id": i,
                "timestamp": i * 0.033,
                "tracker_mode": "KF",
                "radial_error_mrad": 0.2 * i,
            })

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "telemetry.csv")
            json_path = os.path.join(tmpdir, "telemetry.json")

            logger.export_csv(csv_path)
            logger.export_json(json_path)

            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(json_path))

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                self.assertEqual(len(reader), 5)
                self.assertIn("frame_id", reader[0])
                self.assertIn("tracker_mode", reader[0])
                self.assertEqual(reader[0]["tracker_mode"], "KF")

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.assertIn("summary", data)
                self.assertIn("frame_records", data)
                self.assertEqual(len(data["frame_records"]), 5)

    def test_metrics_calculator_comprehensive(self):
        """Verify MetricsCalculator accurate computation of all summary metrics."""
        calc = MetricsCalculator(dt=0.033)
        records = [
            {"frame_id": 0, "tracker_mode": "COAST", "radial_error_mrad": 5.0, "detected": False},
            {"frame_id": 1, "tracker_mode": "COAST", "radial_error_mrad": 4.0, "detected": False},
            {"frame_id": 2, "tracker_mode": "KF", "radial_error_mrad": 1.0, "detected": True},
            {"frame_id": 3, "tracker_mode": "KF", "radial_error_mrad": 0.5, "detected": True},
            {"frame_id": 4, "tracker_mode": "PF", "radial_error_mrad": 0.8, "detected": True},
        ]
        stage_timings = {
            "detection_ms": [2.0, 2.5, 2.1, 2.3, 2.2],
            "tracking_ms": [0.1, 0.1, 0.2, 0.1, 0.3],
            "control_ms": [0.05, 0.05, 0.05, 0.05, 0.05],
        }

        record = calc.compute_metrics(
            frame_records=records,
            track_loss_count=1,
            start_wall_time=0.0,
            end_wall_time=0.1,
            scenario_name="TEST_SCENARIO",
            is_held_out=True,
            stage_timings=stage_timings,
        )

        self.assertEqual(record.total_frames, 5)
        self.assertAlmostEqual(record.simulation_duration, 5 * 0.033)
        self.assertAlmostEqual(record.acquisition_time, 2 * 0.033)  # Acquired at frame 2
        self.assertEqual(record.active_locked_frames, 3)
        self.assertEqual(record.coasting_frames, 2)
        self.assertAlmostEqual(record.lock_retention_rate, 3 / 5)
        self.assertTrue(record.is_held_out)
        self.assertEqual(record.scenario_name, "TEST_SCENARIO")
        self.assertEqual(record.track_loss_count, 1)

        # Check RMSE and errors
        errors = [5.0, 4.0, 1.0, 0.5, 0.8]
        expected_avg = sum(errors) / len(errors)
        expected_max = max(errors)
        expected_rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5

        self.assertAlmostEqual(record.avg_tracking_error, expected_avg, places=4)
        self.assertAlmostEqual(record.max_tracking_error, expected_max, places=4)
        self.assertAlmostEqual(record.rmse_tracking_error, expected_rmse, places=4)

        # Stage timings
        self.assertIn("detection_ms", record.stage_timing_breakdown)
        self.assertAlmostEqual(record.stage_timing_breakdown["detection_ms"], 2.22, places=2)

    def test_batch_runner_execution(self):
        """Verify BatchScenarioRunner runs scenarios, produces records, and exports CSV/JSON."""
        runner = BatchScenarioRunner()
        mini_matrix = [
            {
                "scenario_name": "MINI_STANDARD",
                "is_held_out": False,
                "num_frames": 10,
                "dt": 0.033,
                "target": {"initial_pos": [0.005, 0.002], "velocity": [0.001, 0.001], "distance_km": 5.0},
                "disturbances": {"cn2": 1e-16, "vibration_amplitude": 0.0001, "noise_level": 2.0},
                "control": {"kp": 0.35, "kd": 0.15, "k_ff": 1.0, "latency_frames": 2},
                "reacquisition": {"enable_predictive_search": True, "tier1_budget_frames": 25},
            },
            {
                "scenario_name": "[HELD-OUT] MINI_HELD_OUT",
                "is_held_out": True,
                "num_frames": 10,
                "dt": 0.033,
                "target": {"initial_pos": [0.010, -0.005], "velocity": [0.003, -0.002], "distance_km": 8.0},
                "disturbances": {"cn2": 2e-14, "vibration_amplitude": 0.0003, "noise_level": 5.0},
                "control": {"kp": 0.35, "kd": 0.15, "k_ff": 1.0, "latency_frames": 2},
                "reacquisition": {"enable_predictive_search": True, "tier1_budget_frames": 25},
            },
        ]

        results = runner.run_all(mini_matrix)
        self.assertEqual(len(results), 2)
        self.assertFalse(results[0].is_held_out)
        self.assertTrue(results[1].is_held_out)
        self.assertGreater(results[0].fps, 0.0)
        self.assertIn("detection_ms", results[0].stage_timing_breakdown)

        # Table formatting
        table_str = runner.format_results_table()
        self.assertIn("MINI_STANDARD", table_str)
        self.assertIn("[HELD-OUT]", table_str)

        # Export test
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = os.path.join(tmpdir, "batch.json")
            csv_file = os.path.join(tmpdir, "batch.csv")

            runner.export_batch_json(json_file)
            runner.export_batch_csv(csv_file)

            self.assertTrue(os.path.exists(json_file))
            self.assertTrue(os.path.exists(csv_file))

            with open(csv_file, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["scenario_name"], "MINI_STANDARD")
                self.assertEqual(rows[1]["is_held_out"], "True")


if __name__ == "__main__":
    unittest.main()
