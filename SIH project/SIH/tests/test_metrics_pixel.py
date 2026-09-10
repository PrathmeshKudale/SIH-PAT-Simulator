"""
Unit Tests for Pixel-Space Tracking Error and PS-Exact Terminology Metrics.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Requirements Covered:
1. Validate mrad-to-pixel conversion against independent hand-calculated known geometry cases
   (fixed FOV, resolution, angular error -> expected pixel error computed independently).
2. Validate target_loss_percent against a synthetic sequence with a known number of dropped-lock frames.
3. Validate rmse_px against a synthetic error sequence with known mathematical RMSE.
4. Verify coexistence of mrad-based and pixel-based tracking errors side-by-side.
5. Verify PS-exact terminology fields:
   "Tracking Error", "Target Loss", "Centroiding error", "RMSE",
   "Re-acquisition time", "Lock retention rate", "FPS".
"""

import unittest
import math
import numpy as np

from contracts import MetricsRecord
from metrics.calculator import MetricsCalculator, angular_to_pixel_error
from metrics.logger import MetricsLogger


class TestMetricsPixelAndPSTerminology(unittest.TestCase):
    """Test suite for pixel-space metrics and PS-matching terminology."""

    def test_angular_to_pixel_conversion_hand_calculated_geometry(self):
        """
        Validate mrad-to-pixel conversion against two hand-calculated known geometries,
        computed completely independently of the implementation formula.

        Case 1: Standard PAT Virtual Camera
        - Sensor resolution: W = 640 pixels
        - Angular FOV: theta = 0.040 radians (40 mrad)
        - Spatial scale: 640 px / 0.040 rad = 16,000 px/rad = 16.0 px/mrad
        - Given angular error: 1.500 mrad
        - Hand-calculated expected error: 1.5 * 16.0 = 24.000 pixels.

        Case 2: Custom Optical Setup with FOV in degrees
        - Sensor resolution: W = 1920 pixels
        - Angular FOV: theta = 3.0 degrees = 3.0 * pi / 180 = pi / 60 rad (~0.05235987756 rad)
        - Spatial scale: 1920 / (pi / 60) = 115,200 / pi px/rad (~36669.29889 px/rad)
        - Given angular error: 2.0 mrad = 0.002 rad
        - Hand-calculated expected error: 0.002 * (115200 / pi) = 230.4 / pi = 73.3385977 pixels.
        """
        # Case 1: Standard FOV = 0.040 rad, 640 px, err = 1.5 mrad
        err_px_1 = angular_to_pixel_error(
            angular_error=1.5,
            fov_rad=0.040,
            resolution_px=640,
            angular_is_mrad=True,
        )
        # Assert against hardcoded ground-truth constant 24.0
        self.assertAlmostEqual(err_px_1, 24.0, places=7,
                               msg=f"Case 1 pixel error {err_px_1} deviated from hand-calculated 24.0 px")

        # Case 2: Custom FOV in degrees = 3.0 deg, 1920 px, err = 2.0 mrad
        err_px_2 = angular_to_pixel_error(
            angular_error=2.0,
            fov_deg=3.0,
            resolution_px=1920,
            angular_is_mrad=True,
        )
        expected_case_2 = 230.4 / math.pi  # 73.3385977...
        self.assertAlmostEqual(err_px_2, expected_case_2, places=5,
                               msg=f"Case 2 pixel error {err_px_2} deviated from hand-calculated {expected_case_2} px")

    def test_target_loss_percent_synthetic_dropped_frames(self):
        """
        Validate target_loss_percent against a synthetic sequence with an exact,
        known number of dropped-lock frames:
        - 120 total frames:
          * 90 frames active locked (KF mode, error = 0.4 mrad, detected=True)
          * 30 frames unlocked / lost (COAST / LOST mode, error = 8.0 mrad, detected=False)
        - Expected Target Loss = (30 / 120) * 100 = 25.0%
        - Expected Lock Retention Rate = (90 / 120) = 0.75 (75.0%)
        """
        calc = MetricsCalculator(dt=0.033)
        records = []

        # 90 locked frames
        for f in range(90):
            records.append({
                "frame_id": f,
                "tracker_mode": "KF",
                "detected": True,
                "radial_error_mrad": 0.4,
            })

        # 30 unlocked frames (track lost / coasting)
        for f in range(90, 120):
            records.append({
                "frame_id": f,
                "tracker_mode": "LOST",
                "detected": False,
                "radial_error_mrad": 8.0,
            })

        record = calc.compute_metrics(frame_records=records)

        self.assertEqual(record.total_frames, 120)
        self.assertEqual(record.active_locked_frames, 90)
        self.assertEqual(record.coasting_frames, 0)
        # Lock retention rate
        self.assertAlmostEqual(record.lock_retention_rate, 0.75, places=7)
        # Target loss percent (exact PS item 18 formula)
        self.assertAlmostEqual(record.target_loss_percent, 25.0, places=7)

    def test_rmse_px_synthetic_known_error_sequence(self):
        """
        Validate rmse_px against a synthetic error sequence with known mathematical RMSE.

        Synthetic pixel error sequence: [3.0, 4.0, 5.0, 0.0, 0.0]
        - Sum of squares = 3^2 + 4^2 + 5^2 + 0^2 + 0^2 = 9 + 16 + 25 = 50.0
        - Mean of squares = 50.0 / 5 = 10.0
        - Expected RMSE = sqrt(10.0) ≈ 3.1622776601683795 pixels.
        - Expected Mean = (3 + 4 + 5 + 0 + 0) / 5 = 2.400 pixels.
        - Expected Max = 5.000 pixels.
        """
        # For FOV = 0.040 rad and res = 640 px: 1 mrad = 16 px.
        # Conversely, to generate exactly [3, 4, 5, 0, 0] px:
        # mrad = px / 16.0
        calc = MetricsCalculator(dt=0.033, fov_rad=0.040, resolution_px=640)
        synthetic_px = [3.0, 4.0, 5.0, 0.0, 0.0]

        records = []
        for idx, px in enumerate(synthetic_px):
            mrad_val = px / 16.0
            records.append({
                "frame_id": idx,
                "tracker_mode": "KF",
                "detected": True,
                "radial_error_mrad": mrad_val,
                "tracking_error_px": px,  # Explicitly provide pixel error
            })

        record = calc.compute_metrics(frame_records=records)

        expected_rmse = math.sqrt(10.0)  # 3.16227766...
        expected_avg = 2.400
        expected_max = 5.000

        self.assertAlmostEqual(record.rmse_px, expected_rmse, places=6,
                               msg=f"rmse_px {record.rmse_px} did not match expected {expected_rmse}")
        self.assertAlmostEqual(record.avg_tracking_error_px, expected_avg, places=6)
        self.assertAlmostEqual(record.max_tracking_error_px, expected_max, places=6)

    def test_coexistence_of_mrad_and_pixel_metrics(self):
        """
        Confirm that both mrad and pixel-space tracking metrics coexist side-by-side
        without overwriting or removing original mrad values.
        """
        calc = MetricsCalculator(dt=0.033, fov_rad=0.040, resolution_px=640)
        # 1 mrad = 16 pixels
        records = [
            {"frame_id": 0, "tracker_mode": "KF", "radial_error_mrad": 1.0, "detected": True},
            {"frame_id": 1, "tracker_mode": "KF", "radial_error_mrad": 2.0, "detected": True},
        ]
        rec = calc.compute_metrics(frame_records=records)

        # 1. mrad values remain intact
        self.assertAlmostEqual(rec.avg_tracking_error, 1.5, places=5)
        self.assertAlmostEqual(rec.max_tracking_error, 2.0, places=5)

        # 2. pixel values are correctly computed
        self.assertAlmostEqual(rec.avg_tracking_error_px, 24.0, places=5)  # 1.5 * 16 = 24.0
        self.assertAlmostEqual(rec.max_tracking_error_px, 32.0, places=5)  # 2.0 * 16 = 32.0

        # 3. dictionary serialization contains both
        d = rec.to_dict()
        self.assertIn("avg_tracking_error", d)
        self.assertIn("max_tracking_error", d)
        self.assertIn("rmse_tracking_error", d)
        self.assertIn("tracking_error_px", d)
        self.assertIn("max_tracking_error_px", d)
        self.assertIn("rmse_px", d)

    def test_exact_ps_terminology_in_performance_report(self):
        """
        Confirm that serialized performance report contains all PS-mandated field names:
        "Tracking Error", "Target Loss", "Centroiding error", "RMSE",
        "Re-acquisition time", "Lock retention rate", "FPS".
        """
        calc = MetricsCalculator(dt=0.033, fov_rad=0.040, resolution_px=640)
        records = [
            {
                "frame_id": 0,
                "tracker_mode": "KF",
                "radial_error_mrad": 0.5,
                "centroiding_error_px": 0.12,
                "detected": True,
            },
            {
                "frame_id": 1,
                "tracker_mode": "KF",
                "radial_error_mrad": 0.8,
                "centroiding_error_px": 0.18,
                "detected": True,
            },
        ]
        rec = calc.compute_metrics(frame_records=records, start_wall_time=0.0, end_wall_time=0.066)
        d = rec.to_dict()

        # Check standard snake_case keys
        self.assertIn("tracking_error_px", d)
        self.assertIn("target_loss_percent", d)
        self.assertIn("centroiding_error_px", d)
        self.assertIn("rmse_px", d)
        self.assertIn("acquisition_time_s", d)
        self.assertIn("reacquisition_time_s", d)
        self.assertIn("lock_retention_rate", d)
        self.assertIn("fps", d)

        # Check human-readable exact PS Terminology mapping
        self.assertIn("ps_terminology", d)
        ps_dict = d["ps_terminology"]
        required_ps_terms = [
            "Tracking Error",
            "Target Loss",
            "Centroiding error",
            "RMSE",
            "Re-acquisition time",
            "Lock retention rate",
            "FPS",
        ]
        for term in required_ps_terms:
            self.assertIn(term, ps_dict, f"Missing PS-exact terminology: '{term}'")

        # Verify centroiding error averaging
        self.assertAlmostEqual(rec.avg_centroiding_error_px, 0.15, places=5)
        self.assertEqual(len(rec.centroiding_error_log_px), 2)


if __name__ == "__main__":
    unittest.main()
