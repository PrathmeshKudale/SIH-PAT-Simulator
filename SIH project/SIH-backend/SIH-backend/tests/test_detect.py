"""
Unit tests for detect module: AdaptiveOpticalDetector, Top-Hat filtering,
blob quality scoring, and temporal blinking signature verification.
"""

import unittest
import numpy as np
from detect.detector import AdaptiveOpticalDetector, BeaconVerifier, BlobCandidate
from contracts import FrameData


class TestDetect(unittest.TestCase):

    def test_tophat_and_candidate_extraction(self):
        # Create image with non-uniform gradient background + localized spot
        h, w = 480, 640
        y_grad = np.linspace(20, 80, h).reshape(-1, 1).repeat(w, axis=1)
        spot = 200.0 * np.exp(-((np.arange(w) - 300.0)**2 + (np.arange(h)[:, None] - 200.0)**2) / (2.0 * 2.5**2))
        img = np.clip(y_grad + spot, 0, 255).astype(np.uint8)

        detector = AdaptiveOpticalDetector(tophat_kernel_size=7)
        cands = detector.extract_candidates(img)

        self.assertGreater(len(cands), 0)
        best = cands[0]
        # Spot centroid should be accurately found despite background gradient
        self.assertAlmostEqual(best.x, 300.0, delta=0.2)
        self.assertAlmostEqual(best.y, 200.0, delta=0.2)
        self.assertGreater(best.confidence, 0.5)

    def test_beacon_verifier_rejects_static_clutter(self):
        """Verifier must reject static, non-blinking bright spots as clutter."""
        verifier = BeaconVerifier(min_history_len=6)

        rng = np.random.default_rng(123)
        for i in range(8):
            steady_intensity = 210.0 + rng.normal(0, 1.5)
            cand = BlobCandidate(
                x=150.0, y=100.0,
                peak_intensity=steady_intensity,
                flux=1000.0, area=25,
                contrast=0.8, size_consistency=0.9, confidence=0.85
            )
            verified_blob, report = verifier.update([cand], timestamp=i * 0.033)

        self.assertIsNone(verified_blob)
        self.assertEqual(report["clutter_rejected"], 1)
        self.assertFalse(verifier.tracks[1]["is_verified"])
        self.assertTrue(verifier.tracks[1]["is_clutter"])

    def test_beacon_verifier_accepts_blinking_target(self):
        """Verifier must confirm target blinking periodically at 4 Hz."""
        verifier = BeaconVerifier(min_history_len=6)

        for i in range(10):
            t = i * 0.0333
            mod_intensity = 120.0 + 90.0 * np.sin(2.0 * np.pi * 4.0 * t)
            cand = BlobCandidate(
                x=320.0, y=240.0,
                peak_intensity=mod_intensity,
                flux=1000.0, area=25,
                contrast=0.8, size_consistency=0.9, confidence=0.85
            )
            verified_blob, report = verifier.update([cand], timestamp=t)

        self.assertIsNotNone(verified_blob)
        self.assertTrue(verifier.tracks[1]["is_verified"])
        self.assertFalse(verifier.tracks[1]["is_clutter"])

    def test_verifier_rejects_brighter_clutter_and_selects_dimmer_beacon(self):
        """
        Stress test: When both a saturated static clutter spot (intensity 255.0) and a dimmer
        blinking beacon (peak 180.0) are present simultaneously in the same frames, the verifier
        must reject the brighter spot and lock strictly onto the dimmer blinking beacon.
        """
        verifier = BeaconVerifier(min_history_len=6)
        rng = np.random.default_rng(42)

        target_x, target_y = 350.0, 200.0
        clutter_x, clutter_y = 150.0, 100.0

        for i in range(10):
            t = i * 0.0333
            # Dimmer blinking beacon: peak 180, modulation between 30 and 180
            tgt_intensity = 105.0 + 75.0 * np.sin(2.0 * np.pi * 4.0 * t)
            tgt_cand = BlobCandidate(
                x=target_x, y=target_y,
                peak_intensity=tgt_intensity,
                flux=700.0, area=22,
                contrast=0.7, size_consistency=0.9, confidence=0.75
            )

            # Saturated brighter clutter: steady 255.0 +- 1.0 noise
            clutter_intensity = 255.0 + rng.normal(0, 0.5)
            clutter_cand = BlobCandidate(
                x=clutter_x, y=clutter_y,
                peak_intensity=clutter_intensity,
                flux=1200.0, area=28,
                contrast=0.95, size_consistency=0.95, confidence=0.95
            )

            # Feed both candidates simultaneously
            verified_blob, report = verifier.update([clutter_cand, tgt_cand], timestamp=t)

        # Confirm verifier locked onto the dimmer target, NOT the brighter clutter
        self.assertIsNotNone(verified_blob, "Target was not acquired!")
        self.assertAlmostEqual(verified_blob.x, target_x, delta=1.0)
        self.assertAlmostEqual(verified_blob.y, target_y, delta=1.0)
        self.assertEqual(report["clutter_rejected"], 1)


if __name__ == "__main__":
    unittest.main()
