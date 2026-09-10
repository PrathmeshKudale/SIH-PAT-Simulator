"""
Unit and Integration Tests for Environmental Disturbances and Sensor Noise Models.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Tests:
1. Standalone apply_poisson_noise on flat-field image (valid uint8, dynamic range, variance ≈ mean).
2. Poisson scaling behavior (scale_factor modulation of photon count variance).
3. SensorNoise with Poisson noise alone.
4. Stacked Poisson + Gaussian combined noise (composite variance ≈ Var_poisson + Var_gaussian).
5. Stacked Poisson + Gaussian + Salt & Pepper impulse noise.
6. DisturbanceConfig schema validation, noise_types list serialization, and from_dict.
7. AdaptiveOpticalDetector centroiding and confidence smoke-test under Poisson noise (no NaNs, no crashes).
8. Backward compatibility with legacy scenario configurations.
"""

import unittest
import numpy as np

from contracts import DisturbanceConfig, TargetConfig, FrameData
from disturb.noise import (
    SensorNoise,
    apply_poisson_noise,
    VALID_NOISE_TYPES,
    normalize_noise_type,
)
from sim.camera import Camera
from sim.target import Target
from sim.environment import Environment
from detect.detector import AdaptiveOpticalDetector


class TestSensorDisturbances(unittest.TestCase):
    """Test suite for sensor noise models, Poisson shot noise, and composite pipelines."""

    def test_apply_poisson_noise_flat_field_statistics(self):
        """
        Verify apply_poisson_noise produces valid uint8 output whose variance
        approximates the mean intensity for a flat-field test image when scale_factor=1.0.
        """
        # Create a uniform flat-field test image (e.g. 200x200 = 40,000 pixels at intensity 100)
        mean_intensity = 100.0
        flat_img = np.full((200, 200), int(mean_intensity), dtype=np.uint8)
        
        rng = np.random.default_rng(12345)
        noisy = apply_poisson_noise(flat_img, scale_factor=1.0, rng=rng)

        # 1. Type and bounds verification
        self.assertEqual(noisy.dtype, np.uint8)
        self.assertEqual(noisy.shape, flat_img.shape)
        self.assertGreaterEqual(int(np.min(noisy)), 0)
        self.assertLessEqual(int(np.max(noisy)), 255)

        # 2. Statistical expectation: E[I_noisy] ≈ mean_intensity
        measured_mean = float(np.mean(noisy))
        self.assertAlmostEqual(measured_mean, mean_intensity, delta=1.0,
                               msg=f"Poisson mean {measured_mean} deviated from expected {mean_intensity}")

        # 3. Statistical variance: Var(I_noisy) ≈ mean_intensity (Poisson property: Var = Mean)
        measured_var = float(np.var(noisy))
        # With 40,000 samples, standard error of sample variance is ~0.7
        self.assertAlmostEqual(measured_var, mean_intensity, delta=15.0,
                               msg=f"Poisson variance {measured_var} not close to mean {mean_intensity}")

    def test_poisson_noise_scaling_factors(self):
        """
        Verify that scale_factor scales the effective photon count:
        Var(I_noisy) ≈ mean_intensity / scale_factor.
        """
        mean_intensity = 100.0
        flat_img = np.full((200, 200), int(mean_intensity), dtype=np.uint8)
        rng = np.random.default_rng(42)

        # High photon flux (scale_factor = 4.0) -> lower relative variance ≈ 100 / 4 = 25.0
        noisy_high_flux = apply_poisson_noise(flat_img, scale_factor=4.0, rng=rng)
        var_high_flux = float(np.var(noisy_high_flux))
        self.assertAlmostEqual(var_high_flux, 25.0, delta=6.0)

        # Low photon flux (scale_factor = 0.5) -> higher relative variance ≈ 100 / 0.5 = 200.0
        noisy_low_flux = apply_poisson_noise(flat_img, scale_factor=0.5, rng=rng)
        var_low_flux = float(np.var(noisy_low_flux))
        self.assertAlmostEqual(var_low_flux, 200.0, delta=30.0)

    def test_sensor_noise_poisson_alone(self):
        """Verify SensorNoise initialized with noise_types=['poisson'] applies only Poisson noise."""
        noise = SensorNoise(
            gaussian_std=0.0,
            salt_pepper_prob=0.0,
            poisson_scale=1.0,
            noise_types=["poisson"],
            seed=999,
        )
        self.assertEqual(noise.noise_types, ["poisson"])

        flat_img = np.full((150, 150), 80, dtype=np.uint8)
        noisy = noise.apply_noise(flat_img)

        self.assertEqual(noisy.dtype, np.uint8)
        self.assertAlmostEqual(float(np.mean(noisy)), 80.0, delta=1.5)
        # Var should be close to 80.0
        self.assertAlmostEqual(float(np.var(noisy)), 80.0, delta=15.0)

    def test_stacked_poisson_and_gaussian_noise(self):
        """
        Verify that Poisson and Gaussian noise stack additively/sequentially:
        Total Var ≈ Var(Poisson) + Var(Gaussian) = mean_intensity + (gaussian_std)^2.
        """
        mean_intensity = 100.0
        flat_img = np.full((200, 200), int(mean_intensity), dtype=np.uint8)
        gauss_std = 10.0  # Gaussian variance = 100.0

        # Expected combined variance = 100 (Poisson) + 100 (Gaussian) = 200.0
        noise = SensorNoise(
            gaussian_std=gauss_std,
            poisson_scale=1.0,
            noise_types=["poisson", "gaussian"],
            seed=42,
        )
        noisy = noise.apply_noise(flat_img)

        self.assertEqual(noisy.dtype, np.uint8)
        measured_mean = float(np.mean(noisy))
        measured_var = float(np.var(noisy))

        self.assertAlmostEqual(measured_mean, mean_intensity, delta=2.0)
        self.assertAlmostEqual(measured_var, 200.0, delta=30.0,
                               msg=f"Composite variance {measured_var} expected near 200.0")

    def test_stacked_all_three_noise_models(self):
        """Verify sequential stacking of Poisson + Gaussian + Salt & Pepper impulse noise."""
        flat_img = np.full((100, 100), 128, dtype=np.uint8)
        noise = SensorNoise(
            gaussian_std=5.0,
            poisson_scale=1.0,
            salt_pepper_prob=0.02,  # 2% impulse pixels
            noise_types=["poisson", "gaussian", "salt_pepper"],
            seed=777,
        )
        noisy = noise.apply_noise(flat_img)

        self.assertEqual(noisy.dtype, np.uint8)
        # Check presence of hot pixels (255) and dead pixels (0)
        self.assertIn(255, noisy)
        self.assertIn(0, noisy)

    def test_ps_mandated_extreme_noise_bounds(self):
        """
        Verify PS-mandated extreme stress boundaries:
        - Gaussian readout noise up to 20.0 counts/pixels std-dev
        - Salt-and-pepper impulse density up to 10% (0.10) of total sensor pixels
        Confirms no arithmetic overflow, valid uint8 range [0, 255], and expected corruption ratio.
        """
        img = np.full((200, 200), 128, dtype=np.uint8)
        total_pixels = img.size  # 40,000

        noise = SensorNoise(
            gaussian_std=20.0,
            salt_pepper_prob=0.10,
            noise_types=["gaussian", "salt_pepper"],
            seed=42,
        )
        noisy = noise.apply_noise(img)

        self.assertEqual(noisy.dtype, np.uint8)
        self.assertEqual(noisy.shape, img.shape)
        self.assertEqual(int(np.min(noisy)), 0)
        self.assertEqual(int(np.max(noisy)), 255)

        # Count impulse pixels: salt (255) and pepper (0)
        num_salt = int(np.count_nonzero(noisy == 255))
        num_pepper = int(np.count_nonzero(noisy == 0))
        total_corrupted = num_salt + num_pepper

        # Expected ~10% corrupted: 5% salt (2000), 5% pepper (2000), total ~4000
        corrupted_ratio = total_corrupted / total_pixels
        self.assertAlmostEqual(corrupted_ratio, 0.10, delta=0.015,
                               msg=f"Expected ~10% impulse density, got {corrupted_ratio:.3f}")

    def test_noise_aliases_and_normalization(self):
        """Verify alias normalization and rejection of invalid noise types."""
        self.assertEqual(normalize_noise_type("shot"), "poisson")
        self.assertEqual(normalize_noise_type("photon"), "poisson")
        self.assertEqual(normalize_noise_type("readout"), "gaussian")
        self.assertEqual(normalize_noise_type("normal"), "gaussian")
        self.assertEqual(normalize_noise_type("sp"), "salt_pepper")
        self.assertEqual(normalize_noise_type("salt_and_pepper"), "salt_pepper")

        with self.assertRaises(ValueError):
            normalize_noise_type("cosmic_ray_annihilation")

    def test_disturbance_config_schema_and_serialization(self):
        """Verify DisturbanceConfig serialization, defaults, and from_dict loading."""
        cfg = DisturbanceConfig(
            noise_level=6.0,
            noise_types=["poisson", "gaussian"],
            poisson_scale=1.5,
            salt_pepper_prob=0.001,
        )
        d = cfg.to_dict()
        self.assertEqual(d["noise_level"], 6.0)
        self.assertEqual(d["noise_types"], ["poisson", "gaussian"])
        self.assertEqual(d["poisson_scale"], 1.5)
        self.assertEqual(d["salt_pepper_prob"], 0.001)

        # Deserialize back
        restored = DisturbanceConfig.from_dict(d)
        self.assertEqual(restored.noise_types, ["poisson", "gaussian"])
        self.assertEqual(restored.poisson_scale, 1.5)
        self.assertEqual(restored.noise_level, 6.0)

    def test_legacy_backward_compatibility_when_noise_types_omitted(self):
        """Legacy configs without noise_types default to active models based on parameters."""
        # Gaussian alone when salt_pepper_prob is zero
        noise_gauss_only = SensorNoise(gaussian_std=8.0, salt_pepper_prob=0.0, seed=123)
        self.assertEqual(noise_gauss_only.noise_types, ["gaussian"])

        # Gaussian + Salt & Pepper when both parameters are positive
        noise_legacy = SensorNoise(gaussian_std=8.0, salt_pepper_prob=0.0005, seed=123)
        self.assertEqual(noise_legacy.noise_types, ["gaussian", "salt_pepper"])

        # No noise when noise_std = 0 and salt_pepper_prob = 0
        noise_zero = SensorNoise(gaussian_std=0.0, salt_pepper_prob=0.0)
        self.assertEqual(noise_zero.noise_types, [])

    def test_detector_smoke_test_under_poisson_noise(self):
        """
        Verify that AdaptiveOpticalDetector centroiding and confidence estimation
        function robustly under Poisson shot noise without crashes, NaNs, or infs.
        """
        scenario = {
            "scenario_name": "POISSON_SMOKE_TEST",
            "dt": 0.033,
            "target": {
                "initial_pos": [0.004, -0.002],
                "velocity": [0.0, 0.0],
                "base_intensity": 230.0,
            },
            "disturbances": {
                "noise_types": ["poisson"],
                "poisson_scale": 1.0,
            },
            "seed": 42,
        }
        env = Environment.from_config(scenario)
        detector = AdaptiveOpticalDetector(enable_signature_verification=False)

        # Step simulation for several frames
        confidences = []
        centroids = []
        for f in range(5):
            frame, tgt, cam = env.step(0.033)
            det, _ = detector.detect(frame)
            self.assertIsNotNone(det, f"Target detection failed at frame {f} under Poisson noise")
            self.assertGreater(len(det), 0, f"Target detection list empty at frame {f}")
            
            # Verify no NaNs or Infs in detected centroid or confidence
            self.assertFalse(np.isnan(det.x), f"Centroid x is NaN at frame {f}")
            self.assertFalse(np.isnan(det.y), f"Centroid y is NaN at frame {f}")
            self.assertFalse(np.isnan(det.confidence), f"Confidence is NaN at frame {f}")
            self.assertGreater(det.confidence, 0.0)
            self.assertLessEqual(det.confidence, 1.0)
            
            confidences.append(det.confidence)
            centroids.append((det.x, det.y))

        # Detector should consistently report strong confidence for clean beacon
        avg_conf = float(np.mean(confidences))
        self.assertGreater(avg_conf, 0.5, f"Expected robust detector confidence, got {avg_conf}")


if __name__ == "__main__":
    unittest.main()
