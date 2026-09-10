"""
Unit and Integration Tests for Atmospheric Weather Disturbances and Illumination Conditions.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Tests cover:
1. Normalization of atmospheric condition names and aliases.
2. Directional rain blur kernel synthesis and normalization.
3. Clear condition verification as an exact identity no-op.
4. Haze contrast reduction and gray airlight lift (Koschmieder's law).
5. Fog condition verification as strictly stronger than haze (lower contrast, higher veiling, heavier blur).
6. Rain directional streak convolution and high-frequency precipitation noise.
7. Low-light exposure reduction (>50% attenuation) and noise floor preservation.
8. User-defined parameter overrides (transmission, airlight, exposure_scale, rain streak params).
9. DisturbanceConfig dataclass schema validation and serialization.
10. Environment integration with atmospheric disturbances and profiling timers.
11. AdaptiveOpticalDetector robustness smoke test across all 5 atmospheric conditions (no NaNs, no crashes).
"""

import unittest
import numpy as np
import cv2

from contracts import DisturbanceConfig, FrameData
from disturb.atmospheric import (
    AtmosphericDisturbance,
    VALID_ATMOSPHERIC_CONDITIONS,
    ATMOSPHERIC_CONDITION_ALIASES,
    normalize_atmospheric_condition,
    generate_directional_rain_kernel,
)
from sim.environment import Environment
from detect.detector import AdaptiveOpticalDetector


class TestAtmosphericDisturbances(unittest.TestCase):
    """Test suite for atmospheric conditions: Clear, Haze, Fog, Rain, and Low-light."""

    def test_normalize_atmospheric_condition(self):
        """Verify normalization for canonical names, case insensitivity, and aliases."""
        for cond in VALID_ATMOSPHERIC_CONDITIONS:
            self.assertEqual(normalize_atmospheric_condition(cond), cond)
            self.assertEqual(normalize_atmospheric_condition(cond.upper()), cond)
            self.assertEqual(normalize_atmospheric_condition(f"  {cond}  "), cond)

        # Aliases
        self.assertEqual(normalize_atmospheric_condition("none"), "clear")
        self.assertEqual(normalize_atmospheric_condition("nominal"), "clear")
        self.assertEqual(normalize_atmospheric_condition("mist"), "haze")
        self.assertEqual(normalize_atmospheric_condition("heavy_fog"), "fog")
        self.assertEqual(normalize_atmospheric_condition("dense_fog"), "fog")
        self.assertEqual(normalize_atmospheric_condition("rainy"), "rain")
        self.assertEqual(normalize_atmospheric_condition("precipitation"), "rain")
        self.assertEqual(normalize_atmospheric_condition("night"), "low_light")
        self.assertEqual(normalize_atmospheric_condition("dark"), "low_light")
        self.assertEqual(normalize_atmospheric_condition("low-light"), "low_light")

        with self.assertRaises(ValueError):
            normalize_atmospheric_condition("tornado")

    def test_generate_directional_rain_kernel(self):
        """Verify directional rain streak kernel properties."""
        kernel = generate_directional_rain_kernel(length=15, angle_deg=75.0)
        self.assertEqual(kernel.shape, (15, 15))
        self.assertAlmostEqual(float(np.sum(kernel)), 1.0, places=5)

        # Horizontal streak (angle = 0) should be mostly aligned on center row
        k_horiz = generate_directional_rain_kernel(length=11, angle_deg=0.0)
        self.assertGreater(float(np.sum(k_horiz[5, :])), 0.8)

    def test_clear_is_strict_noop(self):
        """Verify that Clear condition is an exact no-op returning identical pixels."""
        rng = np.random.default_rng(42)
        test_img = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)

        dist = AtmosphericDisturbance(condition="clear")
        out_img = dist.apply(test_img)

        self.assertEqual(out_img.dtype, np.uint8)
        self.assertTrue(np.array_equal(out_img, test_img))
        self.assertEqual(out_img.tobytes(), test_img.tobytes())
        self.assertIs(out_img, test_img)

    def test_haze_contrast_reduction_and_gray_lift(self):
        """
        Verify Haze reduces contrast (lower std dev) and lifts dark background
        pixels toward gray airlight level.
        """
        # Create a high-contrast test image (black background 0 with bright targets 240)
        img = np.zeros((120, 120), dtype=np.uint8)
        img[30:90, 30:90] = 240

        std_clear = float(np.std(img))
        min_clear = float(np.min(img))

        dist = AtmosphericDisturbance(condition="haze", seed=42)
        hazy = dist.apply(img)

        self.assertEqual(hazy.dtype, np.uint8)
        self.assertEqual(hazy.shape, img.shape)

        std_haze = float(np.std(hazy))
        min_haze = float(np.min(hazy))

        # Contrast should drop significantly (std dev lower by at least 15%)
        self.assertLess(std_haze, std_clear * 0.85,
                        f"Expected haze to reduce contrast: std_clear={std_clear}, std_haze={std_haze}")
        # Dark regions should be lifted toward gray airlight (A ~ 60)
        self.assertGreater(min_haze, min_clear + 10.0,
                           f"Expected haze to lift dark floor: min_clear={min_clear}, min_haze={min_haze}")

    def test_fog_strictly_stronger_than_haze(self):
        """
        Verify Fog is strictly stronger than Haze:
        1. Fog contrast (std dev) is strictly lower than Haze contrast.
        2. Fog high-frequency edge response (Laplacian variance) is strictly lower (heavier blur).
        3. Fog background lift is higher (denser veiling airlight).
        """
        # Create a test pattern with edges and high contrast
        img = np.zeros((160, 160), dtype=np.uint8)
        for i in range(0, 160, 20):
            img[:, i:i+10] = 220

        haze_dist = AtmosphericDisturbance(condition="haze", seed=101)
        fog_dist = AtmosphericDisturbance(condition="fog", seed=101)

        haze_img = haze_dist.apply(img)
        fog_img = fog_dist.apply(img)

        std_clear = float(np.std(img))
        std_haze = float(np.std(haze_img))
        std_fog = float(np.std(fog_img))

        # 1. Strict hierarchy of contrast: std_fog < std_haze < std_clear
        self.assertLess(std_haze, std_clear)
        self.assertLess(std_fog, std_haze,
                        f"Fog contrast ({std_fog}) must be strictly lower than Haze ({std_haze})")

        # 2. Heavier spatial diffusion / blur in Fog vs Haze
        lap_haze = cv2.Laplacian(haze_img, cv2.CV_64F).var()
        lap_fog = cv2.Laplacian(fog_img, cv2.CV_64F).var()
        self.assertLess(lap_fog, lap_haze,
                        f"Fog edge sharpness ({lap_fog}) must be strictly lower than Haze ({lap_haze})")

        # 3. Dense veiling luminance is higher in fog
        self.assertGreater(float(np.min(fog_img)), float(np.min(haze_img)))

    def test_rain_streaks_and_variance(self):
        """
        Verify Rain introduces directional streak patterns and high-frequency noise.
        """
        # Uniform flat-field image
        flat = np.full((128, 128), 100, dtype=np.uint8)
        dist = AtmosphericDisturbance(condition="rain", seed=42)
        rain_img = dist.apply(flat)

        self.assertEqual(rain_img.dtype, np.uint8)
        self.assertEqual(rain_img.shape, flat.shape)

        # Variance on uniform image was 0.0, rain must introduce non-trivial variance
        var_rain = float(np.var(rain_img))
        self.assertGreater(var_rain, 5.0, f"Expected rain to add streak/splash variance, got {var_rain}")

    def test_low_light_exposure_reduction(self):
        """
        Verify Low-light condition scales optical exposure down by >50%.
        """
        # Scene with mid-level illumination
        scene = np.full((128, 128), 180, dtype=np.uint8)
        dist = AtmosphericDisturbance(condition="low_light", seed=42)
        dark_img = dist.apply(scene)

        mean_orig = float(np.mean(scene))
        mean_dark = float(np.mean(dark_img))

        # Exposure should drop by at least 50%
        self.assertLess(mean_dark, mean_orig * 0.5,
                        f"Expected low-light to attenuate mean brightness: orig={mean_orig}, dark={mean_dark}")
        # Valid range
        self.assertGreaterEqual(int(np.min(dark_img)), 0)
        self.assertLessEqual(int(np.max(dark_img)), 255)

    def test_parameter_overrides(self):
        """Verify custom user parameters override default values."""
        # Low light with custom exposure scale 0.10
        dist_custom = AtmosphericDisturbance(
            condition="low_light",
            params={"exposure_scale": 0.10, "dark_floor_std": 0.0},
        )
        img = np.full((50, 50), 200, dtype=np.uint8)
        out = dist_custom.apply(img)
        self.assertAlmostEqual(float(np.mean(out)), 20.0, delta=2.0)

        # Haze with custom transmission 0.90 (light haze)
        dist_light = AtmosphericDisturbance(
            condition="haze",
            params={"transmission": 0.90, "airlight": 50.0, "blur_sigma": 0.0},
        )
        out_light = dist_light.apply(img)
        # 200 * 0.9 + 50 * 0.1 = 180 + 5 = 185
        self.assertAlmostEqual(float(np.mean(out_light)), 185.0, delta=2.0)

    def test_disturbance_config_dataclass_and_serialization(self):
        """Verify DisturbanceConfig support for atmospheric_condition and atmospheric_params."""
        cfg = DisturbanceConfig(
            atmospheric_condition="rain",
            atmospheric_params={"rain_density": 0.003, "streak_length": 20},
            noise_types=["gaussian", "poisson"],
        )
        self.assertEqual(cfg.atmospheric_condition, "rain")
        self.assertEqual(cfg.atmospheric_params["streak_length"], 20)

        d = cfg.to_dict()
        self.assertEqual(d["atmospheric_condition"], "rain")
        self.assertEqual(d["atmospheric_params"]["rain_density"], 0.003)

        restored = DisturbanceConfig.from_dict(d)
        self.assertEqual(restored.atmospheric_condition, "rain")
        self.assertEqual(restored.atmospheric_params["streak_length"], 20)

    def test_environment_integration_and_profiling(self):
        """
        Verify Environment.from_config correctly creates AtmosphericDisturbance
        and records execution time in last_step_profile.
        """
        scenario = {
            "scenario_name": "ATMOSPHERIC_INTEGRATION_TEST",
            "dt": 0.033,
            "target": {
                "initial_pos": [0.002, 0.001],
                "velocity": [0.0, 0.0],
                "base_intensity": 220.0,
            },
            "disturbances": {
                "atmospheric_condition": "fog",
            },
            "seed": 42,
        }
        env = Environment.from_config(scenario)
        self.assertIsNotNone(env.atmospheric_dist)
        self.assertEqual(env.atmospheric_dist.condition, "fog")

        frame, tgt_state, cam_state = env.step(0.033)
        self.assertEqual(frame.image.dtype, np.uint8)
        self.assertIn("disturb_atmospheric_ms", env.last_step_profile)
        self.assertGreaterEqual(env.last_step_profile["disturb_atmospheric_ms"], 0.0)

    def test_detector_robustness_across_all_conditions(self):
        """
        Smoke test verifying AdaptiveOpticalDetector processes frames under
        all 5 conditions without crashing, raising exceptions, or producing NaNs.
        """
        detector = AdaptiveOpticalDetector(enable_signature_verification=False)

        for cond in VALID_ATMOSPHERIC_CONDITIONS:
            scenario = {
                "scenario_name": f"DETECTOR_{cond.upper()}_TEST",
                "dt": 0.033,
                "target": {
                    "initial_pos": [0.003, -0.002],
                    "velocity": [0.0, 0.0],
                    "base_intensity": 240.0,
                },
                "disturbances": {
                    "atmospheric_condition": cond,
                },
                "seed": 42,
            }
            env = Environment.from_config(scenario)

            # Advance 3 frames
            for f in range(3):
                frame, tgt_state, cam_state = env.step(0.033)
                det, _ = detector.detect(frame)

                if det is not None:
                    self.assertFalse(np.isnan(det.x), f"Centroid x is NaN under {cond} at frame {f}")
                    self.assertFalse(np.isnan(det.y), f"Centroid y is NaN under {cond} at frame {f}")
                    self.assertFalse(np.isnan(det.confidence), f"Confidence is NaN under {cond} at frame {f}")
                    self.assertFalse(np.isinf(det.x), f"Centroid x is Inf under {cond} at frame {f}")
                    self.assertFalse(np.isinf(det.y), f"Centroid y is Inf under {cond} at frame {f}")
                    self.assertGreaterEqual(det.confidence, 0.0)
                    self.assertLessEqual(det.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
