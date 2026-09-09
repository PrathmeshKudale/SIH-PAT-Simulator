"""
Unit tests for disturb module: KolmogorovTurbulence, PlatformVibration, SensorNoise, and DynamicOccluder.
"""

import unittest
import numpy as np
from disturb.turbulence import KolmogorovTurbulence
from disturb.vibration import PlatformVibration
from disturb.noise import SensorNoise
from disturb.occluder import DynamicOccluder
from sim.target import Target
from sim.camera import Camera
from sim.environment import Environment
from detect.detector import AdaptiveOpticalDetector
from contracts import DisturbanceConfig


class TestDisturbances(unittest.TestCase):

    def test_turbulence_confidence_degrades_with_increasing_cn2(self):
        """
        Confidence score must monotonically drop as atmospheric turbulence Cn2 increases
        (Low -> Medium -> High), proving physical degradation of optical signal.
        """
        camera = Camera(pan=0.0, tilt=0.0)
        detector = AdaptiveOpticalDetector(enable_signature_verification=False)

        cn2_levels = [
            ("low", 1e-16),
            ("medium", 1e-14),
            ("high", 5e-13),
        ]
        results = {}

        for name, cn2 in cn2_levels:
            target = Target(initial_pos=(0.0, 0.0), velocity=(0.0, 0.0), base_intensity=220.0, blink_frequency=0.0, modulation_depth=0.0)
            turb = KolmogorovTurbulence(cn2=cn2, seed=42)
            env = Environment(target=target, camera=camera, turbulence=turb)
            frame, _, _ = env.step(0.033)
            det, _ = detector.detect(frame)
            results[name] = det.confidence if det is not None else 0.0

        # Assert monotonic degradation: low > medium > high
        self.assertGreater(results["low"], results["medium"], "Low Cn2 should yield higher confidence than Medium!")
        self.assertGreater(results["medium"], results["high"], "Medium Cn2 should yield higher confidence than High!")
        self.assertGreater(results["low"], 0.75, "Low turbulence confidence should remain high (>0.75)!")
        self.assertLess(results["high"], 0.40, "High turbulence confidence should drop below 0.40!")

    def test_platform_vibration_jitter(self):
        """Platform vibration should generate sinusoidal + random walk offsets bounded by config."""
        vib = PlatformVibration(amplitude_rad=0.001, frequency_hz=10.0, random_walk_std=0.0001, seed=42)
        jitters = []
        for i in range(50):
            t = i * 0.033
            j_pan, j_tilt = vib.step(t, dt=0.033)
            jitters.append((j_pan, j_tilt))

        # Max jitter should be in the order of 1 to 2 mrad
        max_jitter = max(np.hypot(p, t) for p, t in jitters)
        self.assertLess(max_jitter, 0.003)
        self.assertGreater(max_jitter, 0.0005)

    def test_sensor_noise_application(self):
        """Sensor noise should increase pixel variance and apply impulse noise."""
        noise = SensorNoise(gaussian_std=8.0, salt_pepper_prob=0.01, seed=42)
        clean_img = np.full((100, 100), 50, dtype=np.uint8)
        noisy_img = noise.apply_noise(clean_img)

        # Standard deviation should be close to 8.0
        diff = noisy_img.astype(np.float32) - 50.0
        self.assertGreater(float(np.std(diff)), 6.0)

    def test_dynamic_occluder_blocks_and_recovers(self):
        """Dynamic occluder passing over target must drop detection, then recover."""
        target = Target(initial_pos=(0.0, 0.0), velocity=(0.0, 0.0), base_intensity=220.0, blink_frequency=0.0, modulation_depth=0.0)
        camera = Camera(pan=0.0, tilt=0.0)
        # Starts at -8 mrad, moves at +60 mrad/s (2 mrad/step), radius 4 mrad
        occluder = DynamicOccluder(initial_pos=(-0.008, 0.0), velocity=(0.060, 0.0), radius_rad=0.004, opacity=1.0)
        env = Environment(target=target, camera=camera, occluders=[occluder])
        detector = AdaptiveOpticalDetector(enable_signature_verification=False)

        detection_history = []
        for f in range(9):
            frame, tgt, cam = env.step(0.033)
            det, _ = detector.detect(frame)
            detection_history.append(det is not None)

        # Initial frame (f=0): unoccluded -> True
        self.assertTrue(detection_history[0], "Should be detected initially!")
        # Mid frames (f=3, 4): occluded -> False
        self.assertFalse(detection_history[3], "Should be occluded at frame 3!")
        self.assertFalse(detection_history[4], "Should be occluded at frame 4!")
        # Post frame (f=8): recovered -> True
        self.assertTrue(detection_history[8], "Should recover detection after occluder passes!")


if __name__ == "__main__":
    unittest.main()
