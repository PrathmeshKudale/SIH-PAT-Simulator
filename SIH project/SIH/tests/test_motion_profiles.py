"""
Unit and Integration Tests for Parametric Motion Profiles (Target and Platform Kinematics).
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Tests:
1. Mathematical trajectory correctness for all profiles (straight_line, circular, figure_eight,
   random, spiral, sinusoidal).
2. Radius constancy for circular motion within tight tolerance.
3. Center-crossing and amplitude containment for Lissajous figure-8 lemniscate.
4. Bounded stochastic drift and specular boundary reflection for random walk.
5. TargetConfig schema validation, aliases, defaults, and legacy backward compatibility.
6. Target class integration with selectable motion strategies.
7. Environment platform motion generation reusing MotionProfile strategies.
"""

import unittest
import numpy as np

from contracts import TargetConfig, normalize_scenario_targets, VALID_MOTION_TYPES
from sim.motion_profiles import (
    MotionProfile,
    StraightLineProfile,
    CircularProfile,
    FigureEightProfile,
    RandomWalkProfile,
    SpiralProfile,
    SinusoidalProfile,
    create_motion_profile,
    normalize_motion_type,
)
from sim.target import Target
from sim.camera import Camera
from sim.environment import Environment


class TestMotionProfiles(unittest.TestCase):
    """Test suite for individual kinematic motion profile generators."""

    def test_straight_line_profile(self):
        """Verify linear rectilinear motion advances precisely as x0 + v*t."""
        prof = StraightLineProfile(initial_pos=(0.010, -0.005), velocity=(0.003, -0.002))
        
        # t = 0
        x0, y0 = prof.get_position(0.0)
        self.assertAlmostEqual(x0, 0.010, places=7)
        self.assertAlmostEqual(y0, -0.005, places=7)

        # t = 2.5s
        x2, y2 = prof.get_position(2.5)
        self.assertAlmostEqual(x2, 0.010 + 0.003 * 2.5, places=7)
        self.assertAlmostEqual(y2, -0.005 - 0.002 * 2.5, places=7)

        # Constant velocity
        vx, vy = prof.get_velocity(2.5)
        self.assertAlmostEqual(vx, 0.003, places=7)
        self.assertAlmostEqual(vy, -0.002, places=7)

    def test_circular_profile_radius_and_orthogonality(self):
        """
        Verify circular orbit stays within radius tolerance at all times,
        velocity vector is strictly orthogonal to radius vector, and speed = R * omega.
        """
        center = (0.015, -0.010)
        radius = 0.008
        omega = 1.5  # rad/s
        prof = CircularProfile(center=center, radius=radius, angular_velocity=omega, phase_offset=0.2)

        for t in np.linspace(0.0, 10.0, 100):
            x, y = prof.get_position(t)
            dist = np.hypot(x - center[0], y - center[1])
            self.assertAlmostEqual(dist, radius, delta=1e-7, msg=f"Radius deviation at t={t}")

            # Velocity verification
            vx, vy = prof.get_velocity(t)
            speed = np.hypot(vx, vy)
            self.assertAlmostEqual(speed, radius * omega, delta=1e-6)

            # Dot product (r_vec . v_vec == 0)
            rx = x - center[0]
            ry = y - center[1]
            dot = rx * vx + ry * vy
            self.assertAlmostEqual(dot, 0.0, delta=1e-7)

    def test_figure_eight_center_crossing_and_bounds(self):
        """
        Verify Lissajous figure-8 lemniscate crosses center at periodic intervals
        and remains strictly bounded within (amplitude_x, amplitude_y).
        """
        center = (0.005, 0.002)
        ax = 0.012
        ay = 0.006
        freq = 0.25  # Hz -> period T = 4.0s
        prof = FigureEightProfile(center=center, amplitude_x=ax, amplitude_y=ay, frequency=freq)

        # At t = 0, figure-8 crosses center
        x0, y0 = prof.get_position(0.0)
        self.assertAlmostEqual(x0, center[0], places=7)
        self.assertAlmostEqual(y0, center[1], places=7)

        # At t = half-period (2.0s) and full-period (4.0s), crosses center
        for t_cross in [2.0, 4.0, 6.0]:
            xc, yc = prof.get_position(t_cross)
            self.assertAlmostEqual(xc, center[0], delta=1e-6, msg=f"Center x crossing failed at t={t_cross}")
            self.assertAlmostEqual(yc, center[1], delta=1e-6, msg=f"Center y crossing failed at t={t_cross}")

        # Bounds containment
        for t in np.linspace(0.0, 8.0, 160):
            x, y = prof.get_position(t)
            self.assertLessEqual(abs(x - center[0]), ax + 1e-7)
            self.assertLessEqual(abs(y - center[1]), ay + 1e-7)

    def test_random_walk_boundary_reflection(self):
        """
        Verify bounded stochastic random walk strictly respects boundaries
        via specular reflection over a large step sequence.
        """
        bounds = (-0.015, 0.015, -0.010, 0.010)
        prof = RandomWalkProfile(
            initial_pos=(0.0, 0.0),
            step_variance=0.005,  # High variance to force multiple boundary collisions
            bounds=bounds,
            seed=123,
        )

        min_x, max_x, min_y, max_y = bounds
        for t in np.linspace(0.0, 5.0, 100):
            x, y = prof.get_position(t)
            self.assertGreaterEqual(x, min_x - 1e-9, msg=f"x lower bound violated at t={t}")
            self.assertLessEqual(x, max_x + 1e-9, msg=f"x upper bound violated at t={t}")
            self.assertGreaterEqual(y, min_y - 1e-9, msg=f"y lower bound violated at t={t}")
            self.assertLessEqual(y, max_y + 1e-9, msg=f"y upper bound violated at t={t}")

    def test_random_walk_reproducibility_and_reset(self):
        """Verify seeded random walk is fully reproducible and resets to initial position."""
        prof1 = RandomWalkProfile(initial_pos=(0.002, -0.001), step_variance=0.001, seed=999)
        prof2 = RandomWalkProfile(initial_pos=(0.002, -0.001), step_variance=0.001, seed=999)

        pos1_seq = [prof1.get_position(t) for t in np.linspace(0.0, 2.0, 20)]
        pos2_seq = [prof2.get_position(t) for t in np.linspace(0.0, 2.0, 20)]

        for p1, p2 in zip(pos1_seq, pos2_seq):
            self.assertAlmostEqual(p1[0], p2[0], places=7)
            self.assertAlmostEqual(p1[1], p2[1], places=7)

        # Reset prof1
        prof1.reset()
        x_reset, y_reset = prof1.get_position(0.0)
        self.assertAlmostEqual(x_reset, 0.002, places=7)
        self.assertAlmostEqual(y_reset, -0.001, places=7)

    def test_spiral_profile_expansion(self):
        """Verify spiral profile radius expands at the configured rate."""
        center = (0.0, 0.0)
        r0 = 0.002
        vr = 0.001
        omega = 2.0
        prof = SpiralProfile(center=center, initial_radius=r0, radial_velocity=vr, angular_velocity=omega)

        for t in [0.0, 1.0, 2.5, 4.0]:
            x, y = prof.get_position(t)
            expected_r = r0 + vr * t
            measured_r = np.hypot(x, y)
            self.assertAlmostEqual(measured_r, expected_r, delta=1e-6)

    def test_sinusoidal_profile(self):
        """Verify sinusoidal wave drift and transverse oscillation."""
        prof = SinusoidalProfile(
            initial_pos=(0.0, 0.0),
            velocity=(0.004, 0.0),
            amplitude=0.003,
            frequency=1.0,  # Period 1.0s
            axis="y",
        )
        # At t = 0, y = 0
        x0, y0 = prof.get_position(0.0)
        self.assertAlmostEqual(x0, 0.0)
        self.assertAlmostEqual(y0, 0.0)

        # At t = 0.25s (quarter wave), y = amplitude
        x_q, y_q = prof.get_position(0.25)
        self.assertAlmostEqual(x_q, 0.004 * 0.25, places=6)
        self.assertAlmostEqual(y_q, 0.003, places=6)

    def test_create_motion_profile_factory_and_aliases(self):
        """Verify create_motion_profile normalizes aliases and rejects invalid types."""
        self.assertIsInstance(create_motion_profile("straight_line"), StraightLineProfile)
        self.assertIsInstance(create_motion_profile("linear"), StraightLineProfile)
        self.assertIsInstance(create_motion_profile("circular"), CircularProfile)
        self.assertIsInstance(create_motion_profile("circle"), CircularProfile)
        self.assertIsInstance(create_motion_profile("figure_eight"), FigureEightProfile)
        self.assertIsInstance(create_motion_profile("figure_8"), FigureEightProfile)
        self.assertIsInstance(create_motion_profile("lissajous"), FigureEightProfile)
        self.assertIsInstance(create_motion_profile("random"), RandomWalkProfile)
        self.assertIsInstance(create_motion_profile("random_walk"), RandomWalkProfile)
        self.assertIsInstance(create_motion_profile("spiral"), SpiralProfile)
        self.assertIsInstance(create_motion_profile("sinusoidal"), SinusoidalProfile)
        self.assertIsInstance(create_motion_profile("sine"), SinusoidalProfile)

        with self.assertRaises(ValueError):
            create_motion_profile("invalid_unknown_profile_123")


class TestTargetConfigAndContracts(unittest.TestCase):
    """Test TargetConfig schema, validation, and backward compatibility."""

    def test_target_config_defaults(self):
        """Confirm default motion_type is straight_line and motion_params is empty dict."""
        cfg = TargetConfig()
        self.assertEqual(cfg.motion_type, "straight_line")
        self.assertEqual(cfg.motion_params, {})

    def test_target_config_from_dict_validation(self):
        """Confirm valid motion types are accepted and invalid types raise ValueError."""
        valid_dict = {
            "target_id": "tgt_circ",
            "motion_type": "circular",
            "radius": 0.007,
            "angular_velocity": 0.6,
        }
        cfg = TargetConfig.from_dict(valid_dict)
        self.assertEqual(cfg.motion_type, "circular")
        self.assertEqual(cfg.motion_params["radius"], 0.007)
        self.assertEqual(cfg.motion_params["angular_velocity"], 0.6)

        # Invalid motion type raises ValueError
        invalid_dict = {"motion_type": "hyperbolic_teleport"}
        with self.assertRaises(ValueError):
            TargetConfig.from_dict(invalid_dict)

    def test_legacy_scenario_normalization_backward_compatibility(self):
        """Confirm normalize_scenario_targets retains backward compatibility for legacy configs."""
        legacy_cfg = {
            "scenario_name": "LEGACY_TEST",
            "target": {
                "initial_pos": [0.010, -0.005],
                "velocity": [0.002, 0.001],
            }
        }
        targets_list, primary_id = normalize_scenario_targets(legacy_cfg)
        self.assertEqual(len(targets_list), 1)
        self.assertEqual(primary_id, "target_0")
        self.assertEqual(targets_list[0]["motion_type"], "straight_line")

    def test_multi_target_motion_config(self):
        """Confirm multi-target configs support heterogeneous motion types."""
        multi_cfg = {
            "scenario_name": "HETEROGENEOUS_TARGETS",
            "targets": [
                {"target_id": "tgt_linear", "motion_type": "straight_line", "velocity": [0.001, 0.0]},
                {"target_id": "tgt_circle", "motion_type": "circular", "radius": 0.005},
                {"target_id": "tgt_fig8", "motion_type": "figure_eight", "amplitude_x": 0.008},
            ]
        }
        targets_list, primary_id = normalize_scenario_targets(multi_cfg)
        self.assertEqual(len(targets_list), 3)
        self.assertEqual(targets_list[0]["motion_type"], "straight_line")
        self.assertEqual(targets_list[1]["motion_type"], "circular")
        self.assertEqual(targets_list[2]["motion_type"], "figure_eight")


class TestKinematicsIntegration(unittest.TestCase):
    """Integration tests verifying Target and Environment with motion profiles."""

    def test_target_with_circular_profile(self):
        """Verify Target advances along a circle when stepped."""
        center = (0.010, 0.005)
        radius = 0.006
        target = Target(
            target_id="target_circ",
            motion_type="circular",
            motion_params={"center": center, "radius": radius, "angular_velocity": 1.0},
        )
        dt = 0.033
        for f in range(30):
            st = target.step(dt)
            dist = np.hypot(st.x - center[0], st.y - center[1])
            self.assertAlmostEqual(dist, radius, delta=1e-6)
            self.assertEqual(st.target_id, "target_circ")

    def test_environment_with_platform_motion(self):
        """Verify Environment correctly incorporates platform motion generator."""
        cam = Camera(resolution=(640, 480))
        target = Target(initial_pos=(0.0, 0.0), velocity=(0.0, 0.0))

        # Circular platform motion profile
        plat_prof = CircularProfile(center=(0.0, 0.0), radius=0.002, angular_velocity=2.0)
        env = Environment(
            target=target,
            camera=cam,
            platform_motion=plat_prof,
        )

        frame, tgt_state, cam_state = env.step(dt=0.033)
        self.assertIsNotNone(frame)
        self.assertAlmostEqual(tgt_state.x, 0.0)
        self.assertAlmostEqual(tgt_state.y, 0.0)

    def test_environment_from_config_instantiation(self):
        """Verify Environment.from_config creates proper targets and platform motion."""
        scenario_cfg = {
            "scenario_name": "PARAMETRIC_TEST",
            "targets": [
                {
                    "target_id": "tgt_0",
                    "motion_type": "figure_eight",
                    "amplitude_x": 0.010,
                    "amplitude_y": 0.005,
                    "frequency": 0.5,
                }
            ],
            "platform_motion": {
                "motion_type": "circular",
                "radius": 0.001,
                "angular_velocity": 3.0,
            },
            "disturbances": {
                "cn2": 0.0,
                "vibration_amplitude": 0.0,
                "noise_level": 0.0,
            }
        }
        env = Environment.from_config(scenario_cfg)
        self.assertEqual(len(env.targets), 1)
        self.assertIsInstance(env.targets[0].motion_profile, FigureEightProfile)
        self.assertIsInstance(env.platform_motion, CircularProfile)

        # Step environment without error
        frame, tgt_state, cam_state = env.step(0.033)
        self.assertIsNotNone(frame)
        self.assertEqual(tgt_state.target_id, "tgt_0")


if __name__ == "__main__":
    unittest.main()
