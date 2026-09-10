"""
Unit tests for track module: ConstantVelocityKalmanFilter, ParticleFilter,
SeverityCalculator, and HybridTracker with hysteresis.
"""

import unittest
import numpy as np
from track.kalman import ConstantVelocityKalmanFilter
from track.particle import ParticleFilter
from track.severity import SeverityCalculator
from track.hybrid import HybridTracker
from metrics.logger import MetricsLogger
from contracts import TargetState, CameraState


class TestTrack(unittest.TestCase):

    def test_kalman_filter_state_initialization(self):
        kf = ConstantVelocityKalmanFilter(initial_pos=(0.015, -0.010), initial_vel=(0.002, 0.001))
        self.assertTrue(kf.is_initialized)
        state = kf.get_state()
        self.assertAlmostEqual(state.x, 0.015)
        self.assertAlmostEqual(state.y, -0.010)
        self.assertAlmostEqual(state.vx, 0.002)
        self.assertAlmostEqual(state.vy, 0.001)

    def test_kalman_predict_step(self):
        kf = ConstantVelocityKalmanFilter(initial_pos=(0.0, 0.0), initial_vel=(0.010, -0.005))
        dt = 0.1
        state = kf.predict(dt=dt)
        self.assertAlmostEqual(state.x, 0.001)   # 0.0 + 0.010 * 0.1
        self.assertAlmostEqual(state.y, -0.0005) # 0.0 - 0.005 * 0.1
        self.assertEqual(state.tracker_mode, "KF")

    def test_kalman_filters_measurement_noise(self):
        """Kalman filter should smooth out zero-mean Gaussian measurement jitter."""
        true_pos_x, true_pos_y = 0.020, 0.015
        true_vel_x, true_vel_y = 0.003, -0.001

        kf = ConstantVelocityKalmanFilter(
            q_noise_std=0.01,
            r_noise_std=0.0002,
            initial_pos=(true_pos_x, true_pos_y),
            initial_vel=(0.0, 0.0),
        )

        dt = 0.033
        rng = np.random.default_rng(42)
        raw_errors = []
        kf_errors = []

        cur_x = true_pos_x
        cur_y = true_pos_y

        for step_i in range(80):
            cur_x += true_vel_x * dt
            cur_y += true_vel_y * dt

            # Noisy measurement with 0.2 mrad jitter
            meas_x = cur_x + rng.normal(0, 0.0002)
            meas_y = cur_y + rng.normal(0, 0.0002)

            raw_err = np.hypot(meas_x - cur_x, meas_y - cur_y)
            raw_errors.append(raw_err)

            # Filter step
            est_state = kf.step(dt=dt, measurement=(meas_x, meas_y), confidence=0.9)
            kf_err = np.hypot(est_state.x - cur_x, est_state.y - cur_y)
            kf_errors.append(kf_err)

        raw_steady_rmse = np.sqrt(np.mean(np.square(raw_errors[40:])))
        kf_steady_rmse = np.sqrt(np.mean(np.square(kf_errors[40:])))

        self.assertLess(kf_steady_rmse, raw_steady_rmse)
        self.assertAlmostEqual(kf.x[2, 0], true_vel_x, delta=0.001)
        self.assertAlmostEqual(kf.x[3, 0], true_vel_y, delta=0.001)

    def test_kalman_coast_mode_5_consecutive_dropouts(self):
        """Simulate 5 consecutive dropped frames."""
        true_x, true_y = 0.030, 0.020
        true_vx, true_vy = 0.004, -0.002
        dt = 0.033

        kf = ConstantVelocityKalmanFilter(
            q_noise_std=0.01,
            r_noise_std=0.001,
            initial_pos=(true_x, true_y),
            initial_vel=(true_vx, true_vy),
        )

        for i in range(15):
            t = (i + 1) * dt
            cur_x = true_x + true_vx * t
            cur_y = true_y + true_vy * t
            kf.step(dt=dt, measurement=(cur_x, cur_y), confidence=1.0)

        coast_errors = []
        for i in range(5):
            t = (15 + i + 1) * dt
            cur_true_x = true_x + true_vx * t
            cur_true_y = true_y + true_vy * t

            pred_state = kf.step(dt=dt, measurement=None)
            self.assertEqual(pred_state.tracker_mode, "COAST")
            self.assertEqual(kf.consecutive_coasts, i + 1)

            err = np.hypot(pred_state.x - cur_true_x, pred_state.y - cur_true_y)
            coast_errors.append(err)

        max_coast_err = max(coast_errors)
        self.assertLess(max_coast_err, 0.0001)

        t_resume = 21 * dt
        resume_true_x = true_x + true_vx * t_resume
        resume_true_y = true_y + true_vy * t_resume

        resumed_state = kf.step(dt=dt, measurement=(resume_true_x, resume_true_y), confidence=1.0)
        self.assertEqual(resumed_state.tracker_mode, "KF")
        self.assertEqual(kf.consecutive_coasts, 0)
        post_resume_err = np.hypot(resumed_state.x - resume_true_x, resumed_state.y - resume_true_y)
        self.assertLess(post_resume_err, 0.0001)

    def test_severity_calculator(self):
        """Severity score should reflect confidence, miss count, and turbulence."""
        calc = SeverityCalculator()

        # Clear conditions: high confidence, 0 misses, low turbulence
        s_clear = calc.compute(confidence=0.95, consecutive_misses=0, cn2=1e-16)
        self.assertLess(s_clear, 0.20)

        # Moderate conditions
        s_mod = calc.compute(confidence=0.60, consecutive_misses=1, cn2=1e-14)
        self.assertTrue(0.25 <= s_mod <= 0.50)

        # Severe conditions: low confidence, 3 misses, high turbulence
        s_severe = calc.compute(confidence=0.15, consecutive_misses=3, cn2=5e-13)
        self.assertGreater(s_severe, 0.65)

    def test_kalman_severity_noise_adaptation(self):
        """Higher severity should increase measurement R and process noise Q."""
        kf = ConstantVelocityKalmanFilter(initial_pos=(0.0, 0.0))

        # Predict and update with low severity
        kf.step(dt=0.033, measurement=(0.001, 0.001), confidence=0.95, severity=0.10)
        r_low = float(kf.R[0, 0])
        q_low = float(kf.Q[0, 0])

        # Predict and update with high severity (confidence 0.60 passes the 0.40 gate)
        kf.step(dt=0.033, measurement=(0.002, 0.002), confidence=0.60, severity=0.70)
        r_high = float(kf.R[0, 0])
        q_high = float(kf.Q[0, 0])

        # R and Q must scale up significantly with severity
        self.assertGreater(r_high, r_low * 10.0)
        self.assertGreater(q_high, q_low * 2.0)

    def test_particle_filter_initialization_and_handoff(self):
        """Particle filter initializes properly from Gaussian mean and covariance."""
        pf = ParticleFilter(num_particles=200, seed=42)
        mean = np.array([[0.01], [-0.02], [0.001], [0.002]])
        cov = np.diag([1e-5, 1e-5, 1e-6, 1e-6])

        pf.init_from_distribution(mean=mean, cov=cov)
        self.assertTrue(pf.is_initialized)
        est_mean, est_cov = pf.get_mean_and_cov()

        # Mean should be close to initialization mean
        self.assertAlmostEqual(est_mean[0, 0], 0.01, delta=0.001)
        self.assertAlmostEqual(est_mean[1, 0], -0.02, delta=0.001)

    def test_particle_filter_tracking(self):
        """Particle filter tracks a moving target with non-Gaussian measurement noise."""
        pf = ParticleFilter(num_particles=300, seed=123)
        pf.init_state(pos_x=0.010, pos_y=0.010, vel_x=0.002, vel_y=-0.001)

        dt = 0.033
        cur_x, cur_y = 0.010, 0.010
        vx, vy = 0.002, -0.001
        rng = np.random.default_rng(99)

        errors = []
        for step in range(40):
            cur_x += vx * dt
            cur_y += vy * dt

            # Heavy-tailed noise (occasional 3-sigma outlier)
            noise_x = rng.normal(0, 0.0003) if step % 6 != 0 else rng.normal(0, 0.0015)
            noise_y = rng.normal(0, 0.0003) if step % 6 != 0 else rng.normal(0, 0.0015)

            state = pf.step(dt=dt, measurement=(cur_x + noise_x, cur_y + noise_y), confidence=0.85)
            err = np.hypot(state.x - cur_x, state.y - cur_y)
            errors.append(err)

        steady_err = float(np.mean(errors[20:]))
        self.assertLess(steady_err, 0.001, f"PF tracking error {steady_err*1e3:.3f} mrad exceeded 1.0 mrad")

    def test_hybrid_tracker_hysteresis_and_no_flicker(self):
        """
        Tests:
        1. Low severity -> Mode is KF
        2. High severity -> Switches to PF
        3. Oscillating near boundary (0.45 - 0.52) -> Does NOT switch back, stays PF (NO FLICKER)
        4. Calm conditions sustained for 5 frames -> Switches back to KF
        """
        logger = MetricsLogger()
        tracker = HybridTracker(
            logger=logger,
            severity_switch_to_pf=0.55,
            severity_switch_to_kf=0.30,
            consecutive_calm_to_kf=5,
        )
        tracker.init_state(0.01, 0.01, 0.0, 0.0)

        dt = 0.033

        # Frame 1 to 5: Calm conditions (high confidence, low turbulence)
        for i in range(5):
            state = tracker.step(dt=dt, measurement=(0.01, 0.01), confidence=0.90, cn2=1e-16)
            self.assertEqual(tracker.active_mode, "KF")

        # Frame 6: Severe disturbance (Cn2 jumps, confidence drops)
        state = tracker.step(dt=dt, measurement=(0.01, 0.01), confidence=0.20, cn2=5e-13)
        self.assertEqual(tracker.active_mode, "PF", "Tracker failed to switch to PF on high severity!")
        self.assertEqual(len(logger.switch_events), 1)
        self.assertEqual(logger.switch_events[0].from_mode, "KF")
        self.assertEqual(logger.switch_events[0].to_mode, "PF")

        # Frames 7 to 14: Severity fluctuates in hysteresis band (between 0.35 and 0.52)
        # MUST NOT switch back to KF or flicker!
        for i in range(8):
            conf = 0.55 if i % 2 == 0 else 0.45
            cn2_val = 1e-14 if i % 2 == 0 else 3e-14
            state = tracker.step(dt=dt, measurement=(0.01, 0.01), confidence=conf, cn2=cn2_val)
            self.assertEqual(tracker.active_mode, "PF", f"Tracker flickered to {tracker.active_mode} in hysteresis band at step {i}!")
            self.assertEqual(len(logger.switch_events), 1, "Spurious switch event recorded in hysteresis band!")

        # Frames 15 to 18: Calm frames 1, 2, 3, 4 (< 5 calm frames)
        for i in range(4):
            state = tracker.step(dt=dt, measurement=(0.01, 0.01), confidence=0.92, cn2=1e-16)
            self.assertEqual(tracker.active_mode, "PF", f"Switched back prematurely at calm frame {i+1} < 5!")
            self.assertEqual(len(logger.switch_events), 1)

        # Frame 19: Calm frame 5 (meets M=5 threshold) -> MUST switch to KF!
        state = tracker.step(dt=dt, measurement=(0.01, 0.01), confidence=0.92, cn2=1e-16)
        self.assertEqual(tracker.active_mode, "KF", "Failed to switch back to KF after 5 consecutive calm frames!")
        self.assertEqual(len(logger.switch_events), 2)
        self.assertEqual(logger.switch_events[1].from_mode, "PF")
        self.assertEqual(logger.switch_events[1].to_mode, "KF")

    def test_switch_event_log_records_live_confidence_and_aligned_frame(self):
        """
        Regression Test 1:
        Asserts that the logged confidence in switch-event matches the live detection
        confidence at the switch frame (e.g., 0.364, NOT 0.000 or stale), and that the
        switch event frame_id aligns exactly with the caller's frame counter.
        """
        logger = MetricsLogger()
        tracker = HybridTracker(
            logger=logger,
            severity_switch_to_pf=0.55,
            severity_switch_to_kf=0.30,
            min_valid_confidence=0.40,
        )
        tracker.init_state(0.01, 0.01, 0.0, 0.0)

        dt = 0.033

        # Frames 0 to 4: Calm
        for f in range(5):
            tracker.step(dt=dt, measurement=(0.01, 0.01), confidence=0.90, cn2=1e-16, frame_id=f)
            self.assertEqual(tracker.active_mode, "KF")

        # Frame 5: Detection arrives with confidence 0.364 (below 0.40 gate, elevating severity)
        live_detection_conf = 0.364
        meas = (0.012, 0.012)
        state = tracker.step(dt=dt, measurement=meas, confidence=live_detection_conf, cn2=5e-13, frame_id=5)

        self.assertEqual(tracker.active_mode, "PF")
        self.assertEqual(len(logger.switch_events), 1)

        ev = logger.switch_events[0]
        # 1. Frame alignment: must match caller frame counter (frame 5, not 6)
        self.assertEqual(ev.frame_id, 5, f"Expected switch frame_id=5, got {ev.frame_id}")
        # 2. Live confidence: must record exact live frame confidence (0.364, not 0.000)
        self.assertAlmostEqual(ev.confidence, live_detection_conf, places=3,
                               msg=f"Expected live confidence {live_detection_conf}, got {ev.confidence}")
        self.assertNotEqual(ev.confidence, 0.0, "Switch event logged 0.000 instead of live detection confidence!")

    def test_particle_filter_confidence_gate_triggers_coast_and_rejects_measurement(self):
        """
        Regression Test 2:
        Asserts that when the Particle Filter is active and confidence drops below the
        gate threshold (0.40), tracker_mode reports COAST and no measurement update
        is applied to the PF's particles (even if a corrupt outlier measurement is passed).
        """
        pf = ParticleFilter(
            num_particles=200,
            min_valid_confidence=0.40,
            seed=42,
        )
        pf.init_state(pos_x=0.010, pos_y=0.010, vel_x=0.000, vel_y=0.000)

        # 1. Directly test ParticleFilter: pass a huge outlier at (0.100, 0.100) with confidence 0.35 (< 0.40 gate)
        outlier_meas = (0.100, 0.100)
        low_conf = 0.350
        pf_state = pf.step(dt=0.033, measurement=outlier_meas, confidence=low_conf)

        # Must report COAST
        self.assertEqual(pf_state.tracker_mode, "COAST")
        self.assertEqual(pf.mode, "COAST")
        # Measurement must NOT be absorbed: estimate should stay near (0.010, 0.010) and NOT jump to (0.100, 0.100)
        self.assertLess(pf_state.x, 0.020, f"PF absorbed gated outlier! x jumped to {pf_state.x}")
        self.assertLess(pf_state.y, 0.020, f"PF absorbed gated outlier! y jumped to {pf_state.y}")

        # 2. Test via HybridTracker with active_mode == 'PF'
        logger = MetricsLogger()
        tracker = HybridTracker(
            pf=pf,
            logger=logger,
            initial_mode="PF",
            min_valid_confidence=0.40,
        )
        tracker.init_state(pos_x=0.010, pos_y=0.010, vel_x=0.000, vel_y=0.000)
        self.assertEqual(tracker.active_mode, "PF")

        # Pass corrupt outlier below gate
        hybrid_state = tracker.step(dt=0.033, measurement=outlier_meas, confidence=low_conf, frame_id=10)

        # Mode must be COAST and outlier rejected
        self.assertEqual(hybrid_state.tracker_mode, "COAST")
        self.assertEqual(tracker.current_mode, "COAST")
        self.assertLess(hybrid_state.x, 0.020, "HybridTracker PF absorbed gated outlier!")

        # 3. Now pass a valid measurement above gate (conf = 0.85) near true target
        valid_meas = (0.011, 0.011)
        valid_conf = 0.85
        valid_state = tracker.step(dt=0.033, measurement=valid_meas, confidence=valid_conf, frame_id=11)

        # Must report PF (normal update absorbed)
        self.assertEqual(valid_state.tracker_mode, "PF")
        self.assertEqual(tracker.current_mode, "PF")
        self.assertAlmostEqual(valid_state.x, 0.011, delta=0.002)


if __name__ == "__main__":
    unittest.main()

