# FSOC Coarse PAT Simulator (Backend Engine)
**Smart India Hackathon | Problem Statement 26169 (ISRO / Department of Space)**

Backend simulation, optical detection, adaptive tracking, gimbal control, disturbance generation, and telemetry metrics engine for Free Space Optical Communication (FSOC) coarse Pointing, Acquisition, and Tracking (PAT).

---

## 📁 Project Architecture & Folder Structure

```text
SIH/
├── sim/              # Virtual 2D environment, target beacon kinematics, camera optics model
├── detect/           # Pre-filtering, adaptive thresholding, blob centroiding
├── track/            # Kalman Filter, Particle Filter, and adaptive hybrid switching logic
├── control/          # PID gimbal controller, slew-rate limiting, latency, velocity feedforward
├── disturb/          # Kolmogorov phase-screen turbulence, platform vibration, noise, occluders
├── metrics/          # Structured logging (CSV/JSON) and quantitative performance evaluation
├── config/           # Scenario JSON configurations (trajectories, optical & disturbance settings)
├── tests/            # Standalone unit tests for each module
├── contracts.py      # Core data contracts & plain Python dataclasses (Shared Interface)
└── README.md         # Architecture and interface specification for team members
```

---

## 🤝 Data Contracts (`contracts.py`)

All modules communicate exclusively through typed Python dataclasses defined in [`contracts.py`](contracts.py). 

Frontend / GUI / Dashboard consumers can easily serialize any contract instance to a JSON-compatible dictionary using `.to_dict()`.

### 1. `FrameData`
Represents raw sensor readout at a simulation step.
* `image`: 2D/3D NumPy array of simulated sensor intensities.
* `timestamp`: Float (simulation epoch in seconds).
* `frame_id`: Sequential integer ID.
* `ground_truth_target_pos`: Optional `(x, y)` ground truth coordinate for validation.
* `to_dict()`: Serializes metadata and image dimensions (omits large raw array for fast transmission).

### 2. `TargetState`
Represents the estimated or ground-truth kinematic state of the target optical beacon.
* `x`, `y`: Beacon coordinates (pixels or angular coordinate).
* `vx`, `vy`: Estimated beacon velocity components.
* `confidence`: Float `[0.0, 1.0]` indicating tracking quality / SNR.
* `timestamp`: Float (seconds).
* `tracker_mode`: Active tracker state (`'KF'`, `'PF'`, `'COAST'`, `'LOST'`).

### 3. `CameraState`
Represents the virtual pan-tilt gimbal camera state and optical properties.
* `pan`, `tilt`: Gimbal orientation angles in radians (or degrees).
* `pan_rate`, `tilt_rate`: Angular velocities (rad/s).
* `fov_x`, `fov_y`: Angular field of view spans.
* `focal_length`: Optical focal length (mm).
* `resolution`: Sensor pixel grid dimensions `(width, height)`.
* `fov_bounds`: Computed property `(pan_min, pan_max, tilt_min, tilt_max)` for FOV containment checks.

### 4. `DisturbanceConfig`
Configures environmental and hardware disturbance intensities.
* `cn2`: Refractive index structure constant $C_n^2$ for Kolmogorov atmospheric turbulence.
* `vibration_amplitude`: Amplitude of platform jitter.
* `vibration_frequency`: Dominant jitter frequency (Hz).
* `noise_level`: Standard deviation of additive Gaussian sensor noise.
* `occluder_frequency`: Frequency/probability of dynamic line-of-sight occluders.
* `occluder_size`: Occluder radius / footprint (pixels).
* `random_walk_jitter`: Platform drift rate.

### 5. `MetricsRecord`
Exportable quantitative benchmark record for reporting and dashboards.
* `simulation_duration`: Total elapsed run time (seconds).
* `fps`: Frame processing rate.
* `acquisition_time`: Time taken to achieve initial target lock (seconds).
* `avg_tracking_error`: Mean radial distance from boresight to target.
* `max_tracking_error`: Peak tracking error.
* `lock_retention_rate`: Percentage/fraction of duration target stayed locked `[0.0, 1.0]`.
* `per_frame_processing_time_ms`: Algorithm processing execution time per frame.
* `total_frames`: Total simulated frames.
* `track_loss_count`: Number of lock drops.
* `active_tracker_breakdown`: Time fraction breakdown between tracker modes (e.g. `{'KF': 0.88, 'PF': 0.12}`).

---

## 🚀 Quickstart & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Backend in Editable Mode** (allows importing `contracts`, `sim`, `detect`, `track`, etc. from any script):
   ```bash
   pip install -e .
   ```

3. **Launch the Visual Telemetry Simulator UI**:
   ```bash
   python verify_ui.py
   ```

4. **Run the Extended Combined-Stress Benchmark**:
   ```bash
   python run_validation.py
   ```

---

## 🌐 Launch the Browser Mission Control Frontend

The backend engine now has a zero-build browser frontend that sits on top of the existing simulation modules. It is served locally by a small Python adapter and does not require Node, npm, a database, or an external service.

```bash
# from the repository root
python run_web.py
```

Open `http://127.0.0.1:8787` in a browser. The console provides:

- live SVG optical theater with target trail, camera boresight, FOV, occluders, and predictive search zone;
- live estimator, supervisor, confidence, severity, link, and pipeline status;
- acquisition time, lock retention, average/peak error, and nominal FPS cards;
- error and signal-quality traces;
- scenario switching across the checked-in flight profiles;
- event timeline and disturbance profile cards;
- JSON/CSV telemetry export and an opt-in scenario benchmark action.

The browser adapter uses `verify_ui.SimulationSession`, so playback is driven by the real backend detection, tracking, disturbance, reacquisition, and control loop rather than mock values. The frontend assets live under `frontend/`, while `web_server.py` owns the local JSON API boundary.

## 🧪 Running Tests

To run the complete automated test suite (57 unit tests across all modules):
```bash
python -m unittest discover -s tests
```
