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

### 4. `TargetConfig`
Configures initial kinematics and parametric motion generation for single or multi-target scenarios:
* `target_id`: String identifier for target discrimination.
* `initial_pos`: `(x, y)` coordinates in angular space (radians).
* `velocity`: `(vx, vy)` baseline velocity vector (rad/s).
* `motion_type`: Selectable kinematic motion pattern:
  * `"straight_line"`: Constant velocity rectilinear motion.
  * `"circular"`: Bounded orbit with configurable `radius`, `angular_velocity`, and `center`.
  * `"figure_eight"`: Lissajous 1:2 frequency lemniscate trajectory (`amplitude_x`, `amplitude_y`, `frequency`).
  * `"random"`: Bounded 2D stochastic random-walk with specular boundary reflection.
  * `"spiral"`: Expanding/contracting Archimedean spiral trajectory.
  * `"sinusoidal"`: Transverse harmonic oscillation overlaid on directional drift.
* `blink_frequency`: Modulation signature frequency (Hz) for multi-target identity verification.
* `base_intensity`: Peak optical beacon brightness (counts).

### 5. `DisturbanceConfig`
Configures environmental and hardware disturbance intensities.
* `cn2`: Refractive index structure constant $C_n^2$ for Kolmogorov atmospheric turbulence.
* `vibration_amplitude`: Amplitude of platform jitter.
* `vibration_frequency`: Dominant jitter frequency (Hz).
* `noise_level`: Standard deviation of additive Gaussian sensor readout noise.
* `noise_types`: Selectable list of active sensor noise models (e.g. `["gaussian"]`, `["poisson"]`, `["gaussian", "poisson", "salt_pepper"]`).
* `poisson_scale`: Photon scaling factor for Poisson shot noise (photons per gray level).
* `salt_pepper_prob`: Probability of hot/dead pixel impulse noise defects.
* `occluder_frequency`: Frequency/probability of dynamic line-of-sight occluders.
* `occluder_size`: Occluder radius / footprint (pixels).
* `random_walk_jitter`: Platform drift rate.

### 6. `MetricsRecord`
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
   python run_demo.py
   ```

---

## 🧪 Running Tests

To run the complete automated test suite (114 unit and integration tests across all modules):
```bash
python -m unittest discover -s tests
```

---

## 📦 Standalone Executable Packaging (PyInstaller)

The simulator can be packaged into a completely standalone, portable Windows application that runs without requiring Python or any external packages installed.

### 1. Build the Executable
Run the automated packaging script:
```bash
python build_exe.py
```
Or build directly using the PyInstaller specification:
```bash
python -m PyInstaller fsoc_simulator.spec
```

### 2. Output Location & Structure
The build generates a self-contained `--onedir` distribution:
```text
dist/
└── FSOC_PAT_Simulator/
    ├── FSOC_PAT_Simulator.exe   # Application entry point (~5.1 MB)
    ├── config/                  # Bundled scenario configurations
    │   ├── default_scenario.json
    │   ├── demo_scenarios.json
    │   └── scenarios.json
    ├── data/
    │   └── videos/              # Bundled benchmark test videos (~15 MB)
    └── _internal/               # Bundled Python runtime, OpenCV, NumPy, and Tkinter DLLs
```

### 3. Packaging Mode Tradeoffs: `--onedir` vs `--onefile`
* **`--onedir` (Default & Recommended)**:
  * **Instant Startup (< 1s)**: Windows maps executable binaries and DLLs directly into memory without extraction overhead.
  * **Portable & Inspectable**: The entire `dist/FSOC_PAT_Simulator/` folder can be zipped, moved to any Windows machine or USB drive, and run immediately. Configuration files in `config/` can be directly inspected or customized.
* **`--onefile` (`python build_exe.py --onefile`)**:
  * Bundles everything into a single `.exe`, but incurs a 5–15 second startup penalty on every launch because Windows must decompress the entire archive into a temporary folder (`%TEMP%/_MEIxxxxxx`).

### 4. Running the Standalone Application

* **Interactive Visual HUD (Tkinter GUI)**:
  Double-click `FSOC_PAT_Simulator.exe` or launch from command prompt:
  ```powershell
  .\dist\FSOC_PAT_Simulator\FSOC_PAT_Simulator.exe
  ```

* **Headless / CLI Execution (Automated Verification & Benchmarking)**:
  ```powershell
  # List all available packaged scenarios
  .\dist\FSOC_PAT_Simulator\FSOC_PAT_Simulator.exe --list-scenarios

  # Run a specific scenario headlessly and export performance metrics
  .\dist\FSOC_PAT_Simulator\FSOC_PAT_Simulator.exe --run-scenario DEMO_ACQUISITION_AND_TRACK --output-dir ./logs
  ```


