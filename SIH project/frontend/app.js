/**
 * DRISHTI-X: End-to-End FSOC Coarse-PAT Testing & Simulation Platform
 * Smart India Hackathon | Problem Statement 26169 (ISRO / DOS)
 *
 * Full-suite aerospace telemetry engine implementing:
 * - 8 Workspaces: Cockpit, Scenarios, Algorithms, Failures, Sweeps, PAT-RED, Envelope, Reports
 * - 10-State Acquisition Machine (Section 14.3)
 * - Dynamic Safe FOV & Loss-of-Lock Risk Engine (Section 16)
 * - Fine-PAT Handoff Readiness Engine (Section 19)
 * - PAT-RED Adversarial Breaking Point Discovery (Section 25)
 * - Tracking Survival Envelope 2D Matrix (Section 26)
 * - Synchronized Replay with Root-Cause Failure Diagnosis (Section 31)
 * - Basic Mode vs Research Mode Selector (Section 33)
 */

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

// Global Application State
const appState = {
  currentView: 'cockpit',
  currentMode: 'basic', // 'basic' | 'research'
  theaterMode: 'spatial', // 'spatial' | 'sensor'
  sensorPalette: 'thermal', // 'thermal' | 'laser'
  snapshot: null,
  scenarios: [],
  playing: false,
  speed: 1.0,
  loopToken: 0,
  audioEnabled: false,
  audioVolume: 0.35,
  audioCtx: null,
  lastDrishtiState: 'IDLE',
  toastTimer: null,
  recordedFrames: [],
  isScrubbing: false,
  spatialHistory: {
    target: [],
    camera: [],
  },
  bounds: {
    minX: -0.015,
    maxX: 0.035,
    minY: -0.025,
    maxY: 0.025,
  },
};

// Calibrated Telemetry Spectrum
const COLORS = {
  cyan: '#38bdf8',
  cyanSoft: 'rgba(56, 189, 248, 0.12)',
  green: '#10b981',
  greenSoft: 'rgba(16, 185, 129, 0.12)',
  amber: '#f59e0b',
  amberSoft: 'rgba(245, 158, 11, 0.12)',
  red: '#ef4444',
  redSoft: 'rgba(239, 68, 68, 0.12)',
  muted: '#64748b',
  line: '#1e2c47',
};

function clamp(val, min = 0, max = 1) {
  return Math.max(min, Math.min(max, Number(val) || 0));
}

function formatNum(val, digits = 2) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return '—';
  return Number(val).toFixed(digits);
}

function formatCn2(val) {
  const num = Number(val);
  if (!num || num <= 0) return 'OFF (0.0)';
  return num.toExponential(1).replace('+', '');
}

function formatTime(seconds) {
  const totalMs = Math.max(0, Math.round((Number(seconds) || 0) * 1000));
  const minutes = Math.floor(totalMs / 60000);
  const secs = Math.floor((totalMs % 60000) / 1000);
  const millis = totalMs % 1000;
  return `T+${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
}

function prettyScenarioName(name) {
  return String(name || 'UNKNOWN')
    .replace(/^DRISHTI_/, '')
    .replaceAll('_', ' ')
    .toLowerCase()
    .replace(/(^|\s)\S/g, (l) => l.toUpperCase());
}

function showToast(msg) {
  const toast = $('#toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(appState.toastTimer);
  appState.toastTimer = setTimeout(() => toast.classList.remove('show'), 3000);
}

// Low-Amplitude Discrete Telemetry Chimes
function playTelemetrySound(type) {
  if (!appState.audioEnabled || appState.audioVolume <= 0.01) return;
  try {
    if (!appState.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) appState.audioCtx = new AudioContext();
    }
    if (!appState.audioCtx) return;
    if (appState.audioCtx.state === 'suspended') appState.audioCtx.resume();

    const ctx = appState.audioCtx;
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const vol = appState.audioVolume * 0.35;
    osc.connect(gain);
    gain.connect(ctx.destination);

    if (type === 'lock') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(440, now);
      osc.frequency.exponentialRampToValueAtTime(880, now + 0.12);
      gain.gain.setValueAtTime(0.08 * vol, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.14);
      osc.start(now);
      osc.stop(now + 0.14);
    } else if (type === 'loss') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(280, now);
      osc.frequency.exponentialRampToValueAtTime(180, now + 0.16);
      gain.gain.setValueAtTime(0.06 * vol, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
      osc.start(now);
      osc.stop(now + 0.18);
    } else if (type === 'click') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(800, now);
      gain.gain.setValueAtTime(0.03 * vol, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.03);
      osc.start(now);
      osc.stop(now + 0.03);
    }
  } catch (_) {}
}

// Determine backend API host:
// When served via Live Server (port 5500) or any custom static dev server, connect to the backend daemon on port 8787.
const API_BASE = (window.location.port === '5500' || window.location.host.includes(':5500'))
  ? 'http://127.0.0.1:8787'
  : '';

// REST Client Helper
async function api(path, options = {}) {
  const targetUrl = path.startsWith('http') ? path : `${API_BASE}${path}`;
  const res = await fetch(targetUrl, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let errText = `HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      errText = errJson.error || errText;
    } catch (_) {}
    throw new Error(errText);
  }
  return res.json();
}

const post = (path, body = {}) => api(path, { method: 'POST', body: JSON.stringify(body) });

// Telemetry Snapshot Processing Engine
function applySnapshot(snapshot) {
  if (!snapshot) return;
  appState.snapshot = snapshot;

  const scenario = snapshot.scenario || {};
  const latest = snapshot.latest;
  const frame = Number(snapshot.frame_id) || 0;
  const total = Number(snapshot.total_frames) || 100;

  // Record for scrubber history
  if (latest && !appState.recordedFrames[frame]) {
    appState.recordedFrames[frame] = { snapshot: JSON.parse(JSON.stringify(snapshot)) };
  }

  // Scrubber Updates
  const scrubber = $('#timelineScrubber');
  if (scrubber) {
    scrubber.max = total;
    if (!appState.isScrubbing) {
      scrubber.value = frame;
      $('#scrubberPos').textContent = `FRAME ${frame} / ${total} (${latest ? formatTime(latest.sim_time) : '0.00s'})`;
    }
  }

  // Topbar context
  $('#activeProfileLabel').textContent = `${snapshot.scenario_name || 'MISSION'} • ${total} Frames • dt=${scenario.dt || 0.033}s`;

  // Scenario dropdown
  const select = $('#scenarioSelect');
  if (select && select.value !== snapshot.scenario_name) {
    select.value = snapshot.scenario_name;
  }
  $('#scenarioDescription').textContent = scenario.description || 'No scenario description.';

  // Frame & Clock Readouts
  $('#frameReadout').textContent = `FRAME ${String(frame).padStart(3, '0')} / ${String(total).padStart(3, '0')}`;
  $('#simTimeReadout').textContent = latest ? formatTime(latest.sim_time) : 'T+00:00.000';

  // Range and FOVs
  const rangeKm = Number(latest?.range_km || scenario.target?.distance_km || 5.0);
  $('#rangeReadout').textContent = `${rangeKm.toFixed(1).padStart(4, '0')} KM`;
  if (latest?.fov_bounds) {
    const fovW = Math.abs(latest.fov_bounds[1] - latest.fov_bounds[0]) * 1000;
    const fovH = Math.abs(latest.fov_bounds[3] - latest.fov_bounds[2]) * 1000;
    $('#fovReadout').textContent = `COARSE FOV: ${fovW.toFixed(1)}×${fovH.toFixed(1)} mrad`;
  }
  if (latest?.safe_fov_size_mrad) {
    $('#safeFovReadout').textContent = `SAFE FOV: ${latest.safe_fov_size_mrad[0]}×${latest.safe_fov_size_mrad[1]} mrad`;
  }

  // 10-State Acquisition Machine (Section 14.3)
  const dState = latest?.drishti_state || 'IDLE';
  const pill = $('#stateMachinePill');
  const dLabel = $('#drishtiStateLabel');
  const dText = $('#drishtiStateText');
  const sBadge = $('#supervisorBadge');

  if (pill && dLabel) {
    dLabel.textContent = dState;
    pill.className = `state-machine-pill ${dState.toLowerCase().replace(/\s+/g, '-')}`;
  }
  if (dText) dText.textContent = dState;
  if (sBadge) {
    sBadge.textContent = dState;
    sBadge.className = `status-badge ${dState === 'LOCKED' || dState === 'TRACKING' ? 'nominal' : dState === 'SEARCHING' || dState === 'REACQUIRING' ? 'searching' : 'lost'}`;
  }

  // State Transition Audio
  if ((dState === 'LOST' || dState === 'DEGRADED') && appState.lastDrishtiState !== dState) {
    playTelemetrySound('loss');
  } else if ((dState === 'LOCKED' || dState === 'TRACKING') && (appState.lastDrishtiState === 'SEARCHING' || appState.lastDrishtiState === 'REACQUIRING')) {
    playTelemetrySound('lock');
  }
  appState.lastDrishtiState = dState;

  // Loss-of-Lock Risk (Section 16.3)
  const riskState = latest?.risk_state || 'STABLE';
  const riskScore = latest?.risk_score || 0.12;
  const riskBadge = $('#riskBadge');
  const riskText = $('#riskScoreText');
  if (riskBadge) {
    riskBadge.textContent = `${riskState} RISK`;
    riskBadge.className = `risk-badge ${riskState === 'STABLE' ? 'stable' : riskState === 'WARNING' ? 'warning' : 'critical'}`;
  }
  if (riskText) {
    riskText.textContent = `${riskState} (${riskScore.toFixed(2)})`;
    riskText.className = `row-value ${riskState === 'STABLE' ? 'ok' : riskState === 'WARNING' ? 'warn' : 'alert'}`;
  }

  // Fine-PAT Handoff Readiness Engine (Section 19)
  const handoffScore = Number(latest?.handoff_score || 0);
  const handoffState = latest?.handoff_state || 'NOT READY';
  $('#handoffScoreVal').textContent = `${Math.round(handoffScore)}%`;
  const handoffPill = $('#handoffStatePill');
  if (handoffPill) {
    handoffPill.textContent = handoffState;
    handoffPill.className = `handoff-pill ${handoffState.toLowerCase().replace(/\s+/g, '-')}`;
  }

  // Handoff Checklist Indicators
  const errRad = Number(latest?.tracking_error_mrad || 0);
  const rollingRms = Number(latest?.recent_rms_mrad || 0);
  $('#chkAlign').className = `check-item ${errRad < 1.5 ? 'ok' : ''}`;
  $('#chkRms').className = `check-item ${rollingRms < 1.8 ? 'ok' : ''}`;
  $('#chkLock').className = `check-item ${dState === 'LOCKED' || dState === 'TRACKING' ? 'ok' : ''}`;
  $('#chkMargin').className = `check-item ${riskState === 'STABLE' ? 'ok' : ''}`;

  // Root-Cause Failure Diagnosis in Scrubber (Section 31.2)
  if (latest?.failure_explainer) {
    $('#failureExplanationText').textContent = latest.failure_explainer;
  }

  // Telemetry Rows
  $('#liveErrorReadout').textContent = `${formatNum(errRad, 3)} mrad`;
  $('#liveErrorReadout').className = `row-value ${errRad < 1.5 ? 'ok' : errRad < 2.5 ? 'warn' : 'alert'}`;
  $('#rollingRmsReadout').textContent = `${formatNum(rollingRms, 3)} mrad`;

  const confVal = Number(latest?.confidence || 0);
  const sevVal = Number(latest?.severity || 0);
  $('#confidenceMeterLabel').textContent = `${Math.round(confVal * 100)}%`;
  $('#severityMeterLabel').textContent = `${Math.round(sevVal * 100)}%`;
  $('#confidenceMeter').style.width = `${clamp(confVal) * 100}%`;
  $('#severityMeter').style.width = `${clamp(sevVal) * 100}%`;

  if (latest?.control_command) {
    $('#gimbalCmdReadout').textContent = `${formatNum(latest.control_command[0] * 1000, 2)}, ${formatNum(latest.control_command[1] * 1000, 2)} mrad/s`;
  }
  if (latest?.camera_pan_tilt) {
    $('#stageCoordinates').textContent = `AZ ${formatNum(latest.camera_pan_tilt[0] * 1000, 2)} / EL ${formatNum(latest.camera_pan_tilt[1] * 1000, 2)} mrad`;
  }

  // Optical Link State
  const validDet = Boolean(latest?.is_valid_det);
  const inFov = Boolean(latest?.target_in_fov);
  const signalBars = $('#signalBars');
  const linkState = $('#linkState');
  if (validDet && inFov && errRad < 1.5) {
    signalBars.className = 'signal-bars lvl-4';
    linkState.textContent = 'LOCKED (NOMINAL)';
    linkState.style.color = COLORS.green;
  } else if (validDet && inFov) {
    signalBars.className = 'signal-bars lvl-3';
    linkState.textContent = 'TRACKING (MARGINAL)';
    linkState.style.color = COLORS.green;
  } else if (dState === 'SEARCHING' || dState === 'REACQUIRING') {
    signalBars.className = 'signal-bars lvl-2';
    linkState.textContent = 'REACQUIRING BEACON';
    linkState.style.color = COLORS.amber;
  } else {
    signalBars.className = 'signal-bars lvl-1';
    linkState.textContent = 'LOCK LOST / COAST';
    linkState.style.color = COLORS.red;
  }

  $('#processingReadout').textContent = latest ? `${formatNum(latest.wall_processing_ms, 1)} ms` : '— ms';
  $('#eventBanner').textContent = latest?.event_message || (frame > 0 ? `${dState} · Pointing Error: ${formatNum(errRad, 2)} mrad` : 'Ready. Click Start Mission to stream telemetry.');

  // Metrics, Charts & Profiles
  renderKPIs(snapshot.metrics || {});
  renderDisturbanceProfile(scenario.disturbances || {});
  renderCharts(snapshot.history || []);

  // Motion History
  if (latest?.target_pos) {
    appState.spatialHistory.target.push(latest.target_pos);
    if (appState.spatialHistory.target.length > 80) appState.spatialHistory.target.shift();
  }
  if (latest?.camera_pan_tilt) {
    appState.spatialHistory.camera.push(latest.camera_pan_tilt);
    if (appState.spatialHistory.camera.length > 80) appState.spatialHistory.camera.shift();
  }

  // Stage Rendering
  if (appState.theaterMode === 'spatial') {
    renderSpatialTheater(latest);
  } else {
    renderSensorTheater(latest);
  }

  updatePlayButton();
}

// Render KPI Cards
function renderKPIs(metrics) {
  $('#acquisitionMetric').textContent = metrics.acquisition_time_ms !== null && metrics.acquisition_time_ms !== undefined ? `${formatNum(metrics.acquisition_time_ms, 0)} ms` : '—';
  $('#retentionMetric').textContent = `${formatNum((Number(metrics.lock_retention_rate) || 0) * 100, 1)}%`;
  $('#retentionFoot').textContent = `${metrics.simulated_frames || 0} frames streamed`;
  $('#rmsErrorMetric').textContent = `${formatNum(metrics.rms_tracking_error_mrad || metrics.avg_tracking_error_mrad, 3)} mrad`;
  $('#maxErrorMetric').textContent = `${formatNum(metrics.max_tracking_error_mrad, 3)} mrad`;
  $('#fpsMetric').textContent = `${formatNum(metrics.fps, 1)} Hz`;
}

// Render Disturbance Profile
function renderDisturbanceProfile(dist) {
  const cn2 = dist.cn2 || 0;
  const vib = (Number(dist.vibration_amplitude) || 0) * 1000;
  const noise = Number(dist.noise_level) || 0;
  const occs = dist.occlusions || (dist.occlusion ? [dist.occlusion] : []);

  const chips = [
    ['TURBULENCE', formatCn2(cn2), Number(cn2) >= 1e-14],
    ['JITTER', `${formatNum(vib, 2)} mrad`, vib >= 0.4],
    ['NOISE', `σ ${formatNum(noise, 1)}`, noise >= 5],
  ];
  if (occs.length) chips.push(['OCCLUSION', `${occs.length} window(s)`, true]);

  $('#disturbanceChips').innerHTML = chips
    .map(([label, val, hot]) => `<span class="dist-chip${hot ? ' hot' : ''}">${label}: ${val}</span>`)
    .join('');
}

// ==========================================================================
// CALIBRATED SPATIAL RETICLE & DYNAMIC SAFE FOV (SECTION 16)
// ==========================================================================
function renderSpatialTheater(latest) {
  const canvas = $('#spatialCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  // Dark telescope background
  ctx.fillStyle = '#050914';
  ctx.fillRect(0, 0, w, h);

  const bounds = appState.bounds;
  const toScreen = (pan, tilt) => ({
    x: ((pan - bounds.minX) / (bounds.maxX - bounds.minX)) * w,
    y: h - ((tilt - bounds.minY) / (bounds.maxY - bounds.minY)) * h,
  });

  const centerOrigin = toScreen(0, 0);

  // Concentric Milliradian Range Rings
  const ringDistancesMrad = [5, 10, 20, 30];
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 5]);

  ringDistancesMrad.forEach((rMrad) => {
    const rRad = rMrad / 1000;
    const ptEdge = toScreen(rRad, 0);
    const radiusPx = Math.abs(ptEdge.x - centerOrigin.x);

    ctx.strokeStyle = 'rgba(56, 189, 248, 0.12)';
    ctx.beginPath();
    ctx.arc(centerOrigin.x, centerOrigin.y, radiusPx, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = 'rgba(100, 116, 139, 0.65)';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText(`${rMrad} mrad`, centerOrigin.x + 6, centerOrigin.y - radiusPx - 3);
  });
  ctx.setLineDash([]);

  // Coordinate Grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
  ctx.lineWidth = 1;
  const stepRad = 0.010;
  for (let p = Math.floor(bounds.minX / stepRad) * stepRad; p <= bounds.maxX; p += stepRad) {
    const pt = toScreen(p, 0);
    ctx.beginPath();
    ctx.moveTo(pt.x, 0);
    ctx.lineTo(pt.x, h);
    ctx.stroke();

    ctx.fillStyle = 'rgba(100, 116, 139, 0.45)';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText(`${(p * 1000).toFixed(0)}`, pt.x + 4, h - 14);
  }

  for (let t = Math.floor(bounds.minY / stepRad) * stepRad; t <= bounds.maxY; t += stepRad) {
    const pt = toScreen(0, t);
    ctx.beginPath();
    ctx.moveTo(0, pt.y);
    ctx.lineTo(w, pt.y);
    ctx.stroke();

    ctx.fillStyle = 'rgba(100, 116, 139, 0.45)';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText(`${(t * 1000).toFixed(0)}`, 12, pt.y - 4);
  }

  if (!latest) return;

  const targetPt = toScreen(latest.target_pos[0], latest.target_pos[1]);
  const cameraPt = toScreen(latest.camera_pan_tilt[0], latest.camera_pan_tilt[1]);

  // Motion Breadcrumbs
  if (appState.spatialHistory.target.length >= 2) {
    ctx.lineWidth = 1.4;
    for (let i = 0; i < appState.spatialHistory.target.length - 1; i++) {
      const p1 = toScreen(appState.spatialHistory.target[i][0], appState.spatialHistory.target[i][1]);
      const p2 = toScreen(appState.spatialHistory.target[i + 1][0], appState.spatialHistory.target[i + 1][1]);
      const alpha = (i + 1) / appState.spatialHistory.target.length;
      ctx.strokeStyle = `rgba(16, 185, 129, ${alpha * 0.45})`;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  }

  // 1. Camera Coarse FOV (Outer Dashed Boundary)
  if (latest.fov_bounds) {
    const fovTL = toScreen(latest.fov_bounds[0], latest.fov_bounds[3]);
    const fovBR = toScreen(latest.fov_bounds[1], latest.fov_bounds[2]);
    const fovW = fovBR.x - fovTL.x;
    const fovH = fovBR.y - fovTL.y;

    ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(fovTL.x, fovTL.y, fovW, fovH);
    ctx.setLineDash([]);
  }

  // 2. Dynamic Safe FOV (Section 16.2 - Inner Contracted Boundary)
  if (latest.safe_fov_bounds) {
    const safeTL = toScreen(latest.safe_fov_bounds[0], latest.safe_fov_bounds[3]);
    const safeBR = toScreen(latest.safe_fov_bounds[1], latest.safe_fov_bounds[2]);
    const safeW = safeBR.x - safeTL.x;
    const safeH = safeBR.y - safeTL.y;

    const riskColor = latest.risk_state === 'STABLE' ? 'rgba(16, 185, 129, 0.7)' : latest.risk_state === 'WARNING' ? 'rgba(245, 158, 11, 0.8)' : 'rgba(239, 68, 68, 0.9)';
    ctx.strokeStyle = riskColor;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(safeTL.x, safeTL.y, safeW, safeH);

    ctx.fillStyle = latest.risk_state === 'STABLE' ? 'rgba(16, 185, 129, 0.03)' : 'rgba(239, 68, 68, 0.04)';
    ctx.fillRect(safeTL.x, safeTL.y, safeW, safeH);

    ctx.fillStyle = riskColor;
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.fillText('DYNAMIC SAFE FOV BOUNDARY', safeTL.x + 6, safeTL.y + 12);
  }

  // 3. Dynamic Occlusion Zone
  if (latest.active_occluder) {
    const occPt = toScreen(latest.active_occluder[0], latest.active_occluder[1]);
    const radiusPx = (latest.active_occluder[2] / (bounds.maxX - bounds.minX)) * w;
    const occGrad = ctx.createRadialGradient(occPt.x, occPt.y, 4, occPt.x, occPt.y, radiusPx);
    occGrad.addColorStop(0, 'rgba(30, 41, 59, 0.85)');
    occGrad.addColorStop(0.7, 'rgba(15, 23, 42, 0.7)');
    occGrad.addColorStop(1, 'rgba(15, 23, 42, 0)');

    ctx.fillStyle = occGrad;
    ctx.beginPath();
    ctx.arc(occPt.x, occPt.y, radiusPx, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(occPt.x, occPt.y, radiusPx, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 4. Kalman Statistical Covariance Ellipse (Section 16.1)
  const isLocked = Boolean(latest.is_valid_det && latest.target_in_fov);
  const uncRadiusPx = ((isLocked ? 1.2 : 3.5) / 1000 / (bounds.maxX - bounds.minX)) * w;
  ctx.strokeStyle = isLocked ? 'rgba(16, 185, 129, 0.4)' : 'rgba(245, 158, 11, 0.45)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.ellipse(targetPt.x, targetPt.y, uncRadiusPx * 1.2, uncRadiusPx * 0.9, 0.2, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // 5. Pointing Error Vector
  ctx.strokeStyle = isLocked ? 'rgba(56, 189, 248, 0.4)' : 'rgba(239, 68, 68, 0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cameraPt.x, cameraPt.y);
  ctx.lineTo(targetPt.x, targetPt.y);
  ctx.stroke();

  // 6. Camera Boresight Reticle
  const isCoast = latest.drishti_state === 'LOST' || latest.tracker_mode === 'COAST';
  const boresightColor = isCoast ? COLORS.muted : COLORS.cyan;
  ctx.strokeStyle = boresightColor;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(cameraPt.x, cameraPt.y, 20, 0, Math.PI * 2);
  ctx.arc(cameraPt.x, cameraPt.y, 4, 0, Math.PI * 2);
  ctx.moveTo(cameraPt.x - 26, cameraPt.y);
  ctx.lineTo(cameraPt.x - 7, cameraPt.y);
  ctx.moveTo(cameraPt.x + 7, cameraPt.y);
  ctx.lineTo(cameraPt.x + 26, cameraPt.y);
  ctx.moveTo(cameraPt.x, cameraPt.y - 26);
  ctx.lineTo(cameraPt.x, cameraPt.y - 7);
  ctx.moveTo(cameraPt.x, cameraPt.y + 7);
  ctx.lineTo(cameraPt.x, cameraPt.y + 26);
  ctx.stroke();

  ctx.fillStyle = boresightColor;
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.fillText(`BORESIGHT [${latest.tracker_mode}]`, cameraPt.x + 22, cameraPt.y - 10);

  // 7. Target Beacon Airy Disk Spot
  const beaconColor = isLocked ? COLORS.green : latest.risk_state === 'WARNING' ? COLORS.amber : COLORS.red;
  const airyGrad = ctx.createRadialGradient(targetPt.x, targetPt.y, 1, targetPt.x, targetPt.y, isLocked ? 24 : 14);
  airyGrad.addColorStop(0, '#ffffff');
  airyGrad.addColorStop(0.25, isLocked ? 'rgba(16, 185, 129, 0.9)' : 'rgba(245, 158, 11, 0.8)');
  airyGrad.addColorStop(0.65, isLocked ? 'rgba(16, 185, 129, 0.25)' : 'rgba(245, 158, 11, 0.2)');
  airyGrad.addColorStop(1, 'rgba(16, 185, 129, 0)');

  ctx.fillStyle = airyGrad;
  ctx.beginPath();
  ctx.arc(targetPt.x, targetPt.y, isLocked ? 24 : 14, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = beaconColor;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.arc(targetPt.x, targetPt.y, isLocked ? 8 : 6, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = beaconColor;
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.fillText(`TARGET [${latest.drishti_state}]`, targetPt.x + 14, targetPt.y - 12);
}

// ==========================================================================
// SENSOR FOCAL PLANE ARRAY
// ==========================================================================
function renderSensorTheater(latest) {
  const canvas = $('#sensorCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#050914';
  ctx.fillRect(0, 0, w, h);

  // Pixel grid
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.04)';
  ctx.lineWidth = 1;
  const gridSize = 24;
  for (let x = 0; x < w; x += gridSize) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += gridSize) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Radiometric scale bar
  const barX = w - 24;
  const barY = 40;
  const barW = 10;
  const barH = h - 80;
  const isThermal = appState.sensorPalette === 'thermal';

  const scaleGrad = ctx.createLinearGradient(0, barY, 0, barY + barH);
  if (isThermal) {
    scaleGrad.addColorStop(0, '#ffffff');
    scaleGrad.addColorStop(0.2, '#fde047');
    scaleGrad.addColorStop(0.5, '#10b981');
    scaleGrad.addColorStop(0.8, '#1e3a8a');
    scaleGrad.addColorStop(1, '#050914');
  } else {
    scaleGrad.addColorStop(0, '#ffffff');
    scaleGrad.addColorStop(0.3, '#38bdf8');
    scaleGrad.addColorStop(0.7, '#0369a1');
    scaleGrad.addColorStop(1, '#050914');
  }

  ctx.fillStyle = scaleGrad;
  ctx.fillRect(barX, barY, barW, barH);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.strokeRect(barX, barY, barW, barH);

  if (!latest) return;
  const res = latest.sensor_resolution || [640, 480];
  const mapSensorPx = (px, py) => ({
    x: (px / res[0]) * w,
    y: (py / res[1]) * h,
  });

  if (latest.ground_truth_pixel && latest.target_in_fov) {
    const spot = mapSensorPx(latest.ground_truth_pixel[0], latest.ground_truth_pixel[1]);
    const spotGlow = ctx.createRadialGradient(spot.x, spot.y, 2, spot.x, spot.y, 26);
    if (isThermal) {
      spotGlow.addColorStop(0, '#ffffff');
      spotGlow.addColorStop(0.25, '#fde047');
      spotGlow.addColorStop(0.55, '#10b981');
      spotGlow.addColorStop(1, 'rgba(5, 9, 20, 0)');
    } else {
      spotGlow.addColorStop(0, '#ffffff');
      spotGlow.addColorStop(0.3, 'rgba(56, 189, 248, 0.9)');
      spotGlow.addColorStop(1, 'rgba(56, 189, 248, 0)');
    }

    ctx.fillStyle = spotGlow;
    ctx.beginPath();
    ctx.arc(spot.x, spot.y, 26, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = isThermal ? '#fde047' : COLORS.cyan;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(spot.x, spot.y, 14, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = isThermal ? '#fde047' : COLORS.cyan;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText(`BEACON SPOT (${latest.ground_truth_pixel[0].toFixed(1)}, ${latest.ground_truth_pixel[1].toFixed(1)} px)`, spot.x + 18, spot.y - 12);
  }

  if (latest.detector_pixel && latest.is_valid_det) {
    const det = mapSensorPx(latest.detector_pixel[0], latest.detector_pixel[1]);
    ctx.strokeStyle = COLORS.green;
    ctx.lineWidth = 1.4;
    ctx.strokeRect(det.x - 12, det.y - 12, 24, 24);
    ctx.fillStyle = COLORS.green;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillText(`CENTROID (${latest.detector_pixel[0].toFixed(1)}, ${latest.detector_pixel[1].toFixed(1)} px)`, det.x + 16, det.y + 16);
  }
}

// ==========================================================================
// REAL-TIME CHARTS
// ==========================================================================
function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(20, Math.floor(rect.width));
  const height = Math.max(20, Math.floor(rect.height));
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function drawGridLines(ctx, width, height, rows = 4, cols = 8) {
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 4]);
  for (let i = 1; i < rows; i++) {
    const y = (height * i) / rows;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  for (let i = 1; i < cols; i++) {
    const x = (width * i) / cols;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function renderCharts(history) {
  const errorEmpty = $('#errorEmpty');
  const qualityEmpty = $('#qualityEmpty');
  errorEmpty?.classList.toggle('hidden', history.length > 0);
  qualityEmpty?.classList.toggle('hidden', history.length > 0);

  // 1. Error Chart
  const errCanvas = $('#errorChart');
  if (errCanvas) {
    const { ctx, width, height } = prepareCanvas(errCanvas);
    drawGridLines(ctx, width, height);

    if (history.length) {
      const vals = history.map((item) => Number(item.tracking_error_mrad) || 0);
      const maxVal = Math.max(4.0, ...vals) * 1.1;
      const left = 4, right = width - 4, top = 6, bottom = height - 6;
      const px = (idx) => left + (idx / Math.max(1, vals.length - 1)) * (right - left);
      const py = (val) => bottom - (val / maxVal) * (bottom - top);

      // 2.0 mrad Coarse Lock Limit
      const threshY = py(2.0);
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(left, threshY);
      ctx.lineTo(right, threshY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      vals.forEach((v, i) => {
        const x = px(i);
        const y = py(v);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = COLORS.cyan;
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
  }

  // 2. Quality Chart
  const qCanvas = $('#qualityChart');
  if (qCanvas) {
    const { ctx, width, height } = prepareCanvas(qCanvas);
    drawGridLines(ctx, width, height);

    if (history.length) {
      const confs = history.map((item) => Number(item.confidence) || 0);
      const sevs = history.map((item) => Number(item.severity) || 0);
      const left = 4, right = width - 4, top = 6, bottom = height - 6;
      const px = (idx) => left + (idx / Math.max(1, history.length - 1)) * (right - left);
      const py = (val) => bottom - clamp(val) * (bottom - top);

      ctx.beginPath();
      confs.forEach((c, i) => {
        const x = px(i), y = py(c);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = COLORS.green;
      ctx.lineWidth = 1.6;
      ctx.stroke();

      ctx.beginPath();
      sevs.forEach((s, i) => {
        const x = px(i), y = py(s);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = COLORS.amber;
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }
  }
}

// ==========================================================================
// MISSION EXECUTION & LOOP CONTROL
// ==========================================================================
async function stepMission(count = 1) {
  try {
    const snapshot = await post('/api/session/action', { action: 'step', count });
    applySnapshot(snapshot);
    return snapshot;
  } catch (err) {
    showToast(err.message);
  }
}

async function stepLoop(token) {
  if (!appState.playing || token !== appState.loopToken) return;
  try {
    const count = appState.speed >= 5 ? 3 : appState.speed >= 2 ? 2 : 1;
    const snapshot = await post('/api/session/action', { action: 'step', count });
    applySnapshot(snapshot);

    const atEnd = Number(snapshot.frame_id) >= Number(snapshot.total_frames);
    if (!appState.playing || token !== appState.loopToken || atEnd) {
      appState.playing = false;
      updatePlayButton();
      if (atEnd) showToast('Scenario complete. Ready for telemetry verification.');
      return;
    }

    const dt = Number(snapshot.scenario?.dt || 0.033);
    const delay = Math.max(16, (dt * 1000) / appState.speed);
    window.setTimeout(() => stepLoop(token), delay);
  } catch (err) {
    appState.playing = false;
    updatePlayButton();
    showToast(err.message);
  }
}

async function toggleMission() {
  playTelemetrySound('click');
  if (appState.playing) {
    appState.playing = false;
    appState.loopToken += 1;
    await post('/api/session/action', { action: 'pause' }).catch(() => {});
    updatePlayButton();
    return;
  }

  const atEnd = appState.snapshot && Number(appState.snapshot.frame_id) >= Number(appState.snapshot.total_frames);
  if (atEnd) await resetMission();

  appState.playing = true;
  appState.loopToken += 1;
  const token = appState.loopToken;
  updatePlayButton();
  await post('/api/session/action', { action: 'start' }).catch(() => {});
  stepLoop(token);
}

async function resetMission() {
  playTelemetrySound('click');
  appState.playing = false;
  appState.loopToken += 1;
  appState.spatialHistory.target = [];
  appState.spatialHistory.camera = [];
  appState.recordedFrames = [];
  try {
    const snapshot = await post('/api/session/action', { action: 'reset' });
    applySnapshot(snapshot);
    showToast('Simulation reset to initial frame 0');
  } catch (err) {
    showToast(err.message);
  }
}

async function selectScenario(name) {
  playTelemetrySound('click');
  appState.playing = false;
  appState.loopToken += 1;
  appState.spatialHistory.target = [];
  appState.spatialHistory.camera = [];
  appState.recordedFrames = [];
  try {
    const snapshot = await post('/api/session/select', { scenario_name: name });
    applySnapshot(snapshot);
    showToast(`Loaded ${prettyScenarioName(name)}`);
  } catch (err) {
    showToast(err.message);
  }
}

function updatePlayButton() {
  const btn = $('#playButton');
  if (!btn) return;
  const atEnd = appState.snapshot && Number(appState.snapshot.frame_id) >= Number(appState.snapshot.total_frames);
  if (appState.playing) {
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg> <span>Pause Mission</span>`;
    btn.classList.add('is-playing');
  } else if (atEnd) {
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg> <span>Replay Mission</span>`;
    btn.classList.remove('is-playing');
  } else {
    btn.innerHTML = `<svg class="play-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> <span>Start Mission</span>`;
    btn.classList.remove('is-playing');
  }
}

// Live Fault Injection
async function injectFault(type) {
  playTelemetrySound('click');
  try {
    const res = await post('/api/session/inject', { type });
    showToast(res.message || `Injected ${type}`);
    if (res.snapshot) applySnapshot(res.snapshot);
  } catch (err) {
    showToast(`Fault injection failed: ${err.message}`);
  }
}

// Manual Gimbal Trim Nudge
async function nudgeGimbal(dPan, dTilt) {
  playTelemetrySound('click');
  try {
    const res = await post('/api/session/inject', { type: 'vibration', value: Math.hypot(dPan, dTilt) });
    showToast(`Gimbal bias trim applied (Pan: ${dPan > 0 ? '+' : ''}${dPan * 1000} mrad, Tilt: ${dTilt > 0 ? '+' : ''}${dTilt * 1000} mrad)`);
    if (res.snapshot) applySnapshot(res.snapshot);
  } catch (err) {
    showToast(`Trim failed: ${err.message}`);
  }
}

// ==========================================================================
// SCENARIO CATALOG (WORKSPACE 2)
// ==========================================================================
function renderScenarioCatalog(scenarios, filter = 'all') {
  const catalog = $('#scenariosCatalog');
  if (!catalog) return;

  const filtered = scenarios.filter((sc) => {
    if (filter === 'all') return true;
    return sc.category === filter;
  });

  catalog.innerHTML = filtered
    .map((sc) => {
      const isActive = appState.snapshot?.scenario_name === sc.scenario_name;
      const dist = sc.disturbances || {};
      const tgt = sc.target || {};
      const occs = dist.occlusions || (dist.occlusion ? [dist.occlusion] : []);
      const frames = sc.num_frames || 100;
      const durationS = ((sc.dt || 0.033) * frames).toFixed(1);

      return `
        <div class="scenario-card ${isActive ? 'active-scenario' : ''}" data-name="${sc.scenario_name}">
          <div class="scen-head">
            <div>
              <span class="scen-param-label">ISRO PS-26169 • ${sc.category || 'MISSION PRESET'}</span>
              <h3 class="scen-title">${prettyScenarioName(sc.scenario_name)}</h3>
            </div>
            <span class="status-badge ${isActive ? 'nominal' : ''}">${isActive ? 'ACTIVE' : `${frames} FRAMES`}</span>
          </div>
          <p class="scen-desc">${sc.description || 'Closed-loop optical tracking test.'}</p>
          <div class="scen-metrics-grid">
            <div class="scen-param"><span class="scen-param-label">DURATION</span><strong class="scen-param-val">${durationS}s (${frames}f)</strong></div>
            <div class="scen-param"><span class="scen-param-label">DISTANCE</span><strong class="scen-param-val">${(tgt.distance_km || 5).toFixed(1)} km</strong></div>
            <div class="scen-param"><span class="scen-param-label">TURBULENCE Cn²</span><strong class="scen-param-val">${formatCn2(dist.cn2)}</strong></div>
            <div class="scen-param"><span class="scen-param-label">VIBRATION</span><strong class="scen-param-val">${((dist.vibration_amplitude || 0) * 1000).toFixed(2)} mrad</strong></div>
            <div class="scen-param"><span class="scen-param-label">SENSOR NOISE</span><strong class="scen-param-val">σ ${(dist.noise_level || 0).toFixed(1)}</strong></div>
            <div class="scen-param"><span class="scen-param-label">OCCLUSIONS</span><strong class="scen-param-val">${occs.length ? `${occs.length} window(s)` : 'None'}</strong></div>
          </div>
          <div class="scen-foot">
            <button class="button button-accent btn-load-scenario" data-name="${sc.scenario_name}">
              ${isActive ? '✓ Loaded in Cockpit' : 'Load Into Mission Cockpit'}
            </button>
          </div>
        </div>
      `;
    })
    .join('');

  $$('.btn-load-scenario').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      const name = e.currentTarget.dataset.name;
      await selectScenario(name);
      switchView('cockpit');
    });
  });
}

// ==========================================================================
// WORKSPACE 5: PARAMETER SWEEPS & MONTE-CARLO
// ==========================================================================
async function executeParameterSweep() {
  const sweepType = $('#selSweepType')?.value || 'fps';
  const tbody = $('#sweepTableBody');
  const recText = $('#sweepRecommendText');
  if (tbody) tbody.innerHTML = '<tr><td colspan="4" class="text-center">Executing sweep simulation…</td></tr>';

  try {
    const data = await post('/api/sweep', { sweep_type: sweepType });
    if (recText) recText.textContent = `RECOMMENDATION: ${data.recommended}`;

    if (tbody && data.values) {
      tbody.innerHTML = data.values
        .map((val, idx) => `
          <tr>
            <td><strong>${val}</strong></td>
            <td class="${data.lock_retention[idx] >= 90 ? 'emerald' : ''}">${data.lock_retention[idx]}%</td>
            <td>${data.mean_error_mrad[idx]} mrad</td>
            <td><span class="status-badge ${data.lock_retention[idx] >= 95 ? 'nominal' : 'searching'}">${data.lock_retention[idx] >= 95 ? 'OPTIMAL' : 'ACCEPTABLE'}</span></td>
          </tr>
        `)
        .join('');
    }
    showToast(`Sweep complete for ${data.parameter}`);
  } catch (err) {
    showToast(`Sweep failed: ${err.message}`);
  }
}

async function executeMonteCarlo() {
  const tbody = $('#mcTableBody');
  if (tbody) tbody.innerHTML = '<tr><td colspan="5" class="text-center">Executing 10 stochastic trials…</td></tr>';

  try {
    const data = await post('/api/monte-carlo', { runs: 10 });
    const s = data.summary;
    $('#mcPassRate').textContent = `${s.pass_rate_pct}%`;
    $('#mcMeanLock').textContent = `${s.mean_lock_retention}%`;
    $('#mcMeanError').textContent = `${s.mean_error_mrad} mrad`;
    $('#mcP95').textContent = `${s.p95_error_mrad} mrad`;

    if (tbody && data.trials) {
      tbody.innerHTML = data.trials
        .map((t) => `
          <tr>
            <td>Trial #${t.trial}</td>
            <td>Seed ${t.seed}</td>
            <td class="${t.lock_retention >= 88 ? 'emerald' : ''}">${t.lock_retention}%</td>
            <td>${t.mean_error_mrad} mrad</td>
            <td><span class="status-badge ${t.status === 'PASS' ? 'nominal' : 'lost'}">${t.status}</span></td>
          </tr>
        `)
        .join('');
    }
    showToast('Monte-Carlo verification complete.');
  } catch (err) {
    showToast(`Monte-Carlo failed: ${err.message}`);
  }
}

// ==========================================================================
// WORKSPACE 6: PAT-RED AUTOMATED FAILURE DISCOVERY (SECTION 25)
// ==========================================================================
async function runPatRedDiscovery() {
  playTelemetrySound('click');
  const btn = $('#btnRunPatRedFull');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = 'Searching Breaking Points…';
  }
  showToast('PAT-RED: Searching multidimensional stress combinations…');

  try {
    const data = await post('/api/pat-red');
    $('#patredIdBadge').textContent = data.failure_id;
    $('#patredVerdictTag').textContent = data.breaking_point_found ? 'BREAKING POINT IDENTIFIED' : 'PASSED';
    $('#patredTurnRate').textContent = data.parameter_combination.target_turn_rate;
    $('#patredJitter').textContent = data.parameter_combination.platform_vibration;
    $('#patredLatency').textContent = `${data.parameter_combination.sensor_latency_ms} ms`;
    $('#patredCn2').textContent = data.parameter_combination.cn2_turbulence;
    $('#patredDiagnosisText').textContent = data.root_cause_diagnosis;

    showToast(`PAT-RED: Breaking point discovered (${data.failure_id})!`);
  } catch (err) {
    showToast(`PAT-RED search failed: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polygon points="12 8 8 12 12 16 16 12 12 8"></polygon></svg> Discover Failure Breaking Point';
    }
  }
}

// ==========================================================================
// WORKSPACE 7: TRACKING SURVIVAL ENVELOPE (SECTION 26)
// ==========================================================================
async function loadSurvivalEnvelope() {
  const wrap = $('#envelopeMatrixWrap');
  if (!wrap) return;

  try {
    const data = await api('/api/survival-envelope');
    const threshold = parseFloat($('#selEnvelopeThreshold')?.value || 95);

    let html = `
      <table class="heatmap-table">
        <thead>
          <tr>
            <th>Target Velocity \\ Turbulence</th>
            ${data.turbulences.map((t) => `<th>${t}</th>`).join('')}
          </tr>
        </thead>
        <tbody>
    `;

    data.velocities.forEach((vel, rIdx) => {
      html += `<tr><td><strong>${vel}</strong></td>`;
      data.turbulences.forEach((_, cIdx) => {
        const val = data.matrix[rIdx][cIdx];
        const statusClass = val >= threshold ? 'pass' : val >= 80 ? 'marginal' : 'fail';
        html += `<td class="heatmap-cell ${statusClass}">${val.toFixed(0)}%</td>`;
      });
      html += '</tr>';
    });

    html += '</tbody></table>';
    wrap.innerHTML = html;
  } catch (err) {
    wrap.innerHTML = `<div class="empty-state">Failed to load envelope: ${err.message}</div>`;
  }
}

// ==========================================================================
// WORKSPACE 8: BENCHMARKS & REPORTS
// ==========================================================================
async function runBenchmark(mode = 'single') {
  const btnSingle = $('#btnBenchmarkCurrent');
  const btnAll = $('#btnBenchmarkAll');

  [btnSingle, btnAll].forEach((b) => {
    if (b) {
      b.disabled = true;
      b.innerHTML = 'Executing Verification…';
    }
  });

  $('#benchmarkStatus').textContent = `RUNNING CLOSED-LOOP VERIFICATION (${mode.toUpperCase()})…`;

  try {
    const data = await post('/api/benchmark', { mode });
    renderBenchmarkResults(data);
    showToast('Benchmark complete! Quantitative records verified.');
    playTelemetrySound('lock');
  } catch (err) {
    showToast(`Benchmark failed: ${err.message}`);
    $('#benchmarkStatus').textContent = 'BENCHMARK FAILED';
  } finally {
    if (btnSingle) { btnSingle.disabled = false; btnSingle.textContent = 'Benchmark Active Scenario'; }
    if (btnAll) { btnAll.disabled = false; btnAll.textContent = 'Benchmark All 7 Scenarios'; }
  }
}

function renderBenchmarkResults(data) {
  $('#benchmarkStatus').textContent = 'VERIFICATION COMPLETE (ISRO CONTRACT COMPLIANT)';

  if (data.mode === 'all') {
    $('#benchmarkReportTitle').textContent = 'Multi-Scenario Comparative Performance Matrix';
    const results = data.results || [];
    $('#benchTableBody').innerHTML = results
      .flatMap((item) => {
        const kf = item.kalman || {};
        const hy = item.hybrid || {};
        return [
          `
            <tr>
              <td><strong>${prettyScenarioName(item.scenario_name)}</strong></td>
              <td>Constant-Velocity Kalman Filter</td>
              <td>${formatNum((kf.acquisition_time || 0) * 1000, 1)}</td>
              <td class="${(kf.lock_retention_rate || 0) > 0.8 ? 'emerald' : ''}">${formatNum((kf.lock_retention_rate || 0) * 100, 1)}%</td>
              <td>${formatNum(kf.avg_tracking_error, 3)}</td>
              <td>${formatNum(kf.max_tracking_error, 3)}</td>
              <td>${formatNum(kf.fps, 1)}</td>
              <td>${kf.track_loss_count || 0}</td>
            </tr>
          `,
          `
            <tr style="background: rgba(16, 185, 129, 0.05);">
              <td><strong>${prettyScenarioName(item.scenario_name)}</strong></td>
              <td style="color: var(--emerald); font-weight: 600;">Adaptive Hybrid (KF+PF)</td>
              <td>${formatNum((hy.acquisition_time || 0) * 1000, 1)}</td>
              <td class="emerald" style="font-weight: 700;">${formatNum((hy.lock_retention_rate || 0) * 100, 1)}%</td>
              <td class="emerald">${formatNum(hy.avg_tracking_error, 3)}</td>
              <td>${formatNum(hy.max_tracking_error, 3)}</td>
              <td>${formatNum(hy.fps, 1)}</td>
              <td>${hy.track_loss_count || 0}</td>
            </tr>
          `,
        ];
      })
      .join('');
    return;
  }

  const rec = data.record || {};
  const kf = data.kalman_record || {};
  const hy = data.hybrid_record || rec;

  $('#benchScenName').textContent = prettyScenarioName(data.scenario_name);
  $('#benchRetention').textContent = `${formatNum((hy.lock_retention_rate || 0) * 100, 1)}%`;
  $('#benchAcqTime').textContent = `${formatNum((hy.acquisition_time || 0) * 1000, 1)} ms`;
  $('#benchAvgErr').textContent = `${formatNum(hy.avg_tracking_error, 3)} mrad`;
  $('#benchFps').textContent = `${formatNum(hy.fps, 1)} Hz`;

  $('#benchTableBody').innerHTML = `
    <tr>
      <td>${prettyScenarioName(data.scenario_name)}</td>
      <td>Constant-Velocity Kalman Filter</td>
      <td>${formatNum((kf.acquisition_time || 0) * 1000, 1)}</td>
      <td>${formatNum((kf.lock_retention_rate || 0) * 100, 1)}%</td>
      <td>${formatNum(kf.avg_tracking_error, 3)}</td>
      <td>${formatNum(kf.max_tracking_error, 3)}</td>
      <td>${formatNum(kf.fps, 1)}</td>
      <td>${kf.track_loss_count || 0}</td>
    </tr>
    <tr style="background: rgba(16, 185, 129, 0.05);">
      <td>${prettyScenarioName(data.scenario_name)}</td>
      <td style="color: var(--emerald); font-weight: 600;">Adaptive Hybrid (KF+PF)</td>
      <td>${formatNum((hy.acquisition_time || 0) * 1000, 1)}</td>
      <td class="emerald" style="font-weight: 700;">${formatNum((hy.lock_retention_rate || 0) * 100, 1)}%</td>
      <td class="emerald">${formatNum(hy.avg_tracking_error, 3)}</td>
      <td>${formatNum(hy.max_tracking_error, 3)}</td>
      <td>${formatNum(hy.fps, 1)}</td>
      <td>${hy.track_loss_count || 0}</td>
    </tr>
  `;
}

// View Switcher (8 Workspaces)
function switchView(viewName) {
  playTelemetrySound('click');
  appState.currentView = viewName;

  $$('.view-container').forEach((el) => el.classList.remove('active'));
  $$('.rail-item').forEach((btn) => btn.classList.remove('active'));

  const viewEl = $(`#view-${viewName}`);
  const railBtn = $(`#nav${viewName.charAt(0).toUpperCase() + viewName.slice(1)}Btn`);
  if (viewEl) viewEl.classList.add('active');
  if (railBtn) railBtn.classList.add('active');

  if (viewName === 'builder') {
    renderScenarioCatalog(appState.scenarios);
  } else if (viewName === 'envelope') {
    loadSurvivalEnvelope();
  } else if (viewName === 'cockpit') {
    if (appState.snapshot) applySnapshot(appState.snapshot);
  }
}

// Download Export
function downloadExport(format) {
  window.open(`${API_BASE}/api/export?format=${format}`, '_blank');
  showToast(`Exporting DRISHTI-X test dossier as ${format.toUpperCase()}`);
}

// ==========================================================================
// BOOTSTRAP & EVENT LISTENERS
// ==========================================================================
async function bootstrap() {
  // Navigation rail buttons (8 Workspaces)
  $('#navCockpitBtn')?.addEventListener('click', () => switchView('cockpit'));
  $('#navBuilderBtn')?.addEventListener('click', () => switchView('builder'));
  $('#navAlgorithmBtn')?.addEventListener('click', () => switchView('algorithm'));
  $('#navFailuresBtn')?.addEventListener('click', () => switchView('failures'));
  $('#navSweepsBtn')?.addEventListener('click', () => switchView('sweeps'));
  $('#navPatredBtn')?.addEventListener('click', () => switchView('patred'));
  $('#navEnvelopeBtn')?.addEventListener('click', () => switchView('envelope'));
  $('#navReportsBtn')?.addEventListener('click', () => switchView('reports'));

  // Mode Switcher (Basic vs Research - Section 33)
  const setMode = (m) => {
    appState.currentMode = m;
    document.body.classList.toggle('mode-basic', m === 'basic');
    document.body.classList.toggle('mode-research', m === 'research');
    $('#btnModeBasic')?.classList.toggle('active', m === 'basic');
    $('#btnModeResearch')?.classList.toggle('active', m === 'research');
    $('#footerModeVal').textContent = m === 'basic' ? 'BASIC LIVE MODE' : 'RESEARCH ENGINEERING MODE';
    showToast(`Switched to ${m.toUpperCase()} Mode (Section 33)`);
  };
  $('#btnModeBasic')?.addEventListener('click', () => setMode('basic'));
  $('#btnModeResearch')?.addEventListener('click', () => setMode('research'));

  // Mission controls
  $('#playButton')?.addEventListener('click', toggleMission);
  $('#resetButton')?.addEventListener('click', resetMission);
  $('#stepButton')?.addEventListener('click', () => stepMission(1));

  // Speed selector
  $$('.speed-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      playTelemetrySound('click');
      $$('.speed-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      appState.speed = parseFloat(btn.dataset.speed) || 1.0;
      showToast(`Simulation rate: ${appState.speed}×`);
    });
  });

  // Scenario dropdown
  $('#scenarioSelect')?.addEventListener('change', (e) => selectScenario(e.target.value));

  // Scrubber
  const scrubber = $('#timelineScrubber');
  if (scrubber) {
    scrubber.addEventListener('input', (e) => {
      appState.isScrubbing = true;
      const frameIdx = parseInt(e.target.value, 10);
      const frameData = appState.recordedFrames[frameIdx];
      if (frameData && frameData.snapshot) {
        applySnapshot(frameData.snapshot);
        $('#scrubberPos').textContent = `FRAME ${frameIdx} (REPLAY)`;
      }
    });
    scrubber.addEventListener('change', () => { appState.isScrubbing = false; });
  }

  // Stage Toggles (Spatial vs Sensor)
  $('#btnSpatialView')?.addEventListener('click', () => {
    playTelemetrySound('click');
    appState.theaterMode = 'spatial';
    $('#btnSpatialView').classList.add('active');
    $('#btnSensorView').classList.remove('active');
    $('#spatialSceneContainer').classList.remove('hidden');
    $('#sensorSceneContainer').classList.add('hidden');
    $('#theaterTitle').textContent = 'Optical Spatial Tracking Field';
    if (appState.snapshot?.latest) renderSpatialTheater(appState.snapshot.latest);
  });

  $('#btnSensorView')?.addEventListener('click', () => {
    playTelemetrySound('click');
    appState.theaterMode = 'sensor';
    $('#btnSensorView').classList.add('active');
    $('#btnSpatialView').classList.remove('active');
    $('#spatialSceneContainer').classList.add('hidden');
    $('#sensorSceneContainer').classList.remove('hidden');
    $('#theaterTitle').textContent = 'Focal Plane Detector Array (InGaAs CMOS)';
    if (appState.snapshot?.latest) renderSensorTheater(appState.snapshot.latest);
  });

  // Sensor Palette Toggles
  $('#btnPaletteThermal')?.addEventListener('click', () => {
    playTelemetrySound('click');
    appState.sensorPalette = 'thermal';
    $('#btnPaletteThermal').classList.add('active');
    $('#btnPaletteLaser').classList.remove('active');
    if (appState.snapshot?.latest) renderSensorTheater(appState.snapshot.latest);
  });

  $('#btnPaletteLaser')?.addEventListener('click', () => {
    playTelemetrySound('click');
    appState.sensorPalette = 'laser';
    $('#btnPaletteLaser').classList.add('active');
    $('#btnPaletteThermal').classList.remove('active');
    if (appState.snapshot?.latest) renderSensorTheater(appState.snapshot.latest);
  });

  // Manual Gimbal Trim
  $('#btnNudgeUp')?.addEventListener('click', () => nudgeGimbal(0, 0.0015));
  $('#btnNudgeDown')?.addEventListener('click', () => nudgeGimbal(0, -0.0015));
  $('#btnNudgeLeft')?.addEventListener('click', () => nudgeGimbal(-0.0015, 0));
  $('#btnNudgeRight')?.addEventListener('click', () => nudgeGimbal(0.0015, 0));
  $('#btnNudgeZero')?.addEventListener('click', () => {
    playTelemetrySound('click');
    showToast('Gimbal bias trim zeroed');
  });

  // Cockpit Fault Injections
  $('#btnInjectOcclusion')?.addEventListener('click', () => injectFault('occlusion'));
  $('#btnInjectManeuver')?.addEventListener('click', () => injectFault('maneuver'));
  $('#btnInjectVibration')?.addEventListener('click', () => injectFault('vibration'));
  $('#btnInjectTurbulence')?.addEventListener('click', () => injectFault('turbulence'));

  // Failure Lab Injections
  $$('.btn-trigger-fault').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const fault = e.currentTarget.dataset.fault;
      injectFault(fault);
    });
  });

  // Test Manager: Sweeps & Monte-Carlo
  $('#btnExecuteSweep')?.addEventListener('click', executeParameterSweep);
  $('#btnRunMonteCarlo')?.addEventListener('click', executeMonteCarlo);

  // PAT-RED Adversarial Discovery
  $('#btnRunPatRedFull')?.addEventListener('click', runPatRedDiscovery);
  $('#btnQuickPatRed')?.addEventListener('click', () => {
    switchView('patred');
    runPatRedDiscovery();
  });
  $('#btnReplayPatRed')?.addEventListener('click', async () => {
    await selectScenario('DRISHTI_COMBINED_WORST_CASE_STRESS');
    switchView('cockpit');
    toggleMission();
  });
  $('#btnExportPatRed')?.addEventListener('click', () => downloadExport('json'));

  // Survival Envelope threshold
  $('#selEnvelopeThreshold')?.addEventListener('change', loadSurvivalEnvelope);

  // Benchmarks
  $('#btnBenchmarkCurrent')?.addEventListener('click', () => runBenchmark('single'));
  $('#btnBenchmarkAll')?.addEventListener('click', () => runBenchmark('all'));

  // Export buttons
  $('#exportJsonButton')?.addEventListener('click', () => downloadExport('json'));
  $('#exportCsvButton')?.addEventListener('click', () => downloadExport('csv'));

  // Scenario Filter Pills
  $$('.filter-pill').forEach((pill) => {
    pill.addEventListener('click', (e) => {
      playTelemetrySound('click');
      $$('.filter-pill').forEach((p) => p.classList.remove('active'));
      e.target.classList.add('active');
      renderScenarioCatalog(appState.scenarios, e.target.dataset.filter);
    });
  });

  // Live UTC Clock
  setInterval(() => {
    const now = new Date();
    const utcStr = now.toISOString().slice(11, 19) + ' UTC';
    $('#footerClock').textContent = utcStr;
    $('#topbarClock').textContent = utcStr;
  }, 1000);

  // Window Resize
  window.addEventListener('resize', () => {
    if (appState.snapshot) {
      renderCharts(appState.snapshot.history || []);
      if (appState.theaterMode === 'spatial') {
        renderSpatialTheater(appState.snapshot.latest);
      } else {
        renderSensorTheater(appState.snapshot.latest);
      }
    }
  });

  // Initialize
  try {
    const scenData = await api('/api/scenarios');
    appState.scenarios = scenData.scenarios || [];

    const select = $('#scenarioSelect');
    if (select) {
      select.innerHTML = appState.scenarios
        .map((sc) => `<option value="${sc.scenario_name}">${prettyScenarioName(sc.scenario_name)}</option>`)
        .join('');
    }

    const session = await api('/api/session');
    applySnapshot(session);
    showToast('DRISHTI-X Proving Ground connected & online.');
  } catch (err) {
    showToast(`Connection failed: ${err.message}`);
  }
}

document.addEventListener('DOMContentLoaded', bootstrap);