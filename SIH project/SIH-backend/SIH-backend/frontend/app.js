const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const appState = {
  snapshot: null,
  scenarios: [],
  playing: false,
  speed: 1,
  loopToken: 0,
  lastEventCount: 0,
  toastTimer: null,
};

const COLORS = {
  cyan: '#79b9aa',
  green: '#b7c77e',
  amber: '#d39a55',
  red: '#c86d5f',
  muted: '#7d8a82',
  grid: 'rgba(125, 144, 130, .16)',
};

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, Number(value) || 0));
}

function prettyScenarioName(name) {
  return String(name || 'UNKNOWN').replace(/^DEMO_/, '').replaceAll('_', ' ').toLowerCase().replace(/(^|\s)\S/g, (letter) => letter.toUpperCase());
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

function formatCn2(value) {
  const number = Number(value);
  if (!number) return 'OFF';
  return number.toExponential(1).replace('+', '');
}

function formatTime(seconds) {
  const totalMs = Math.max(0, Math.round((Number(seconds) || 0) * 1000));
  const minutes = Math.floor(totalMs / 60000);
  const secs = Math.floor((totalMs % 60000) / 1000);
  const millis = totalMs % 1000;
  return `T+${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
}

function showToast(message) {
  const toast = $('#toast');
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(appState.toastTimer);
  appState.toastTimer = setTimeout(() => toast.classList.remove('show'), 2800);
}

async function api(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

async function post(path, body = {}) {
  return api(path, { method: 'POST', body: JSON.stringify(body) });
}

function setConnection(connected, label = connected ? 'CONNECTED' : 'OFFLINE') {
  $('#connectionLabel').textContent = label;
  const dot = document.querySelector('.status-dot');
  dot.classList.toggle('offline', !connected);
}

function applySnapshot(snapshot) {
  appState.snapshot = snapshot;
  setConnection(Boolean(snapshot.backend_ready), snapshot.backend_ready ? 'CONNECTED' : 'BACKEND ERROR');
  renderScenarioDetails(snapshot);
  renderLatest(snapshot);
  renderMetrics(snapshot.metrics || {});
  renderEvents(snapshot.events || []);
  renderTrackingSvg(snapshot);
  renderCharts(snapshot.history || []);
  updatePlayButton();
  $('#footerClock').textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function renderScenarioDetails(snapshot) {
  const scenario = snapshot.scenario || {};
  const name = snapshot.scenario_name || scenario.scenario_name || 'UNKNOWN';
  const select = $('#scenarioSelect');
  if (select.value !== name) select.value = name;
  $('#scenarioDescription').textContent = scenario.description || 'No scenario description available.';
  const disturbance = scenario.disturbances || {};
  const chips = [
    ['ATMOSPHERE', formatCn2(disturbance.cn2), Number(disturbance.cn2) >= 1e-14],
    ['VIBRATION', `${formatNumber((Number(disturbance.vibration_amplitude) || 0) * 1000, 2)} mrad`, Number(disturbance.vibration_amplitude) >= 0.0004],
    ['SENSOR NOISE', `σ ${formatNumber(disturbance.noise_level, 1)}`, Number(disturbance.noise_level) >= 5],
  ];
  const occlusions = disturbance.occlusions || (disturbance.occlusion ? [disturbance.occlusion] : []);
  if (occlusions.length) chips.push(['OCCLUSION', `${occlusions.length} window${occlusions.length > 1 ? 's' : ''}`, true]);
  $('#disturbanceChips').innerHTML = chips.map(([label, value, hot]) => `<span class="dist-chip${hot ? ' hot' : ''}">${label} · ${value}</span>`).join('');

  const profileLevel = Number(disturbance.cn2) >= 1e-14 || Number(disturbance.noise_level) >= 5 || occlusions.length ? 'STRESS TEST' : (Number(disturbance.vibration_amplitude) > 0.0003 ? 'JITTER TEST' : 'BASELINE');
  $('#profileTag').textContent = profileLevel;
  $('#profileCaption').textContent = scenario.description || 'Select a scenario to inspect its stress model.';
  const target = scenario.target || {};
  const profileRows = [
    ['Atmospheric Cn²', Number(disturbance.cn2) > 0 ? Math.min(100, Math.round((Math.log10(Number(disturbance.cn2)) + 16) * 20)) : 0, formatCn2(disturbance.cn2)],
    ['Platform vibration', Math.min(100, Math.round((Number(disturbance.vibration_amplitude) || 0) / 0.0006 * 100)), `${formatNumber((Number(disturbance.vibration_amplitude) || 0) * 1000, 2)} mrad`],
    ['Sensor noise', Math.min(100, Math.round((Number(disturbance.noise_level) || 0) / 8 * 100)), `σ ${formatNumber(disturbance.noise_level, 1)}`],
    ['Target velocity', Math.min(100, Math.round(Math.hypot(Number(target.velocity?.[0]) || 0, Number(target.velocity?.[1]) || 0) / 0.006 * 100)), `${formatNumber(Math.hypot(Number(target.velocity?.[0]) || 0, Number(target.velocity?.[1]) || 0) * 1000, 2)} mrad/s`],
  ];
  $('#profileList').innerHTML = profileRows.map(([label, percent, value]) => `<div class="profile-row"><label>${label}</label><div class="profile-bar"><span style="width:${clamp(percent, 0, 100)}%"></span></div><div class="profile-value">${value}</div></div>`).join('');
}

function renderLatest(snapshot) {
  const sourceLatest = snapshot.latest;
  const preview = !sourceLatest;
  const latest = sourceLatest || buildPreviewTelemetry(snapshot);
  const frame = Number(snapshot.frame_id || 0);
  const total = Number(snapshot.total_frames || 0);
  $('#frameReadout').textContent = preview ? 'NO TELEMETRY / STANDBY' : `FRAME ${String(frame).padStart(3, '0')} / ${String(total).padStart(3, '0')}`;
  $('#simTimeReadout').textContent = preview ? 'STATIC 3D SCENE' : formatTime(latest.sim_time);
  const mode = latest.tracker_mode || 'KF';
  const supervisor = preview ? 'STANDBY' : (latest.supervisor_state || (frame ? 'TRACKING' : 'STANDBY'));
  const confidence = Number(latest.confidence || 0);
  const severity = Number(latest.severity || 0);
  const inFov = latest.target_in_fov;
  const valid = latest.is_valid_det;

  $('#trackerMode').textContent = mode.replace('_', ' ');
  $('#trackerSub').textContent = preview ? 'Awaiting telemetry stream' : mode === 'PF' ? 'Particle filter / adaptive recovery' : mode.includes('COAST') ? 'Predictive coast / awaiting signal' : mode === 'KF' ? 'Kalman filter / active lock' : 'Acquisition path / searching';
  $('#supervisorState').textContent = supervisor;
  $('#supervisorState').className = preview ? 'preview-state' : supervisor === 'TRACKING' || supervisor === 'REACQUIRED' ? 'ok' : supervisor === 'SEARCHING' ? 'warn' : '';
  $('#confidenceValue').textContent = latest ? formatNumber(confidence, 3) : '—';
  $('#severityValue').textContent = latest ? formatNumber(severity, 3) : '—';
  $('#confidenceMeterLabel').textContent = `${Math.round(confidence * 100)}%`;
  $('#severityMeterLabel').textContent = `${Math.round(severity * 100)}%`;
  $('#confidenceMeter').style.width = `${clamp(confidence) * 100}%`;
  $('#severityMeter').style.width = `${clamp(severity) * 100}%`;
  $('#fovState').textContent = latest ? (inFov ? 'YES' : 'NO') : '—';
  $('#fovState').className = latest ? (inFov ? 'ok' : 'warn') : '';
  $('#fovReadout').textContent = latest?.fov_bounds ? `FOV ${(Math.abs(latest.fov_bounds[1] - latest.fov_bounds[0]) * 1000).toFixed(0)} × ${(Math.abs(latest.fov_bounds[3] - latest.fov_bounds[2]) * 1000).toFixed(0)} mrad` : 'FOV 40 × 30 mrad';
  const rangeKm = Number(appState.snapshot?.scenario?.target?.distance_km || 5);
  $('#rangeReadout').textContent = `${rangeKm.toFixed(1).padStart(4, '0')} KM`;
  $('#stageMode').textContent = preview ? 'STANDBY / NO TELEMETRY' : `${mode} / ${severity > .65 ? 'HIGH SEVERITY' : severity > .35 ? 'ADAPTIVE' : 'NOMINAL'}`;
  $('#stageMode').style.color = preview ? COLORS.muted : severity > .65 ? COLORS.red : severity > .35 ? COLORS.amber : COLORS.green;
  $('#linkState').textContent = preview ? 'STANDBY' : valid && inFov ? 'LOCKED' : supervisor === 'SEARCHING' ? 'SEARCHING' : 'COAST';
  $('#linkState').style.color = preview ? COLORS.muted : valid && inFov ? COLORS.green : supervisor === 'SEARCHING' ? COLORS.amber : COLORS.red;
  $('#processingReadout').textContent = preview ? '— ms' : latest ? `${formatNumber(latest.wall_processing_ms, 2)} ms` : '— ms';
  $('#eventBanner').textContent = preview ? 'Awaiting backend telemetry. No live frame received.' : latest?.event_message || (frame ? `${supervisor} · ${mode} estimator active · target ${inFov ? 'inside' : 'outside'} camera FOV` : 'System ready. Start a run to stream live telemetry.');
  $('#stageCoordinates').textContent = latest?.camera_pan_tilt ? `PAN ${formatNumber(latest.camera_pan_tilt[0] * 1000, 2)} / TILT ${formatNumber(latest.camera_pan_tilt[1] * 1000, 2)} mrad` : 'PAN +0.00 / TILT +0.00 mrad';
  const badge = $('#supervisorBadge');
  badge.textContent = supervisor;
  badge.className = `status-badge ${preview ? 'standby' : supervisor === 'TRACKING' || supervisor === 'REACQUIRED' ? 'nominal' : supervisor === 'SEARCHING' ? 'searching' : 'lost'}`;
}

function renderMetrics(metrics) {
  $('#acquisitionMetric').textContent = metrics.acquisition_time_ms === null || metrics.acquisition_time_ms === undefined ? '—' : `${formatNumber(metrics.acquisition_time_ms, 0)} ms`;
  $('#retentionMetric').textContent = `${formatNumber((Number(metrics.lock_retention_rate) || 0) * 100, 1)}%`;
  $('#retentionFoot').textContent = `${metrics.simulated_frames || 0} streamed frames`;
  $('#avgErrorMetric').textContent = `${formatNumber(metrics.avg_tracking_error_mrad, 2)} mrad`;
  $('#maxErrorMetric').textContent = `${formatNumber(metrics.max_tracking_error_mrad, 2)} mrad`;
  $('#fpsMetric').textContent = `${formatNumber(metrics.fps, 1)} fps`;
}

function renderEvents(events) {
  const list = $('#eventList');
  $('#eventCount').textContent = `${events.length} event${events.length === 1 ? '' : 's'}`;
  if (!events.length) {
    list.innerHTML = '<div class="empty-state">No events yet. Telemetry events will appear here.</div>';
    return;
  }
  list.innerHTML = events.slice().reverse().map((event) => {
    const message = String(event.message || 'Telemetry update');
    const isWarn = /LOST|SEARCH|severity/i.test(message);
    const isOk = /ACQUIRED|complete|Loaded|reset/i.test(message);
    const kind = isWarn ? 'WARN' : isOk ? 'OK' : 'INFO';
    const kindClass = isWarn ? 'warn' : isOk ? 'ok' : '';
    return `<div class="event-entry"><span class="event-time">${formatTime(event.sim_time)}</span><span class="event-message" title="${message.replaceAll('"', '&quot;')}">${message}</span><span class="event-kind ${kindClass}">${kind}</span></div>`;
  }).join('');
}

function buildPreviewTelemetry(snapshot) {
  const distance = Number(snapshot.scenario?.target?.distance_km || 5);
  return {
    frame_id: 0,
    sim_time: 0,
    target_pos: [0.012, 0.006],
    camera_pan_tilt: [0.0, 0.0],
    fov_bounds: [-0.02, 0.02, -0.015, 0.015],
    is_valid_det: false,
    target_in_fov: false,
    confidence: 0,
    severity: 0,
    tracker_mode: '—',
    supervisor_state: 'STANDBY',
    active_occluder: null,
    reacq_zone: null,
    preview_only: true,
    range_km: distance,
  };
}

function renderTrackingSvg(snapshot) {
  const sourceLatest = snapshot.latest;
  const latest = sourceLatest || buildPreviewTelemetry(snapshot);
  const isPreview = !sourceLatest;
  const history = snapshot.history || [];
  const layers = {
    stars: $('#starLayer'),
    depthGrid: $('#depthGridLayer'),
    trajectory: $('#trajectoryLayer'),
    frustum: $('#frustumLayer'),
    reacq: $('#reacqLayer'),
    fov: $('#fovLayer'),
    cameraBody: $('#cameraBodyLayer'),
    targetBody: $('#targetBodyLayer'),
    target: $('#targetLayer'),
    boresight: $('#boresightLayer'),
    annotation: $('#annotationLayer'),
  };
  if (!appState._stars) {
    const random = mulberry32(26169);
    appState._stars = Array.from({ length: 72 }, () => ({ x: random() * 900, y: random() * 520, r: .4 + random() * 1.2, o: .25 + random() * .55 }));
    layers.stars.innerHTML = appState._stars.map((star) => `<circle cx="${star.x.toFixed(1)}" cy="${star.y.toFixed(1)}" r="${star.r.toFixed(1)}" fill="#b7d9ec" opacity="${star.o.toFixed(2)}" />`).join('');
  }
  Object.values(layers).forEach((layer) => { if (layer !== layers.stars) layer.innerHTML = ''; });
  if (!latest) return;

  const points = history.flatMap((item) => [item.target_pos, item.camera_pan_tilt]).filter(Boolean);
  const bounds = computeBounds(points, latest.fov_bounds, latest.reacq_zone);
  const xy = ([pan, tilt]) => ({ x: ((pan - bounds.minX) / (bounds.maxX - bounds.minX)) * 900, y: 520 - ((tilt - bounds.minY) / (bounds.maxY - bounds.minY)) * 520 });
  const target = xy(latest.target_pos);
  const camera = xy(latest.camera_pan_tilt);
  const fovA = xy([latest.fov_bounds[0], latest.fov_bounds[3]]);
  const fovB = xy([latest.fov_bounds[1], latest.fov_bounds[2]]);
  const fovW = Math.max(25, fovB.x - fovA.x);
  const fovH = Math.max(20, fovB.y - fovA.y);
  const mode = latest.tracker_mode || 'KF';
  const locked = latest.is_valid_det && latest.target_in_fov;
  const targetColor = locked ? COLORS.green : latest.supervisor_state === 'SEARCHING' ? COLORS.amber : COLORS.muted;

  drawDepthGrid(layers.depthGrid, bounds, camera, target);
  drawOpticalFrustum(layers.frustum, camera, target, fovA, fovB, locked);
  drawCameraBody(layers.cameraBody, camera, mode);
  drawTargetBody(layers.targetBody, target, targetColor, locked);

  const trail = history.slice(-70).map((item) => xy(item.target_pos));
  if (trail.length > 1) {
    const path = trail.map((point, index) => `${index ? 'L' : 'M'} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(' ');
    layers.trajectory.innerHTML = `<path d="${path}" fill="none" stroke="${COLORS.green}" stroke-opacity=".34" stroke-width="1.5" stroke-dasharray="3 6" />`;
  }

  layers.fov.innerHTML = `<rect x="${fovA.x.toFixed(1)}" y="${fovA.y.toFixed(1)}" width="${fovW.toFixed(1)}" height="${fovH.toFixed(1)}" rx="3" fill="rgba(73,214,255,.025)" stroke="${mode.includes('COAST') ? COLORS.muted : COLORS.cyan}" stroke-width="1.5" stroke-dasharray="7 5" /><text x="${(fovA.x + 8).toFixed(1)}" y="${(fovA.y + 17).toFixed(1)}" fill="${COLORS.cyan}" font-family="monospace" font-size="9" opacity=".85">CAMERA FOV</text>`;
  layers.boresight.innerHTML = `<circle cx="${camera.x.toFixed(1)}" cy="${camera.y.toFixed(1)}" r="20" fill="none" stroke="${COLORS.cyan}" stroke-opacity=".16" /><circle cx="${camera.x.toFixed(1)}" cy="${camera.y.toFixed(1)}" r="7" fill="none" stroke="${COLORS.cyan}" stroke-width="1.5" /><path d="M ${(camera.x - 28).toFixed(1)} ${camera.y.toFixed(1)} H ${(camera.x - 9).toFixed(1)} M ${(camera.x + 9).toFixed(1)} ${camera.y.toFixed(1)} H ${(camera.x + 28).toFixed(1)} M ${camera.x.toFixed(1)} ${(camera.y - 28).toFixed(1)} V ${(camera.y - 9).toFixed(1)} M ${camera.x.toFixed(1)} ${(camera.y + 9).toFixed(1)} V ${(camera.y + 28).toFixed(1)}" stroke="${mode.includes('COAST') ? COLORS.muted : COLORS.cyan}" stroke-width="1.5" /><text x="${(camera.x + 31).toFixed(1)}" y="${(camera.y - 15).toFixed(1)}" fill="${mode.includes('COAST') ? COLORS.muted : COLORS.cyan}" font-family="monospace" font-size="9">BORESIGHT / ${mode}</text>`;
  layers.target.innerHTML = `<circle cx="${target.x.toFixed(1)}" cy="${target.y.toFixed(1)}" r="${locked ? 34 : 24}" fill="url(#targetGlow)" opacity="${locked ? .7 : .35}" /><circle cx="${target.x.toFixed(1)}" cy="${target.y.toFixed(1)}" r="${locked ? 7 : 5}" fill="${targetColor}" stroke="#f0f4d8" stroke-width="1.2" /><circle cx="${target.x.toFixed(1)}" cy="${target.y.toFixed(1)}" r="${locked ? 13 : 10}" fill="none" stroke="${targetColor}" stroke-opacity=".7" stroke-width="1" stroke-dasharray="2 3" /><text x="${(target.x + 17).toFixed(1)}" y="${(target.y - 14).toFixed(1)}" fill="${targetColor}" font-family="monospace" font-size="9">${locked ? 'BEACON / LOCKED' : latest.supervisor_state === 'SEARCHING' ? 'BEACON / SEARCH' : 'BEACON / LOW CONF'}</text>`;

  if (latest.reacq_zone) {
    const zone = latest.reacq_zone;
    const center = xy([zone.center_pan, zone.center_tilt]);
    const rX = Math.max(20, (zone.search_radius / (bounds.maxX - bounds.minX)) * 900);
    const rY = Math.max(18, (zone.search_radius / (bounds.maxY - bounds.minY)) * 520);
    layers.reacq.innerHTML = `<ellipse cx="${center.x.toFixed(1)}" cy="${center.y.toFixed(1)}" rx="${rX.toFixed(1)}" ry="${rY.toFixed(1)}" fill="rgba(245,189,98,.05)" stroke="${COLORS.amber}" stroke-width="1.5" stroke-dasharray="4 4" /><text x="${(center.x - rX).toFixed(1)}" y="${(center.y - rY - 8).toFixed(1)}" fill="${COLORS.amber}" font-family="monospace" font-size="9">PREDICTIVE SEARCH ZONE</text>`;
  }
  if (latest.active_occluder) {
    const occluder = xy([latest.active_occluder[0], latest.active_occluder[1]]);
    const radius = Math.max(12, (latest.active_occluder[2] / (bounds.maxX - bounds.minX)) * 900);
    layers.annotation.innerHTML += `<circle cx="${occluder.x.toFixed(1)}" cy="${occluder.y.toFixed(1)}" r="${radius.toFixed(1)}" fill="rgba(104,127,151,.22)" stroke="#7e8e82" stroke-width="1" stroke-dasharray="3 3" /><text x="${(occluder.x - 26).toFixed(1)}" y="${(occluder.y + 3).toFixed(1)}" fill="#9aa79b" font-family="monospace" font-size="8">OCCLUDER</text>`;
  }
  layers.annotation.innerHTML += `<line x1="450" y1="0" x2="450" y2="520" stroke="#4f7698" stroke-opacity=".08" /><line x1="0" y1="260" x2="900" y2="260" stroke="#4f7698" stroke-opacity=".08" />`;
}

function drawDepthGrid(layer, bounds, camera, target) {
  if (!layer) return;
  const horizon = 245;
  const floor = 520;
  const lines = [];
  for (let index = 0; index <= 10; index += 1) {
    const x = -40 + index * 94;
    lines.push(`<path d="M 450 ${horizon} L ${x.toFixed(1)} ${floor}" stroke="#3789a8" stroke-opacity="${index % 2 ? '.18' : '.28'}" stroke-width="1" />`);
  }
  for (let index = 0; index < 8; index += 1) {
    const progress = index / 8;
    const y = horizon + Math.pow(progress, 1.65) * (floor - horizon);
    const inset = progress * 46;
    lines.push(`<path d="M ${inset.toFixed(1)} ${y.toFixed(1)} Q 450 ${(y - progress * 7).toFixed(1)} ${(900 - inset).toFixed(1)} ${y.toFixed(1)}" fill="none" stroke="#3789a8" stroke-opacity="${.13 + progress * .12}" stroke-width="1" />`);
  }
  const axisX = 450;
  const targetY = Math.max(115, Math.min(360, target.y));
  const cameraY = Math.max(330, Math.min(438, camera.y + 48));
  lines.push(`<path d="M ${axisX} ${horizon} L ${axisX} ${floor}" stroke="#b7c77e" stroke-opacity=".28" stroke-dasharray="2 6" />`);
  lines.push(`<path d="M 34 ${targetY.toFixed(1)} L 866 ${targetY.toFixed(1)}" stroke="#79b9aa" stroke-opacity=".08" stroke-dasharray="2 8" />`);
  lines.push(`<circle cx="${camera.x.toFixed(1)}" cy="${cameraY.toFixed(1)}" r="3" fill="#79b9aa" opacity=".8" />`);
  lines.push(`<circle cx="${target.x.toFixed(1)}" cy="${targetY.toFixed(1)}" r="2" fill="#b7c77e" opacity=".9" />`);
  layer.innerHTML = lines.join('');
}

function drawOpticalFrustum(layer, camera, target, fovA, fovB, locked) {
  if (!layer) return;
  const originX = camera.x;
  const originY = Math.max(320, Math.min(430, camera.y + 38));
  const farY = Math.max(100, target.y - 32);
  const spread = Math.max(70, Math.min(220, Math.abs(target.x - camera.x) * .72 + 58));
  const farTop = farY - 58;
  const farBottom = farY + 46;
  const tone = locked ? '#79b9aa' : '#d39a55';
  layer.innerHTML = `
    <polygon points="${originX.toFixed(1)},${(originY - 15).toFixed(1)} ${(target.x + spread).toFixed(1)},${farTop.toFixed(1)} ${(target.x + spread * .82).toFixed(1)},${farBottom.toFixed(1)} ${(originX).toFixed(1)},${(originY + 17).toFixed(1)} ${(target.x - spread * .82).toFixed(1)},${farBottom.toFixed(1)} ${(target.x - spread).toFixed(1)},${farTop.toFixed(1)}" fill="url(#frustumGradient)" stroke="${tone}" stroke-opacity=".33" stroke-width="1.2" stroke-dasharray="6 5" />
    <ellipse cx="${target.x.toFixed(1)}" cy="${((farTop + farBottom) / 2).toFixed(1)}" rx="${spread.toFixed(1)}" ry="${((farBottom - farTop) / 2).toFixed(1)}" fill="none" stroke="${tone}" stroke-opacity=".32" stroke-width="1" stroke-dasharray="3 6" />
    <path d="M ${originX.toFixed(1)} ${originY.toFixed(1)} L ${target.x.toFixed(1)} ${target.y.toFixed(1)}" stroke="${tone}" stroke-opacity=".55" stroke-width="1.2" stroke-dasharray="2 5" />
    <path d="M ${(originX - 12).toFixed(1)} ${(originY - 12).toFixed(1)} L ${(target.x - spread).toFixed(1)} ${farTop.toFixed(1)} M ${(originX + 12).toFixed(1)} ${(originY - 12).toFixed(1)} L ${(target.x + spread).toFixed(1)} ${farTop.toFixed(1)}" stroke="${tone}" stroke-opacity=".62" stroke-width="1.3" />
    <text x="${(originX + 18).toFixed(1)}" y="${(originY + 2).toFixed(1)}" fill="${tone}" fill-opacity=".72" font-family="monospace" font-size="8">OPTICAL PATH / ${locked ? 'TRACK' : 'SEARCH'}</text>`;
}

function drawCameraBody(layer, camera, mode) {
  if (!layer) return;
  const x = camera.x;
  const y = Math.max(310, Math.min(408, camera.y + 38));
  const blind = mode.includes('COAST') || mode === 'LOST';
  const outline = blind ? '#7d8a82' : '#79b9aa';
  layer.innerHTML = `
    <ellipse cx="${x.toFixed(1)}" cy="${(y + 48).toFixed(1)}" rx="68" ry="16" fill="#050e19" stroke="#2c6481" stroke-opacity=".78" />
    <ellipse cx="${x.toFixed(1)}" cy="${(y + 31).toFixed(1)}" rx="55" ry="25" fill="none" stroke="${outline}" stroke-opacity=".24" stroke-width="1" />
    <path d="M ${(x - 55).toFixed(1)} ${(y + 31).toFixed(1)} A 55 24 0 0 0 ${(x + 55).toFixed(1)} ${(y + 31).toFixed(1)}" fill="none" stroke="${outline}" stroke-opacity=".6" stroke-width="1.2" stroke-dasharray="7 5" />
    <path d="M ${(x - 43).toFixed(1)} ${(y + 28).toFixed(1)} L ${(x - 27).toFixed(1)} ${(y - 2).toFixed(1)} L ${(x + 27).toFixed(1)} ${(y - 2).toFixed(1)} L ${(x + 43).toFixed(1)} ${(y + 28).toFixed(1)} Z" fill="url(#gimbalMetal)" stroke="${outline}" stroke-opacity=".7" stroke-width="1.3" />
    <path d="M ${(x - 27).toFixed(1)} ${(y - 2).toFixed(1)} L ${(x - 27).toFixed(1)} ${(y + 28).toFixed(1)} L ${(x + 43).toFixed(1)} ${(y + 28).toFixed(1)} L ${(x + 27).toFixed(1)} ${(y - 2).toFixed(1)}" fill="url(#terminalSide)" opacity=".92" />
    <path d="M ${(x - 29).toFixed(1)} ${(y + 5).toFixed(1)} H ${(x + 29).toFixed(1)}" stroke="#d4dfbf" stroke-opacity=".25" />
    <ellipse cx="${x.toFixed(1)}" cy="${(y - 5).toFixed(1)}" rx="31" ry="17" fill="#0b2e4b" stroke="${outline}" stroke-opacity=".86" stroke-width="1.5" />
    <ellipse cx="${x.toFixed(1)}" cy="${(y - 7).toFixed(1)}" rx="22" ry="14" fill="url(#lensGlow)" opacity="${blind ? '.25' : '.9'}" />
    <ellipse cx="${x.toFixed(1)}" cy="${(y - 7).toFixed(1)}" rx="9" ry="6" fill="#c8fbff" opacity="${blind ? '.3' : '.96'}" />
    <path d="M ${(x - 50).toFixed(1)} ${(y + 41).toFixed(1)} H ${(x + 50).toFixed(1)}" stroke="#b2d9e6" stroke-opacity=".3" stroke-width="1" />
    <text x="${(x + 58).toFixed(1)}" y="${(y + 39).toFixed(1)}" fill="${outline}" fill-opacity=".82" font-family="monospace" font-size="8">VIRTUAL GIMBAL</text>`;
}

function drawTargetBody(layer, target, color, locked) {
  if (!layer) return;
  const x = target.x;
  const y = target.y;
  const size = locked ? 18 : 15;
  layer.innerHTML = `
    <ellipse cx="${x.toFixed(1)}" cy="${(y + size * 1.2).toFixed(1)}" rx="${(size * 1.7).toFixed(1)}" ry="${(size * .48).toFixed(1)}" fill="#071822" stroke="${color}" stroke-opacity=".42" />
    <path d="M ${x.toFixed(1)} ${(y - size).toFixed(1)} L ${(x + size).toFixed(1)} ${(y - size * .15).toFixed(1)} L ${x.toFixed(1)} ${(y + size * .65).toFixed(1)} L ${(x - size).toFixed(1)} ${(y - size * .15).toFixed(1)} Z" fill="url(#terminalFace)" stroke="${color}" stroke-width="1.2" />
    <path d="M ${(x + size).toFixed(1)} ${(y - size * .15).toFixed(1)} L ${(x + size).toFixed(1)} ${(y + size * .65).toFixed(1)} L ${x.toFixed(1)} ${(y + size * 1.12).toFixed(1)} L ${x.toFixed(1)} ${(y + size * .65).toFixed(1)} Z" fill="url(#terminalSide)" opacity=".92" />
    <path d="M ${(x - size).toFixed(1)} ${(y - size * .15).toFixed(1)} L ${x.toFixed(1)} ${(y + size * .65).toFixed(1)} L ${x.toFixed(1)} ${(y + size * 1.12).toFixed(1)} L ${(x - size).toFixed(1)} ${(y + size * .28).toFixed(1)} Z" fill="#115567" opacity=".86" />
    <circle cx="${x.toFixed(1)}" cy="${(y - size * .16).toFixed(1)}" r="${(size * .24).toFixed(1)}" fill="#f0f4d8" opacity=".9" />`;
}

function computeBounds(points, fovBounds, reacqZone) {
  const valuesX = points.map((point) => point[0]);
  const valuesY = points.map((point) => point[1]);
  if (fovBounds) { valuesX.push(fovBounds[0], fovBounds[1]); valuesY.push(fovBounds[2], fovBounds[3]); }
  if (reacqZone) { valuesX.push(reacqZone.center_pan - reacqZone.search_radius, reacqZone.center_pan + reacqZone.search_radius); valuesY.push(reacqZone.center_tilt - reacqZone.search_radius, reacqZone.center_tilt + reacqZone.search_radius); }
  const minX = Math.min(...valuesX, -.03) - .004;
  const maxX = Math.max(...valuesX, .03) + .004;
  const minY = Math.min(...valuesY, -.025) - .004;
  const maxY = Math.max(...valuesY, .025) + .004;
  const minSpan = .04;
  return { minX, maxX: Math.max(maxX, minX + minSpan), minY, maxY: Math.max(maxY, minY + minSpan) };
}

function renderCharts(history) {
  const errorEmpty = $('#errorEmpty');
  const qualityEmpty = $('#qualityEmpty');
  errorEmpty.classList.toggle('hidden', history.length > 0);
  qualityEmpty.classList.toggle('hidden', history.length > 0);
  drawLineChart($('#errorChart'), history.map((item) => Number(item.tracking_error_mrad) || 0), { color: COLORS.cyan, threshold: 2, maxFloor: 4 });
  drawMultiChart($('#qualityChart'), [
    { values: history.map((item) => Number(item.confidence) || 0), color: COLORS.green },
    { values: history.map((item) => Number(item.severity) || 0), color: COLORS.amber },
  ], { min: 0, max: 1 });
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(20, Math.floor(rect.width));
  const height = Math.max(20, Math.floor(rect.height));
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext('2d');
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

function drawGrid(context, width, height, rows = 4, cols = 8) {
  context.clearRect(0, 0, width, height);
  context.strokeStyle = COLORS.grid;
  context.lineWidth = 1;
  context.setLineDash([2, 5]);
  for (let i = 1; i < rows; i += 1) { const y = (height * i) / rows; context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  for (let i = 1; i < cols; i += 1) { const x = (width * i) / cols; context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
  context.setLineDash([]);
}

function drawLineChart(canvas, values, options) {
  const { context, width, height } = prepareCanvas(canvas);
  drawGrid(context, width, height);
  if (!values.length) return;
  const maxValue = Math.max(options.maxFloor || 1, ...values) * 1.12;
  const left = 3, top = 5, right = width - 3, bottom = height - 5;
  const px = (index) => left + (index / Math.max(1, values.length - 1)) * (right - left);
  const py = (value) => bottom - (value / maxValue) * (bottom - top);
  if (options.threshold !== undefined) {
    const y = py(options.threshold);
    context.strokeStyle = '#7890a8'; context.lineWidth = 1; context.setLineDash([4, 5]); context.beginPath(); context.moveTo(left, y); context.lineTo(right, y); context.stroke(); context.setLineDash([]);
  }
  const gradient = context.createLinearGradient(0, top, 0, bottom); gradient.addColorStop(0, 'rgba(73,214,255,.22)'); gradient.addColorStop(1, 'rgba(73,214,255,0)');
  context.beginPath(); values.forEach((value, index) => { const x = px(index), y = py(value); index ? context.lineTo(x, y) : context.moveTo(x, y); }); context.lineTo(right, bottom); context.lineTo(left, bottom); context.closePath(); context.fillStyle = gradient; context.fill();
  context.beginPath(); values.forEach((value, index) => { const x = px(index), y = py(value); index ? context.lineTo(x, y) : context.moveTo(x, y); }); context.strokeStyle = options.color; context.lineWidth = 2; context.lineJoin = 'round'; context.stroke();
  const lastX = px(values.length - 1), lastY = py(values[values.length - 1]); context.fillStyle = options.color; context.shadowColor = options.color; context.shadowBlur = 8; context.beginPath(); context.arc(lastX, lastY, 3, 0, Math.PI * 2); context.fill(); context.shadowBlur = 0;
}

function drawMultiChart(canvas, series, options) {
  const { context, width, height } = prepareCanvas(canvas);
  drawGrid(context, width, height);
  const length = Math.max(0, ...series.map((item) => item.values.length));
  if (!length) return;
  const left = 3, top = 5, right = width - 3, bottom = height - 5;
  const px = (index) => left + (index / Math.max(1, length - 1)) * (right - left);
  const py = (value) => bottom - ((value - options.min) / (options.max - options.min)) * (bottom - top);
  series.forEach((item) => {
    context.beginPath(); item.values.forEach((value, index) => { const x = px(index), y = py(value); index ? context.lineTo(x, y) : context.moveTo(x, y); }); context.strokeStyle = item.color; context.lineWidth = 1.8; context.lineJoin = 'round'; context.stroke();
  });
}

function updatePlayButton() {
  const button = $('#playButton');
  const atEnd = appState.snapshot && Number(appState.snapshot.frame_id) >= Number(appState.snapshot.total_frames);
  if (appState.playing) { button.textContent = 'Ⅱ Pause mission'; button.classList.add('is-playing'); }
  else if (atEnd) { button.textContent = '↻ Replay mission'; button.classList.remove('is-playing'); }
  else { button.textContent = '▶ Start mission'; button.classList.remove('is-playing'); }
}

async function stepLoop(token) {
  if (!appState.playing || token !== appState.loopToken) return;
  try {
    const snapshot = await post('/api/session/action', { action: 'step' });
    applySnapshot(snapshot);
    if (!appState.playing || token !== appState.loopToken || Number(snapshot.frame_id) >= Number(snapshot.total_frames)) {
      appState.playing = false;
      updatePlayButton();
      return;
    }
    const dt = Number(snapshot.scenario?.dt || 0.033);
    window.setTimeout(() => stepLoop(token), Math.max(16, (dt * 1000) / appState.speed));
  } catch (error) {
    appState.playing = false; updatePlayButton(); setConnection(false, 'OFFLINE'); showToast(error.message); }
}

async function toggleMission() {
  if (!appState.snapshot?.backend_ready) { showToast('Backend is not ready. Check the Python environment.'); return; }
  if (appState.playing) {
    appState.playing = false; appState.loopToken += 1; await post('/api/session/action', { action: 'pause' }).catch(() => {}); updatePlayButton(); return;
  }
  if (Number(appState.snapshot.frame_id) >= Number(appState.snapshot.total_frames)) {
    await resetMission();
  }
  appState.playing = true; appState.loopToken += 1; const token = appState.loopToken; updatePlayButton(); await post('/api/session/action', { action: 'start' }).then(applySnapshot); stepLoop(token);
}

async function resetMission() {
  appState.playing = false; appState.loopToken += 1;
  try { const snapshot = await post('/api/session/action', { action: 'reset' }); applySnapshot(snapshot); showToast('Simulation reset to frame 0'); } catch (error) { showToast(error.message); }
}

async function selectScenario(event) {
  appState.playing = false; appState.loopToken += 1;
  try { const snapshot = await post('/api/session/select', { scenario_name: event.target.value }); applySnapshot(snapshot); showToast(`Loaded ${prettyScenarioName(event.target.value)}`); } catch (error) { showToast(error.message); }
}

async function runBenchmark() {
  if (!appState.snapshot?.backend_ready) { showToast('Connect the backend before running a benchmark.'); return; }
  const button = $('#benchmarkButton'); button.disabled = true; button.innerHTML = '<span>◌</span> Running…';
  try {
    const result = await post('/api/benchmark', {});
    const record = result.record || {};
    showToast(`Benchmark complete · ${formatNumber(record.fps, 1)} fps · ${formatNumber(Number(record.lock_retention_rate) * 100, 1)}% retention`);
  } catch (error) { showToast(`Benchmark failed · ${error.message}`); }
  finally { button.disabled = false; button.innerHTML = '<span>◈</span> Run benchmark'; }
}

function downloadExport(format) { window.open(`/api/export?format=${format}`, '_blank'); showToast(`Preparing ${format.toUpperCase()} telemetry export`); }

function mulberry32(seed) { return function random() { let t = seed += 0x6D2B79F5; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

async function bootstrap() {
  $('#scenarioSelect').addEventListener('change', selectScenario);
  $('#playButton').addEventListener('click', toggleMission);
  $('#resetButton').addEventListener('click', resetMission);
  $('#benchmarkButton').addEventListener('click', runBenchmark);
  $('#exportJsonButton').addEventListener('click', () => downloadExport('json'));
  $('#exportCsvButton').addEventListener('click', () => downloadExport('csv'));
  $('#themeButton').addEventListener('click', () => { document.body.classList.toggle('light-mode'); localStorage.setItem('fsoc-theme', document.body.classList.contains('light-mode') ? 'light' : 'dark'); });
  if (localStorage.getItem('fsoc-theme') === 'light') document.body.classList.add('light-mode');
  $$('.rail-item').forEach((item) => item.addEventListener('click', () => { $$('.rail-item').forEach((button) => button.classList.remove('active')); item.classList.add('active'); if (item.dataset.view !== 'live') showToast(`${item.textContent.trim()} view is available from the live console after a run.`); }));
  window.addEventListener('resize', () => { if (appState.snapshot) renderCharts(appState.snapshot.history || []); });
  const theaterStage = $('#theaterStage');
  theaterStage.addEventListener('pointermove', (event) => {
    if (event.pointerType === 'touch') return;
    const rect = theaterStage.getBoundingClientRect();
    const nx = (event.clientX - rect.left) / rect.width - 0.5;
    const ny = (event.clientY - rect.top) / rect.height - 0.5;
    theaterStage.style.setProperty('--scene-tilt-x', `${(-nx * 3.2).toFixed(2)}deg`);
    theaterStage.style.setProperty('--scene-tilt-y', `${(ny * 2.4).toFixed(2)}deg`);
  });
  theaterStage.addEventListener('pointerleave', () => {
    theaterStage.style.setProperty('--scene-tilt-x', '0deg');
    theaterStage.style.setProperty('--scene-tilt-y', '0deg');
  });

  try {
    const scenarios = await api('/api/scenarios');
    appState.scenarios = scenarios.scenarios || [];
    $('#scenarioSelect').innerHTML = appState.scenarios.map((scenario) => `<option value="${scenario.scenario_name}">${prettyScenarioName(scenario.scenario_name)}</option>`).join('');
    const snapshot = await api('/api/session');
    applySnapshot(snapshot);
  } catch (error) {
    setConnection(false, 'OFFLINE'); showToast(`Unable to connect · ${error.message}`);
  }
}

document.addEventListener('DOMContentLoaded', bootstrap);
