"""
Browser mission-control frontend server for the FSOC Coarse PAT Simulator.

This adapter intentionally sits beside the existing simulation engine. It does not
replace the verified backend modules; it exposes their live session telemetry through
small local JSON endpoints and serves the zero-build frontend in ./frontend.
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
SCENARIO_PATH = ROOT / "config" / "demo_scenarios.json"

try:
    from verify_ui import SimulationSession
    BACKEND_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # Keep the visual shell available when optional CV deps are missing.
    SimulationSession = None  # type: ignore[assignment,misc]
    BACKEND_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def load_scenarios() -> List[Dict[str, Any]]:
    with SCENARIO_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_safe(value: Any) -> Any:
    """Convert backend dataclasses, tuples, numpy scalars and arrays to JSON data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


class MissionState:
    """Single-user local demo state. The app is intentionally local and session-scoped."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.scenarios = load_scenarios()
        self.current_config: Dict[str, Any] = self.scenarios[0] if self.scenarios else {}
        self.session: Any = None
        self.running = False
        self.history: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.last_wall_step_ms = 0.0
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._create_session()

    def _create_session(self) -> None:
        self.history = []
        self.events = []
        self.last_wall_step_ms = 0.0
        self.running = False
        if SimulationSession is not None and self.current_config:
            self.session = SimulationSession(self.current_config)
        else:
            self.session = None

    @property
    def scenario_name(self) -> str:
        return str(self.current_config.get("scenario_name", "NO_SCENARIO"))

    def select(self, scenario_name: str) -> None:
        match = next((item for item in self.scenarios if item.get("scenario_name") == scenario_name), None)
        if match is None:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        self.current_config = match
        self._create_session()
        self.events.append({"kind": "scenario", "message": f"Loaded {scenario_name}", "frame_id": 0, "sim_time": 0.0})

    def reset(self) -> None:
        if self.session is not None:
            self.session.reset()
        self.history = []
        self.events = [{"kind": "system", "message": "Simulation reset to frame 0", "frame_id": 0, "sim_time": 0.0}]
        self.last_wall_step_ms = 0.0
        self.running = False

    def step(self) -> Optional[Dict[str, Any]]:
        if self.session is None:
            raise RuntimeError(BACKEND_IMPORT_ERROR or "Backend session is unavailable")
        if self.session.frame_id >= self.session.num_frames:
            self.running = False
            return self.history[-1] if self.history else None

        wall_start = time.perf_counter()
        telemetry = json_safe(self.session.step())
        self.last_wall_step_ms = (time.perf_counter() - wall_start) * 1000.0
        telemetry["wall_processing_ms"] = round(self.last_wall_step_ms, 3)
        self.history.append(telemetry)

        message = telemetry.get("event_message")
        if message:
            self.events.append({
                "kind": "event",
                "message": message,
                "frame_id": telemetry.get("frame_id", 0),
                "sim_time": telemetry.get("sim_time", 0.0),
                "severity": telemetry.get("severity", 0.0),
            })
        if self.session.frame_id >= self.session.num_frames:
            self.running = False
            self.events.append({
                "kind": "system",
                "message": "Scenario complete — telemetry ready for export",
                "frame_id": telemetry.get("frame_id", 0),
                "sim_time": telemetry.get("sim_time", 0.0),
            })
        return telemetry

    def metrics(self) -> Dict[str, Any]:
        frames = self.history
        total = len(frames)
        errors = [float(item.get("tracking_error_mrad", 0.0)) for item in frames]
        confidence = [float(item.get("confidence", 0.0)) for item in frames]
        processing = [float(item.get("wall_processing_ms", 0.0)) for item in frames]
        locked = [
            bool(item.get("is_valid_det")) and item.get("tracker_mode") in ("KF", "PF")
            for item in frames
        ]
        acquisition_index = next((idx for idx, value in enumerate(locked) if value), None)
        dt = float(self.current_config.get("dt", 0.033) or 0.033)
        duration = total * dt
        nominal_fps = 1.0 / dt if dt > 0 else 0.0
        return {
            "total_frames": int(self.session.frame_id if self.session is not None else total),
            "simulated_frames": total,
            "simulation_duration_s": round(duration, 3),
            "fps": round(nominal_fps, 1),
            "acquisition_time_ms": round((acquisition_index or 0) * dt * 1000.0, 1) if acquisition_index is not None else None,
            "avg_tracking_error_mrad": round(sum(errors) / len(errors), 3) if errors else 0.0,
            "max_tracking_error_mrad": round(max(errors), 3) if errors else 0.0,
            "lock_retention_rate": round(sum(locked) / total, 4) if total else 0.0,
            "processing_time_ms": round(sum(processing) / len(processing), 3) if processing else 0.0,
            "peak_confidence": round(max(confidence), 3) if confidence else 0.0,
            "track_loss_count": sum(1 for event in self.events if event.get("kind") == "event" and "TRACK LOST" in event.get("message", "")),
        }

    def snapshot(self) -> Dict[str, Any]:
        latest = self.history[-1] if self.history else None
        return {
            "service": "FSOC PAT Mission Control",
            "backend_ready": self.session is not None,
            "backend_error": BACKEND_IMPORT_ERROR,
            "scenario": json_safe(self.current_config),
            "scenario_name": self.scenario_name,
            "running": self.running,
            "frame_id": int(self.session.frame_id if self.session is not None else 0),
            "total_frames": int(self.session.num_frames if self.session is not None else self.current_config.get("num_frames", 0)),
            "latest": latest,
            "history": self.history[-240:],
            "events": self.events[-60:],
            "metrics": self.metrics(),
            "server_time": datetime.now(timezone.utc).isoformat(),
        }

    def export_json(self) -> bytes:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scenario": json_safe(self.current_config),
            "metrics": self.metrics(),
            "events": self.events,
            "frames": self.history,
        }
        return json.dumps(payload, indent=2).encode("utf-8")

    def export_csv(self) -> bytes:
        output = io.StringIO()
        fields = [
            "frame_id", "sim_time", "tracking_error_mrad", "confidence", "severity",
            "tracker_mode", "supervisor_state", "is_valid_det", "target_in_fov", "wall_processing_ms",
        ]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.history)
        return output.getvalue().encode("utf-8")


STATE = MissionState()


class FrontendHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the terminal useful during a demo; static asset noise is suppressed.
        if self.path.startswith("/api/"):
            super().log_message(format, *args)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(json_safe(payload)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, filename: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"ok": True, "backend_ready": STATE.session is not None, "backend_error": BACKEND_IMPORT_ERROR})
            return
        if parsed.path == "/api/scenarios":
            self._send_json({"scenarios": STATE.scenarios})
            return
        if parsed.path == "/api/session":
            with STATE.lock:
                self._send_json(STATE.snapshot())
            return
        if parsed.path == "/api/export":
            fmt = parse_qs(parsed.query).get("format", ["json"])[0].lower()
            with STATE.lock:
                if fmt == "csv":
                    self._send_bytes(STATE.export_csv(), "text/csv; charset=utf-8", "fsoc-pat-telemetry.csv")
                else:
                    self._send_bytes(STATE.export_json(), "application/json; charset=utf-8", "fsoc-pat-telemetry.json")
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self._send_json({"error": "Not found"}, 404)
            return
        try:
            body = self._read_json()
            with STATE.lock:
                if parsed.path == "/api/session/select":
                    STATE.select(str(body.get("scenario_name", "")))
                elif parsed.path == "/api/session/action":
                    action = str(body.get("action", "")).lower()
                    if action == "step":
                        STATE.step()
                    elif action == "start":
                        STATE.running = True
                    elif action == "pause":
                        STATE.running = False
                    elif action == "reset":
                        STATE.reset()
                    else:
                        raise ValueError(f"Unknown action: {action}")
                elif parsed.path == "/api/benchmark":
                    # Batch benchmarking is opt-in from the frontend; it never runs during playback.
                    from metrics.batch_runner import BatchScenarioRunner
                    runner = BatchScenarioRunner(config_path=str(SCENARIO_PATH), use_hybrid_tracker=False)
                    record = runner.run_scenario(STATE.current_config)
                    self._send_json({"record": record.to_dict()})
                    return
                else:
                    self._send_json({"error": "Not found"}, 404)
                    return
                self._send_json(STATE.snapshot())
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main() -> None:
    if not FRONTEND_DIR.exists():
        raise FileNotFoundError(f"Missing frontend assets at {FRONTEND_DIR}")
    port = int(os.environ.get("FSOC_PORT", "8787"))
    server = ThreadingHTTPServer(("127.0.0.1", port), FrontendHandler)
    print(f"FSOC PAT Mission Control running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
