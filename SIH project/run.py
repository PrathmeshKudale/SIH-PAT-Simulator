#!/usr/bin/env python3
"""
FSOC Coarse PAT Simulator — Unified Mission Control Launcher
Smart India Hackathon | Problem Statement 26169 (ISRO / DOS)

Single-command entrypoint to launch the Python simulation engine,
HTTP/API adapter, and browser-based flight cockpit.
"""
import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

# Resolve project directories
ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "SIH" if (ROOT / "SIH").exists() else (ROOT / "SIH-backend" / "SIH-backend")
FRONTEND_DIR = ROOT / "frontend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure frontend dir environment variable points to root frontend
os.environ["FSOC_FRONTEND_DIR"] = str(FRONTEND_DIR)

BANNER = r"""
================================================================================
       ISRO / DOS -- SMART INDIA HACKATHON | PROBLEM STATEMENT 26169
      FSOC COARSE POINTING, ACQUISITION & TRACKING (PAT) MISSION CONTROL
================================================================================
 [OK] Backend Engine       : Online (NumPy / SciPy / OpenCV Verified)
 [OK] Closed-Loop Modules  : Optics / Kalman-Particle Hybrid / PID / Slew / Reacq
 [OK] Mission Cockpit HUD  : Serving at http://127.0.0.1:{port}
================================================================================
"""


def open_browser(port: int) -> None:
    time.sleep(0.8)
    url = f"http://127.0.0.1:{port}"
    print(f"[*] Opening mission cockpit in default browser: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as exc:
        print(f"[!] Could not auto-open browser ({exc}). Please visit {url} manually.")


def main() -> None:
    port = int(os.environ.get("FSOC_PORT", "8787"))
    print(BANNER.format(port=port))
    
    # Launch browser opener daemon thread
    opener_thread = threading.Thread(target=open_browser, args=(port,), daemon=True)
    opener_thread.start()

    try:
        import web_server
        web_server.main()
    except KeyboardInterrupt:
        print("\n[*] Mission Control shutdown gracefully.")
    except Exception as exc:
        print(f"\n[!] Failed to launch server: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
