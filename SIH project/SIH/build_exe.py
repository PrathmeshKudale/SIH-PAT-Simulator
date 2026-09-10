"""
Automated Standalone Application Packaging Script with PyInstaller.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Packages verify_ui.py and the complete PAT simulation backend into a standalone,
portable Windows executable directory (--onedir) including all dependencies
(OpenCV, NumPy, Scipy, Tkinter) and bundled configuration/video assets.
"""

import os
import sys
import shutil
import time
import argparse


def format_size(bytes_val: int) -> str:
    """Format byte count to human-readable string (KB, MB, GB)."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def get_dir_size(dir_path: str) -> int:
    """Calculate total byte size of all files in directory tree."""
    total = 0
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def build(clean: bool = True, onefile: bool = False) -> int:
    """
    Execute PyInstaller build.
    
    Args:
        clean: Whether to wipe previous build and dist directories.
        onefile: If True, build single-file executable; else build onedir bundle.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("=" * 70)
    print("  FSOC PAT SIMULATOR — PYINSTALLER STANDALONE BUILD SCRIPT")
    print("  ISRO / DOS PS 26169 | Entry Point: verify_ui.py")
    print("=" * 70)

    # 1. Clean previous artifacts if requested
    if clean:
        for folder in ["build", "dist"]:
            p = os.path.join(project_root, folder)
            if os.path.exists(p):
                print(f"[*] Cleaning previous '{folder}/' directory...")
                shutil.rmtree(p, ignore_errors=True)

    # 2. Check PyInstaller
    try:
        import PyInstaller
        import PyInstaller.__main__
        print(f"[*] Found PyInstaller version {PyInstaller.__version__}")
    except ImportError:
        print("[ERROR] PyInstaller is not installed in the active Python environment.")
        print("        Install it via: python -m pip install pyinstaller")
        return 1

    t0 = time.perf_counter()

    # 3. Determine build mode
    if onefile:
        print("[*] Building in --onefile mode (single self-extracting executable)...")
        spec_or_args = [
            "verify_ui.py",
            "--onefile",
            "--name=FSOC_PAT_Simulator",
            "--add-data=config;config",
            "--add-data=data/videos;data/videos",
            "--hidden-import=contracts",
            "--hidden-import=resources",
            "--hidden-import=sim",
            "--hidden-import=detect",
            "--hidden-import=track",
            "--hidden-import=control",
            "--hidden-import=disturb",
            "--hidden-import=metrics",
            "--hidden-import=cv2",
            "--hidden-import=numpy",
            "--hidden-import=scipy",
            "--hidden-import=scipy.spatial",
            "--hidden-import=scipy.ndimage",
            "--hidden-import=tkinter",
            "--hidden-import=tkinter.ttk",
            "--console",
            "--noconfirm",
        ]
    else:
        spec_path = os.path.join(project_root, "fsoc_simulator.spec")
        if os.path.exists(spec_path):
            print(f"[*] Building using specification: {spec_path} (--onedir mode)...")
            spec_or_args = [spec_path, "--noconfirm"]
        else:
            print("[*] Building using CLI flags (--onedir mode)...")
            spec_or_args = [
                "verify_ui.py",
                "--onedir",
                "--name=FSOC_PAT_Simulator",
                "--add-data=config;config",
                "--add-data=data/videos;data/videos",
                "--hidden-import=contracts",
                "--hidden-import=resources",
                "--hidden-import=sim",
                "--hidden-import=detect",
                "--hidden-import=track",
                "--hidden-import=control",
                "--hidden-import=disturb",
                "--hidden-import=metrics",
                "--hidden-import=cv2",
                "--hidden-import=numpy",
                "--hidden-import=scipy",
                "--hidden-import=tkinter",
                "--console",
                "--noconfirm",
            ]

    print(f"[*] Running PyInstaller command: python -m PyInstaller {' '.join(spec_or_args)}")
    try:
        PyInstaller.__main__.run(spec_or_args)
    except SystemExit as e:
        if e.code != 0:
            print(f"[ERROR] PyInstaller failed with exit code: {e.code}")
            return e.code
    except Exception as ex:
        print(f"[ERROR] Exception during PyInstaller build: {ex}")
        return 1

    build_time = time.perf_counter() - t0

    # 4. Verify outputs
    dist_dir = os.path.join(project_root, "dist")
    if onefile:
        exe_path = os.path.join(dist_dir, "FSOC_PAT_Simulator.exe")
        target_dir = dist_dir
    else:
        target_dir = os.path.join(dist_dir, "FSOC_PAT_Simulator")
        exe_path = os.path.join(target_dir, "FSOC_PAT_Simulator.exe")

    if not os.path.exists(exe_path):
        print(f"[ERROR] Expected executable not found at: {exe_path}")
        return 1

    exe_size = os.path.getsize(exe_path)
    total_size = get_dir_size(target_dir)

    print("\n" + "=" * 70)
    print("  BUILD SUCCESSFUL!")
    print("=" * 70)
    print(f"  Build Elapsed Time : {build_time:.2f} seconds")
    print(f"  Packaging Mode     : {'--onefile' if onefile else '--onedir (recommended for fast startup)'}")
    print(f"  Executable Path    : {exe_path}")
    print(f"  Executable Binary  : {format_size(exe_size)} ({exe_size:,} bytes)")
    if not onefile:
        print(f"  Bundle Directory   : {target_dir}")
        print(f"  Total Bundle Size  : {format_size(total_size)} ({total_size:,} bytes)")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package FSOC PAT Simulator as standalone executable.")
    parser.add_argument("--no-clean", action="store_true", help="Do not wipe build/ and dist/ prior to building.")
    parser.add_argument("--onefile", action="store_true", help="Package into a single .exe (slower startup vs onedir).")
    args = parser.parse_args()

    exit_code = build(clean=not args.no_clean, onefile=args.onefile)
    sys.exit(exit_code)
