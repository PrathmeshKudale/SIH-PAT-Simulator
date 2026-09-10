"""
Evaluation and Stress-Testing of FSOC PAT Pipeline on Real and Adversarial Videos.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Tests detector generalization, confidence distributions, and tracker stability
on genuine real-world and unfamiliar synthetic footage:
1. Real handheld camera footage of a moving laser spot against a room background (real_laser_pointer.mp4).
2. Real European Southern Observatory (ESO) Adaptive Optics laser guide star footage (real_paranal_ao_laser.mp4).
3. Adversarial astigmatic/elliptical spot with non-uniform ambient illumination gradient and CMOS banding.
4. Stress scenario with static background glints/clutter and a 13-frame total optical occlusion dropout.
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import cv2

# Ensure project root is on sys.path when invoked directly as a script
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from detect.detector import AdaptiveOpticalDetector
from sim.video_source import VideoFrameSource
from metrics.batch_runner import BatchScenarioRunner


def analyze_video_raw_detections(video_path: str, max_frames: int = 60) -> Dict[str, Any]:
    """Inspect raw frame-by-frame candidate extraction and confidence characteristics."""
    source = VideoFrameSource(video_path)
    detector = AdaptiveOpticalDetector(enable_signature_verification=False)

    total_cands_list = []
    top_confs = []
    top_areas = []
    top_peaks = []
    top_positions = []

    for f in range(max_frames):
        frame = source.get_frame(f)
        if frame is None:
            break
        cands = detector.extract_candidates(frame.image)
        total_cands_list.append(len(cands))
        if cands:
            top = cands[0]
            top_confs.append(top.confidence)
            top_areas.append(top.area)
            top_peaks.append(top.peak_intensity)
            top_positions.append((top.x, top.y))
        else:
            top_confs.append(0.0)
            top_areas.append(0)
            top_peaks.append(0.0)
            top_positions.append(None)

    source.close()

    valid_confs = [c for c in top_confs if c > 0]
    valid_positions = [p for p in top_positions if p is not None]

    # Compute centroid jitter between consecutive frames
    jitters = []
    for i in range(len(valid_positions) - 1):
        p1 = valid_positions[i]
        p2 = valid_positions[i + 1]
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        jitters.append(dist)

    return {
        "frames_evaluated": len(total_cands_list),
        "mean_candidates_per_frame": float(np.mean(total_cands_list)) if total_cands_list else 0.0,
        "max_candidates_per_frame": int(np.max(total_cands_list)) if total_cands_list else 0,
        "detection_fraction": float(len(valid_confs) / len(total_cands_list)) if total_cands_list else 0.0,
        "confidence_min": float(np.min(valid_confs)) if valid_confs else 0.0,
        "confidence_max": float(np.max(valid_confs)) if valid_confs else 0.0,
        "confidence_mean": float(np.mean(valid_confs)) if valid_confs else 0.0,
        "confidence_std": float(np.std(valid_confs)) if valid_confs else 0.0,
        "peak_intensity_mean": float(np.mean(top_peaks)) if top_peaks else 0.0,
        "blob_area_mean": float(np.mean([a for a in top_areas if a > 0])) if any(a > 0 for a in top_areas) else 0.0,
        "mean_interframe_displacement_px": float(np.mean(jitters)) if jitters else 0.0,
        "max_interframe_displacement_px": float(np.max(jitters)) if jitters else 0.0,
    }


def run_pipeline_test(video_path: str, scenario_name: str, num_frames: int = 50, use_hybrid: bool = False) -> Dict[str, Any]:
    """Execute the full closed-loop PAT simulator against the video."""
    runner = BatchScenarioRunner(use_hybrid_tracker=use_hybrid)
    cfg = {
        "scenario_name": scenario_name,
        "frame_source": "video_file",
        "video_path": video_path,
        "num_frames": num_frames,
        "control": {
            "kp": 0.35,
            "ki": 0.0,
            "kd": 0.15,
            "k_ff": 1.0,
            "latency_frames": 1,
        },
    }

    rec = runner.run_scenario(cfg)
    logger = runner.last_logger

    confs = [f.get("confidence", 0.0) for f in logger.frame_records]
    detected_frames = sum(1 for f in logger.frame_records if f.get("detected"))
    states = [f.get("state") for f in logger.frame_records]
    modes = [f.get("tracker_mode") for f in logger.frame_records]
    cam_pans = [f.get("cam_pan_mrad", 0.0) for f in logger.frame_records]
    cam_tilts = [f.get("cam_tilt_mrad", 0.0) for f in logger.frame_records]

    state_counts = {s: states.count(s) for s in set(states)}
    mode_counts = {m: modes.count(m) for m in set(modes)}

    return {
        "scenario_name": scenario_name,
        "tracker_type": "Hybrid (KF/PF)" if use_hybrid else "Standalone KF",
        "total_frames": rec.total_frames,
        "pipeline_errors": rec.pipeline_errors,
        "fps": rec.fps,
        "per_frame_processing_time_ms": rec.per_frame_processing_time_ms,
        "lock_retention_rate": rec.lock_retention_rate,
        "active_locked_frames": rec.active_locked_frames,
        "coasting_frames": rec.coasting_frames,
        "track_loss_count": rec.track_loss_count,
        "tracking_error_reported": rec.avg_tracking_error,
        "detected_fraction": detected_frames / len(confs) if confs else 0.0,
        "conf_min": float(np.min(confs)) if confs else 0.0,
        "conf_max": float(np.max(confs)) if confs else 0.0,
        "conf_mean": float(np.mean(confs)) if confs else 0.0,
        "state_breakdown": state_counts,
        "mode_breakdown": mode_counts,
        "camera_pan_span_mrad": float(np.max(cam_pans) - np.min(cam_pans)) if cam_pans else 0.0,
        "camera_tilt_span_mrad": float(np.max(cam_tilts) - np.min(cam_tilts)) if cam_tilts else 0.0,
    }


def main():
    videos = [
        {
            "id": "real_laser_pointer",
            "name": "Real Handheld Laser Pointer (Wall & Room Ambient Clutter)",
            "path": "data/videos/real_laser_pointer.mp4",
            "frames": 50,
        },
        {
            "id": "real_paranal_laser",
            "name": "Real ESO Paranal Adaptive Optics Laser Guide Star",
            "path": "data/videos/real_paranal_ao_laser.mp4",
            "frames": 50,
        },
        {
            "id": "stress_astigmatic",
            "name": "Adversarial Non-Circular Astigmatic Spot + Skyglow Gradient",
            "path": "data/videos/stress_test_astigmatic.mp4",
            "frames": 60,
        },
        {
            "id": "stress_clutter_occlusion",
            "name": "Moving Beacon + Static Background Glints + 13-Frame Dropout",
            "path": "data/videos/stress_test_clutter_occlusion.mp4",
            "frames": 80,
        },
    ]

    print("=" * 88)
    print("  FSOC PAT BENCHMARK-2 VIDEO GENERALIZATION & STRESS EVALUATION")
    print("=" * 88)

    all_raw_stats = {}
    all_pipeline_stats = []

    for v in videos:
        print(f"\nEvaluating: {v['name']}")
        print(f"Path: {v['path']}")
        
        if not os.path.exists(v["path"]):
            print(f"  [ERROR] File does not exist: {v['path']}")
            continue

        raw = analyze_video_raw_detections(v["path"], max_frames=v["frames"])
        all_raw_stats[v["id"]] = raw
        print(f"  Raw Detections: {raw['detection_fraction']*100:.1f}% valid frames")
        print(f"  Mean candidate blobs/frame: {raw['mean_candidates_per_frame']:.1f} (max: {raw['max_candidates_per_frame']})")
        print(f"  Confidence: [{raw['confidence_min']:.3f}, {raw['confidence_max']:.3f}], mean: {raw['confidence_mean']:.3f} +/- {raw['confidence_std']:.3f}")
        print(f"  Blob peak: {raw['peak_intensity_mean']:.1f}/255, area: {raw['blob_area_mean']:.1f} px")
        print(f"  Mean interframe motion: {raw['mean_interframe_displacement_px']:.2f} px (max: {raw['max_interframe_displacement_px']:.2f} px)")

        # Run with Standalone KF
        kf_res = run_pipeline_test(v["path"], f"{v['id']}_KF", num_frames=v["frames"], use_hybrid=False)
        all_pipeline_stats.append(kf_res)
        print(f"  -> KF Closed Loop: Lock Ret={kf_res['lock_retention_rate']*100:.1f}%, TrackLoss={kf_res['track_loss_count']}, States={kf_res['state_breakdown']}, Cam Pan Span={kf_res['camera_pan_span_mrad']:.1f} mrad")

        # Run with Hybrid Tracker (KF/PF)
        hyb_res = run_pipeline_test(v["path"], f"{v['id']}_HYBRID", num_frames=v["frames"], use_hybrid=True)
        all_pipeline_stats.append(hyb_res)
        print(f"  -> Hybrid (KF/PF): Lock Ret={hyb_res['lock_retention_rate']*100:.1f}%, Modes={hyb_res['mode_breakdown']}, States={hyb_res['state_breakdown']}")

    # Save results to json
    os.makedirs("results", exist_ok=True)
    summary_path = "results/video_stress_test_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"raw_stats": all_raw_stats, "pipeline_runs": all_pipeline_stats}, f, indent=2)

    print("\n" + "=" * 88)
    print(f"Comprehensive evaluation completed. Output saved to {summary_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
