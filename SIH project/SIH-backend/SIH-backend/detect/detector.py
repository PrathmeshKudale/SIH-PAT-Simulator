"""
Advanced Optical Beacon Detector with Morphological Top-Hat Filtering,
Adaptive Thresholding, Blob Quality Metrics, and Temporal Blinking Verification.
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import numpy as np
import cv2
from contracts import FrameData, TargetState, CameraState


@dataclass
class BlobCandidate:
    """Represents an extracted optical spot candidate."""
    x: float
    y: float
    peak_intensity: float
    flux: float
    area: int
    contrast: float
    size_consistency: float
    confidence: float


class BeaconVerifier:
    """
    Tracks candidate blobs over consecutive frames and validates temporal blinking signature
    to reliably reject non-blinking clutter and glints before declaring target acquisition.
    Compensates for gimbal motion between frames.
    """

    def __init__(
        self,
        min_history_len: int = 6,
        max_history_len: int = 15,
        min_intensity_std: float = 25.0,  # Minimum intensity variation for blinking
        max_clutter_std: float = 12.0,    # Threshold below which blob is flagged as static clutter
        assoc_dist_px: float = 40.0,
    ):
        self.min_history_len = min_history_len
        self.max_history_len = max_history_len
        self.min_intensity_std = min_intensity_std
        self.max_clutter_std = max_clutter_std
        self.assoc_dist_px = assoc_dist_px

        # Active candidate tracks: track_id -> dict
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self._next_track_id = 1
        self.confirmed_target_id: Optional[int] = None
        self._last_cam_pan: Optional[float] = None
        self._last_cam_tilt: Optional[float] = None

    def reset(self) -> None:
        """Reset verifier state."""
        self.tracks.clear()
        self.confirmed_target_id = None
        self._next_track_id = 1
        self._last_cam_pan = None
        self._last_cam_tilt = None

    def update(
        self,
        candidates: List[BlobCandidate],
        timestamp: float,
        camera_state: Optional[CameraState] = None,
    ) -> Tuple[Optional[BlobCandidate], Dict[str, Any]]:
        """
        Associate candidates with tracks using ego-motion compensation,
        analyze temporal intensity profile, and return verified target blob (if seen in current frame).
        """
        report: Dict[str, Any] = {
            "status": "SEARCHING",
            "candidates_count": len(candidates),
            "clutter_rejected": 0,
            "evaluating_count": 0,
            "verified_id": self.confirmed_target_id,
            "details": [],
        }

        # Calculate camera ego-motion pixel shift if camera_state provided
        shift_u = 0.0
        shift_v = 0.0
        if camera_state is not None:
            if self._last_cam_pan is not None and self._last_cam_tilt is not None:
                d_pan = camera_state.pan - self._last_cam_pan
                d_tilt = camera_state.tilt - self._last_cam_tilt
                w, h = camera_state.resolution
                # Camera pan right -> image shifts left (negative u)
                shift_u = -d_pan * (w / camera_state.fov_x)
                shift_v = -d_tilt * (h / camera_state.fov_y)

            self._last_cam_pan = camera_state.pan
            self._last_cam_tilt = camera_state.tilt

        # Associate candidates to existing tracks
        matched_tracks = set()
        matched_candidates = {}
        unmatched_candidates = list(range(len(candidates)))

        for c_idx, cand in enumerate(candidates):
            best_id = None
            best_dist = float("inf")

            for t_id, track in self.tracks.items():
                if t_id in matched_tracks:
                    continue
                last_pos = track["positions"][-1]
                # Predict position after camera ego-motion
                pred_x = last_pos[0] + shift_u
                pred_y = last_pos[1] + shift_v

                dist = np.hypot(cand.x - pred_x, cand.y - pred_y)
                if dist < self.assoc_dist_px and dist < best_dist:
                    best_dist = dist
                    best_id = t_id

            if best_id is not None:
                matched_tracks.add(best_id)
                matched_candidates[best_id] = cand
                if c_idx in unmatched_candidates:
                    unmatched_candidates.remove(c_idx)

                track = self.tracks[best_id]
                track["positions"].append((cand.x, cand.y))
                track["intensities"].append(cand.peak_intensity)
                track["confidences"].append(cand.confidence)
                track["last_candidate"] = cand
                track["missed_frames"] = 0
                track["frames_seen"] += 1
                if len(track["intensities"]) > self.max_history_len:
                    track["intensities"].pop(0)
                    track["positions"].pop(0)
                    track["confidences"].pop(0)

        # Handle missed tracks
        for t_id, track in list(self.tracks.items()):
            if t_id not in matched_tracks:
                track["missed_frames"] = track.get("missed_frames", 0) + 1
                # Drop tracks that disappear for > 5 frames
                if track["missed_frames"] > 5:
                    if self.confirmed_target_id == t_id:
                        self.confirmed_target_id = None
                    del self.tracks[t_id]

        # Create new tracks for unmatched candidates
        for c_idx in unmatched_candidates:
            cand = candidates[c_idx]
            new_id = self._next_track_id
            self._next_track_id += 1
            self.tracks[new_id] = {
                "id": new_id,
                "positions": [(cand.x, cand.y)],
                "intensities": [cand.peak_intensity],
                "confidences": [cand.confidence],
                "last_candidate": cand,
                "frames_seen": 1,
                "missed_frames": 0,
                "is_verified": False,
                "is_clutter": False,
            }

        # Analyze tracks
        verified_blob: Optional[BlobCandidate] = None

        for t_id, track in list(self.tracks.items()):
            intensities = track["intensities"]
            n_samples = len(intensities)

            if n_samples < self.min_history_len:
                report["evaluating_count"] += 1
                report["details"].append({
                    "track_id": t_id,
                    "status": "EVALUATING",
                    "samples": n_samples,
                    "pos": track["positions"][-1],
                })
                continue

            # Calculate temporal statistics
            int_std = float(np.std(intensities))
            int_range = float(np.ptp(intensities))

            # 1. Clutter Check: Static, non-blinking spot has minimal intensity variation
            if int_std < self.max_clutter_std:
                track["is_clutter"] = True
                track["is_verified"] = False
                report["clutter_rejected"] += 1
                report["details"].append({
                    "track_id": t_id,
                    "status": "REJECTED_CLUTTER_STATIC",
                    "std": round(int_std, 2),
                    "pos": track["positions"][-1],
                })
                continue

            # 2. Blinking Signature Check: Significant periodic variation
            if int_std >= self.min_intensity_std and int_range >= 40.0:
                mean_val = np.mean(intensities)
                centered = np.array(intensities) - mean_val
                crossings = np.sum(np.diff(np.signbit(centered)) != 0)

                if crossings >= 2:
                    track["is_verified"] = True
                    track["is_clutter"] = False
                    self.confirmed_target_id = t_id

                    report["status"] = "VERIFIED_LOCKED"
                    report["verified_id"] = t_id
                    report["details"].append({
                        "track_id": t_id,
                        "status": "VERIFIED_BLINKING_BEACON",
                        "std": round(int_std, 2),
                        "crossings": int(crossings),
                        "pos": track["positions"][-1],
                    })

                    # ONLY return candidate if it was matched in THIS current frame!
                    if t_id in matched_candidates:
                        verified_blob = matched_candidates[t_id]

        # If we have an already confirmed target and it was seen in this frame:
        if verified_blob is None and self.confirmed_target_id in matched_candidates:
            verified_blob = matched_candidates[self.confirmed_target_id]
            report["status"] = "VERIFIED_LOCKED"
            report["verified_id"] = self.confirmed_target_id

        return verified_blob, report


class AdaptiveOpticalDetector:
    """
    Upgraded detector combining Morphological Top-Hat, Adaptive Thresholding,
    Connected-Component Centroiding, and Blinking Signature Verification.
    """

    def __init__(
        self,
        tophat_kernel_size: int = 7,
        min_blob_area: int = 3,
        max_blob_area: int = 250,
        enable_signature_verification: bool = True,
    ):
        self.tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tophat_kernel_size, tophat_kernel_size)
        )
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.enable_signature_verification = enable_signature_verification
        self.verifier = BeaconVerifier()

    def extract_candidates(self, image: np.ndarray) -> List[BlobCandidate]:
        """
        Extract all valid optical spot candidates from frame using Top-Hat + Connected Components.
        """
        if image is None or image.size == 0:
            return []

        # 1. Morphological Top-Hat filter: isolates bright localized spots
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, self.tophat_kernel)

        # 2. Adaptive thresholding on top-hat filtered image
        th_mean, th_std = float(np.mean(tophat)), float(np.std(tophat))
        thresh_level = max(18.0, th_mean + 2.8 * th_std)
        _, binary = cv2.threshold(tophat, thresh_level, 255, cv2.THRESH_BINARY)

        # 3. Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        candidates: List[BlobCandidate] = []
        if num_labels <= 1:
            return candidates

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.min_blob_area or area > self.max_blob_area:
                continue

            mask = (labels == label)
            ys, xs = np.nonzero(mask)
            weights = image[ys, xs].astype(np.float64)
            total_weight = np.sum(weights)
            if total_weight <= 0:
                continue

            u_c = float(np.sum(xs * weights) / total_weight)
            v_c = float(np.sum(ys * weights) / total_weight)

            peak_intensity = float(np.max(weights))
            flux = float(total_weight)

            local_bg = th_mean
            contrast = float(np.clip((peak_intensity - local_bg) / 255.0, 0.0, 1.0))
            size_consistency = float(np.exp(-abs(area - 25) / 35.0))

            confidence = float(
                0.40 * (peak_intensity / 255.0)
                + 0.35 * contrast
                + 0.25 * size_consistency
            )

            candidates.append(
                BlobCandidate(
                    x=u_c,
                    y=v_c,
                    peak_intensity=peak_intensity,
                    flux=flux,
                    area=area,
                    contrast=contrast,
                    size_consistency=size_consistency,
                    confidence=confidence,
                )
            )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def detect(
        self,
        frame: FrameData,
        camera_state: Optional[CameraState] = None,
    ) -> Tuple[Optional[TargetState], Dict[str, Any]]:
        """
        Process frame: extract candidates, run signature verification, and return
        verified TargetState along with detection diagnostic metadata.
        """
        candidates = self.extract_candidates(frame.image)

        if not self.enable_signature_verification:
            if not candidates:
                return None, {"status": "NO_BLOB"}
            best = candidates[0]
            target = TargetState(
                x=best.x,
                y=best.y,
                confidence=best.confidence,
                timestamp=frame.timestamp,
                tracker_mode="DETECTED",
            )
            return target, {"status": "DETECTED", "candidates_count": len(candidates)}

        verified_blob, verifier_report = self.verifier.update(
            candidates, frame.timestamp, camera_state=camera_state
        )

        if verified_blob is not None:
            target = TargetState(
                x=verified_blob.x,
                y=verified_blob.y,
                confidence=verified_blob.confidence,
                timestamp=frame.timestamp,
                tracker_mode="ACQUIRED",
            )
            return target, verifier_report

        return None, verifier_report


# Backward compatibility alias
SimpleDetector = AdaptiveOpticalDetector
