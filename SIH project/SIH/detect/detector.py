"""
Advanced Optical Beacon Detector with Morphological Top-Hat Filtering,
Adaptive Thresholding, Blob Quality Metrics, and Temporal Blinking Verification.
Supports multi-target discrimination and signature-based identity tagging.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
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
    target_id: Optional[Union[str, int]] = None


class VerifiedBlobList(list):
    """
    List of verified BlobCandidate objects representing all signature-verified blobs.
    Delegates attribute access to the primary blob (the first element) for complete
    backward compatibility with single-target consumers.
    """
    def __init__(self, items: Optional[List[BlobCandidate]] = None):
        super().__init__(items if items is not None else [])

    @property
    def primary(self) -> Optional[BlobCandidate]:
        return self[0] if len(self) > 0 else None

    @property
    def x(self) -> float:
        return self[0].x

    @property
    def y(self) -> float:
        return self[0].y

    @property
    def peak_intensity(self) -> float:
        return self[0].peak_intensity

    @property
    def flux(self) -> float:
        return self[0].flux

    @property
    def area(self) -> int:
        return self[0].area

    @property
    def contrast(self) -> float:
        return self[0].contrast

    @property
    def size_consistency(self) -> float:
        return self[0].size_consistency

    @property
    def confidence(self) -> float:
        return self[0].confidence

    @property
    def target_id(self) -> Optional[Union[str, int]]:
        return self[0].target_id if len(self) > 0 else None


class VerifiedTargetList(list):
    """
    List of verified TargetState objects representing all validated detections in the frame.
    Delegates attribute access to the primary target (the first element) for complete
    backward compatibility with single-target consumers.
    """
    def __init__(self, items: Optional[List[TargetState]] = None):
        super().__init__(items if items is not None else [])

    @property
    def primary(self) -> Optional[TargetState]:
        return self[0] if len(self) > 0 else None

    @property
    def x(self) -> float:
        return self[0].x

    @property
    def y(self) -> float:
        return self[0].y

    @property
    def vx(self) -> float:
        return self[0].vx

    @property
    def vy(self) -> float:
        return self[0].vy

    @property
    def confidence(self) -> float:
        return self[0].confidence

    @property
    def timestamp(self) -> float:
        return self[0].timestamp

    @property
    def tracker_mode(self) -> str:
        return self[0].tracker_mode

    @property
    def target_id(self) -> Union[str, int]:
        return self[0].target_id

    def to_dict(self) -> Dict[str, Any]:
        return self[0].to_dict() if len(self) > 0 else {}


class BeaconVerifier:
    """
    Tracks candidate blobs over consecutive frames and validates temporal blinking signature
    to reliably reject non-blinking clutter and glints before declaring target acquisition.
    Supports discriminating multiple targets by their unique blink frequencies.
    Compensates for gimbal motion between frames.
    """

    def __init__(
        self,
        min_history_len: int = 6,
        max_history_len: int = 15,
        min_intensity_std: float = 25.0,  # Minimum intensity variation for blinking
        max_clutter_std: float = 12.0,    # Threshold below which blob is flagged as static clutter
        assoc_dist_px: float = 40.0,
        known_signatures: Optional[Dict[Union[str, int], float]] = None,
        targets: Optional[List[Any]] = None,
        primary_target_id: Optional[Union[str, int]] = None,
        freq_tolerance: float = 2.0,
    ):
        self.min_history_len = min_history_len
        self.max_history_len = max_history_len
        self.min_intensity_std = min_intensity_std
        self.max_clutter_std = max_clutter_std
        self.assoc_dist_px = assoc_dist_px
        self.freq_tolerance = freq_tolerance
        self.primary_target_id = primary_target_id

        self.known_signatures: Dict[Union[str, int], float] = {}
        if known_signatures:
            self.known_signatures.update(known_signatures)
        if targets:
            self.register_targets(targets)

        # Active candidate tracks: track_id -> dict
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self._next_track_id = 1
        self.confirmed_target_id: Optional[Any] = None
        self._last_cam_pan: Optional[float] = None
        self._last_cam_tilt: Optional[float] = None

    def reset(self) -> None:
        """Reset verifier state."""
        self.tracks.clear()
        self.confirmed_target_id = None
        self._next_track_id = 1
        self._last_cam_pan = None
        self._last_cam_tilt = None

    def set_known_signatures(self, signatures: Dict[Union[str, int], float]) -> None:
        """Set known target signatures mapping target_id to blink frequency in Hz."""
        self.known_signatures = dict(signatures)

    def register_target(self, target_id: Union[str, int], blink_frequency: float) -> None:
        """Register a known target signature."""
        self.known_signatures[target_id] = float(blink_frequency)

    def register_targets(self, targets: List[Any]) -> None:
        """Register multiple targets from Target instances, TargetConfig instances, or dicts."""
        for t in targets:
            if hasattr(t, "target_id") and hasattr(t, "blink_frequency"):
                self.known_signatures[t.target_id] = float(t.blink_frequency)
            elif isinstance(t, dict):
                tid = t.get("target_id", "target_0")
                bfreq = float(t.get("blink_frequency", 4.0))
                self.known_signatures[tid] = bfreq

    def update(
        self,
        candidates: List[BlobCandidate],
        timestamp: float,
        camera_state: Optional[CameraState] = None,
    ) -> Tuple[Optional[VerifiedBlobList], Dict[str, Any]]:
        """
        Associate candidates with tracks using ego-motion compensation,
        analyze temporal intensity profile, verify signatures, and return all
        verified target blobs (primary first) along with diagnostic report.
        """
        report: Dict[str, Any] = {
            "status": "SEARCHING",
            "candidates_count": len(candidates),
            "clutter_rejected": 0,
            "evaluating_count": 0,
            "verified_id": self.confirmed_target_id,
            "verified_count": 0,
            "verified_target_ids": [],
            "verified_blobs": [],
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
                shift_u = -d_pan * (w / camera_state.fov_x)
                shift_v = -d_tilt * (h / camera_state.fov_y)

            self._last_cam_pan = camera_state.pan
            self._last_cam_tilt = camera_state.tilt

        # Associate candidates to existing tracks
        matched_tracks = set()
        matched_candidates: Dict[int, BlobCandidate] = {}
        unmatched_candidates = list(range(len(candidates)))

        for c_idx, cand in enumerate(candidates):
            best_id = None
            best_dist = float("inf")

            for t_id, track in self.tracks.items():
                if t_id in matched_tracks:
                    continue
                last_pos = track["positions"][-1]
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
                track["timestamps"].append(timestamp)
                track["confidences"].append(cand.confidence)
                track["last_candidate"] = cand
                track["missed_frames"] = 0
                track["frames_seen"] += 1
                if len(track["intensities"]) > self.max_history_len:
                    track["intensities"].pop(0)
                    track["positions"].pop(0)
                    track["timestamps"].pop(0)
                    track["confidences"].pop(0)

        # Handle missed tracks
        for t_id, track in list(self.tracks.items()):
            if t_id not in matched_tracks:
                track["missed_frames"] = track.get("missed_frames", 0) + 1
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
                "timestamps": [timestamp],
                "confidences": [cand.confidence],
                "last_candidate": cand,
                "frames_seen": 1,
                "missed_frames": 0,
                "is_verified": False,
                "is_clutter": False,
                "matched_target_id": None,
            }

        # Analyze tracks
        for t_id, track in list(self.tracks.items()):
            intensities = track["intensities"]
            timestamps = track.get("timestamps", [])
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

            int_std = float(np.std(intensities))
            int_range = float(np.ptp(intensities))

            # 1. Clutter Check: Static non-blinking spot has minimal intensity variation
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
                    # Match against configured target signatures if available
                    if self.known_signatures:
                        t_arr = (
                            np.array(timestamps)
                            if len(timestamps) == len(intensities)
                            else np.arange(len(intensities)) * 0.0333
                        )
                        best_tid = None
                        best_score = -1.0

                        for tid, f_tgt in self.known_signatures.items():
                            if f_tgt > 0:
                                c = np.sum(centered * np.cos(2.0 * np.pi * f_tgt * t_arr))
                                s = np.sum(centered * np.sin(2.0 * np.pi * f_tgt * t_arr))
                                amp = (2.0 / len(intensities)) * np.hypot(c, s)
                                score = float(amp / (np.sqrt(2.0) * int_std)) if int_std > 0 else 0.0
                            else:
                                score = 0.0

                            if score > best_score:
                                best_score = score
                                best_tid = tid

                        if best_score >= 0.50 and best_tid is not None:
                            track["is_verified"] = True
                            track["is_clutter"] = False
                            track["matched_target_id"] = best_tid
                            track["match_score"] = best_score
                            report["details"].append({
                                "track_id": t_id,
                                "status": "VERIFIED_BLINKING_BEACON",
                                "target_id": best_tid,
                                "match_score": round(best_score, 2),
                                "std": round(int_std, 2),
                                "crossings": int(crossings),
                                "pos": track["positions"][-1],
                            })
                        else:
                            # Blob does NOT match any configured target's signature -> reject as clutter
                            track["is_clutter"] = True
                            track["is_verified"] = False
                            report["clutter_rejected"] += 1
                            report["details"].append({
                                "track_id": t_id,
                                "status": "REJECTED_UNKNOWN_SIGNATURE",
                                "best_score": round(best_score, 2),
                                "pos": track["positions"][-1],
                            })
                    else:
                        # Legacy fallback: no specific signatures configured -> accept any oscillating beacon
                        default_tid = self.primary_target_id or "target_0"
                        track["is_verified"] = True
                        track["is_clutter"] = False
                        track["matched_target_id"] = default_tid
                        report["details"].append({
                            "track_id": t_id,
                            "status": "VERIFIED_BLINKING_BEACON",
                            "target_id": default_tid,
                            "std": round(int_std, 2),
                            "crossings": int(crossings),
                            "pos": track["positions"][-1],
                        })

        # Collect verified blobs matched in current frame
        verified_blobs: List[BlobCandidate] = []
        for t_id, track in self.tracks.items():
            if track.get("is_verified", False) and t_id in matched_candidates:
                cand = matched_candidates[t_id]
                cand.target_id = track.get("matched_target_id", "target_0")
                verified_blobs.append(cand)

        # Sort: primary target first (if present), then by confidence descending
        primary_id = self.primary_target_id
        if primary_id is not None:
            verified_blobs.sort(
                key=lambda b: (0 if b.target_id == primary_id else 1, -b.confidence)
            )
        else:
            verified_blobs.sort(key=lambda b: -b.confidence)

        if verified_blobs:
            report["status"] = "VERIFIED_LOCKED"
            report["verified_count"] = len(verified_blobs)
            report["verified_target_ids"] = [b.target_id for b in verified_blobs]
            report["verified_id"] = verified_blobs[0].target_id
            report["verified_blobs"] = verified_blobs
            self.confirmed_target_id = verified_blobs[0].target_id
            return VerifiedBlobList(verified_blobs), report

        report["verified_blobs"] = []
        return None, report

    def update_multi(
        self,
        candidates: List[BlobCandidate],
        timestamp: float,
        camera_state: Optional[CameraState] = None,
    ) -> Tuple[List[BlobCandidate], Dict[str, Any]]:
        """
        Explicit multi-target update method returning a list of all verified BlobCandidates.
        """
        res, report = self.update(candidates, timestamp, camera_state=camera_state)
        if res is None:
            return [], report
        return list(res), report


class AdaptiveOpticalDetector:
    """
    Upgraded detector combining Morphological Top-Hat, Adaptive Thresholding,
    Connected-Component Centroiding, and Multi-Target Blinking Signature Verification.
    """

    def __init__(
        self,
        tophat_kernel_size: int = 7,
        min_blob_area: int = 3,
        max_blob_area: int = 450,  # Accommodate target sizes up to 20x20 px (400 px^2) per PS spec
        enable_signature_verification: bool = True,
        known_signatures: Optional[Dict[Union[str, int], float]] = None,
        targets: Optional[List[Any]] = None,
        primary_target_id: Optional[Union[str, int]] = None,
    ):
        self.tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tophat_kernel_size, tophat_kernel_size)
        )
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.enable_signature_verification = enable_signature_verification
        self.primary_target_id = primary_target_id
        self.verifier = BeaconVerifier(
            known_signatures=known_signatures,
            targets=targets,
            primary_target_id=primary_target_id,
        )

    def set_known_signatures(self, signatures: Dict[Union[str, int], float]) -> None:
        """Set known target blink signatures."""
        self.verifier.set_known_signatures(signatures)

    def register_target(self, target_id: Union[str, int], blink_frequency: float) -> None:
        """Register an individual known target signature."""
        self.verifier.register_target(target_id, blink_frequency)

    def register_targets(self, targets: List[Any]) -> None:
        """Register multiple known targets."""
        self.verifier.register_targets(targets)

    def set_primary_target(self, target_id: Union[str, int]) -> None:
        """Designate the primary target ID."""
        self.primary_target_id = target_id
        self.verifier.primary_target_id = target_id

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
        return_all: bool = False,
    ) -> Tuple[Optional[Union[TargetState, List[TargetState]]], Dict[str, Any]]:
        """
        Process frame: extract candidates, run signature verification, and return
        verified TargetState(s) along with detection diagnostic metadata.

        In single-target mode (default), returns (VerifiedTargetList, report) where
        VerifiedTargetList delegates attribute access (x, y, vx, vy, etc.) to the primary target.
        If return_all=True, returns (List[TargetState], report).
        """
        candidates = self.extract_candidates(frame.image)

        if not self.enable_signature_verification:
            if not candidates:
                report = {"status": "NO_BLOB", "verified_targets": [], "all_detections": []}
                return ([] if return_all else None), report

            states = [
                TargetState(
                    x=c.x,
                    y=c.y,
                    confidence=c.confidence,
                    timestamp=frame.timestamp,
                    tracker_mode="DETECTED",
                    target_id=c.target_id or f"target_{i}",
                )
                for i, c in enumerate(candidates)
            ]
            report = {
                "status": "DETECTED",
                "candidates_count": len(candidates),
                "verified_targets": states,
                "all_detections": states,
            }
            if return_all:
                return states, report
            return VerifiedTargetList(states), report

        verified_blobs, verifier_report = self.verifier.update(
            candidates, frame.timestamp, camera_state=camera_state
        )

        if verified_blobs is not None and len(verified_blobs) > 0:
            target_states = [
                TargetState(
                    x=b.x,
                    y=b.y,
                    confidence=b.confidence,
                    timestamp=frame.timestamp,
                    tracker_mode="ACQUIRED",
                    target_id=b.target_id or "target_0",
                )
                for b in verified_blobs
            ]
            verifier_report["verified_targets"] = target_states
            verifier_report["all_detections"] = target_states
            if return_all:
                return target_states, verifier_report
            return VerifiedTargetList(target_states), verifier_report

        verifier_report["verified_targets"] = []
        verifier_report["all_detections"] = []
        if return_all:
            return [], verifier_report
        return None, verifier_report

    def detect_multi(
        self,
        frame: FrameData,
        camera_state: Optional[CameraState] = None,
    ) -> Tuple[List[TargetState], Dict[str, Any]]:
        """
        Explicit multi-target detection returning a list of all validated TargetState instances.
        """
        res, report = self.detect(frame, camera_state=camera_state, return_all=True)
        return list(res) if res is not None else [], report


# Backward compatibility alias
SimpleDetector = AdaptiveOpticalDetector

