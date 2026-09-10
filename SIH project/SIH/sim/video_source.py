"""
Video Frame Source for Free Space Optical Communication (FSOC) coarse PAT Simulator.
Supports consuming real pre-recorded video files (.mp4, .avi) for Benchmark Performance-2.
"""

from __future__ import annotations
import os
from typing import Optional, Iterator, Tuple
import cv2
import numpy as np
from contracts import FrameData, get_resource_path


class VideoFrameSource:
    """
    Decodes frames from a video file (.mp4, etc.) via OpenCV VideoCapture and
    yields typed FrameData objects matching the optical detector's expected
    monochrome single-channel uint8 array format.

    Timestamps are derived dynamically from the file's actual container FPS
    (cv2.CAP_PROP_FPS) rather than assuming a fixed frame rate.
    """

    def __init__(self, video_path: str, loop: bool = False):
        """
        Args:
            video_path: Filepath to the video file.
            loop: Whether to loop back to the start upon reaching end of stream.
        """
        resolved = get_resource_path(str(video_path))
        self.video_path = resolved if os.path.exists(resolved) else str(video_path)
        self.loop = bool(loop)

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found at path: '{self.video_path}'")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video file via OpenCV VideoCapture: '{self.video_path}'")

        raw_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = raw_fps if (raw_fps > 0.0 and np.isfinite(raw_fps)) else 30.0
        self.frame_duration = 1.0 / self.fps

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_id = 0

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return (width, height) resolution tuple."""
        return (self.width, self.height)

    def get_frame(self, frame_id: Optional[int] = None) -> Optional[FrameData]:
        """
        Retrieve the next decoded frame from the video stream.

        Args:
            frame_id: Optional frame index to seek to. If None, retrieves next sequential frame.

        Returns:
            FrameData with 2D monochrome image array and FPS-derived timestamp,
            or None when end-of-video is reached.
        """
        if not self.cap.isOpened():
            return None

        if frame_id is not None and frame_id != self.current_frame_id:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_id))
            self.current_frame_id = frame_id

        ret, raw_frame = self.cap.read()
        if not ret or raw_frame is None:
            if self.loop and self.total_frames > 0:
                self.reset()
                ret, raw_frame = self.cap.read()
                if not ret or raw_frame is None:
                    return None
            else:
                return None

        # Convert multi-channel BGR/RGB/BGRA to single-channel monochrome grayscale
        if raw_frame.ndim == 3:
            if raw_frame.shape[2] == 3:
                gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            elif raw_frame.shape[2] == 4:
                gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2GRAY)
            else:
                gray = raw_frame[:, :, 0]
        else:
            gray = raw_frame

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        active_id = self.current_frame_id
        timestamp = float(active_id * self.frame_duration)
        self.current_frame_id += 1

        return FrameData(
            image=gray,
            timestamp=timestamp,
            frame_id=active_id,
            ground_truth_target_pos=None,
            ground_truth_targets=None,
        )

    def __iter__(self) -> Iterator[FrameData]:
        return self

    def __next__(self) -> FrameData:
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame

    def reset(self) -> None:
        """Rewind video stream back to frame index 0."""
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        self.current_frame_id = 0

    def close(self) -> None:
        """Release OpenCV VideoCapture stream handle."""
        if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self) -> "VideoFrameSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
