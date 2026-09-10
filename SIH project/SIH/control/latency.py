"""
Control Latency Simulation introducing configurable N-frame actuation delay.
Smart India Hackathon - Problem Statement 26169 (ISRO / DOS)

Models optical sensor readout, image processing, and mechanical gimbal latency
by buffering control commands in a FIFO delay queue before actuation.
"""

from typing import Tuple
from collections import deque


class ControlDelayQueue:
    """
    Simulates discrete frame-level latency between optical detection/command computation
    and physical gimbal motor actuation.
    """

    def __init__(self, delay_frames: int = 2):
        self.delay_frames = max(0, int(delay_frames))
        # Initialize FIFO buffer with zero commands
        self._queue = deque([(0.0, 0.0)] * self.delay_frames, maxlen=self.delay_frames + 1)

    def reset(self) -> None:
        """Clear queue and re-initialize with zero commands."""
        self._queue = deque([(0.0, 0.0)] * self.delay_frames, maxlen=self.delay_frames + 1)

    def step(self, cmd_pan: float, cmd_tilt: float) -> Tuple[float, float]:
        """
        Push new command and pop the command delayed by N frames.

        Args:
            cmd_pan: Commanded pan adjustment computed at current frame (rad).
            cmd_tilt: Commanded tilt adjustment computed at current frame (rad).

        Returns:
            Delayed (actuated_pan, actuated_tilt) from N frames ago.
        """
        if self.delay_frames <= 0:
            return (float(cmd_pan), float(cmd_tilt))

        # Push current command
        self._queue.append((float(cmd_pan), float(cmd_tilt)))

        # Pop delayed command from N frames ago
        delayed_pan, delayed_tilt = self._queue.popleft()
        return (delayed_pan, delayed_tilt)
