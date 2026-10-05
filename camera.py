"""
camera.py
---------
Webcam handling: open/close the camera, grab frames, run face detection and
compute a smoothed FPS value. The GUI calls read() repeatedly (via Tk's after()).
"""

import os
import time
from datetime import datetime

import cv2

from face_detector import (CAPTURED_DIR, DetectionError, ensure_dirs,
                           unique_path, write_image)


class Camera:
    def __init__(self, detector, index=0):
        self.detector = detector
        self.index = index
        self.cap = None
        self._reset_stats()

    def _reset_stats(self):
        self.last_frame = None   # most recent annotated frame (for Capture)
        self.last_count = 0
        self.max_faces = 0
        self.frames = 0
        self._fps = 0.0
        self._last_time = None

    @property
    def running(self):
        return self.cap is not None and self.cap.isOpened()

    def start(self):
        """Open the webcam or raise a user-friendly DetectionError."""
        if self.running:
            return
        # DirectShow opens much faster on Windows than the default backend.
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            raise DetectionError(
                "Webcam unavailable.\n\nCheck that a camera is connected, that no "
                "other app is using it, and that camera access is allowed "
                "(Windows: Settings > Privacy > Camera).")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise DetectionError(
                "The camera opened but returned no image.\n\nIt may be in use by "
                "another application or camera permission may be denied.")
        self.cap = cap
        self._reset_stats()

    def read(self):
        """Grab one frame, detect faces. Returns (frame, face_count, fps)."""
        if not self.running:
            raise DetectionError("The camera is not running.")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise DetectionError("Lost connection to the webcam.")

        frame, count = self.detector.process(frame)

        # FPS: exponential moving average so the number doesn't jitter.
        now = time.perf_counter()
        if self._last_time is not None:
            instant = 1.0 / max(now - self._last_time, 1e-6)
            self._fps = instant if self._fps == 0 else 0.9 * self._fps + 0.1 * instant
        self._last_time = now

        self.last_frame, self.last_count = frame, count
        self.max_faces = max(self.max_faces, count)
        self.frames += 1
        return frame, count, self._fps

    def capture(self):
        """Save the current frame into captured/. Returns (path, face_count)."""
        if self.last_frame is None:
            raise DetectionError("No frame available to capture yet.")
        ensure_dirs()
        name = f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        path = unique_path(CAPTURED_DIR, name)
        write_image(path, self.last_frame)
        return path, self.last_count

    def stop(self):
        """Release the webcam. Returns a small session summary."""
        summary = {"frames": self.frames, "max_faces": self.max_faces}
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return summary
