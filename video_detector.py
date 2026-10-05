"""
video_detector.py
-----------------
Frame-by-frame video face detection. The GUI calls step() repeatedly (via
Tk's after()) so the window stays responsive. Every processed frame is also
written to a new video file in results/.
"""

import math
import os
import shutil

import cv2

from face_detector import (RESULTS_DIR, UPLOADS_DIR, VIDEO_EXTS, DetectionError,
                           ensure_dirs, safe_filename, unique_path,
                           validate_media_path)


class VideoProcessor:
    def __init__(self, detector):
        self.detector = detector
        self.cap = None
        self.writer = None
        self.output_path = None
        self._reset()

    def _reset(self):
        self.frames = 0
        self.max_faces = 0
        self.total_frames = 0
        self._pending = None   # first frame, read while validating the file
        self._size = None

    @property
    def active(self):
        return self.cap is not None

    @property
    def progress(self):
        """Fraction 0..1 (0 if the container doesn't report a frame count)."""
        if self.total_frames <= 0:
            return 0.0
        return min(self.frames / self.total_frames, 1.0)

    @staticmethod
    def validate(path):
        validate_media_path(path, VIDEO_EXTS, "video")

    def open(self, path):
        """Prepare reading + writing. Raises DetectionError on any problem."""
        self.close()
        self.validate(path)
        ensure_dirs()

        # Work from an ASCII-named copy: OpenCV can fail on Unicode paths (Windows).
        local = unique_path(UPLOADS_DIR, safe_filename(os.path.basename(path)))
        try:
            shutil.copy2(path, local)
        except OSError as exc:
            raise DetectionError(f"Could not copy the video into uploads/:\n{exc}")

        cap = cv2.VideoCapture(local)
        ok, first = cap.read() if cap.isOpened() else (False, None)
        if not ok or first is None:
            cap.release()
            raise DetectionError("The video could not be opened. It may be "
                                 "corrupted or use an unsupported codec.")

        height, width = first.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps < 1:
            fps = 25.0

        stem = os.path.splitext(safe_filename(os.path.basename(path)))[0]
        out = unique_path(RESULTS_DIR, f"{stem}_detected.mp4")
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (width, height))
        if not writer.isOpened():
            cap.release()
            raise DetectionError("Could not create the output video file.")

        self._reset()
        self.cap, self.writer, self.output_path = cap, writer, out
        self._pending, self._size = first, (width, height)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.total_frames = total if total > 0 else 0

    def step(self):
        """Process the next frame. Returns (frame, faces) or None when finished."""
        if not self.active:
            return None
        if self._pending is not None:
            frame, self._pending = self._pending, None
        else:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return None
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)

        frame, count = self.detector.process(frame, overlay=True)
        self.writer.write(frame)
        self.frames += 1
        self.max_faces = max(self.max_faces, count)
        return frame, count

    def close(self):
        """Release everything. Returns a summary dict (or None if nothing was open)."""
        if not self.active and self.writer is None:
            return None
        summary = {"frames": self.frames, "max_faces": self.max_faces,
                   "output": self.output_path}
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        if self.frames == 0 and self.output_path and os.path.exists(self.output_path):
            try:
                os.remove(self.output_path)   # nothing was written; drop empty file
            except OSError:
                pass
        self.cap = self.writer = None
        self._pending = None
        return summary
