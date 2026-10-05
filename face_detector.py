"""
face_detector.py
----------------
Core face-detection logic plus small shared helpers (paths, safe file names,
image reading/writing, custom error class) used by every other module.

Detection pipeline:
    Image/Frame -> Grayscale -> Haar Cascade -> detectMultiScale()
                -> Detected faces -> Draw bounding boxes -> Face count
"""

import os
import re

import cv2
import numpy as np

# ---------------------------------------------------------------- paths -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_NAME = "haarcascade_frontalface_default.xml"
CASCADE_PATH = os.path.join(BASE_DIR, CASCADE_NAME)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
CAPTURED_DIR = os.path.join(BASE_DIR, "captured")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
HISTORY_DIR = os.path.join(BASE_DIR, "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


class DetectionError(Exception):
    """Error whose message is safe to show directly to the user."""


# -------------------------------------------------------------- helpers -----
def ensure_dirs():
    """Create all working folders if they do not exist yet."""
    for folder in (RESULTS_DIR, CAPTURED_DIR, UPLOADS_DIR, HISTORY_DIR):
        os.makedirs(folder, exist_ok=True)


def safe_filename(name):
    """Return a filesystem-safe version of a file name (no paths, no odd chars)."""
    name = os.path.basename(str(name).replace("\\", "/"))
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("._") or "file"
    ext = re.sub(r"[^A-Za-z0-9.]", "", ext).lower()
    return stem[:80] + ext


def unique_path(directory, filename):
    """Return a path inside `directory` that does not exist yet (never overwrites)."""
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base}_{counter}{ext}")
        counter += 1
    return candidate


def validate_media_path(path, allowed_exts, kind):
    """Check that a path is non-empty, exists, is non-empty and has a valid extension."""
    if not path or not str(path).strip():
        raise DetectionError(f"No {kind} was selected.")
    if not os.path.isfile(path):
        raise DetectionError(f"File not found:\n{path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in allowed_exts:
        raise DetectionError(
            f"Unsupported {kind} format '{ext or 'unknown'}'.\n"
            f"Supported formats: {', '.join(allowed_exts)}")
    if os.path.getsize(path) == 0:
        raise DetectionError(f"The {kind} file is empty or corrupted.")


def read_image(path):
    """Read an image as BGR. np.fromfile + imdecode also works with non-ASCII
    Windows paths, where cv2.imread often fails."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError as exc:
        raise DetectionError(f"Could not read the image file:\n{exc}")
    if image is None or image.size == 0:
        raise DetectionError("The image could not be read. It may be corrupted "
                             "or not a valid image.")
    return image


def write_image(path, image):
    """Write a BGR image to disk (Unicode-path safe)."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise DetectionError("Could not encode the image for saving.")
    try:
        buffer.tofile(path)
    except OSError as exc:
        raise DetectionError(f"Could not save the image:\n{exc}")


# ------------------------------------------------------------- detector -----
EYE_CASCADE_NAME = "haarcascade_eye.xml"


def _load_cascade_file(name, local_path=None):
    """Load a Haar cascade from the project folder, else from OpenCV's bundle."""
    candidates = [local_path] if local_path else []
    data = getattr(cv2, "data", None)
    if data is not None:
        candidates.append(os.path.join(data.haarcascades, name))

    found_file = False
    for path in candidates:
        if os.path.isfile(path):
            found_file = True
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade
    if found_file:
        raise DetectionError(f"{name} exists but could not be loaded. "
                             "The file may be corrupted.")
    raise DetectionError(f"Haar Cascade file '{name}' is missing.\n"
                         f"Place it in:\n{BASE_DIR}")


class FaceDetector:
    """Frontal face detection with Haar Cascades.

    Haar cascades only look for light/dark *patterns*, so objects such as coins,
    clocks or sockets can be mistaken for faces. In strict mode (default) each
    candidate is therefore verified:
      1. higher minNeighbors and a minimum size relative to the image,
      2. a plausible width/height ratio,
      3. enough skin-coloured pixels (YCrCb colour range),
      4. at least one eye found in the upper part of the face.
    """

    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30),
                 strict=True):
        self.scale_factor = scale_factor    # image shrink step between scans
        self.min_neighbors = min_neighbors  # higher = fewer false positives
        self.min_size = min_size            # ignore anything smaller (pixels)
        self.strict = strict                # verify candidates (fewer false hits)
        self.cascade = _load_cascade_file(CASCADE_NAME, CASCADE_PATH)
        self.eye_cascade = _load_cascade_file(EYE_CASCADE_NAME)

    # -- verification helpers (strict mode) ---------------------------------
    @staticmethod
    def _skin_ratio(frame, x, y, w, h):
        """Fraction of skin-coloured pixels in the central part of a candidate."""
        x0, x1 = x + w // 5, x + w - w // 5
        y0, y1 = y + h // 5, y + h - h // 10
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        return float(cv2.countNonZero(mask)) / mask.size

    def _has_eye(self, gray, x, y, w, h):
        """True if at least one eye is found in the upper 60% of the candidate."""
        roi = gray[y:y + int(h * 0.6), x:x + w]
        if roi.size == 0:
            return False
        eyes = self.eye_cascade.detectMultiScale(
            roi, scaleFactor=1.1, minNeighbors=4,
            minSize=(max(8, w // 12), max(8, w // 12)))
        return len(eyes) > 0

    def detect_faces(self, frame):
        """Return a list of (x, y, w, h) face rectangles for a BGR frame."""
        if frame is None or frame.size == 0:
            raise DetectionError("Empty image/frame received.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Haar works on grayscale
        gray = cv2.equalizeHist(gray)                    # better in uneven light

        min_neighbors, min_size = self.min_neighbors, self.min_size
        if self.strict:
            min_neighbors = max(min_neighbors, 8)
            side = max(self.min_size[0], int(0.06 * min(gray.shape[:2])))
            min_size = (side, side)

        candidates = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor,
            minNeighbors=min_neighbors, minSize=min_size)
        if not self.strict:
            return [tuple(int(v) for v in f) for f in candidates]

        faces = []
        for (x, y, w, h) in candidates:
            x, y, w, h = int(x), int(y), int(w), int(h)
            if not 0.75 <= w / float(h) <= 1.3:
                continue
            if self._skin_ratio(frame, x, y, w, h) < 0.25:
                continue
            if not self._has_eye(gray, x, y, w, h):
                continue
            faces.append((x, y, w, h))
        return faces

    @staticmethod
    def draw_faces(frame, faces, overlay=False):
        """Draw green boxes (and optionally a face-count banner) in place."""
        thickness = max(2, frame.shape[1] // 400)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), thickness)
        if overlay:
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def process(self, frame, overlay=False):
        """Detect + draw. Returns (annotated_frame, face_count)."""
        faces = self.detect_faces(frame)
        self.draw_faces(frame, faces, overlay)
        return frame, len(faces)
