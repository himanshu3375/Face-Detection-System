"""
image_detector.py
-----------------
Load an image, run face detection and save the annotated result.
(File *selection* happens in the GUI; this module does the work.)
"""

import os
import shutil

from face_detector import (IMAGE_EXTS, RESULTS_DIR, UPLOADS_DIR, DetectionError,
                           ensure_dirs, read_image, safe_filename, unique_path,
                           validate_media_path, write_image)


class ImageDetector:
    def __init__(self, detector):
        self.detector = detector

    def load(self, path):
        """Validate + read the image. A safe copy is stored in uploads/."""
        validate_media_path(path, IMAGE_EXTS, "image")
        image = read_image(path)
        ensure_dirs()
        try:
            shutil.copy2(path, unique_path(UPLOADS_DIR,
                                           safe_filename(os.path.basename(path))))
        except OSError:
            pass  # the upload copy is a convenience; detection can continue
        return image

    def detect(self, image):
        """Return (annotated_copy, face_count); the original stays untouched."""
        return self.detector.process(image.copy())

    def save(self, annotated, source_name):
        """Save into results/ without overwriting. Returns the saved path."""
        if annotated is None:
            raise DetectionError("There is no processed image to save.")
        ensure_dirs()
        stem, ext = os.path.splitext(safe_filename(source_name))
        ext = ".jpg" if ext in ("", ".jpeg") else ext
        path = unique_path(RESULTS_DIR, f"{stem}_detected{ext}")
        write_image(path, annotated)
        return path
