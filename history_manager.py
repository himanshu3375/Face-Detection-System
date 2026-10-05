"""
history_manager.py
------------------
Stores every detection in history/history.json.

Entry format:
    {"date": "2026-10-02", "time": "16:30:20", "type": "Image",
     "file": "person.jpg", "faces": 2}
"""

import json
import os
from datetime import datetime

from face_detector import HISTORY_DIR, HISTORY_FILE, DetectionError


def read_history():
    """Return the list of saved entries (oldest first). Never raises on bad data."""
    if not os.path.isfile(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        # Corrupted file: keep a backup so nothing is silently lost.
        try:
            os.replace(HISTORY_FILE, HISTORY_FILE + ".bak")
        except OSError:
            pass
        return []


def _write(entries):
    """Write atomically: temp file first, then replace the real file."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    tmp = HISTORY_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=4)
        os.replace(tmp, HISTORY_FILE)
    except OSError as exc:
        raise DetectionError(f"Could not save detection history:\n{exc}")


def add_entry(detection_type, file_name, faces):
    """Append one detection record and return it."""
    now = datetime.now()
    entry = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "type": detection_type,
        "file": str(file_name),
        "faces": int(faces),
    }
    entries = read_history()
    entries.append(entry)
    _write(entries)
    return entry


def clear_history():
    """Delete all saved entries."""
    _write([])
