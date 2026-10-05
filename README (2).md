# Face Detection System

A desktop application that detects human faces from a **live webcam**, **images** and **videos** using **Python, OpenCV and the Haar Cascade classifier**, with a Tkinter GUI and a saved detection history.

## Features

- **Dashboard** with sidebar navigation and quick-access cards
- **Live camera detection** with face count, FPS, Start / Stop / Capture buttons
- **Image detection** (JPG, JPEG, PNG, BMP) with preview and *Save Result*
- **Video detection** (MP4, AVI, MOV, MKV) with progress bar, Start / Stop, and the processed video saved to `results/`
- **Detection history** table (`history/history.json`) with Refresh and Clear
- **About** page explaining face detection, OpenCV, Haar Cascade and real-world uses
- Friendly error messages (no webcam, bad/corrupted/unsupported files, missing cascade, empty selection)
- Safe file names, automatic folder creation, no overwriting of existing results

## Technologies

Python 3 - OpenCV (`cv2`) - NumPy - Tkinter - Pillow - JSON

## Installation

```bash
pip install -r requirements.txt
```

Tkinter ships with Python on Windows and macOS. On some Linux distributions install it separately, e.g. `sudo apt install python3-tk`.

## How to run

```bash
python main.py
```

## How Haar Cascade works

```
Image / Frame -> Grayscale -> Haar Cascade Classifier -> detectMultiScale()
              -> Detected faces -> Draw bounding boxes -> Display face count
```

A Haar Cascade (Viola-Jones, 2001) is trained on thousands of face and non-face images. It stores simple light/dark patterns ("Haar features") in `haarcascade_frontalface_default.xml`. `detectMultiScale()` slides a window across the image at many sizes; every window passes through a series of stages, and windows that fail an early stage are discarded quickly. Parameters used here (see `face_detector.py`):

| Parameter | Value | Meaning |
|---|---|---|
| `scaleFactor` | 1.1 | Size step between scans (smaller = more thorough, slower) |
| `minNeighbors` | 5 (8 in strict mode) | Overlapping hits required to accept a face (higher = fewer false positives) |
| `minSize` | 30 x 30 | Smallest face to look for, in pixels |

## Project structure

```
Face-Detection-System/
├── main.py                 GUI, navigation, page logic
├── face_detector.py        Haar Cascade loading, detection, drawing, shared helpers
├── camera.py               Webcam handling, real-time detection, FPS
├── image_detector.py       Image loading, detection, saving
├── video_detector.py       Frame-by-frame video processing and saving
├── history_manager.py      Save / read / clear history.json
├── requirements.txt
├── README.md
├── haarcascade_frontalface_default.xml
├── results/                Saved images and processed videos
├── captured/               Webcam captures
├── uploads/                Safe copies of selected images and videos
└── history/history.json
```

## Usage instructions

1. **Live Camera:** open *Live Camera*, press *Start Camera*. Faces are boxed and the face count and FPS update live. *Capture Image* saves the current frame to `captured/`. *Stop Camera* releases the webcam.
2. **Image:** press *Select Image*, choose a file. The processed image appears with "Faces Detected: X". Press *Save Result* to store it in `results/`.
3. **Video:** press *Select Video*, then *Start Processing*. Use *Stop Processing* to end early; the processed video (so far) is still saved to `results/`.
4. **History:** every detection adds a record (date, time, type, file, faces). For videos and webcam sessions the face number is the **most faces seen in a single frame**.

## Applications

CCTV surveillance - security systems - mobile cameras - attendance systems - smart access control - video monitoring - human-computer interaction

## Strict mode (fewer false detections)

Haar Cascades match light/dark patterns, so objects (coins, clocks, sockets, patterned items) can occasionally be mistaken for faces. **Strict mode** (ON by default, checkbox in the sidebar) verifies each candidate: stricter `minNeighbors`, a minimum size relative to the image, a sensible width/height ratio, enough skin-coloured pixels, and at least one eye detected inside the face. Turn it off if real faces are being missed (for example sunglasses, very small or side-on faces).

## Limitations

Haar Cascades work best on frontal, well-lit faces. Profile views, masks, sunglasses and poor lighting reduce accuracy.

## Future improvements

- Face recognition
- Multiple face tracking
- Mask detection
- Age and gender estimation
- Attendance using face recognition
- Deep learning-based detection
- YOLO face detection
- Database integration
