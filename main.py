"""
main.py
-------
Face Detection System - Tkinter desktop application.

Run with:  python main.py

Pages (sidebar navigation): Home, Live Camera, Image, Video, History, About.
All detection logic lives in the other modules; this file only builds the GUI.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import history_manager as history
from camera import Camera
from face_detector import (IMAGE_EXTS, VIDEO_EXTS, DetectionError, FaceDetector,
                           ensure_dirs)
from image_detector import ImageDetector
from video_detector import VideoProcessor

# ------------------------------------------------------------- styling ------
SIDEBAR = "#1e293b"
SIDEBAR_HOVER = "#334155"
ACCENT = "#3b82f6"
GREEN = "#16a34a"
RED = "#dc2626"
GREY = "#64748b"
BG = "#f1f5f9"
CARD = "#ffffff"
TEXT = "#0f172a"
MUTED = "#64748b"
VIEW_BG = "#0f172a"

FONT = "Segoe UI"
F_TITLE = (FONT, 22, "bold")
F_HEAD = (FONT, 16, "bold")
F_BODY = (FONT, 11)
F_SMALL = (FONT, 10)
F_BTN = (FONT, 10, "bold")


def shade(hex_color, factor=0.85):
    """Darken a #rrggbb color (used for hover effects)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (int(r * factor), int(g * factor), int(b * factor))


class FlatButton(tk.Button):
    """Flat coloured button with hover effect and a visible disabled state."""

    def __init__(self, master, text, command, color=ACCENT, **kwargs):
        super().__init__(master, text=text, command=command, bg=color, fg="white",
                         activebackground=shade(color), activeforeground="white",
                         disabledforeground="#e2e8f0", relief="flat", bd=0,
                         cursor="hand2", font=F_BTN, padx=16, pady=8, **kwargs)
        self.color = color
        self.bind("<Enter>", lambda e: self["state"] == "normal"
                  and self.config(bg=shade(self.color)))
        self.bind("<Leave>", lambda e: self["state"] == "normal"
                  and self.config(bg=self.color))

    def set_enabled(self, enabled):
        self.config(state="normal" if enabled else "disabled",
                    bg=self.color if enabled else "#94a3b8",
                    cursor="hand2" if enabled else "arrow")


class ImageView(tk.Frame):
    """Dark panel that shows a BGR (OpenCV) image scaled to fit, keeping its
    aspect ratio. It re-scales automatically when the window is resized."""

    def __init__(self, master, placeholder="No image"):
        super().__init__(master, bg=VIEW_BG)
        self.placeholder = placeholder
        self.label = tk.Label(self, bg=VIEW_BG, fg="#94a3b8", text=placeholder,
                              font=F_BODY, justify="center")
        self.label.place(relx=0.5, rely=0.5, anchor="center")
        self._frame = None
        self._photo = None      # keep a reference or Tk garbage-collects the image
        self._job = None
        self.bind("<Configure>", self._on_resize)

    def show(self, bgr):
        self._frame = bgr
        self._render()

    def clear(self):
        self._frame = None
        self._photo = None
        self.label.config(image="", text=self.placeholder)

    def _on_resize(self, _event):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(60, self._render)

    def _render(self):
        self._job = None
        if self._frame is None:
            return
        width, height = self.winfo_width(), self.winfo_height()
        if width < 20 or height < 20:
            return
        # OpenCV uses BGR, Pillow/Tk expect RGB.
        image = Image.fromarray(cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB))
        scale = min(width / image.width, height / image.height, 1.0)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(size, Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(image)
        self.label.config(image=self._photo, text="")


# ----------------------------------------------------------- About text -----
ABOUT = [
    ("What is Face Detection?",
     "Face detection is a computer-vision task that finds where human faces are "
     "located in an image or video. The output is a set of rectangles (x, y, "
     "width, height) around each face. It does not tell you who the person is."),
    ("What is OpenCV?",
     "OpenCV (Open Source Computer Vision Library) is a free library for image and "
     "video processing. It provides camera access, image filters, object "
     "detection and much more, and works from Python through the cv2 module."),
    ("What is Haar Cascade?",
     "A Haar Cascade is a classic machine-learning detector (Viola-Jones, 2001). "
     "It is trained with many face and non-face images and stores simple "
     "light/dark patterns called Haar features in an XML file. This app uses "
     "haarcascade_frontalface_default.xml."),
    ("How face detection works",
     "1. The frame is converted to grayscale.\n"
     "2. detectMultiScale() slides a window over the image at many scales.\n"
     "3. Each window passes through a cascade of stages. Windows that fail an "
     "early stage are rejected quickly; only face-like windows pass all stages.\n"
     "4. Overlapping hits are merged (minNeighbors) and returned as rectangles.\n"
     "5. The app draws boxes and counts the faces.\n\n"
     "Parameters: scaleFactor = size step between scans (1.1), minNeighbors = "
     "how many overlapping hits confirm a face (5), minSize = smallest face "
     "in pixels (30 x 30)."),
    ("Face Detection vs Face Recognition",
     "Detection answers \"Is there a face, and where?\". Recognition answers "
     "\"Whose face is it?\" by comparing facial features against known people. "
     "Detection is usually the first step of a recognition system."),
    ("Real-world applications",
     "- CCTV surveillance\n- Security systems\n- Mobile cameras (auto-focus, "
     "portrait mode)\n- Attendance systems\n- Smart access control\n"
     "- Video monitoring\n- Human-computer interaction"),
]


class App(tk.Tk):
    def __init__(self, detector):
        super().__init__()
        self.title("Face Detection System")
        self.configure(bg=BG)
        self._center(1150, 720)
        self.minsize(960, 620)

        ensure_dirs()
        self.detector = detector
        self.camera = Camera(detector)
        self.image_detector = ImageDetector(detector)
        self.video = VideoProcessor(detector)

        # State for the image and video pages
        self.current_annotated = None
        self.current_image_name = ""
        self.video_path = None
        self.video_name = ""
        self._cam_job = None
        self._vid_job = None
        self.current_page = None
        self.pages = {}
        self.nav_buttons = {}

        self._setup_styles()
        self._build_layout()
        self._build_home_page()
        self._build_camera_page()
        self._build_image_page()
        self._build_video_page()
        self._build_history_page()
        self._build_about_page()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_page("home")

    # ------------------------------------------------------------ layout ----
    def _center(self, width, height):
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        width, height = min(width, screen_w - 80), min(height, screen_h - 120)
        x, y = (screen_w - width) // 2, (screen_h - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", rowheight=28, font=F_SMALL, background="white",
                        fieldbackground="white", foreground=TEXT, borderwidth=0)
        style.configure("Treeview.Heading", font=F_BTN, background=SIDEBAR,
                        foreground="white", relief="flat", padding=6)
        style.map("Treeview.Heading", background=[("active", SIDEBAR_HOVER)])
        style.map("Treeview", background=[("selected", ACCENT)])
        style.configure("Blue.Horizontal.TProgressbar", troughcolor="#cbd5e1",
                        background=ACCENT, borderwidth=0)

    def _build_layout(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Sidebar -----------------------------------------------------------
        side = tk.Frame(self, bg=SIDEBAR, width=230)
        side.grid(row=0, column=0, rowspan=2, sticky="ns")
        side.grid_propagate(False)
        tk.Label(side, text="\u25C9 Face Detection", bg=SIDEBAR, fg="white",
                 font=(FONT, 15, "bold")).pack(anchor="w", padx=20, pady=(26, 0))
        tk.Label(side, text="OpenCV  \u2022  Haar Cascade", bg=SIDEBAR, fg="#94a3b8",
                 font=F_SMALL).pack(anchor="w", padx=20, pady=(0, 24))

        items = [("home", "\u2302   Home"), ("camera", "\u25C9   Live Camera"),
                 ("image", "\u25A3   Image Detection"), ("video", "\u25B6   Video Detection"),
                 ("history", "\u2263   Detection History"), ("about", "\u24D8   About")]
        for key, label in items:
            self.nav_buttons[key] = self._nav_button(side, label,
                                                     lambda k=key: self.show_page(k))
            self.nav_buttons[key].pack(fill="x")
        exit_btn = self._nav_button(side, "\u2715   Exit", self.on_close)
        exit_btn.pack(fill="x", side="bottom", pady=(0, 16))

        # Strict mode: verifies each detection (eyes + skin colour) to avoid
        # mistaking objects for faces. Applies to all pages immediately.
        self.strict_var = tk.BooleanVar(value=self.detector.strict)
        tk.Checkbutton(side, text="Strict mode (fewer false\ndetections)",
                       variable=self.strict_var, command=self._toggle_strict,
                       bg=SIDEBAR, fg="white", selectcolor=SIDEBAR_HOVER,
                       activebackground=SIDEBAR, activeforeground="white",
                       font=F_SMALL, justify="left", anchor="w", bd=0,
                       highlightthickness=0, cursor="hand2").pack(
            side="bottom", fill="x", padx=18, pady=(0, 10))

        # Content area + status bar -----------------------------------------
        self.content = tk.Frame(self, bg=BG)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

        status = tk.Frame(self, bg="#e2e8f0")
        status.grid(row=1, column=1, sticky="ew")
        self.status_label = tk.Label(status, text="Ready", bg="#e2e8f0", fg=TEXT,
                                     font=F_SMALL, anchor="w", padx=14, pady=6)
        self.status_label.pack(side="left", fill="x", expand=True)
        self.busy_bar = ttk.Progressbar(status, mode="indeterminate", length=120,
                                        style="Blue.Horizontal.TProgressbar")

    def _toggle_strict(self):
        self.detector.strict = self.strict_var.get()
        self.set_status("Strict mode ON: faces are verified (fewer false detections)."
                        if self.detector.strict else
                        "Strict mode OFF: more sensitive, but objects may be detected.")

    def _nav_button(self, parent, text, command):
        btn = tk.Button(parent, text=text, command=command, anchor="w", bg=SIDEBAR,
                        fg="white", activebackground=ACCENT, activeforeground="white",
                        relief="flat", bd=0, font=(FONT, 11), padx=22, pady=11,
                        cursor="hand2")
        btn.bind("<Enter>", lambda e: btn["bg"] != ACCENT and btn.config(bg=SIDEBAR_HOVER))
        btn.bind("<Leave>", lambda e: btn["bg"] != ACCENT and btn.config(bg=SIDEBAR))
        return btn

    def _new_page(self, key, title, subtitle):
        page = tk.Frame(self.content, bg=BG)
        page.grid(row=0, column=0, sticky="nsew")
        self.pages[key] = page
        inner = tk.Frame(page, bg=BG, padx=26, pady=20)
        inner.pack(fill="both", expand=True)
        tk.Label(inner, text=title, font=F_HEAD, bg=BG, fg=TEXT).pack(anchor="w")
        tk.Label(inner, text=subtitle, font=F_SMALL, bg=BG, fg=MUTED).pack(
            anchor="w", pady=(0, 14))
        return inner

    # --------------------------------------------------- status & busy ------
    def set_status(self, text, level="info"):
        colors = {"info": TEXT, "ok": GREEN, "error": RED}
        self.status_label.config(text=text, fg=colors.get(level, TEXT))

    def show_busy(self, text):
        self.set_status(text)
        self.busy_bar.pack(side="right", padx=14)
        self.busy_bar.start(12)
        self.update_idletasks()

    def hide_busy(self):
        self.busy_bar.stop()
        self.busy_bar.pack_forget()

    # -------------------------------------------------------- navigation ----
    def show_page(self, key):
        if key != self.current_page:
            # Leaving a page stops anything running on it.
            if self.camera.running:
                self.stop_camera()
            if self.video.active:
                self.finish_video(completed=False, quiet=True)
        self.current_page = key
        self.pages[key].tkraise()
        for name, btn in self.nav_buttons.items():
            btn.config(bg=ACCENT if name == key else SIDEBAR)
        if key == "history":
            self.refresh_history()
        self.set_status("Ready")

    # ------------------------------------------------------------- home -----
    def _build_home_page(self):
        page = tk.Frame(self.content, bg=BG)
        page.grid(row=0, column=0, sticky="nsew")
        self.pages["home"] = page
        inner = tk.Frame(page, bg=BG, padx=34, pady=30)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text="Face Detection System", font=F_TITLE, bg=BG,
                 fg=TEXT).pack(anchor="w")
        tk.Label(inner, wraplength=760, justify="left", font=F_BODY, bg=BG, fg=MUTED,
                 text="Detect human faces in live webcam video, photos and video "
                      "files using OpenCV and the Haar Cascade classifier. Every "
                      "detection is stored in the history so you can review it "
                      "later.").pack(anchor="w", pady=(6, 24))

        cards = [
            ("camera", "\u25C9", "Live Camera Detection", "Real-time detection from your webcam."),
            ("image", "\u25A3", "Image Detection", "Find faces in JPG, PNG or BMP images."),
            ("video", "\u25B6", "Video Detection", "Process MP4, AVI, MOV or MKV videos."),
            ("history", "\u2263", "Detection History", "Review every past detection."),
            ("about", "\u24D8", "About", "Learn how face detection works."),
            ("exit", "\u2715", "Exit", "Close the application."),
        ]
        grid = tk.Frame(inner, bg=BG)
        grid.pack(fill="both", expand=True)
        for col in range(3):
            grid.columnconfigure(col, weight=1, uniform="card")
        for row in range(2):
            grid.rowconfigure(row, weight=1, uniform="card")

        for i, (key, icon, title, desc) in enumerate(cards):
            card = tk.Frame(grid, bg=CARD, padx=18, pady=16,
                            highlightbackground="#e2e8f0", highlightthickness=1)
            card.grid(row=i // 3, column=i % 3, sticky="nsew", padx=8, pady=8)
            tk.Label(card, text=icon, font=(FONT, 26), bg=CARD,
                     fg=RED if key == "exit" else ACCENT).pack(anchor="w")
            tk.Label(card, text=title, font=(FONT, 12, "bold"), bg=CARD,
                     fg=TEXT).pack(anchor="w", pady=(4, 0))
            tk.Label(card, text=desc, font=F_SMALL, bg=CARD, fg=MUTED, wraplength=200,
                     justify="left").pack(anchor="w", pady=(2, 10))
            action = self.on_close if key == "exit" else (lambda k=key: self.show_page(k))
            FlatButton(card, "Open" if key != "exit" else "Exit", action,
                       RED if key == "exit" else ACCENT).pack(anchor="w", side="bottom")

    # ----------------------------------------------------------- camera -----
    def _build_camera_page(self):
        page = self._new_page("camera", "Live Camera Detection",
                              "Real-time face detection from your webcam.")
        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", pady=(0, 10))
        self.btn_cam_start = FlatButton(bar, "\u25B6  Start Camera", self.start_camera, GREEN)
        self.btn_cam_stop = FlatButton(bar, "\u25A0  Stop Camera", self.stop_camera, RED)
        self.btn_cam_capture = FlatButton(bar, "\u25CE  Capture Image", self.capture_image, ACCENT)
        for btn in (self.btn_cam_start, self.btn_cam_stop, self.btn_cam_capture):
            btn.pack(side="left", padx=(0, 8))
        self.btn_cam_stop.set_enabled(False)
        self.btn_cam_capture.set_enabled(False)

        self.cam_faces_var = tk.StringVar(value="Faces: 0")
        self.cam_fps_var = tk.StringVar(value="FPS: 0.0")
        for var in (self.cam_fps_var, self.cam_faces_var):
            tk.Label(bar, textvariable=var, font=(FONT, 12, "bold"), bg=BG,
                     fg=TEXT).pack(side="right", padx=(16, 0))

        self.cam_view = ImageView(page, "Camera is off.\nPress \u201CStart Camera\u201D.")
        self.cam_view.pack(fill="both", expand=True)

    def start_camera(self):
        if self.camera.running:
            return
        self.show_busy("Opening camera...")
        try:
            self.camera.start()
        except DetectionError as exc:
            self.hide_busy()
            self.set_status("Camera could not be started.", "error")
            messagebox.showerror("Camera error", str(exc))
            return
        self.hide_busy()
        self.btn_cam_start.set_enabled(False)
        self.btn_cam_stop.set_enabled(True)
        self.btn_cam_capture.set_enabled(True)
        self.set_status("Camera running.", "ok")
        self._camera_tick()

    def _camera_tick(self):
        """Called repeatedly while the camera is on (Tk's replacement for a loop)."""
        if not self.camera.running:
            return
        try:
            frame, count, fps = self.camera.read()
        except DetectionError as exc:
            self.stop_camera()
            messagebox.showerror("Camera error", str(exc))
            return
        self.cam_view.show(frame)
        self.cam_faces_var.set(f"Faces: {count}")
        self.cam_fps_var.set(f"FPS: {fps:.1f}")
        self._cam_job = self.after(10, self._camera_tick)

    def stop_camera(self):
        if self._cam_job:
            self.after_cancel(self._cam_job)
            self._cam_job = None
        if self.camera.running:
            summary = self.camera.stop()              # releases the webcam
            if summary["frames"] > 0:
                self._safe_history("Live Camera", "Webcam session", summary["max_faces"])
        self.btn_cam_start.set_enabled(True)
        self.btn_cam_stop.set_enabled(False)
        self.btn_cam_capture.set_enabled(False)
        self.cam_view.clear()
        self.cam_faces_var.set("Faces: 0")
        self.cam_fps_var.set("FPS: 0.0")
        self.set_status("Camera stopped.")

    def capture_image(self):
        try:
            path, count = self.camera.capture()
        except DetectionError as exc:
            messagebox.showerror("Capture error", str(exc))
            return
        self._safe_history("Camera Capture", os.path.basename(path), count)
        self.set_status(f"Image captured: {path}", "ok")

    # ------------------------------------------------------------ image -----
    def _build_image_page(self):
        page = self._new_page("image", "Image Face Detection",
                              "Supported formats: JPG, JPEG, PNG, BMP.")
        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", pady=(0, 10))
        FlatButton(bar, "\u25A3  Select Image", self.select_image, ACCENT).pack(side="left", padx=(0, 8))
        self.btn_img_save = FlatButton(bar, "\u2B07  Save Result", self.save_image_result, GREEN)
        self.btn_img_save.pack(side="left")
        self.btn_img_save.set_enabled(False)

        self.img_faces_var = tk.StringVar(value="Faces Detected: 0")
        tk.Label(bar, textvariable=self.img_faces_var, font=(FONT, 13, "bold"), bg=BG,
                 fg=TEXT).pack(side="right")

        self.img_view = ImageView(page, "No image loaded.\nPress \u201CSelect Image\u201D.")
        self.img_view.pack(fill="both", expand=True)

    def select_image(self):
        patterns = " ".join(f"*{e}" for e in IMAGE_EXTS)
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", patterns), ("All files", "*.*")])
        if not path:
            messagebox.showwarning("No selection", "No image was selected.")
            return

        self.show_busy("Detecting faces...")
        try:
            original = self.image_detector.load(path)
            annotated, count = self.image_detector.detect(original)
        except DetectionError as exc:
            self.hide_busy()
            self.set_status("Image could not be processed.", "error")
            messagebox.showerror("Image error", str(exc))
            return
        self.hide_busy()

        self.current_annotated = annotated
        self.current_image_name = os.path.basename(path)
        self.img_view.show(annotated)
        self.img_faces_var.set(f"Faces Detected: {count}")
        self.btn_img_save.set_enabled(True)
        self._safe_history("Image", self.current_image_name, count)
        self.set_status(f"{self.current_image_name}: {count} face(s) detected.", "ok")

    def save_image_result(self):
        try:
            path = self.image_detector.save(self.current_annotated, self.current_image_name)
        except DetectionError as exc:
            messagebox.showerror("Save error", str(exc))
            return
        self.set_status(f"Result saved: {path}", "ok")
        messagebox.showinfo("Saved", f"Result saved to:\n{path}")

    # ------------------------------------------------------------ video -----
    def _build_video_page(self):
        page = self._new_page("video", "Video Face Detection",
                              "Supported formats: MP4, AVI, MOV, MKV. The processed "
                              "video is saved in the results folder.")
        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", pady=(0, 8))
        self.btn_vid_select = FlatButton(bar, "\u25B6  Select Video", self.select_video, ACCENT)
        self.btn_vid_start = FlatButton(bar, "Start Processing", self.start_video, GREEN)
        self.btn_vid_stop = FlatButton(bar, "\u25A0  Stop Processing", self.stop_video, RED)
        for btn in (self.btn_vid_select, self.btn_vid_start, self.btn_vid_stop):
            btn.pack(side="left", padx=(0, 8))
        self.btn_vid_start.set_enabled(False)
        self.btn_vid_stop.set_enabled(False)

        self.vid_faces_var = tk.StringVar(value="Faces: 0")
        tk.Label(bar, textvariable=self.vid_faces_var, font=(FONT, 12, "bold"), bg=BG,
                 fg=TEXT).pack(side="right")

        info = tk.Frame(page, bg=BG)
        info.pack(fill="x")
        self.vid_name_var = tk.StringVar(value="No video selected")
        self.vid_frames_var = tk.StringVar(value="")
        tk.Label(info, textvariable=self.vid_name_var, font=F_SMALL, bg=BG,
                 fg=MUTED).pack(side="left")
        tk.Label(info, textvariable=self.vid_frames_var, font=F_SMALL, bg=BG,
                 fg=MUTED).pack(side="right")

        self.vid_progress = ttk.Progressbar(page, mode="determinate", maximum=100,
                                            style="Blue.Horizontal.TProgressbar")
        self.vid_progress.pack(fill="x", pady=(4, 10))

        self.vid_view = ImageView(page, "No video loaded.\nPress \u201CSelect Video\u201D.")
        self.vid_view.pack(fill="both", expand=True)

    def select_video(self):
        patterns = " ".join(f"*{e}" for e in VIDEO_EXTS)
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Videos", patterns), ("All files", "*.*")])
        if not path:
            messagebox.showwarning("No selection", "No video was selected.")
            return
        try:
            self.video.validate(path)
        except DetectionError as exc:
            self.set_status("Invalid video.", "error")
            messagebox.showerror("Video error", str(exc))
            return
        self.video_path = path
        self.video_name = os.path.basename(path)
        self.vid_name_var.set(f"Selected: {self.video_name}")
        self.vid_frames_var.set("")
        self.vid_progress["value"] = 0
        self.vid_faces_var.set("Faces: 0")
        self.vid_view.clear()
        self.btn_vid_start.set_enabled(True)
        self.set_status("Video selected. Press Start Processing.")

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No video", "Please select a video first.")
            return
        self.show_busy("Preparing video...")
        try:
            self.video.open(self.video_path)
        except DetectionError as exc:
            self.hide_busy()
            self.set_status("Video could not be opened.", "error")
            messagebox.showerror("Video error", str(exc))
            return
        self.hide_busy()
        self.btn_vid_select.set_enabled(False)
        self.btn_vid_start.set_enabled(False)
        self.btn_vid_stop.set_enabled(True)
        self.set_status("Processing video...")
        self._video_tick()

    def _video_tick(self):
        if not self.video.active:
            return
        try:
            result = self.video.step()
        except DetectionError as exc:
            self.finish_video(completed=False, quiet=True)
            messagebox.showerror("Video error", str(exc))
            return
        if result is None:                       # end of the video
            self.finish_video(completed=True)
            return
        frame, count = result
        self.vid_view.show(frame)
        self.vid_faces_var.set(f"Faces: {count}")
        self.vid_progress["value"] = self.video.progress * 100
        total = self.video.total_frames or "?"
        self.vid_frames_var.set(f"Frame {self.video.frames} / {total}")
        self._vid_job = self.after(1, self._video_tick)

    def stop_video(self):
        self.finish_video(completed=False)

    def finish_video(self, completed, quiet=False):
        """Close reader/writer, log to history and report what happened."""
        if self._vid_job:
            self.after_cancel(self._vid_job)
            self._vid_job = None
        summary = self.video.close()
        self.btn_vid_select.set_enabled(True)
        self.btn_vid_start.set_enabled(self.video_path is not None)
        self.btn_vid_stop.set_enabled(False)
        if completed:
            self.vid_progress["value"] = 100
        if not summary or summary["frames"] == 0:
            self.set_status("Video processing stopped.")
            return
        self._safe_history("Video", self.video_name, summary["max_faces"])
        self.set_status("Video finished." if completed else "Processing stopped.", "ok")
        if not quiet:
            messagebox.showinfo(
                "Video processed" if completed else "Processing stopped",
                f"Frames processed: {summary['frames']}\n"
                f"Most faces in one frame: {summary['max_faces']}\n\n"
                f"Saved to:\n{summary['output']}")

    # ---------------------------------------------------------- history -----
    def _build_history_page(self):
        page = self._new_page("history", "Detection History",
                              "Every detection is saved to history/history.json.")
        bar = tk.Frame(page, bg=BG)
        bar.pack(fill="x", pady=(0, 10))
        FlatButton(bar, "\u21BB  Refresh History", self.refresh_history, ACCENT).pack(side="left", padx=(0, 8))
        FlatButton(bar, "\u2715  Clear History", self.clear_history, RED).pack(side="left")
        self.hist_count_var = tk.StringVar(value="0 records")
        tk.Label(bar, textvariable=self.hist_count_var, font=F_SMALL, bg=BG,
                 fg=MUTED).pack(side="right")

        holder = tk.Frame(page, bg=BG)
        holder.pack(fill="both", expand=True)
        columns = ("date", "time", "type", "file", "faces")
        self.tree = ttk.Treeview(holder, columns=columns, show="headings")
        widths = {"date": 110, "time": 90, "type": 130, "file": 320, "faces": 70}
        for col in columns:
            self.tree.heading(col, text=col.capitalize())
            self.tree.column(col, width=widths[col],
                             anchor="w" if col == "file" else "center")
        scroll = ttk.Scrollbar(holder, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def refresh_history(self):
        self.tree.delete(*self.tree.get_children())
        entries = history.read_history()
        for e in reversed(entries):              # newest first
            self.tree.insert("", "end", values=(
                e.get("date", ""), e.get("time", ""), e.get("type", ""),
                e.get("file", ""), e.get("faces", 0)))
        self.hist_count_var.set(f"{len(entries)} record(s)")
        self.set_status("History refreshed.")

    def clear_history(self):
        if not messagebox.askyesno("Clear history",
                                   "Delete all detection history? This cannot be undone."):
            return
        try:
            history.clear_history()
        except DetectionError as exc:
            messagebox.showerror("History error", str(exc))
            return
        self.refresh_history()
        self.set_status("History cleared.", "ok")

    def _safe_history(self, kind, file_name, faces):
        """Save a history entry; a disk problem must never crash detection."""
        try:
            history.add_entry(kind, file_name, faces)
        except DetectionError as exc:
            self.set_status(str(exc).splitlines()[0], "error")

    # ------------------------------------------------------------ about -----
    def _build_about_page(self):
        page = self._new_page("about", "About", "Face detection concepts and applications.")
        holder = tk.Frame(page, bg=CARD, highlightbackground="#e2e8f0", highlightthickness=1)
        holder.pack(fill="both", expand=True)
        text = tk.Text(holder, wrap="word", bg=CARD, fg=TEXT, font=F_BODY, relief="flat",
                       padx=22, pady=16, spacing3=4, cursor="arrow")
        scroll = ttk.Scrollbar(holder, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        text.tag_configure("h", font=(FONT, 13, "bold"), foreground=ACCENT,
                           spacing1=14, spacing3=4)
        for heading, body in ABOUT:
            text.insert("end", heading + "\n", "h")
            text.insert("end", body + "\n")
        text.config(state="disabled")             # read-only

    # ------------------------------------------------------------- exit -----
    def on_close(self):
        """Release the webcam and any open video before quitting."""
        if self.camera.running:
            self.stop_camera()
        if self.video.active:
            self.finish_video(completed=False, quiet=True)
        self.destroy()


def main():
    # Load the cascade first so a missing file gives a clear message box.
    probe = tk.Tk()
    probe.withdraw()
    try:
        detector = FaceDetector()
    except DetectionError as exc:
        messagebox.showerror("Startup error", str(exc))
        probe.destroy()
        return
    probe.destroy()

    App(detector).mainloop()


if __name__ == "__main__":
    main()
