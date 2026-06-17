"""
eye.py — Production-grade YOLOv8 Object Detection
==================================================
Features
--------
- Full context-manager protocol on every public class
- GUI mode  : CustomTkinter dark-theme window with live controls
- Headless  : simple_detect() OpenCV window, no GUI dependency
- Hot-swap  : switch YOLO model variant without restarting
- Class filter, confidence slider, heatmap overlay, detection log
- Video recording + screenshot with lock-safe frame access
- FPS counter (EMA, last 30 frames)
- Config persistence to ~/.yolov8_app.json
- Thread-safe design: all widget mutations via self.after()
- Legacy aliases: EYE() → simple_detect(), OpenEYE() → launch_gui()

Context-manager patterns
------------------------
# GUI (blocks until window closed)
with EyeSession.gui() as session:
    session.run()
    print(session.last_detections)

# Headless (blocks until target seen or 'q')
with EyeSession.headless(target_class="person") as session:
    detected = session.run()

# Low-level components
with CameraManager(0) as cam:
    with ObjectDetector.from_model("yolov8n.pt") as det:
        ret, frame = cam.read()
        classes, annotated, count, confs = det.detect(frame)

# One-liner functional API
detected = simple_detect("person")
launch_gui()

Bug fixes over original
-----------------------
1. Thread-safety  : widget mutations dispatched via self.after(0, fn).
2. CPU spin       : paused branch sleeps 33 ms instead of hot-looping.
3. Annotation mismatch: inference on 416×416, boxes scaled back to original.
4. Recording corruption: writer always receives original-res annotated frame.
5. Invalid CTk colour  : "darkred" → "#8B0000".
6. Raw-frame race       : screenshot + video thread share frame under Lock.
7. No thread join       : destroy() waits for video thread (2 s timeout).
8. Unbounded FPS        : target-FPS cap via time.sleep.
9. names KeyError       : model.names.get(id, fallback) guard.
10. macOS imshow        : documented; simple_detect() guards caller.
"""

from __future__ import annotations

import contextlib
import cv2
import json
import logging
import importlib.resources
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from queue import Queue, Empty
from types import TracebackType
from typing import (
    Deque, Dict, Generator, List, Optional, Set, Tuple, Type
)

import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_CONFIG_PATH = Path.home() / ".yolov8_app.json"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    """
    Serialisable configuration for all detection parameters.

    Persisted automatically to ~/.yolov8_app.json between sessions.
    """

    model_name:           str              = "yolov8n.pt"
    target_class:         str              = "person"
    confidence_threshold: float            = 0.5
    frame_width:          int              = 640
    frame_height:         int              = 480
    detection_size:       Tuple[int, int]  = (416, 416)
    camera_index:         int              = 0
    target_fps:           int              = 30
    show_heatmap:         bool             = False
    filter_classes:       List[str]        = field(default_factory=list)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path = _CONFIG_PATH) -> None:
        try:
            d = asdict(self)
            d["detection_size"] = list(d["detection_size"])
            path.write_text(json.dumps(d, indent=2))
        except OSError as exc:
            logger.warning("Could not save config: %s", exc)

    @classmethod
    def load(cls, path: Path = _CONFIG_PATH) -> "DetectionConfig":
        try:
            d = json.loads(path.read_text())
            d["detection_size"] = tuple(d.get("detection_size", [416, 416]))
            d["filter_classes"] = d.get("filter_classes", [])
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        except Exception:
            return cls()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class ModelLoader:
    """Loads YOLO models with four cascading fallback strategies."""

    @staticmethod
    def load_model(model_name: str = "yolov8n.pt") -> Optional[YOLO]:
        strategies = [
            ModelLoader._load_from_package,
            ModelLoader._load_from_script_dir,
            ModelLoader._load_from_current_dir,
            ModelLoader._download_model,
        ]
        for strategy in strategies:
            try:
                model = strategy(model_name)
                if model is not None:
                    logger.info("Loaded model via %s", strategy.__name__)
                    return model
            except Exception as exc:
                logger.debug("Strategy %s failed: %s", strategy.__name__, exc)
        logger.error("All model loading strategies exhausted for '%s'", model_name)
        return None

    @staticmethod
    def _load_from_package(model_name: str) -> Optional[YOLO]:
        pkg = __package__ or ""
        if pkg:
            try:
                path = importlib.resources.files(pkg) / model_name
                if Path(str(path)).exists():
                    return YOLO(str(path))
            except Exception:
                pass
        return None

    @staticmethod
    def _load_from_script_dir(model_name: str) -> Optional[YOLO]:
        base = Path(__file__).parent
        for candidate in [base / model_name, base / "eye" / model_name, base.parent / model_name]:
            if candidate.exists():
                return YOLO(str(candidate))
        return None

    @staticmethod
    def _load_from_current_dir(model_name: str) -> Optional[YOLO]:
        p = Path(model_name)
        return YOLO(str(p)) if p.exists() else None

    @staticmethod
    def _download_model(model_name: str) -> Optional[YOLO]:
        logger.info("Downloading %s from Ultralytics …", model_name)
        return YOLO(model_name)


# ---------------------------------------------------------------------------
# CameraManager  —  context-manager aware
# ---------------------------------------------------------------------------

class CameraManager:
    """
    Manages webcam capture with resource-safe open/release.

    Context-manager usage (recommended)
    ------------------------------------
    with CameraManager(device=0, width=640, height=480) as cam:
        ret, frame = cam.read()

    Manual usage
    ------------
    cam = CameraManager(0)
    cam.open()
    ret, frame = cam.read()
    cam.release()
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "CameraManager":
        if not self.open():
            raise RuntimeError(f"Cannot open camera device {self.camera_index}")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self.release()
        return False  # never suppress exceptions

    @classmethod
    @contextmanager
    def open_device(
        cls, camera_index: int = 0, width: int = 640, height: int = 480
    ) -> Generator["CameraManager", None, None]:
        """
        Convenience factory context manager.

        with CameraManager.open_device(0) as cam:
            ret, frame = cam.read()
        """
        cam = cls(camera_index, width, height)
        try:
            if not cam.open():
                raise RuntimeError(f"Cannot open camera device {camera_index}")
            yield cam
        finally:
            cam.release()

    # ── public API ────────────────────────────────────────────────────────────

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error("Failed to open camera %d", self.camera_index)
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera %d opened at %dx%d", self.camera_index, actual_w, actual_h)
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap and self.cap.isOpened():
            return self.cap.read()
        return False, None

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info("Camera %d released", self.camera_index)

    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def __del__(self) -> None:
        self.release()


# ---------------------------------------------------------------------------
# ObjectDetector  —  context-manager aware
# ---------------------------------------------------------------------------

class ObjectDetector:
    """
    Runs YOLOv8 inference; overlays bounding boxes at original resolution.

    Context-manager usage
    ---------------------
    with ObjectDetector.from_model("yolov8n.pt", config) as det:
        classes, annotated, count, confs = det.detect(frame)

    Manual usage
    ------------
    det = ObjectDetector(model, config)
    classes, annotated, count, confs = det.detect(frame)
    """

    def __init__(self, model: YOLO, config: DetectionConfig):
        self.model = model
        self.config = config
        self._heatmap: Optional[np.ndarray] = None

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "ObjectDetector":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self._heatmap = None   # release heatmap accumulator memory
        return False

    @classmethod
    @contextmanager
    def from_model(
        cls,
        model_name: str = "yolov8n.pt",
        config: Optional[DetectionConfig] = None,
    ) -> Generator["ObjectDetector", None, None]:
        """
        Convenience factory context manager.

        with ObjectDetector.from_model("yolov8n.pt") as det:
            classes, annotated, count, confs = det.detect(frame)
        """
        cfg = config or DetectionConfig()
        model = ModelLoader.load_model(model_name)
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_name}")
        det = cls(model, cfg)
        try:
            yield det
        finally:
            det._heatmap = None

    # ── inference ─────────────────────────────────────────────────────────────

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[Set[str], np.ndarray, int, Dict[str, float]]:
        """
        Run YOLOv8 on *frame* and return annotated full-resolution output.

        Returns
        -------
        detected_classes  : set[str]
        annotated_frame   : np.ndarray  (same H×W as input, BGR)
        object_count      : int
        class_confidences : dict[class_name → max confidence]
        """
        h_orig, w_orig = frame.shape[:2]

        # Inference on small frame
        det_w, det_h = self.config.detection_size
        small = cv2.resize(frame, (det_w, det_h))
        results = self.model(small, conf=self.config.confidence_threshold, verbose=False)[0]

        # Scale factors back to original resolution
        sx, sy = w_orig / det_w, h_orig / det_h

        detected_classes: Set[str] = set()
        class_confidences: Dict[str, float] = {}
        annotated = frame.copy()
        filter_set = set(self.config.filter_classes)

        for box in results.boxes:
            cls_id     = int(box.cls[0])
            class_name = self.model.names.get(cls_id, f"class_{cls_id}")
            conf       = float(box.conf[0])

            if filter_set and class_name not in filter_set:
                continue

            detected_classes.add(class_name)
            class_confidences[class_name] = max(class_confidences.get(class_name, 0.0), conf)

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, x2 = int(x1 * sx), int(x2 * sx)
            y1, y2 = int(y1 * sy), int(y2 * sy)

            color = self._class_color(cls_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )

            if self.config.show_heatmap:
                if self._heatmap is None or self._heatmap.shape[:2] != (h_orig, w_orig):
                    self._heatmap = np.zeros((h_orig, w_orig, 3), dtype=np.float32)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                r = max(1, min(x2 - x1, y2 - y1) // 2)
                cv2.circle(self._heatmap, (cx, cy), r, (0, 0, 80), -1)

        if self.config.show_heatmap and self._heatmap is not None:
            self._heatmap *= 0.95
            heat_u8 = np.clip(self._heatmap, 0, 255).astype(np.uint8)
            cv2.addWeighted(annotated, 1.0, heat_u8, 0.4, 0, annotated)

        return detected_classes, annotated, len(results.boxes), class_confidences

    @staticmethod
    def _class_color(cls_id: int) -> Tuple[int, int, int]:
        palette = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
            (207, 210, 49), (72, 249, 10),  (146, 204, 23), (61, 219, 134),
            (26, 147, 52),  (0, 212, 187),  (44, 153, 168), (0, 194, 255),
            (52, 69, 147),  (100, 115, 255),(0, 24, 236),   (132, 56, 255),
            (82, 0, 133),   (203, 56, 255), (255, 149, 200),(255, 55, 199),
        ]
        r, g, b = palette[cls_id % len(palette)]
        return (b, g, r)


# ---------------------------------------------------------------------------
# FramePacket
# ---------------------------------------------------------------------------

@dataclass
class FramePacket:
    rgb_image:        Image.Image
    detected_classes: Set[str]
    class_confidences: Dict[str, float]
    obj_count:        int
    fps:              float
    raw_frame:        np.ndarray


# ---------------------------------------------------------------------------
# EyeSession  —  unified context-manager facade
# ---------------------------------------------------------------------------

class EyeSession:
    """
    High-level context manager that owns the full lifecycle: model load →
    camera open → run → camera release → config save.

    GUI mode (blocks until window closed)
    -------------------------------------
    with EyeSession.gui() as session:
        session.run()
        print(session.last_detections)

    Headless mode (blocks until target seen or 'q' pressed)
    --------------------------------------------------------
    with EyeSession.headless(target_class="person") as session:
        detected = session.run()

    Custom config
    -------------
    cfg = DetectionConfig(model_name="yolov8s.pt", show_heatmap=True)
    with EyeSession.gui(config=cfg) as session:
        session.run()
    """

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        *,
        mode: str = "gui",          # "gui" | "headless"
        target_class: str = "person",
    ) -> None:
        self.config       = config or DetectionConfig.load()
        self.mode         = mode
        self.target_class = target_class
        self.last_detections: List[str] = []

        self._app: Optional[DetectionApp] = None

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "EyeSession":
        logger.info("EyeSession.__enter__ (mode=%s)", self.mode)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        logger.info("EyeSession.__exit__")
        if self._app is not None:
            with contextlib.suppress(Exception):
                self._app._close_app()
            self._app = None
        return False

    # ── factory class-methods ─────────────────────────────────────────────────

    @classmethod
    @contextmanager
    def gui(
        cls,
        config: Optional[DetectionConfig] = None,
    ) -> Generator["EyeSession", None, None]:
        """
        Context manager that launches the full CustomTkinter GUI.

        with EyeSession.gui() as session:
            session.run()        # blocks until window closed
        """
        session = cls(config=config, mode="gui")
        try:
            yield session
        finally:
            if session._app is not None:
                with contextlib.suppress(Exception):
                    session._app._close_app()

    @classmethod
    @contextmanager
    def headless(
        cls,
        target_class: str = "person",
        config: Optional[DetectionConfig] = None,
    ) -> Generator["EyeSession", None, None]:
        """
        Context manager for headless (no GUI) detection.

        with EyeSession.headless("car") as session:
            detected = session.run()   # blocks; returns list[str]
        """
        session = cls(config=config, mode="headless", target_class=target_class)
        try:
            yield session
        finally:
            cv2.destroyAllWindows()

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self) -> List[str]:
        """
        Start the session.  Blocks until the window is closed (GUI) or the
        target class is detected / 'q' is pressed (headless).

        Returns
        -------
        list[str]  All class names detected at the time of exit.
        """
        if self.mode == "gui":
            self._run_gui()
        else:
            self.last_detections = simple_detect(
                target_class=self.target_class,
                config=self.config,
            )
        return self.last_detections

    def _run_gui(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self._app = DetectionApp(self.config)
        self._app.mainloop()


# ---------------------------------------------------------------------------
# DetectionApp GUI
# ---------------------------------------------------------------------------

class DetectionApp(ctk.CTk):
    """
    Main CustomTkinter application window.

    Thread model
    ------------
    Main thread  : tkinter event loop + all widget mutations (via self.after).
    Video thread : camera read → YOLO inference → FramePacket → Queue(maxsize=2).
    Main thread  : polls queue at ~60 Hz via self.after(16, _poll_queue).
    """

    MODEL_VARIANTS = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
    ]

    # ── context manager (wraps mainloop) ─────────────────────────────────────

    def __enter__(self) -> "DetectionApp":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        with contextlib.suppress(Exception):
            self._close_app()
        return False

    @classmethod
    @contextmanager
    def run_app(
        cls,
        config: Optional[DetectionConfig] = None,
    ) -> Generator["DetectionApp", None, None]:
        """
        Context manager that builds, runs, and tears down the GUI.

        with DetectionApp.run_app(config) as app:
            app.mainloop()     # blocks
        """
        cfg = config or DetectionConfig.load()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        app = cls(cfg)
        try:
            yield app
        finally:
            with contextlib.suppress(Exception):
                app._close_app()

    # ── constructor ───────────────────────────────────────────────────────────

    def __init__(self, config: DetectionConfig) -> None:
        super().__init__()
        self.config = config

        self.running   = False
        self.paused    = False
        self.recording = False

        self._frame_lock       = threading.Lock()
        self._current_raw_frame: Optional[np.ndarray] = None
        self._frame_queue: Queue[FramePacket] = Queue(maxsize=2)
        self._video_thread: Optional[threading.Thread] = None

        self.frame_count     = 0
        self.detection_count = 0
        self._fps_times: Deque[float]  = deque(maxlen=30)
        self._detection_log: Deque[str] = deque(maxlen=200)

        self.model:        Optional[YOLO]              = None
        self.camera:       Optional[CameraManager]     = None
        self.detector:     Optional[ObjectDetector]    = None
        self.video_writer: Optional[cv2.VideoWriter]   = None

        self._setup_ui()
        self._setup_keyboard_shortcuts()
        self._load_components()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        self.title("YOLOv8 Object Detection")
        self.geometry("1060x780")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._close_app)

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        # ── left panel (video) ────────────────────────────────────────────────
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(left, text="Initialising…", width=640, height=480)
        self.video_label.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")

        stat_row = ctk.CTkFrame(left, fg_color="transparent")
        stat_row.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))

        self.fps_label = ctk.CTkLabel(stat_row, text="FPS: --",    font=("Courier New", 12, "bold"))
        self.fps_label.pack(side="left", padx=8)
        self.obj_label = ctk.CTkLabel(stat_row, text="Objects: 0", font=("Courier New", 12))
        self.obj_label.pack(side="left", padx=8)
        self.frame_label = ctk.CTkLabel(stat_row, text="Frames: 0", font=("Courier New", 12))
        self.frame_label.pack(side="left", padx=8)
        self.detected_label = ctk.CTkLabel(
            stat_row, text="Detected: —",
            font=("Courier New", 13, "bold"), text_color="#00CFFF",
        )
        self.detected_label.pack(side="right", padx=8)

        # ── right panel (controls + log) ─────────────────────────────────────
        right = ctk.CTkScrollableFrame(self, width=300)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 5))

        self._section(right, "⚙  Model")
        self.model_var = ctk.StringVar(value=self.config.model_name)
        ctk.CTkOptionMenu(
            right, values=self.MODEL_VARIANTS,
            variable=self.model_var, command=self._on_model_change,
        ).pack(fill="x", padx=8, pady=4)

        self._section(right, "🎯  Confidence")
        conf_row = ctk.CTkFrame(right, fg_color="transparent")
        conf_row.pack(fill="x", padx=8, pady=2)
        self.confidence_slider = ctk.CTkSlider(
            conf_row, from_=0.1, to=0.95, number_of_steps=85,
            command=self._update_confidence,
        )
        self.confidence_slider.set(self.config.confidence_threshold)
        self.confidence_slider.pack(side="left", fill="x", expand=True)
        self.conf_val_label = ctk.CTkLabel(
            conf_row, text=f"{self.config.confidence_threshold:.2f}", width=40
        )
        self.conf_val_label.pack(side="right", padx=4)

        self._section(right, "🔍  Class Filter")
        self.filter_entry = ctk.CTkEntry(right, placeholder_text="Add class (e.g. car)")
        self.filter_entry.pack(fill="x", padx=8, pady=4)
        self.filter_entry.bind("<Return>", self._add_filter_class)
        ctk.CTkButton(right, text="Add Filter", command=self._add_filter_class, height=28).pack(
            fill="x", padx=8, pady=2
        )
        ctk.CTkButton(
            right, text="Clear Filter (show all)", command=self._clear_filter,
            height=28, fg_color="#444",
        ).pack(fill="x", padx=8, pady=2)
        self.filter_tags_frame = ctk.CTkFrame(right, fg_color="#2a2a2a")
        self.filter_tags_frame.pack(fill="x", padx=8, pady=4)
        self._refresh_filter_tags()

        self._section(right, "🛠  Options")
        self.heatmap_var = ctk.BooleanVar(value=self.config.show_heatmap)
        ctk.CTkCheckBox(
            right, text="Heat-map overlay",
            variable=self.heatmap_var, command=self._toggle_heatmap,
        ).pack(anchor="w", padx=8, pady=4)

        self._section(right, "🎮  Controls")
        btn_cfg: dict = dict(height=36, font=("Courier New", 13, "bold"))

        self.pause_btn = ctk.CTkButton(
            right, text="⏸  Pause  [Space]", command=self._toggle_pause, **btn_cfg
        )
        self.pause_btn.pack(fill="x", padx=8, pady=3)

        self.screenshot_btn = ctk.CTkButton(
            right, text="📷  Screenshot  [S]", command=self._take_screenshot, **btn_cfg
        )
        self.screenshot_btn.pack(fill="x", padx=8, pady=3)

        self.record_btn = ctk.CTkButton(
            right, text="⏺  Record  [R]", command=self._toggle_recording,
            fg_color="#8B0000", **btn_cfg,
        )
        self.record_btn.pack(fill="x", padx=8, pady=3)

        ctk.CTkButton(
            right, text="✖  Close  [Q]", command=self._close_app,
            fg_color="#333", **btn_cfg,
        ).pack(fill="x", padx=8, pady=3)

        self._section(right, "📋  Detection Log")
        self.log_box = ctk.CTkTextbox(
            right, height=180, font=("Courier New", 10), state="disabled"
        )
        self.log_box.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(
            right, text="Clear Log", height=24, fg_color="#333", command=self._clear_log,
        ).pack(fill="x", padx=8, pady=2)

        # ── status bar ────────────────────────────────────────────────────────
        status_bar = ctk.CTkFrame(self, height=28, fg_color="#1a1a1a")
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 6))

        self.status_label = ctk.CTkLabel(
            status_bar, text="Initialising…",
            font=("Courier New", 11), anchor="w",
        )
        self.status_label.pack(side="left", padx=10)

        self.rec_indicator = ctk.CTkLabel(
            status_bar, text="", font=("Courier New", 11), text_color="#FF4444"
        )
        self.rec_indicator.pack(side="right", padx=10)

    @staticmethod
    def _section(parent: ctk.CTkScrollableFrame, title: str) -> None:
        ctk.CTkLabel(parent, text=title, font=("Courier New", 13, "bold"), anchor="w").pack(
            fill="x", padx=8, pady=(10, 2)
        )
        ctk.CTkFrame(parent, height=1, fg_color="#444").pack(fill="x", padx=8, pady=(0, 4))

    def _setup_keyboard_shortcuts(self) -> None:
        self.bind("<space>", lambda _: self._toggle_pause())
        self.bind("<s>",     lambda _: self._take_screenshot())
        self.bind("<S>",     lambda _: self._take_screenshot())
        self.bind("<r>",     lambda _: self._toggle_recording())
        self.bind("<R>",     lambda _: self._toggle_recording())
        self.bind("<q>",     lambda _: self._close_app())
        self.bind("<Q>",     lambda _: self._close_app())

    # ── component init ────────────────────────────────────────────────────────

    def _load_components(self) -> None:
        self._set_status("Loading model…")
        model = ModelLoader.load_model(self.config.model_name)
        if not model:
            self._show_error("Failed to load YOLO model.")
            return

        self.model    = model
        self.camera   = CameraManager(
            self.config.camera_index,
            self.config.frame_width,
            self.config.frame_height,
        )
        self.detector = ObjectDetector(self.model, self.config)

        if not self.camera.open():
            self._show_error("Failed to open camera.")
            return

        self._set_status(
            f"Camera {self.config.camera_index} | "
            f"{self.config.frame_width}×{self.config.frame_height} | "
            f"Model: {self.config.model_name}"
        )
        self.running      = True
        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._video_thread.start()
        self.after(16, self._poll_queue)

    # ── video worker thread ───────────────────────────────────────────────────

    def _video_loop(self) -> None:
        """
        Daemon thread: camera read → YOLO inference → FramePacket → queue.
        Never touches any CTk widget directly.
        """
        target_interval = 1.0 / max(1, self.config.target_fps)
        last_time = time.perf_counter()

        while self.running:
            if self.paused:
                time.sleep(0.033)
                continue

            now     = time.perf_counter()
            elapsed = now - last_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            last_time = time.perf_counter()

            ret, frame = self.camera.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame")
                time.sleep(0.05)
                continue

            with self._frame_lock:
                self._current_raw_frame = frame.copy()

            self.frame_count += 1
            self._fps_times.append(time.perf_counter())

            try:
                detected, annotated, obj_count, confs = self.detector.detect(frame)
                self.detection_count = obj_count
            except Exception as exc:
                logger.error("Detection error: %s", exc, exc_info=True)
                continue

            fps = 0.0
            if len(self._fps_times) >= 2:
                fps = (len(self._fps_times) - 1) / (
                    self._fps_times[-1] - self._fps_times[0]
                )

            if self.recording and self.video_writer:
                self.video_writer.write(annotated)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((640, 480), Image.LANCZOS)

            if detected:
                ts = datetime.now().strftime("%H:%M:%S")
                for cls in sorted(detected):
                    self._detection_log.appendleft(
                        f"[{ts}] {cls} ({confs.get(cls, 0):.2f})"
                    )

            packet = FramePacket(
                rgb_image=pil,
                detected_classes=detected,
                class_confidences=confs,
                obj_count=obj_count,
                fps=fps,
                raw_frame=annotated,
            )
            try:
                self._frame_queue.put_nowait(packet)
            except Exception:
                pass  # queue full → consumer slow → drop frame

    # ── main-thread queue consumer ────────────────────────────────────────────

    def _poll_queue(self) -> None:
        """All widget mutations happen here — only called from main thread."""
        if not self.running:
            return

        try:
            packet = self._frame_queue.get_nowait()
        except Empty:
            self.after(16, self._poll_queue)
            return

        imgtk = ImageTk.PhotoImage(image=packet.rgb_image)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text="")

        self.fps_label.configure(text=f"FPS: {packet.fps:.1f}")
        self.obj_label.configure(text=f"Objects: {packet.obj_count}")
        self.frame_label.configure(text=f"Frames: {self.frame_count}")
        det_text = (
            ", ".join(sorted(packet.detected_classes)) if packet.detected_classes else "—"
        )
        self.detected_label.configure(text=f"Detected: {det_text}")

        self._refresh_log()

        if self.recording:
            dot = "⏺ REC" if int(time.time() * 2) % 2 == 0 else "   REC"
            self.rec_indicator.configure(text=dot)
        else:
            self.rec_indicator.configure(text="")

        self.after(16, self._poll_queue)

    # ── UI callbacks ─────────────────────────────────────────────────────────

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_btn.configure(
            text="▶  Resume  [Space]" if self.paused else "⏸  Pause  [Space]"
        )

    def _take_screenshot(self) -> None:
        with self._frame_lock:
            frame = (
                self._current_raw_frame.copy()
                if self._current_raw_frame is not None else None
            )
        if frame is None:
            self._set_status("No frame available for screenshot")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"screenshot_{ts}.jpg"
        try:
            cv2.imwrite(fn, frame)
            self._set_status(f"Screenshot saved: {fn}")
            logger.info("Screenshot saved: %s", fn)
        except Exception as exc:
            self._set_status(f"Screenshot failed: {exc}")

    def _toggle_recording(self) -> None:
        if not self.recording:
            ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn     = f"recording_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                fn, fourcc, float(self.config.target_fps),
                (self.config.frame_width, self.config.frame_height),
            )
            self.recording = True
            self.record_btn.configure(text="⏹  Stop Rec  [R]", fg_color="#550000")
            self._set_status(f"Recording → {fn}")
        else:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            self.record_btn.configure(text="⏺  Record  [R]", fg_color="#8B0000")
            self._set_status("Recording stopped")

    def _update_confidence(self, value: float) -> None:
        self.config.confidence_threshold = float(value)
        self.conf_val_label.configure(text=f"{value:.2f}")

    def _on_model_change(self, model_name: str) -> None:
        if model_name == self.config.model_name:
            return
        self._set_status(f"Loading {model_name}…")
        self.config.model_name = model_name

        def _load() -> None:
            new_model = ModelLoader.load_model(model_name)
            if new_model:
                self.model = new_model
                if self.detector:
                    self.detector.model = new_model
                self.after(0, lambda: self._set_status(f"Model: {model_name}"))
            else:
                self.after(0, lambda: self._set_status(f"Failed to load {model_name}"))

        threading.Thread(target=_load, daemon=True).start()

    def _add_filter_class(self, _event=None) -> None:
        cls = self.filter_entry.get().strip().lower()
        if cls and cls not in self.config.filter_classes:
            self.config.filter_classes.append(cls)
            self.filter_entry.delete(0, "end")
            self._refresh_filter_tags()

    def _remove_filter_class(self, cls: str) -> None:
        if cls in self.config.filter_classes:
            self.config.filter_classes.remove(cls)
            self._refresh_filter_tags()

    def _clear_filter(self) -> None:
        self.config.filter_classes.clear()
        self._refresh_filter_tags()

    def _refresh_filter_tags(self) -> None:
        for w in self.filter_tags_frame.winfo_children():
            w.destroy()
        if not self.config.filter_classes:
            ctk.CTkLabel(
                self.filter_tags_frame, text="(all classes shown)",
                font=("Courier New", 10), text_color="#888",
            ).pack(padx=4, pady=2)
        else:
            for cls in self.config.filter_classes:
                row = ctk.CTkFrame(self.filter_tags_frame, fg_color="#333")
                row.pack(fill="x", padx=2, pady=1)
                ctk.CTkLabel(row, text=cls, font=("Courier New", 11)).pack(side="left", padx=6)
                ctk.CTkButton(
                    row, text="✕", width=24, height=20, fg_color="#555",
                    command=lambda c=cls: self._remove_filter_class(c),
                ).pack(side="right", padx=2)

    def _toggle_heatmap(self) -> None:
        self.config.show_heatmap = self.heatmap_var.get()

    def _refresh_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.insert("end", "\n".join(list(self._detection_log)[:60]))
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self._detection_log.clear()

    def _set_status(self, msg: str) -> None:
        self.status_label.configure(text=msg)

    def _show_error(self, msg: str) -> None:
        self.video_label.configure(text=f"⚠  {msg}", image=None)
        self._set_status(f"ERROR: {msg}")

    def _close_app(self) -> None:
        """Orderly shutdown: stop thread → release resources → save config → destroy."""
        logger.info("Closing application…")
        self.running = False

        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join(timeout=2.0)

        if self.video_writer:
            self.video_writer.release()

        if self.camera:
            self.camera.release()

        if self.detector:
            self.detector._heatmap = None

        self.config.save()
        cv2.destroyAllWindows()
        self.destroy()


# ---------------------------------------------------------------------------
# Headless simple_detect
# ---------------------------------------------------------------------------

def simple_detect(
    target_class: str = "person",
    config: Optional[DetectionConfig] = None,
) -> List[str]:
    """
    Headless detection loop.  Runs until *target_class* is detected or 'q'
    is pressed.  Uses context managers internally for safe resource cleanup.

    Note: cv2.imshow() must be called from the main thread on macOS.

    Returns
    -------
    list[str] — detected class names at exit.
    """
    cfg = config or DetectionConfig(target_class=target_class)
    detected_objects: List[str] = []

    with CameraManager(cfg.camera_index, cfg.frame_width, cfg.frame_height) as cam:
        with ObjectDetector.from_model(cfg.model_name, cfg) as det:
            logger.info("Detecting '%s'… press 'q' to quit.", target_class)
            target_interval = 1.0 / max(1, cfg.target_fps)

            while True:
                t0 = time.perf_counter()
                ret, frame = cam.read()
                if not ret or frame is None:
                    break

                detected, annotated, _, _ = det.detect(frame)
                cv2.imshow("YOLOv8 Detection", annotated)

                key = cv2.waitKey(1) & 0xFF
                if target_class in detected or key == ord("q"):
                    detected_objects = sorted(detected)
                    break

                elapsed = time.perf_counter() - t0
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)

    cv2.destroyAllWindows()
    return detected_objects


# ---------------------------------------------------------------------------
# High-level launchers
# ---------------------------------------------------------------------------

def launch_gui(config: Optional[DetectionConfig] = None) -> None:
    """Launch the full GUI application."""
    cfg = config or DetectionConfig.load()
    with DetectionApp.run_app(cfg) as app:
        app.mainloop()


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases
# ---------------------------------------------------------------------------

def EYE() -> List[str]:
    """Legacy alias → simple_detect()."""
    return simple_detect()


def OpenEYE() -> None:
    """Legacy alias → launch_gui()."""
    launch_gui()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    launch_gui()