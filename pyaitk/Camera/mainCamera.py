"""
mainCamera.py — Production-grade camera module with QR scanning,
image capture, video recording, and full context-manager support.

Usage (context manager):
    with Camera.open() as cam:
        cam.run()
        data = cam.scanned_data

Usage (functional):
    data = Start()
"""

from __future__ import annotations

import logging
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from subprocess import run as _run
from types import TracebackType
from typing import Generator, Optional, Set

import cv2
import tkinter as tk
from tkinter import Label, Button, messagebox
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Optional dependency: pyzbar
# ---------------------------------------------------------------------------
try:
    from pyzbar import pyzbar as _pyzbar
    _PYZBAR_AVAILABLE = True
except ImportError:
    _run([sys.executable, "-m", "pip", "install", "pyzbar"], check=False)
    try:
        from pyzbar import pyzbar as _pyzbar  # type: ignore[no-redef]
        _PYZBAR_AVAILABLE = True
    except ImportError:
        _pyzbar = None  # type: ignore[assignment]
        _PYZBAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mainCamera")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FRAME_DELAY_MS: int = 10          # ms between frame updates (~100 fps cap)
_DEFAULT_FPS: float = 20.0
_DEFAULT_FOURCC: str = "XVID"
_CAPTURE_EXT: str = ".png"
_RECORD_EXT: str = ".avi"
_THEME = {
    "bg":      "#1e1e1e",
    "fg":      "#ffffff",
    "green":   "#4CAF50",
    "green_a": "#45a049",
    "blue":    "#2196F3",
    "blue_a":  "#1976D2",
    "orange":  "#FF5722",
    "red":     "#f44336",
    "red_a":   "#e53935",
}


# ---------------------------------------------------------------------------
# CameraError
# ---------------------------------------------------------------------------
class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a critical failure occurs."""


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class Camera:
    """
    Tkinter-based camera viewer with QR scanning, image capture, and
    video recording.

    Supports the context-manager protocol:

        with Camera.open(device=0) as cam:
            cam.run()          # blocks until the window is closed
            print(cam.scanned_data)

    Or instantiate manually and call :meth:`run` / :meth:`close` yourself.
    """

    # ------------------------------------------------------------------
    # Construction / teardown
    # ------------------------------------------------------------------

    def __init__(
        self,
        window: tk.Tk,
        *,
        device: int = 0,
        output_dir: Path | str = ".",
    ) -> None:
        """
        Parameters
        ----------
        window:     A ``tk.Tk`` (or ``tk.Toplevel``) root window.
        device:     OpenCV capture device index (default 0).
        output_dir: Directory where captured images and recordings are saved.
        """
        self._window = window
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._scanned_data: Set[str] = set()
        self._data_lock = threading.Lock()

        self._recording = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_lock = threading.Lock()

        self._closed = False

        # Open camera
        logger.info("Opening camera device %d …", device)
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise CameraError(f"Cannot open camera device {device}.")
        logger.info("Camera device %d opened.", device)

        self._build_ui()
        self._schedule_next_frame()

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self.close()
        return False  # do not suppress exceptions

    @classmethod
    @contextmanager
    def open(
        cls,
        *,
        device: int = 0,
        output_dir: Path | str = ".",
    ) -> Generator["Camera", None, None]:
        """
        Convenience context manager that creates the Tk root, instantiates
        the camera, yields it, and cleans up afterwards.

        Example::

            with Camera.open(device=0, output_dir="./recordings") as cam:
                cam.run()
            print(cam.scanned_data)
        """
        root = tk.Tk()
        cam = cls(root, device=device, output_dir=output_dir)
        try:
            yield cam
        finally:
            cam.close()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def scanned_data(self) -> Set[str]:
        """Thread-safe snapshot of all decoded QR/barcode payloads."""
        with self._data_lock:
            return set(self._scanned_data)

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_closed(self) -> bool:
        return self._closed

    def run(self) -> None:
        """Enter the Tk main-loop (blocks until the window is destroyed)."""
        if self._closed:
            raise CameraError("Camera has already been closed.")
        logger.info("Entering main loop.")
        self._window.mainloop()
        logger.info("Main loop exited.")

    def capture_image(self) -> Optional[Path]:
        """
        Capture a single frame and write it to *output_dir*.

        Returns the saved :class:`~pathlib.Path`, or ``None`` on failure.
        """
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("capture_image: failed to grab frame.")
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._output_dir / f"captured_{ts}{_CAPTURE_EXT}"
        cv2.imwrite(str(path), frame)
        logger.info("Image saved → %s", path)
        return path

    def start_recording(self) -> Optional[Path]:
        """
        Begin writing frames to a video file.

        Returns the output :class:`~pathlib.Path`, or ``None`` if already
        recording or the writer could not be created.
        """
        with self._writer_lock:
            if self._recording:
                logger.warning("start_recording called while already recording.")
                return None
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self._output_dir / f"output_{ts}{_RECORD_EXT}"
            fourcc = cv2.VideoWriter_fourcc(*_DEFAULT_FOURCC)
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._writer = cv2.VideoWriter(str(path), fourcc, _DEFAULT_FPS, (w, h))
            if not self._writer.isOpened():
                logger.error("VideoWriter could not be opened for %s", path)
                self._writer = None
                return None
            self._recording = True
            logger.info("Recording started → %s", path)
            return path

    def stop_recording(self) -> None:
        """Stop an in-progress recording and flush the file."""
        with self._writer_lock:
            if not self._recording:
                logger.warning("stop_recording called while not recording.")
                return
            self._recording = False
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            logger.info("Recording stopped.")

    def close(self) -> None:
        """Release all resources. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        logger.info("Closing camera …")
        if self._recording:
            self.stop_recording()
        self._cap.release()
        logger.info("Camera device released.")
        try:
            if self._window.winfo_exists():
                self._window.destroy()
        except tk.TclError:
            pass  # window already gone

    # ------------------------------------------------------------------
    # Private — UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        w = self._window
        w.title("📷 Camera")
        w.configure(bg=_THEME["bg"])
        w.resizable(False, False)
        w.protocol("WM_DELETE_WINDOW", self._on_close_request)

        self._video_label = Label(w, bg=_THEME["bg"])
        self._video_label.pack(padx=10, pady=10)

        btn_cfg = dict(
            font=("Segoe UI", 12, "bold"),
            fg=_THEME["fg"],
            relief="flat",
            width=22,
        )

        self._capture_btn = Button(
            w,
            text="📸  Capture Image",
            command=self._on_capture,
            bg=_THEME["green"],
            activebackground=_THEME["green_a"],
            **btn_cfg,
        )
        self._capture_btn.pack(pady=5)

        self._record_btn = Button(
            w,
            text="🎥  Start Recording",
            command=self._on_toggle_recording,
            bg=_THEME["blue"],
            activebackground=_THEME["blue_a"],
            **btn_cfg,
        )
        self._record_btn.pack(pady=5)

        self._quit_btn = Button(
            w,
            text="❌  Quit",
            command=self._on_close_request,
            bg=_THEME["red"],
            activebackground=_THEME["red_a"],
            **btn_cfg,
        )
        self._quit_btn.pack(pady=10)

    # ------------------------------------------------------------------
    # Private — frame loop
    # ------------------------------------------------------------------

    def _schedule_next_frame(self) -> None:
        if not self._closed:
            self._window.after(_FRAME_DELAY_MS, self._update_frame)

    def _update_frame(self) -> None:
        if self._closed:
            return

        ret, frame = self._cap.read()
        if not ret:
            logger.debug("Frame grab failed; retrying …")
            self._schedule_next_frame()
            return

        # QR / barcode decoding
        if _PYZBAR_AVAILABLE and _pyzbar is not None:
            try:
                codes = _pyzbar.decode(frame)
                if codes:
                    with self._data_lock:
                        for code in codes:
                            payload = code.data.decode("utf-8", errors="replace")
                            if payload not in self._scanned_data:
                                self._scanned_data.add(payload)
                                logger.info("QR/barcode scanned: %s", payload)
            except Exception as exc:  # noqa: BLE001
                logger.debug("QR decode error: %s", exc)

        # Write to video if recording
        with self._writer_lock:
            if self._recording and self._writer is not None:
                self._writer.write(frame)

        # Display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self._video_label.imgtk = imgtk  # type: ignore[attr-defined]
        self._video_label.configure(image=imgtk)

        self._schedule_next_frame()

    # ------------------------------------------------------------------
    # Private — button handlers
    # ------------------------------------------------------------------

    def _on_capture(self) -> None:
        path = self.capture_image()
        if path:
            messagebox.showinfo("Captured", f"Image saved:\n{path}")
        else:
            messagebox.showerror("Error", "Could not capture image.")

    def _on_toggle_recording(self) -> None:
        if not self._recording:
            path = self.start_recording()
            if path:
                self._record_btn.config(
                    text="⏹️  Stop Recording", bg=_THEME["orange"]
                )
            else:
                messagebox.showerror("Error", "Could not start recording.")
        else:
            self.stop_recording()
            self._record_btn.config(
                text="🎥  Start Recording", bg=_THEME["blue"]
            )

    def _on_close_request(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Functional entry-point (backwards-compatible)
# ---------------------------------------------------------------------------

def Start(*, device: int = 0, output_dir: Path | str = ".") -> Set[str]:
    """
    Launch the camera GUI and return all scanned QR/barcode payloads
    once the window is closed.

    This is the simplest way to use the module::

        data = Start()
        for item in data:
            print(item)
    """
    with Camera.open(device=device, output_dir=output_dir) as cam:
        cam.run()
        return cam.scanned_data