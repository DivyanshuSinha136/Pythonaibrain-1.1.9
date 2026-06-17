# EYE — YOLOv8 Real-Time Object Detection

A production-ready webcam object detection application built on YOLOv8 (Ultralytics) and CustomTkinter. Supports a full GUI with live controls, headless CLI detection, a unified `EyeSession` facade, and full context-manager support on every public class — with extensive threading fixes and feature additions over a naive implementation.

---

## What Is This?

`eye.py` is a fully-featured real-time object detection module that provides:

- **Full context-manager protocol** — every public class (`CameraManager`, `ObjectDetector`, `DetectionApp`, `EyeSession`) supports `with` blocks with guaranteed resource cleanup
- **`EyeSession` facade** — unified entry point with `.gui()` and `.headless()` factory context managers covering the complete lifecycle
- **Live webcam detection** — reads frames from any connected camera and runs YOLOv8 inference at a capped FPS
- **Full GUI application** — dark-mode CustomTkinter UI with video feed, control panel, and detection log
- **Headless CLI mode** — `simple_detect()` uses context managers internally for safe camera/detector teardown
- **Hot-swappable model variants** — switch between `yolov8n/s/m/l/x.pt` at runtime without restarting
- **Class filter** — show bounding boxes only for specific classes (e.g. `person`, `car`)
- **Confidence threshold** — adjustable slider (0.1–0.95), persisted to `~/.yolov8_app.json`
- **Detection heat-map** — alpha-blended overlay that accumulates and decays detection positions
- **Video recording** — save annotated footage to `.mp4` at camera resolution
- **Screenshot** — save the current annotated frame as a timestamped JPG
- **Live FPS counter** — exponential moving average over the last 30 frames
- **Detection log** — scrollable timestamped list of the last 200 detection events
- **Keyboard shortcuts** — Space (pause), S (screenshot), R (record), Q (quit)
- **Serialisable config** — `DetectionConfig` saves/loads all settings as JSON
- **Thread-safe design** — all widget mutations dispatched via `self.after()`, frame buffer protected with `threading.Lock`
- **Legacy aliases** — `EYE()` and `OpenEYE()` for backwards compatibility with `core.py` / pyaitk

---

## Installation

```bash
pip install ultralytics opencv-python customtkinter pillow numpy
```

> YOLOv8 model weights are downloaded automatically on first use (~6 MB for `yolov8n.pt`).
> An internet connection is required for the initial download; subsequent runs work offline.

---

## Context-Manager Patterns

All four public classes now support the full context-manager protocol. Resources are always released — even when exceptions are raised.

### `EyeSession` — recommended top-level API

```python
from eye import EyeSession

# GUI mode (blocks until window closed)
with EyeSession.gui() as session:
    session.run()
    print(session.last_detections)

# Headless mode (blocks until target seen or 'q' pressed)
with EyeSession.headless(target_class="person") as session:
    detected = session.run()
    print(detected)

# GUI with custom config
from eye import DetectionConfig
cfg = DetectionConfig(model_name="yolov8s.pt", show_heatmap=True, confidence_threshold=0.6)
with EyeSession.gui(config=cfg) as session:
    session.run()
```

### `CameraManager` — camera resource

```python
from eye import CameraManager

# __enter__ / __exit__
with CameraManager(camera_index=0, width=640, height=480) as cam:
    ret, frame = cam.read()

# Factory context manager
with CameraManager.open_device(0, width=1280, height=720) as cam:
    ret, frame = cam.read()
    print(cam.is_opened)   # True
```

### `ObjectDetector` — inference resource

```python
from eye import ObjectDetector, DetectionConfig

cfg = DetectionConfig()

# __enter__ / __exit__ (provide your own model)
from eye import ModelLoader
model = ModelLoader.load_model("yolov8n.pt")
with ObjectDetector(model, cfg) as det:
    classes, annotated, count, confs = det.detect(frame)

# Factory context manager (loads model internally)
with ObjectDetector.from_model("yolov8n.pt", cfg) as det:
    classes, annotated, count, confs = det.detect(frame)
```

### `DetectionApp` — GUI application

```python
from eye import DetectionApp, DetectionConfig
import customtkinter as ctk

# __enter__ / __exit__
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
with DetectionApp(DetectionConfig()) as app:
    app.mainloop()

# Factory context manager (sets up CTk theme automatically)
with DetectionApp.run_app(DetectionConfig()) as app:
    app.mainloop()
```

### Composing context managers (low-level pipeline)

```python
from eye import CameraManager, ObjectDetector
import cv2

with CameraManager(0) as cam:
    with ObjectDetector.from_model("yolov8n.pt") as det:
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            classes, annotated, count, confs = det.detect(frame)
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
cv2.destroyAllWindows()
# Camera and detector released automatically on exit
```

---

## How to Use

### 1. Launch the GUI (simplest)

```python
from eye import launch_gui

launch_gui()
```

Or run directly:

```bash
python eye.py
```

### 2. GUI with custom config

```python
from eye import launch_gui, DetectionConfig

config = DetectionConfig(
    model_name="yolov8s.pt",
    confidence_threshold=0.6,
    target_fps=25,
    show_heatmap=True,
    filter_classes=["person", "car"],
)
launch_gui(config)
```

### 3. Headless detection (no GUI)

`simple_detect()` opens an OpenCV window and runs until the target class is seen or `q` is pressed. Uses `CameraManager` and `ObjectDetector` context managers internally — camera is always released.

```python
from eye import simple_detect

detected = simple_detect(target_class="person")
print("Detected:", detected)
```

> **macOS note:** `cv2.imshow` must be called from the main thread. Only call `simple_detect()` from the main thread.

### 4. Headless with custom config

```python
from eye import simple_detect, DetectionConfig

config = DetectionConfig(
    model_name="yolov8m.pt",
    confidence_threshold=0.45,
    camera_index=1,
    target_fps=20,
)
detected = simple_detect(target_class="car", config=config)
print(detected)
```

### 5. Legacy aliases (pyaitk / core.py compatibility)

```python
from eye import EYE, OpenEYE

detected = EYE()      # → simple_detect()
OpenEYE()             # → launch_gui()
```

### 6. Low-level components with context managers

```python
from eye import CameraManager, ObjectDetector, DetectionConfig
import cv2

config = DetectionConfig()

with CameraManager(camera_index=0, width=640, height=480) as cam:
    with ObjectDetector.from_model("yolov8n.pt", config) as det:
        ret, frame = cam.read()
        if ret:
            detected_classes, annotated, count, confidences = det.detect(frame)
            print(f"Detected {count} objects: {detected_classes}")
            cv2.imshow("Frame", annotated)
            cv2.waitKey(0)
cv2.destroyAllWindows()
# No explicit release needed — context managers handle it
```

---

## GUI Controls Reference

| Control | Description |
|---|---|
| Model selector | Switch between `yolov8n/s/m/l/x.pt` (hot-swap, no restart needed) |
| Confidence slider | Detection threshold from 0.10 to 0.95 |
| Class filter field | Type a class name and press Enter or "Add Filter" |
| Clear Filter button | Remove all filters (show all classes) |
| Heatmap toggle | Enable/disable the alpha-blended detection heat-map |
| Pause / Resume button | Freeze/unfreeze the video loop |
| Screenshot button | Save annotated frame as `screenshot_YYYYMMDD_HHMMSS.jpg` |
| Record button | Start/stop saving annotated video as `recording_*.mp4` |
| Detection log | Scrollable list of last 200 timestamped events |
| Clear Log button | Empty the detection log |
| Status bar | Camera info, resolution, active model name |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | Pause / Resume |
| `S` | Screenshot |
| `R` | Start / Stop recording |
| `Q` | Quit |

---

## API Reference

### `EyeSession`

Unified high-level facade with two factory context managers.

#### `EyeSession.gui(config?)` → context manager

Launches the full CustomTkinter GUI. Blocks on `session.run()` until the window is closed. Tears down the app on exit.

```python
with EyeSession.gui(config=DetectionConfig(model_name="yolov8s.pt")) as session:
    session.run()
    print(session.last_detections)
```

#### `EyeSession.headless(target_class, config?)` → context manager

Headless detection. Blocks on `session.run()` until target is detected or `q` is pressed. Calls `cv2.destroyAllWindows()` on exit.

```python
with EyeSession.headless("car") as session:
    detected = session.run()
```

| Property | Type | Description |
|---|---|---|
| `last_detections` | `list[str]` | Class names detected at the time of exit |

---

### `launch_gui(config)` → `None`

Functional launcher. Uses `DetectionApp.run_app()` context manager internally.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `DetectionConfig` or `None` | `None` | Load from `~/.yolov8_app.json` if `None` |

---

### `simple_detect(target_class, config)` → `list[str]`

Headless OpenCV detection loop. Uses `CameraManager` and `ObjectDetector` context managers internally.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_class` | `str` | `"person"` | Class name to trigger exit on |
| `config` | `DetectionConfig` or `None` | `None` | Uses defaults if `None` |

**Returns:** sorted list of all class names detected at the time of exit.

---

### `CameraManager(camera_index, width, height)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `camera_index` | `int` | `0` | OpenCV device index |
| `width` | `int` | `640` | Requested frame width |
| `height` | `int` | `480` | Requested frame height |

| Method / Property | Returns | Description |
|---|---|---|
| `open()` | `bool` | Open the device; returns `True` on success |
| `read()` | `tuple[bool, ndarray\|None]` | Read one frame |
| `release()` | `None` | Release the capture device |
| `is_opened` | `bool` | `True` if device is currently open |
| `__enter__` / `__exit__` | — | Opens on enter, releases on exit |
| `CameraManager.open_device(idx, w, h)` | context manager | Factory `@contextmanager` |

---

### `ObjectDetector(model, config)`

| Method | Returns | Description |
|---|---|---|
| `detect(frame)` | `tuple` | Run inference; see return table below |
| `__enter__` / `__exit__` | — | Clears heatmap accumulator on exit |
| `ObjectDetector.from_model(name, config?)` | context manager | Loads model + yields detector |

**`detect()` return values:**

| Field | Type | Description |
|---|---|---|
| `detected_classes` | `set[str]` | All class names above confidence threshold |
| `annotated_frame` | `np.ndarray` | Original-resolution BGR frame with bounding boxes |
| `object_count` | `int` | Total number of detections |
| `class_confidences` | `dict[str, float]` | Highest confidence per class name |

---

### `DetectionApp(config)`

| Method | Description |
|---|---|
| `mainloop()` | Start the Tk event loop (blocks) |
| `__enter__` / `__exit__` | Calls `_close_app()` on exit |
| `DetectionApp.run_app(config?)` | Factory context manager; sets CTk theme |

---

### `DetectionConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"yolov8n.pt"` | YOLOv8 weight file name |
| `target_class` | `str` | `"person"` | Class to watch for in headless mode |
| `confidence_threshold` | `float` | `0.5` | Minimum detection confidence (0.0–1.0) |
| `frame_width` | `int` | `640` | Requested camera frame width |
| `frame_height` | `int` | `480` | Requested camera frame height |
| `detection_size` | `tuple[int,int]` | `(416, 416)` | Resize resolution for YOLO inference |
| `camera_index` | `int` | `0` | OpenCV camera index |
| `target_fps` | `int` | `30` | FPS cap for the detection loop |
| `show_heatmap` | `bool` | `False` | Enable detection heat-map overlay |
| `filter_classes` | `list[str]` | `[]` | Restrict boxes to these classes; empty = all |

**Persistence:**

```python
config = DetectionConfig(confidence_threshold=0.7)
config.save()                 # → ~/.yolov8_app.json

config = DetectionConfig.load()   # restore from file
```

---

### `ModelLoader.load_model(model_name)` → `YOLO | None`

Tries four strategies in order:

1. Package resource (if running inside a package)
2. Script directory and adjacent `eye/` subfolder
3. Current working directory
4. Auto-download from Ultralytics

---

## Architecture Overview

```
EyeSession
├── EyeSession.gui()       → DetectionApp.run_app() → DetectionApp.__enter__/__exit__
└── EyeSession.headless()  → simple_detect()
                                ├── CameraManager.__enter__/__exit__
                                └── ObjectDetector.from_model().__enter__/__exit__

DetectionApp (GUI)
├── Main thread   ──→  CTk event loop + all widget mutations (via self.after())
│                             ↑
│                       polls frame queue at ~60 Hz
│                             ↑
└── Video thread  ──→  camera.read() → detector.detect() → FramePacket → Queue(maxsize=2)

Resource ownership
──────────────────
CameraManager.__exit__    → cap.release()
ObjectDetector.__exit__   → _heatmap = None
DetectionApp.__exit__     → thread.join(2s) → writer.release() → camera.release() → destroy()
EyeSession.__exit__       → app._close_app() (if GUI) / cv2.destroyAllWindows() (if headless)
```

---

## Examples Summary

```python
# ── EyeSession (recommended) ─────────────────────────────────────────────────

from eye import EyeSession, DetectionConfig

# GUI
with EyeSession.gui() as session:
    session.run()
    print(session.last_detections)

# GUI with custom config
cfg = DetectionConfig(model_name="yolov8s.pt", show_heatmap=True)
with EyeSession.gui(config=cfg) as session:
    session.run()

# Headless
with EyeSession.headless("car") as session:
    detected = session.run()

# ── Functional API ───────────────────────────────────────────────────────────

from eye import launch_gui, simple_detect
launch_gui()
detected = simple_detect("person")

# ── Low-level context managers ───────────────────────────────────────────────

from eye import CameraManager, ObjectDetector
import cv2

with CameraManager(0) as cam:
    with ObjectDetector.from_model("yolov8n.pt") as det:
        ret, frame = cam.read()
        classes, annotated, count, confs = det.detect(frame)
cv2.destroyAllWindows()

# ── DetectionApp directly ────────────────────────────────────────────────────

from eye import DetectionApp, DetectionConfig
with DetectionApp.run_app(DetectionConfig()) as app:
    app.mainloop()

# ── Legacy aliases ───────────────────────────────────────────────────────────

from eye import EYE, OpenEYE
EYE()       # headless detect
OpenEYE()   # launch GUI
```