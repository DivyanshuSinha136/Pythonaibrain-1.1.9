# Camera — Production-grade Camera Module

A Tkinter-based camera viewer with live QR/barcode scanning, image capture, video recording, and full context-manager support. Wraps OpenCV and pyzbar behind a clean class-based API with thread-safe state management.

---

## What Is This?

`mainCamera.py` provides a production-quality camera module that combines:

- **Live video preview** — Tkinter GUI window displaying frames from any OpenCV-compatible device
- **QR / barcode scanning** — auto-decodes QR codes and barcodes from every frame via `pyzbar`; accumulates all unique payloads thread-safely
- **Image capture** — saves the current frame as a timestamped `.png` with one method call
- **Video recording** — start/stop writing annotated frames to a timestamped `.avi` file using XVID codec
- **Context-manager protocol** — `Camera.open()` creates the Tk root, opens the device, and releases everything on exit
- **Functional entry-point** — `Start()` launches the GUI and returns all scanned data in one call
- **Auto-install of `pyzbar`** — silently attempts `pip install pyzbar` if the library is missing; degrades gracefully if it still can't be imported
- **Thread-safe design** — separate `RLock` guards for scanned data and video writer; frame loop never blocks on recording

---

## Installation

```bash
pip install opencv-python pillow pyzbar
```

> On Linux, `pyzbar` also needs the native `zbar` library:
> ```bash
> sudo apt install libzbar0
> ```
> On Windows, the `pyzbar` wheel bundles the DLL automatically.

---

## How to Use

### 1. Simplest — functional one-liner

Launches the GUI; blocks until the window is closed; returns all scanned payloads.

```python
from mainCamera import Start

data = Start()
for item in data:
    print(item)
```

### 2. Context manager (recommended)

```python
from mainCamera import Camera

with Camera.open() as cam:
    cam.run()              # blocks until the window is closed

print(cam.scanned_data)   # set of all scanned QR/barcode strings
```

### 3. Custom device and output directory

```python
from mainCamera import Camera

with Camera.open(device=1, output_dir="./recordings") as cam:
    cam.run()
```

### 4. Programmatic capture and recording

```python
from mainCamera import Camera
import tkinter as tk

root = tk.Tk()
cam = Camera(root, device=0, output_dir="./output")

# Capture a single frame immediately
path = cam.capture_image()
print(f"Saved: {path}")

# Start and stop recording
record_path = cam.start_recording()
print(f"Recording to: {record_path}")

# … run some frames …
cam.run()       # blocks until window closed

cam.stop_recording()
cam.close()
```

### 5. Inspect scanned data before closing

```python
from mainCamera import Camera
import threading

with Camera.open() as cam:
    # Poll scanned_data from another thread
    def monitor():
        import time
        while not cam.is_closed:
            print("Scanned so far:", cam.scanned_data)
            time.sleep(2.0)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    cam.run()

print("Final scanned data:", cam.scanned_data)
```

### 6. State checks

```python
with Camera.open() as cam:
    print(cam.is_recording)   # False
    cam.start_recording()
    print(cam.is_recording)   # True
    cam.stop_recording()
    print(cam.is_closed)      # False
    cam.run()

print(cam.is_closed)          # True
```

---

## GUI Reference

The camera window opens with three buttons:

| Button | Action |
|---|---|
| 📸 **Capture Image** | Saves current frame as `captured_YYYYMMDD_HHMMSS.png` in `output_dir`; shows a confirmation dialog |
| 🎥 **Start Recording** / ⏹️ **Stop Recording** | Toggles video recording; saves to `output_YYYYMMDD_HHMMSS.avi` in `output_dir` |
| ❌ **Quit** | Stops recording (if active), releases the camera, and destroys the window |

Closing the window via the title bar `×` button also triggers a clean shutdown.

---

## API Reference

### `Start(device, output_dir)` → `set[str]`

Functional entry-point. Launches the GUI, blocks until the window is closed, and returns all scanned QR/barcode payloads.

| Parameter    | Type             | Default | Description                            |
|--------------|------------------|---------|----------------------------------------|
| `device`     | `int`            | `0`     | OpenCV camera device index             |
| `output_dir` | `Path` or `str`  | `"."`   | Directory for saved images and videos  |

---

### `Camera.open(device, output_dir)` → context manager

Class-level `@contextmanager` factory. Creates a `tk.Tk` root, instantiates `Camera`, yields it, and calls `close()` on exit.

```python
with Camera.open(device=0, output_dir="./out") as cam:
    cam.run()
```

---

### `Camera(window, device, output_dir)`

Direct constructor for advanced use when you manage the Tk root yourself.

| Parameter    | Type             | Default | Description                              |
|--------------|------------------|---------|------------------------------------------|
| `window`     | `tk.Tk`          | —       | Root Tkinter window                      |
| `device`     | `int`            | `0`     | OpenCV capture device index              |
| `output_dir` | `Path` or `str`  | `"."`   | Directory for saved images and videos    |

**Raises:** `CameraError` if the device cannot be opened.

---

### Instance methods

| Method | Returns | Description |
|---|---|---|
| `run()` | `None` | Enter the Tk main-loop (blocks until window closed) |
| `capture_image()` | `Path` or `None` | Capture current frame to `output_dir`; returns saved path |
| `start_recording()` | `Path` or `None` | Begin writing frames to `.avi`; returns output path |
| `stop_recording()` | `None` | Stop recording and flush the video file |
| `close()` | `None` | Release all resources; safe to call multiple times |

### Properties

| Property | Type | Description |
|---|---|---|
| `scanned_data` | `set[str]` | Thread-safe snapshot of all decoded QR/barcode payloads |
| `is_recording` | `bool` | `True` while a video is being written |
| `is_closed` | `bool` | `True` after `close()` has been called |

---

## Output Files

All files are written to `output_dir` (default: current directory).

| File pattern | Format | Created by |
|---|---|---|
| `captured_YYYYMMDD_HHMMSS.png` | PNG | `capture_image()` / 📸 button |
| `output_YYYYMMDD_HHMMSS.avi` | AVI (XVID, 20 fps) | `start_recording()` / 🎥 button |

---

## Exception Reference

```
CameraError(RuntimeError)
├── Raised when the camera device cannot be opened
└── Raised when run() is called after the camera is already closed
```

---

## Architecture Notes

- **Frame loop** — driven by `window.after(_FRAME_DELAY_MS, _update_frame)` (10 ms ≈ 100 fps cap); never blocks the Tk event loop
- **QR decoding** — runs on every frame inside the frame loop; new unique payloads are added to `_scanned_data` under `_data_lock`
- **Video writer** — guarded by `_writer_lock` so the recording flag and `cv2.VideoWriter` are always consistent across threads
- **`pyzbar` graceful degradation** — if not installed, a `pip install` is attempted once at import time; if still unavailable, QR scanning is silently disabled and the rest of the module works normally
- **Clean shutdown** — `close()` stops recording, releases the OpenCV capture device, and destroys the Tk window; idempotent (safe to call multiple times)

---

## Examples Summary

```python
# Simplest: launch and collect scanned QR data
from mainCamera import Start
data = Start()

# Context manager
from mainCamera import Camera
with Camera.open(device=0, output_dir="./out") as cam:
    cam.run()
print(cam.scanned_data)

# Custom device
with Camera.open(device=1) as cam:
    cam.run()

# Programmatic capture before running
import tkinter as tk
from mainCamera import Camera
root = tk.Tk()
cam = Camera(root, output_dir="./shots")
cam.capture_image()          # save a frame immediately
cam.start_recording()        # start video
cam.run()                    # blocks
cam.stop_recording()
cam.close()

# State checks
with Camera.open() as cam:
    print(cam.is_recording)  # False
    cam.start_recording()
    print(cam.is_recording)  # True
    cam.run()
```