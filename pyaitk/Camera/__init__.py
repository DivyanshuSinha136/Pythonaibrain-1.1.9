"""
Camera package — QR-scanning camera with recording and context-manager support.

Quick start::

    from camera import Start
    data = Start()

Advanced::

    from camera import Camera

    with Camera.open(device=0, output_dir="./recordings") as cam:
        cam.run()
    print(cam.scanned_data)
"""

from .mainCamera import Camera, CameraError, Start

__all__ = ["Camera", "CameraError", "Start"]