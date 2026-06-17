"""
TTI_core.py
===========
Core pixel-level image engine for the TTI system.

Replaces byte_image32.py with a fully modular, configurable design.
Provides:
  - TTIImage      — main image class (BMP/PNG/JPEG I/O, pixel ops)
  - ImageCanvas   — high-level drawing surface with coordinate helpers
  - ColorUtils    — colour manipulation helpers
  - ImageIO       — multi-format file I/O
  - ImageValidator— runtime integrity checks

All classes are config-aware via TTI_config.TTIConfig.
"""

from __future__ import annotations

import math
import struct
import zlib
import copy
import random
from pathlib import Path
from typing import (
    Callable, Iterator, List, Optional, Tuple, Union
)

import numpy as np

from .TTI_config import TTIConfig, get_config

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Color   = Union[Tuple[int, int, int], Tuple[int, int, int, int]]
Color24 = Tuple[int, int, int]
Color32 = Tuple[int, int, int, int]
Pixels  = np.ndarray          # shape (H, W, C), dtype=uint8


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TTIError(Exception):
    """Base exception for TTI core errors."""

class TTIImageError(TTIError):
    """Raised for image-level errors."""

class TTIIOError(TTIError):
    """Raised for file I/O errors."""


# ---------------------------------------------------------------------------
# ColorUtils
# ---------------------------------------------------------------------------

class ColorUtils:
    """Static colour helper methods."""

    @staticmethod
    def clamp(value: int, lo: int = 0, hi: int = 255) -> int:
        return max(lo, min(hi, value))

    @staticmethod
    def to_rgba(color: Color) -> Color32:
        if len(color) == 3:
            return (*color, 255)           # type: ignore[return-value]
        return color                       # type: ignore[return-value]

    @staticmethod
    def lerp(c1: Color, c2: Color, t: float) -> Color24:
        """Linear interpolate between two colours."""
        return tuple(
            int(a * (1 - t) + b * t)
            for a, b in zip(c1[:3], c2[:3])
        )  # type: ignore[return-value]

    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> Color24:
        """Convert HSV (0-1 each) to RGB (0-255 each)."""
        if s == 0.0:
            iv = int(v * 255)
            return (iv, iv, iv)
        i = int(h * 6)
        f = h * 6 - i
        p, q, t_ = v*(1-s), v*(1-s*f), v*(1-s*(1-f))
        i %= 6
        seg = [(v, t_, p), (q, v, p), (p, v, t_),
               (p, q, v), (t_, p, v), (v, p, q)]
        r, g, b = seg[i]
        return (int(r*255), int(g*255), int(b*255))

    @staticmethod
    def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
        r_, g_, b_ = r/255.0, g/255.0, b/255.0
        cmax, cmin = max(r_, g_, b_), min(r_, g_, b_)
        delta = cmax - cmin
        h = 0.0
        if delta:
            if cmax == r_:   h = ((g_ - b_) / delta) % 6
            elif cmax == g_: h = (b_ - r_) / delta + 2
            else:            h = (r_ - g_) / delta + 4
            h /= 6
        s = 0.0 if cmax == 0 else delta / cmax
        return (h, s, cmax)

    @staticmethod
    def luminance(color: Color) -> float:
        r, g, b = color[:3]
        return 0.299*r + 0.587*g + 0.114*b

    @staticmethod
    def blend(c1: Color, c2: Color, alpha: float = 0.5) -> Color24:
        return ColorUtils.lerp(c1, c2, alpha)

    @staticmethod
    def invert(color: Color) -> Color24:
        return tuple(255 - c for c in color[:3])   # type: ignore[return-value]

    @staticmethod
    def grayscale(color: Color) -> Color24:
        g = int(ColorUtils.luminance(color))
        return (g, g, g)


# ---------------------------------------------------------------------------
# TTIImage  (core pixel buffer)
# ---------------------------------------------------------------------------

class TTIImage:
    """
    Core image class backed by a numpy uint8 array.

    Parameters
    ----------
    width, height : int
    bpp           : int     24 (RGB) or 32 (RGBA)
    background    : Color   fill colour (default white)
    config        : TTIConfig | None

    Pixel storage is top-left origin, shape (height, width, channels).
    """

    def __init__(
        self,
        width:      int,
        height:     int,
        bpp:        int = 24,
        background: Optional[Color] = None,
        config:     Optional[TTIConfig] = None,
    ) -> None:
        self._cfg = config or get_config()
        if bpp not in (24, 32):
            raise TTIImageError(f"bpp must be 24 or 32, got {bpp}")
        self.width    = width
        self.height   = height
        self.bpp      = bpp
        self.channels = bpp // 8
        bg = background or self._cfg.image.background_color
        if len(bg) < self.channels:
            bg = (*bg, 255) if self.channels == 4 else bg[:3]
        arr_bg = np.array(bg[:self.channels], dtype=np.uint8)
        self._pixels: Pixels = np.broadcast_to(
            arr_bg, (height, width, self.channels)
        ).copy()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def array(self) -> Pixels:
        """Direct access to the underlying numpy array (H×W×C, uint8)."""
        return self._pixels

    # ------------------------------------------------------------------ #
    # Pixel access
    # ------------------------------------------------------------------ #

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            c = color[:self.channels]
            self._pixels[y, x, :len(c)] = c

    def get_pixel(self, x: int, y: int) -> Optional[Color]:
        if 0 <= x < self.width and 0 <= y < self.height:
            return tuple(int(v) for v in self._pixels[y, x])  # type: ignore
        return None

    def fill(self, color: Color) -> None:
        """Fill entire image with *color*."""
        c = np.array(color[:self.channels], dtype=np.uint8)
        self._pixels[:] = c

    def fill_region(
        self,
        x0: int, y0: int,
        x1: int, y1: int,
        color: Color,
    ) -> None:
        """Fill a rectangular region [x0:x1, y0:y1] with *color*."""
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(self.width, x1), min(self.height, y1)
        c = np.array(color[:self.channels], dtype=np.uint8)
        self._pixels[y0c:y1c, x0c:x1c] = c

    # ------------------------------------------------------------------ #
    # Array-level helpers
    # ------------------------------------------------------------------ #

    def from_array(self, arr: np.ndarray) -> "TTIImage":
        """
        Replace internal buffer with *arr* (H×W×C or H×W×3/4, uint8).
        Returns self for chaining.
        """
        if arr.shape[:2] != (self.height, self.width):
            raise TTIImageError(
                f"Array shape {arr.shape[:2]} ≠ image size {(self.height, self.width)}"
            )
        self._pixels = arr.astype(np.uint8)
        return self

    def copy(self) -> "TTIImage":
        img = TTIImage(self.width, self.height, self.bpp, config=self._cfg)
        img._pixels = self._pixels.copy()
        return img

    def crop(self, x0: int, y0: int, x1: int, y1: int) -> "TTIImage":
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.width, x1), min(self.height, y1)
        img = TTIImage(x1-x0, y1-y0, self.bpp, config=self._cfg)
        img._pixels = self._pixels[y0:y1, x0:x1].copy()
        return img

    def resize(self, new_w: int, new_h: int) -> "TTIImage":
        """Nearest-neighbour resize."""
        xs = (np.arange(new_w) * self.width  / new_w).astype(int)
        ys = (np.arange(new_h) * self.height / new_h).astype(int)
        arr = self._pixels[np.ix_(ys, xs)]
        img = TTIImage(new_w, new_h, self.bpp, config=self._cfg)
        img._pixels = arr.copy()
        return img

    def flip_horizontal(self) -> "TTIImage":
        img = self.copy()
        img._pixels = np.fliplr(self._pixels)
        return img

    def flip_vertical(self) -> "TTIImage":
        img = self.copy()
        img._pixels = np.flipud(self._pixels)
        return img

    def rotate_90(self, clockwise: bool = True) -> "TTIImage":
        k = 3 if clockwise else 1
        arr = np.rot90(self._pixels, k=k)
        img = TTIImage(arr.shape[1], arr.shape[0], self.bpp, config=self._cfg)
        img._pixels = arr.copy()
        return img

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(
        self,
        filepath: Union[str, Path],
        fmt: Optional[str] = None,
    ) -> Path:
        """
        Save image.  Format inferred from extension or *fmt*.
        Supported: .bmp, .png, .jpg/.jpeg
        """
        return ImageIO.save(self, filepath, fmt=fmt)

    @classmethod
    def load(cls, filepath: Union[str, Path], config: Optional[TTIConfig] = None) -> "TTIImage":
        return ImageIO.load(filepath, config=config)

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"TTIImage({self.width}×{self.height}, "
            f"{self.bpp}bpp, {self.channels}ch)"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TTIImage):
            return NotImplemented
        return np.array_equal(self._pixels, other._pixels)


# ---------------------------------------------------------------------------
# ImageCanvas  — drawing surface
# ---------------------------------------------------------------------------

class ImageCanvas:
    """
    High-level drawing surface wrapping TTIImage.
    Provides primitive drawing operations (line, rect, circle, ellipse, polygon).
    """

    def __init__(self, img: TTIImage) -> None:
        self.img = img

    # ------------------------------------------------------------------ #
    # Primitives
    # ------------------------------------------------------------------ #

    def line(
        self,
        x0: int, y0: int,
        x1: int, y1: int,
        color: Color,
        thickness: int = 1,
    ) -> None:
        """Bresenham line, optional thickness via perpendicular offset."""
        self._bresenham_line(x0, y0, x1, y1, color)
        if thickness > 1:
            half = thickness // 2
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy) or 1
            px, py = int(-dy / length), int(dx / length)
            for t in range(1, half + 1):
                self._bresenham_line(
                    x0 + px*t, y0 + py*t,
                    x1 + px*t, y1 + py*t, color
                )
                self._bresenham_line(
                    x0 - px*t, y0 - py*t,
                    x1 - px*t, y1 - py*t, color
                )

    def _bresenham_line(
        self,
        x0: int, y0: int,
        x1: int, y1: int,
        color: Color,
    ) -> None:
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self.img.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x0 += sx
            if e2 < dx:
                err += dx; y0 += sy

    def rect(
        self,
        x0: int, y0: int,
        x1: int, y1: int,
        color: Color,
        filled: bool = False,
        thickness: int = 1,
    ) -> None:
        if filled:
            self.img.fill_region(x0, y0, x1+1, y1+1, color)
        else:
            self.line(x0, y0, x1, y0, color, thickness)
            self.line(x1, y0, x1, y1, color, thickness)
            self.line(x1, y1, x0, y1, color, thickness)
            self.line(x0, y1, x0, y0, color, thickness)

    def circle(
        self,
        cx: int, cy: int,
        radius: int,
        color: Color,
        filled: bool = True,
    ) -> None:
        if filled:
            # Vectorised scanline via numpy
            ys = np.arange(max(0, cy-radius), min(self.img.height, cy+radius+1))
            for y in ys:
                h = int(math.sqrt(max(0, radius*radius - (y-cy)*(y-cy))))
                x0, x1 = max(0, cx-h), min(self.img.width, cx+h+1)
                if x0 < x1:
                    self.img._pixels[y, x0:x1] = color[:self.img.channels]
        else:
            x, y, err = radius, 0, 0
            while x >= y:
                for px, py in [
                    (cx+x, cy+y), (cx+y, cy+x),
                    (cx-y, cy+x), (cx-x, cy+y),
                    (cx-x, cy-y), (cx-y, cy-x),
                    (cx+y, cy-x), (cx+x, cy-y),
                ]:
                    self.img.set_pixel(px, py, color)
                y += 1; err += 1 + 2*y
                if 2*(err-x) + 1 > 0:
                    x -= 1; err += 1 - 2*x

    def ellipse(
        self,
        cx: int, cy: int,
        rx: int, ry: int,
        color: Color,
        filled: bool = True,
    ) -> None:
        for y in range(max(0, cy-ry), min(self.img.height, cy+ry+1)):
            ratio = (y-cy)/ry if ry else 0
            w = int(rx * math.sqrt(max(0, 1 - ratio**2)))
            if filled:
                x0, x1 = max(0, cx-w), min(self.img.width, cx+w+1)
                if x0 < x1:
                    self.img._pixels[y, x0:x1] = color[:self.img.channels]
            else:
                self.img.set_pixel(cx-w, y, color)
                self.img.set_pixel(cx+w, y, color)

    def polygon(
        self,
        points: List[Tuple[int, int]],
        color: Color,
        filled: bool = False,
    ) -> None:
        if not points:
            return
        if filled:
            self._fill_polygon(points, color)
        else:
            n = len(points)
            for i in range(n):
                x0, y0 = points[i]
                x1, y1 = points[(i+1) % n]
                self.line(x0, y0, x1, y1, color)

    def _fill_polygon(self, points: List[Tuple[int, int]], color: Color) -> None:
        """Scanline polygon fill."""
        if len(points) < 3:
            return
        ys_all = [p[1] for p in points]
        y_min, y_max = max(0, min(ys_all)), min(self.img.height-1, max(ys_all))
        n = len(points)
        for y in range(y_min, y_max+1):
            intersections: List[float] = []
            for i in range(n):
                x0, y0 = points[i]
                x1, y1 = points[(i+1) % n]
                if (y0 <= y < y1) or (y1 <= y < y0):
                    if y1 != y0:
                        x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                        intersections.append(x)
            intersections.sort()
            for k in range(0, len(intersections)-1, 2):
                xa = max(0, int(intersections[k]))
                xb = min(self.img.width, int(intersections[k+1])+1)
                if xa < xb:
                    self.img._pixels[y, xa:xb] = color[:self.img.channels]

    def text_pixel(
        self,
        x: int, y: int,
        char: str,
        color: Color,
        scale: int = 1,
    ) -> None:
        """
        Render a single ASCII character using a built-in 5×7 bitmap font.
        *scale* multiplies each pixel dot size.
        """
        glyph = _FONT5x7.get(char, _FONT5x7.get('?', []))
        for row_i, row_bits in enumerate(glyph):
            for col_i in range(5):
                if row_bits & (1 << (4 - col_i)):
                    for dy in range(scale):
                        for dx in range(scale):
                            self.img.set_pixel(
                                x + col_i*scale + dx,
                                y + row_i*scale + dy,
                                color,
                            )

    def text(
        self,
        x: int, y: int,
        text: str,
        color: Color,
        scale: int = 1,
    ) -> None:
        """Render a string left-to-right using the pixel font."""
        for i, ch in enumerate(text):
            self.text_pixel(x + i * (6*scale), y, ch, color, scale)

    # ------------------------------------------------------------------ #
    # Gradients
    # ------------------------------------------------------------------ #

    def linear_gradient(
        self,
        c1: Color, c2: Color,
        direction: str = "horizontal",   # "horizontal" | "vertical" | "diagonal"
    ) -> None:
        w, h = self.img.width, self.img.height
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        if direction == "horizontal":
            t = xs / (w - 1) if w > 1 else xs
        elif direction == "vertical":
            t = ys / (h - 1) if h > 1 else ys
        else:  # diagonal
            xx, yy = np.meshgrid(xs / (w-1), ys / (h-1))
            t = (xx + yy) / 2

        c1a = np.array(c1[:3], dtype=np.float32)
        c2a = np.array(c2[:3], dtype=np.float32)

        if direction == "horizontal":
            # t shape (W,) → broadcast over rows
            arr = (c1a[None, None, :] * (1 - t)[None, :, None] +
                   c2a[None, None, :] * t[None, :, None])
        elif direction == "vertical":
            arr = (c1a[None, None, :] * (1 - t)[:, None, None] +
                   c2a[None, None, :] * t[:, None, None])
        else:
            arr = (c1a[None, None, :] * (1 - t)[:, :, None] +
                   c2a[None, None, :] * t[:, :, None])

        self.img._pixels[:, :, :3] = arr.clip(0, 255).astype(np.uint8)

    def radial_gradient(
        self,
        center_color: Color,
        edge_color:   Color,
        cx: Optional[int] = None,
        cy: Optional[int] = None,
    ) -> None:
        w, h = self.img.width, self.img.height
        cx_ = cx if cx is not None else w // 2
        cy_ = cy if cy is not None else h // 2
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        dist = np.sqrt((xs - cx_)**2 + (ys - cy_)**2).astype(np.float32)
        max_d = dist.max() or 1
        t = (dist / max_d).clip(0, 1)
        c1 = np.array(center_color[:3], dtype=np.float32)
        c2 = np.array(edge_color[:3],   dtype=np.float32)
        arr = (c1[None, None, :] * (1 - t)[:, :, None] +
               c2[None, None, :] * t[:, :, None])
        self.img._pixels[:, :, :3] = arr.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ImageIO
# ---------------------------------------------------------------------------

class ImageIO:
    """Multi-format image file I/O."""

    # ---- BMP ---------------------------------------------------------------

    @staticmethod
    def _write_bmp(img: TTIImage, f) -> None:
        w, h = img.width, img.height
        bpp  = img.bpp
        bpx  = bpp // 8
        row_size = ((w * bpx + 3) // 4) * 4
        padding  = row_size - w * bpx
        px_size  = row_size * h
        # File header (14) + Info header (40)
        f.write(b'BM')
        f.write(struct.pack('<I', 54 + px_size))
        f.write(b'\x00\x00\x00\x00')
        f.write(struct.pack('<I', 54))
        # BITMAPINFOHEADER
        f.write(struct.pack('<IiiHHIIiiII',
            40, w, -h,   # negative height → top-down
            1, bpp, 0, px_size, 2835, 2835, 0, 0
        ))
        # Pixel data (BGR / BGRA order)
        for y in range(h):
            row = img._pixels[y]
            for x in range(w):
                px = row[x]
                if bpp == 24:
                    f.write(bytes([px[2], px[1], px[0]]))
                else:
                    f.write(bytes([px[2], px[1], px[0], px[3]]))
            f.write(b'\x00' * padding)

    @staticmethod
    def _read_bmp(f, config: Optional[TTIConfig]) -> TTIImage:
        data = f.read()
        if data[:2] != b'BM':
            raise TTIIOError("Not a BMP file")
        w    = struct.unpack_from('<i', data, 18)[0]
        h_s  = struct.unpack_from('<i', data, 22)[0]
        bpp  = struct.unpack_from('<H', data, 28)[0]
        flipped = h_s > 0
        h    = abs(h_s)
        bpx  = bpp // 8
        row_size = ((w * bpx + 3) // 4) * 4
        off  = struct.unpack_from('<I', data, 10)[0]
        img  = TTIImage(w, h, bpp, config=config)
        for y in range(h):
            src_y = (h-1-y) if flipped else y
            base  = off + src_y * row_size
            for x in range(w):
                p = base + x * bpx
                if bpp == 24:
                    img._pixels[y, x] = [data[p+2], data[p+1], data[p]]
                else:
                    img._pixels[y, x] = [data[p+2], data[p+1], data[p], data[p+3]]
        return img

    # ---- PNG ---------------------------------------------------------------

    @staticmethod
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + chunk_type + data + struct.pack('>I', crc)

    @staticmethod
    def _write_png(img: TTIImage, f) -> None:
        w, h  = img.width, img.height
        ch    = img.channels
        ctype = 6 if ch == 4 else 2   # RGBA or RGB
        raw   = bytearray()
        for y in range(h):
            raw.append(0)           # filter byte: none
            raw.extend(img._pixels[y].flatten().tolist())
        compressed = zlib.compress(bytes(raw), 9)
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(ImageIO._chunk(b'IHDR',
            struct.pack('>IIBBBBB', w, h, 8, ctype, 0, 0, 0)
        ))
        f.write(ImageIO._chunk(b'IDAT', compressed))
        f.write(ImageIO._chunk(b'IEND', b''))

    @staticmethod
    def _read_png(filepath: Union[str, Path], config: Optional[TTIConfig]) -> TTIImage:
        """Delegate to Pillow for PNG reading."""
        try:
            from PIL import Image as PILImage
            pil = PILImage.open(filepath).convert("RGB")
            arr = np.array(pil, dtype=np.uint8)
            img = TTIImage(pil.width, pil.height, 24, config=config)
            img._pixels = arr
            return img
        except ImportError:
            raise TTIIOError("Pillow required to read PNG. Install: pip install pillow")

    # ---- JPEG --------------------------------------------------------------

    @staticmethod
    def _write_jpeg(img: TTIImage, filepath: Union[str, Path], quality: int) -> None:
        try:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(img._pixels[:, :, :3], mode="RGB")
            pil.save(str(filepath), "JPEG", quality=quality)
        except ImportError:
            raise TTIIOError("Pillow required for JPEG. Install: pip install pillow")

    # ---- Public API --------------------------------------------------------

    @staticmethod
    def save(
        img: TTIImage,
        filepath: Union[str, Path],
        fmt: Optional[str] = None,
    ) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = (fmt or path.suffix.lstrip('.')).lower()
        if ext == "bmp":
            with open(path, 'wb') as f:
                ImageIO._write_bmp(img, f)
        elif ext == "png":
            with open(path, 'wb') as f:
                ImageIO._write_png(img, f)
        elif ext in ("jpg", "jpeg"):
            q = img._cfg.image.jpeg_quality
            ImageIO._write_jpeg(img, path, q)
        else:
            raise TTIIOError(f"Unsupported format: '{ext}'")
        return path

    @staticmethod
    def load(
        filepath: Union[str, Path],
        config: Optional[TTIConfig] = None,
    ) -> TTIImage:
        path = Path(filepath)
        if not path.exists():
            raise TTIIOError(f"File not found: {path}")
        ext = path.suffix.lstrip('.').lower()
        if ext == "bmp":
            with open(path, 'rb') as f:
                return ImageIO._read_bmp(f, config)
        elif ext in ("png", "jpg", "jpeg"):
            return ImageIO._read_png(path, config)
        else:
            raise TTIIOError(f"Unsupported format: '{ext}'")

    @staticmethod
    def convert(
        src: Union[str, Path],
        dst: Union[str, Path],
        config: Optional[TTIConfig] = None,
    ) -> Path:
        """Convert any supported format to any other."""
        img = ImageIO.load(src, config)
        return ImageIO.save(img, dst)


# ---------------------------------------------------------------------------
# ImageValidator
# ---------------------------------------------------------------------------

class ImageValidator:
    """Runtime integrity checks for TTIImage objects."""

    @staticmethod
    def assert_valid(img: TTIImage) -> None:
        assert isinstance(img, TTIImage), "Expected TTIImage"
        assert img.width > 0 and img.height > 0, "Dimensions must be positive"
        assert img.bpp in (24, 32), f"bpp must be 24|32, got {img.bpp}"
        expected = (img.height, img.width, img.channels)
        assert img._pixels.shape == expected, (
            f"Pixel buffer shape {img._pixels.shape} ≠ {expected}"
        )

    @staticmethod
    def same_size(a: TTIImage, b: TTIImage) -> bool:
        return a.size == b.size

    @staticmethod
    def same_mode(a: TTIImage, b: TTIImage) -> bool:
        return a.bpp == b.bpp


# ---------------------------------------------------------------------------
# Minimal 5×7 bitmap font (ASCII 32-126)
# ---------------------------------------------------------------------------

_FONT5x7: dict = {
    ' ': [0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    '!': [0x04,0x04,0x04,0x04,0x04,0x00,0x04],
    '"': [0x0A,0x0A,0x00,0x00,0x00,0x00,0x00],
    '#': [0x0A,0x1F,0x0A,0x0A,0x1F,0x0A,0x00],
    '$': [0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04],
    '%': [0x18,0x19,0x02,0x04,0x08,0x13,0x03],
    '&': [0x0C,0x12,0x14,0x08,0x15,0x12,0x0D],
    "'": [0x04,0x04,0x00,0x00,0x00,0x00,0x00],
    '(': [0x02,0x04,0x08,0x08,0x08,0x04,0x02],
    ')': [0x08,0x04,0x02,0x02,0x02,0x04,0x08],
    '*': [0x00,0x04,0x15,0x0E,0x15,0x04,0x00],
    '+': [0x00,0x04,0x04,0x1F,0x04,0x04,0x00],
    ',': [0x00,0x00,0x00,0x00,0x06,0x04,0x08],
    '-': [0x00,0x00,0x00,0x1F,0x00,0x00,0x00],
    '.': [0x00,0x00,0x00,0x00,0x00,0x06,0x06],
    '/': [0x00,0x01,0x02,0x04,0x08,0x10,0x00],
    '0': [0x0E,0x11,0x13,0x15,0x19,0x11,0x0E],
    '1': [0x04,0x0C,0x04,0x04,0x04,0x04,0x0E],
    '2': [0x0E,0x11,0x01,0x02,0x04,0x08,0x1F],
    '3': [0x1F,0x02,0x04,0x02,0x01,0x11,0x0E],
    '4': [0x02,0x06,0x0A,0x12,0x1F,0x02,0x02],
    '5': [0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E],
    '6': [0x06,0x08,0x10,0x1E,0x11,0x11,0x0E],
    '7': [0x1F,0x01,0x02,0x04,0x08,0x08,0x08],
    '8': [0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E],
    '9': [0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C],
    ':': [0x00,0x06,0x06,0x00,0x06,0x06,0x00],
    ';': [0x00,0x06,0x06,0x00,0x06,0x04,0x08],
    '<': [0x02,0x04,0x08,0x10,0x08,0x04,0x02],
    '=': [0x00,0x00,0x1F,0x00,0x1F,0x00,0x00],
    '>': [0x08,0x04,0x02,0x01,0x02,0x04,0x08],
    '?': [0x0E,0x11,0x01,0x02,0x04,0x00,0x04],
    '@': [0x0E,0x11,0x17,0x15,0x17,0x10,0x0E],
    'A': [0x04,0x0A,0x11,0x11,0x1F,0x11,0x11],
    'B': [0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E],
    'C': [0x0E,0x11,0x10,0x10,0x10,0x11,0x0E],
    'D': [0x1C,0x12,0x11,0x11,0x11,0x12,0x1C],
    'E': [0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F],
    'F': [0x1F,0x10,0x10,0x1E,0x10,0x10,0x10],
    'G': [0x0E,0x11,0x10,0x10,0x17,0x11,0x0F],
    'H': [0x11,0x11,0x11,0x1F,0x11,0x11,0x11],
    'I': [0x0E,0x04,0x04,0x04,0x04,0x04,0x0E],
    'J': [0x07,0x02,0x02,0x02,0x02,0x12,0x0C],
    'K': [0x11,0x12,0x14,0x18,0x14,0x12,0x11],
    'L': [0x10,0x10,0x10,0x10,0x10,0x10,0x1F],
    'M': [0x11,0x1B,0x15,0x11,0x11,0x11,0x11],
    'N': [0x11,0x11,0x19,0x15,0x13,0x11,0x11],
    'O': [0x0E,0x11,0x11,0x11,0x11,0x11,0x0E],
    'P': [0x1E,0x11,0x11,0x1E,0x10,0x10,0x10],
    'Q': [0x0E,0x11,0x11,0x11,0x15,0x12,0x0D],
    'R': [0x1E,0x11,0x11,0x1E,0x14,0x12,0x11],
    'S': [0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E],
    'T': [0x1F,0x04,0x04,0x04,0x04,0x04,0x04],
    'U': [0x11,0x11,0x11,0x11,0x11,0x11,0x0E],
    'V': [0x11,0x11,0x11,0x11,0x11,0x0A,0x04],
    'W': [0x11,0x11,0x11,0x15,0x15,0x1B,0x11],
    'X': [0x11,0x11,0x0A,0x04,0x0A,0x11,0x11],
    'Y': [0x11,0x11,0x0A,0x04,0x04,0x04,0x04],
    'Z': [0x1F,0x01,0x02,0x04,0x08,0x10,0x1F],
    '[': [0x0E,0x08,0x08,0x08,0x08,0x08,0x0E],
    '\\': [0x00,0x10,0x08,0x04,0x02,0x01,0x00],
    ']': [0x0E,0x02,0x02,0x02,0x02,0x02,0x0E],
    '^': [0x04,0x0A,0x11,0x00,0x00,0x00,0x00],
    '_': [0x00,0x00,0x00,0x00,0x00,0x00,0x1F],
    '`': [0x08,0x04,0x00,0x00,0x00,0x00,0x00],
    'a': [0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F],
    'b': [0x10,0x10,0x1E,0x11,0x11,0x11,0x1E],
    'c': [0x00,0x00,0x0E,0x11,0x10,0x11,0x0E],
    'd': [0x01,0x01,0x0F,0x11,0x11,0x11,0x0F],
    'e': [0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E],
    'f': [0x06,0x09,0x08,0x1C,0x08,0x08,0x08],
    'g': [0x00,0x0F,0x11,0x11,0x0F,0x01,0x0E],
    'h': [0x10,0x10,0x1E,0x11,0x11,0x11,0x11],
    'i': [0x04,0x00,0x0C,0x04,0x04,0x04,0x0E],
    'j': [0x02,0x00,0x06,0x02,0x02,0x12,0x0C],
    'k': [0x10,0x10,0x11,0x12,0x1C,0x12,0x11],
    'l': [0x0C,0x04,0x04,0x04,0x04,0x04,0x0E],
    'm': [0x00,0x00,0x1A,0x15,0x15,0x11,0x11],
    'n': [0x00,0x00,0x1E,0x11,0x11,0x11,0x11],
    'o': [0x00,0x00,0x0E,0x11,0x11,0x11,0x0E],
    'p': [0x00,0x1E,0x11,0x11,0x1E,0x10,0x10],
    'q': [0x00,0x0F,0x11,0x11,0x0F,0x01,0x01],
    'r': [0x00,0x00,0x16,0x19,0x10,0x10,0x10],
    's': [0x00,0x00,0x0E,0x10,0x0E,0x01,0x0E],
    't': [0x08,0x08,0x1E,0x08,0x08,0x09,0x06],
    'u': [0x00,0x00,0x11,0x11,0x11,0x11,0x0F],
    'v': [0x00,0x00,0x11,0x11,0x11,0x0A,0x04],
    'w': [0x00,0x00,0x11,0x11,0x15,0x15,0x0A],
    'x': [0x00,0x00,0x11,0x0A,0x04,0x0A,0x11],
    'y': [0x00,0x11,0x11,0x0F,0x01,0x11,0x0E],
    'z': [0x00,0x00,0x1F,0x02,0x04,0x08,0x1F],
    '{': [0x06,0x08,0x08,0x10,0x08,0x08,0x06],
    '|': [0x04,0x04,0x04,0x04,0x04,0x04,0x04],
    '}': [0x0C,0x02,0x02,0x01,0x02,0x02,0x0C],
    '~': [0x08,0x15,0x02,0x00,0x00,0x00,0x00],
}
