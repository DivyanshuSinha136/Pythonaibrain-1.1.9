"""
TTI_art.py
==========
Advanced procedural art, visual effects, streaming, animation and
format-conversion for the TTI system.

Replaces byte_image_advanced.py with a fully modular, config-driven design.

Public classes
--------------
ProceduralArt       — fractals, patterns, geometric art
VisualEffects       — filters (blur, sharpen, edge, noise, blend …)
StreamingWriter     — row-by-row large-image writer
AnimationEngine     — frame sequence generator
CustomBitDepth      — experimental N-channel/N-bit pixel format
"""

from __future__ import annotations

import math
import random
import struct
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .TTI_config import TTIConfig, get_config
from .TTI_core import (
    TTIImage, ImageCanvas, ColorUtils,
    Color, Color24, Pixels,
    TTIImageError, TTIIOError,
)


# ---------------------------------------------------------------------------
# ProceduralArt
# ---------------------------------------------------------------------------

class ProceduralArt:
    """
    Generate procedural & fractal art into TTIImage objects.

    All methods are static and return (or modify in-place) TTIImage objects.
    """

    # ---- Primitives (delegate to ImageCanvas) --------------------------

    @staticmethod
    def draw_circle(
        img: TTIImage,
        cx: int, cy: int,
        radius: int,
        color: Color,
        filled: bool = True,
    ) -> None:
        ImageCanvas(img).circle(cx, cy, radius, color, filled)

    @staticmethod
    def draw_line(
        img: TTIImage,
        x0: int, y0: int,
        x1: int, y1: int,
        color: Color,
        thickness: int = 1,
    ) -> None:
        ImageCanvas(img).line(x0, y0, x1, y1, color, thickness)

    @staticmethod
    def draw_triangle(
        img: TTIImage,
        x0: int, y0: int,
        x1: int, y1: int,
        x2: int, y2: int,
        color: Color,
        filled: bool = True,
    ) -> None:
        ImageCanvas(img).polygon([(x0,y0),(x1,y1),(x2,y2)], color, filled)

    @staticmethod
    def draw_rect(
        img: TTIImage,
        x0: int, y0: int,
        x1: int, y1: int,
        color: Color,
        filled: bool = False,
        thickness: int = 1,
    ) -> None:
        ImageCanvas(img).rect(x0, y0, x1, y1, color, filled, thickness)

    @staticmethod
    def draw_ellipse(
        img: TTIImage,
        cx: int, cy: int,
        rx: int, ry: int,
        color: Color,
        filled: bool = True,
    ) -> None:
        ImageCanvas(img).ellipse(cx, cy, rx, ry, color, filled)

    @staticmethod
    def draw_polygon(
        img: TTIImage,
        points: List[Tuple[int, int]],
        color: Color,
        filled: bool = True,
    ) -> None:
        ImageCanvas(img).polygon(points, color, filled)

    # ---- Fractals ----------------------------------------------------------

    @staticmethod
    def mandelbrot_set(
        width:          int,
        height:         int,
        max_iterations: Optional[int] = None,
        zoom:           float = 1.0,
        center_x:       float = -0.5,
        center_y:       float = 0.0,
        config:         Optional[TTIConfig] = None,
    ) -> TTIImage:
        """Generate Mandelbrot set — fully vectorised via numpy."""
        cfg = config or get_config()
        itr = max_iterations or cfg.art.fractal_max_iter
        img = TTIImage(width, height, 24, config=cfg)

        x0s = np.linspace(
            center_x - 1.5/zoom, center_x + 1.5/zoom, width,  dtype=np.float64
        )
        y0s = np.linspace(
            center_y - 1.0/zoom, center_y + 1.0/zoom, height, dtype=np.float64
        )
        C = x0s[None, :] + 1j * y0s[:, None]
        Z = np.zeros_like(C)
        M = np.zeros(C.shape, dtype=np.int32)

        for i in range(itr):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + C[mask]
            M[mask] += 1

        ratio = M.astype(np.float32) / itr
        r = np.clip(9  * (1-ratio) * ratio**3          * 255, 0, 255)
        g = np.clip(15 * (1-ratio)**2 * ratio**2       * 255, 0, 255)
        b = np.clip(8.5 * (1-ratio)**3 * ratio         * 255, 0, 255)
        arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
        img.from_array(arr)
        return img

    @staticmethod
    def julia_set(
        width:          int,
        height:         int,
        c_real:         float = -0.7,
        c_imag:         float = 0.27015,
        max_iterations: Optional[int] = None,
        zoom:           float = 1.0,
        config:         Optional[TTIConfig] = None,
    ) -> TTIImage:
        """Generate Julia set — fully vectorised."""
        cfg = config or get_config()
        itr = max_iterations or cfg.art.fractal_max_iter
        img = TTIImage(width, height, 24, config=cfg)
        c   = complex(c_real, c_imag)

        xs = np.linspace(-1.5/zoom, 1.5/zoom, width,  dtype=np.float64)
        ys = np.linspace(-1.0/zoom, 1.0/zoom, height, dtype=np.float64)
        Z  = xs[None, :] + 1j * ys[:, None]
        M  = np.zeros(Z.shape, dtype=np.int32)

        for i in range(itr):
            mask = np.abs(Z) < 2
            Z[mask] = Z[mask]**2 + c
            M[mask] += 1

        ratio = M.astype(np.float32) / itr
        r = np.clip(255 * ratio,       0, 255)
        g = np.clip(255 * ratio**2,    0, 255)
        b = np.clip(255 * ratio**0.5,  0, 255)
        arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
        img.from_array(arr)
        return img

    @staticmethod
    def sierpinski_triangle(
        width:  int,
        height: int,
        depth:  int = 7,
        color:  Color = (255, 255, 255),
        bg:     Color = (0, 0, 0),
        config: Optional[TTIConfig] = None,
    ) -> TTIImage:
        """Recursive Sierpiński triangle."""
        cfg = config or get_config()
        img = TTIImage(width, height, 24, background=bg[:3], config=cfg)
        canvas = ImageCanvas(img)

        def draw(ax,ay, bx,by, cx,cy, d):
            if d == 0:
                canvas.polygon([(ax,ay),(bx,by),(cx,cy)], color, filled=True)
                return
            mx1,my1 = (ax+bx)//2, (ay+by)//2
            mx2,my2 = (bx+cx)//2, (by+cy)//2
            mx3,my3 = (ax+cx)//2, (ay+cy)//2
            draw(ax,ay,  mx1,my1, mx3,my3, d-1)
            draw(mx1,my1, bx,by,  mx2,my2, d-1)
            draw(mx3,my3, mx2,my2, cx,cy,  d-1)

        pad = 20
        draw(width//2, pad,
             width-pad, height-pad,
             pad,        height-pad, depth)
        return img

    @staticmethod
    def generate_pattern(
        width:        int,
        height:       int,
        pattern_func: Callable[[int, int, int, int], Color],
        config:       Optional[TTIConfig] = None,
    ) -> TTIImage:
        """
        Generate a pixel pattern via a user-supplied function
        ``(x, y, width, height) → Color``.
        """
        cfg = config or get_config()
        img = TTIImage(width, height, 24, config=cfg)
        xs  = np.arange(width)
        ys  = np.arange(height)
        for y in ys:
            for x in xs:
                img.set_pixel(int(x), int(y), pattern_func(int(x), int(y), width, height))
        return img

    @staticmethod
    def plasma(
        width:  int,
        height: int,
        scale:  float = 0.05,
        config: Optional[TTIConfig] = None,
    ) -> TTIImage:
        """Classic plasma / interference-wave effect."""
        cfg = config or get_config()
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        v  = (
            np.sin(xs * scale) +
            np.sin(ys * scale) +
            np.sin((xs + ys) * scale * 0.7) +
            np.sin(np.sqrt(xs**2 + ys**2) * scale * 0.5)
        )
        v  = (v - v.min()) / (v.max() - v.min() + 1e-9)
        r  = np.clip(np.sin(v * np.pi * 2)           * 127 + 128, 0, 255)
        g  = np.clip(np.sin(v * np.pi * 2 + 2*np.pi/3) * 127+128, 0, 255)
        b  = np.clip(np.sin(v * np.pi * 2 + 4*np.pi/3) * 127+128, 0, 255)
        arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
        img = TTIImage(width, height, 24, config=cfg)
        img.from_array(arr)
        return img

    @staticmethod
    def voronoi(
        width:    int,
        height:   int,
        n_cells:  int = 20,
        seed:     Optional[int] = None,
        config:   Optional[TTIConfig] = None,
    ) -> TTIImage:
        """Voronoi diagram with random cell colours."""
        cfg  = config or get_config()
        rng  = np.random.default_rng(seed)
        pts  = rng.integers([[0,0]], [[width,height]], size=(n_cells, 2))
        cols = rng.integers(0, 255, size=(n_cells, 3), dtype=np.uint8)
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        # Find nearest seed for each pixel
        best_d = np.full((height, width), np.inf)
        best_i = np.zeros((height, width), dtype=np.int32)
        for i, (px, py) in enumerate(pts):
            d = (xs - px)**2 + (ys - py)**2
            closer = d < best_d
            best_d[closer] = d[closer]
            best_i[closer] = i
        arr = cols[best_i]
        img = TTIImage(width, height, 24, config=cfg)
        img.from_array(arr)
        return img

    @staticmethod
    def perlin_noise_image(
        width:  int,
        height: int,
        scale:  float = 0.05,
        octaves: int = 4,
        config: Optional[TTIConfig] = None,
    ) -> TTIImage:
        """
        Simple Perlin-like noise using layered sinusoids.
        Pure numpy — no external noise lib required.
        """
        cfg = config or get_config()
        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32)
        )
        noise = np.zeros((height, width), dtype=np.float32)
        amp, freq = 1.0, scale
        for _ in range(octaves):
            phase_x = np.random.uniform(0, 2*np.pi)
            phase_y = np.random.uniform(0, 2*np.pi)
            noise += amp * (
                np.sin(xs * freq + phase_x) * np.cos(ys * freq + phase_y)
            )
            amp  *= 0.5
            freq *= 2.0
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
        gray  = (noise * 255).astype(np.uint8)
        arr   = np.stack([gray, gray, gray], axis=-1)
        img   = TTIImage(width, height, 24, config=cfg)
        img.from_array(arr)
        return img


# ---------------------------------------------------------------------------
# VisualEffects
# ---------------------------------------------------------------------------

class VisualEffects:
    """Pixel-level visual effects using numpy convolutions."""

    @staticmethod
    def _convolve(img: TTIImage, kernel: np.ndarray) -> TTIImage:
        """Apply a 2D convolution kernel to each channel."""
        from numpy.lib.stride_tricks import sliding_window_view
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        src   = img._pixels.astype(np.float32)
        pad   = np.pad(src, ((ph,ph),(pw,pw),(0,0)), mode='edge')
        out   = np.zeros_like(src)
        for dy in range(kh):
            for dx in range(kw):
                out += kernel[dy, dx] * pad[dy:dy+img.height, dx:dx+img.width]
        result = TTIImage(img.width, img.height, img.bpp, config=img._cfg)
        result.from_array(np.clip(out, 0, 255).astype(np.uint8))
        return result

    @staticmethod
    def apply_blur(img: TTIImage, radius: Optional[int] = None) -> TTIImage:
        """Box blur."""
        r   = radius if radius is not None else get_config().art.blur_default_radius
        sz  = 2 * r + 1
        k   = np.ones((sz, sz), dtype=np.float32) / (sz * sz)
        return VisualEffects._convolve(img, k)

    @staticmethod
    def apply_gaussian_blur(img: TTIImage, sigma: float = 1.0) -> TTIImage:
        """Gaussian blur."""
        radius = max(1, int(3 * sigma))
        sz = 2 * radius + 1
        k  = np.zeros((sz, sz), dtype=np.float32)
        cx, cy = sz // 2, sz // 2
        for y in range(sz):
            for x in range(sz):
                k[y,x] = math.exp(-((x-cx)**2+(y-cy)**2)/(2*sigma**2))
        k /= k.sum()
        return VisualEffects._convolve(img, k)

    @staticmethod
    def apply_sharpen(img: TTIImage) -> TTIImage:
        k = np.array([[ 0,-1, 0],
                      [-1, 5,-1],
                      [ 0,-1, 0]], dtype=np.float32)
        return VisualEffects._convolve(img, k)

    @staticmethod
    def apply_edge_detect(img: TTIImage) -> TTIImage:
        """Sobel edge detection (returns grayscale-ish edge image)."""
        kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        ky = kx.T
        ex = VisualEffects._convolve(img, kx)._pixels.astype(np.float32)
        ey = VisualEffects._convolve(img, ky)._pixels.astype(np.float32)
        edge = np.sqrt(ex**2 + ey**2)
        result = TTIImage(img.width, img.height, img.bpp, config=img._cfg)
        result.from_array(np.clip(edge, 0, 255).astype(np.uint8))
        return result

    @staticmethod
    def apply_emboss(img: TTIImage) -> TTIImage:
        k = np.array([[-2,-1,0],
                      [-1, 1,1],
                      [ 0, 1,2]], dtype=np.float32)
        return VisualEffects._convolve(img, k)

    @staticmethod
    def apply_grayscale(img: TTIImage) -> TTIImage:
        arr  = img._pixels.astype(np.float32)
        gray = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2])
        gray = gray.astype(np.uint8)
        out  = np.stack([gray,gray,gray], axis=-1)
        if img.channels == 4:
            out = np.concatenate([out, img._pixels[:,:,3:4]], axis=-1)
        result = TTIImage(img.width, img.height, img.bpp, config=img._cfg)
        result.from_array(out)
        return result

    @staticmethod
    def apply_sepia(img: TTIImage) -> TTIImage:
        arr = img._pixels.astype(np.float32)
        r   = np.clip(arr[:,:,0]*0.393 + arr[:,:,1]*0.769 + arr[:,:,2]*0.189, 0,255)
        g   = np.clip(arr[:,:,0]*0.349 + arr[:,:,1]*0.686 + arr[:,:,2]*0.168, 0,255)
        b   = np.clip(arr[:,:,0]*0.272 + arr[:,:,1]*0.534 + arr[:,:,2]*0.131, 0,255)
        out = np.stack([r,g,b], axis=-1).astype(np.uint8)
        result = TTIImage(img.width, img.height, img.bpp, config=img._cfg)
        result.from_array(out)
        return result

    @staticmethod
    def apply_invert(img: TTIImage) -> TTIImage:
        arr = img._pixels.copy()
        arr[:,:,:3] = 255 - arr[:,:,:3]
        result = TTIImage(img.width, img.height, img.bpp, config=img._cfg)
        result.from_array(arr)
        return result

    @staticmethod
    def adjust_brightness(img: TTIImage, factor: float = 1.0) -> TTIImage:
        arr = np.clip(img._pixels.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        result = TTIImage(img.width, img.height, img.bpp, config=img._cfg)
        result.from_array(arr)
        return result

    @staticmethod
    def adjust_contrast(img: TTIImage, factor: float = 1.0) -> TTIImage:
        arr = img._pixels.astype(np.float32)
        mean = arr[:,:,:3].mean()
        arr[:,:,:3] = np.clip((arr[:,:,:3] - mean) * factor + mean, 0, 255)
        result = TTIImage(img.width, img.height, img.bpp, config=img._cfg)
        result.from_array(arr.astype(np.uint8))
        return result

    @staticmethod
    def add_noise(
        img:       TTIImage,
        intensity: Optional[float] = None,
        noise_type: str = "gaussian",   # "gaussian" | "salt_pepper"
    ) -> TTIImage:
        cfg = img._cfg
        amt = intensity if intensity is not None else cfg.art.noise_default_intensity
        arr = img._pixels.astype(np.float32).copy()
        if noise_type == "gaussian":
            noise = np.random.normal(0, amt * 255, arr[:,:,:3].shape)
            arr[:,:,:3] = np.clip(arr[:,:,:3] + noise, 0, 255)
        else:  # salt & pepper
            mask_s = np.random.rand(*arr.shape[:2]) < amt / 2
            mask_p = np.random.rand(*arr.shape[:2]) < amt / 2
            arr[mask_s, :3] = 255
            arr[mask_p, :3] = 0
        result = TTIImage(img.width, img.height, img.bpp, config=img._cfg)
        result.from_array(arr.astype(np.uint8))
        return result

    @staticmethod
    def blend_images(
        img1:  TTIImage,
        img2:  TTIImage,
        alpha: float = 0.5,
    ) -> TTIImage:
        if img1.size != img2.size:
            raise TTIImageError("Images must be the same size to blend")
        a1 = img1._pixels.astype(np.float32)
        a2 = img2._pixels.astype(np.float32)
        if img1.channels != img2.channels:
            a1 = a1[:,:,:3]
            a2 = a2[:,:,:3]
        out = np.clip(a1*(1-alpha) + a2*alpha, 0, 255).astype(np.uint8)
        result = TTIImage(img1.width, img1.height,
                          min(img1.bpp, img2.bpp), config=img1._cfg)
        result.from_array(out)
        return result

    @staticmethod
    def overlay(
        base:    TTIImage,
        overlay: TTIImage,
        x:       int = 0,
        y:       int = 0,
        alpha:   float = 1.0,
    ) -> TTIImage:
        """Paste *overlay* on top of *base* at offset (x,y)."""
        result = base.copy()
        ow, oh = overlay.width, overlay.height
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(base.width,  x + ow), min(base.height, y + oh)
        ox0, oy0 = x0 - x, y0 - y
        ox1, oy1 = ox0 + (x1-x0), oy0 + (y1-y0)
        src = overlay._pixels[oy0:oy1, ox0:ox1, :3].astype(np.float32)
        dst = result._pixels[y0:y1, x0:x1, :3].astype(np.float32)
        result._pixels[y0:y1, x0:x1, :3] = np.clip(
            dst*(1-alpha) + src*alpha, 0, 255
        ).astype(np.uint8)
        return result

    @staticmethod
    def create_radial_gradient(
        width:        int,
        height:       int,
        center_color: Color,
        edge_color:   Color,
        cx:           Optional[int] = None,
        cy:           Optional[int] = None,
        config:       Optional[TTIConfig] = None,
    ) -> TTIImage:
        cfg = config or get_config()
        img = TTIImage(width, height, 24, config=cfg)
        ImageCanvas(img).radial_gradient(
            center_color, edge_color,
            cx if cx is not None else width//2,
            cy if cy is not None else height//2,
        )
        return img

    @staticmethod
    def create_linear_gradient(
        width:     int,
        height:    int,
        c1:        Color,
        c2:        Color,
        direction: str = "horizontal",
        config:    Optional[TTIConfig] = None,
    ) -> TTIImage:
        cfg = config or get_config()
        img = TTIImage(width, height, 24, config=cfg)
        ImageCanvas(img).linear_gradient(c1, c2, direction)
        return img

    @staticmethod
    def pixelate(img: TTIImage, block_size: int = 10) -> TTIImage:
        """Pixelate by averaging block_size×block_size tiles."""
        arr  = img._pixels.astype(np.float32)
        out  = arr.copy()
        h, w = img.height, img.width
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                tile = arr[y:y+block_size, x:x+block_size]
                mean = tile.mean(axis=(0,1), keepdims=True)
                out[y:y+block_size, x:x+block_size] = mean
        result = TTIImage(w, h, img.bpp, config=img._cfg)
        result.from_array(out.astype(np.uint8))
        return result

    @staticmethod
    def vignette(
        img:      TTIImage,
        strength: float = 0.5,
    ) -> TTIImage:
        """Apply a radial vignette (darken edges)."""
        w, h = img.width, img.height
        xs, ys = np.meshgrid(
            np.linspace(-1, 1, w),
            np.linspace(-1, 1, h)
        )
        dist = np.sqrt(xs**2 + ys**2)
        mask = 1 - np.clip(dist * strength, 0, 1)
        arr  = img._pixels.astype(np.float32)
        arr[:,:,:3] = np.clip(arr[:,:,:3] * mask[:,:,None], 0, 255)
        result = TTIImage(w, h, img.bpp, config=img._cfg)
        result.from_array(arr.astype(np.uint8))
        return result


# ---------------------------------------------------------------------------
# StreamingWriter
# ---------------------------------------------------------------------------

class StreamingWriter:
    """
    Write BMP images row-by-row for memory-efficient large-image generation.
    Supports images of arbitrary size without holding the full buffer in RAM.

    Usage (context-manager)::

        with StreamingWriter("big.bmp", 8000, 8000) as w:
            w.generate_rows(lambda y: [(r,g,b)] * 8000)
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        width:    int,
        height:   int,
        bpp:      int = 24,
        config:   Optional[TTIConfig] = None,
    ) -> None:
        self._cfg     = config or get_config()
        self.filepath = Path(filepath)
        self.width    = width
        self.height   = height
        self.bpp      = bpp
        self.bpx      = bpp // 8
        self.row_size = ((width * self.bpx + 3) // 4) * 4
        self.padding  = self.row_size - width * self.bpx
        self._file    = None

    def __enter__(self) -> "StreamingWriter":
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, 'wb')
        self._write_headers()
        return self

    def __exit__(self, *_) -> None:
        if self._file:
            self._file.close()

    def _write_headers(self) -> None:
        px_size   = self.row_size * self.height
        file_size = 54 + px_size
        h_bytes   = struct.pack('<i', -self.height)  # negative = top-down
        self._file.write(b'BM')
        self._file.write(struct.pack('<I', file_size))
        self._file.write(b'\x00\x00\x00\x00')
        self._file.write(struct.pack('<I', 54))
        self._file.write(struct.pack('<I', 40))
        self._file.write(struct.pack('<i', self.width))
        self._file.write(h_bytes)
        self._file.write(struct.pack('<H', 1))
        self._file.write(struct.pack('<H', self.bpp))
        self._file.write(struct.pack('<I', 0))
        self._file.write(struct.pack('<I', px_size))
        self._file.write(struct.pack('<ii', 2835, 2835))
        self._file.write(b'\x00' * 8)

    def write_row(self, row: List[Color]) -> None:
        buf = bytearray()
        for c in row:
            if self.bpx == 3:
                buf += bytes([c[2], c[1], c[0]])
            else:
                buf += bytes([c[2], c[1], c[0], c[3] if len(c)>3 else 255])
        buf += b'\x00' * self.padding
        self._file.write(buf)

    def generate_rows(
        self,
        row_func: Callable[[int], List[Color]],
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        for y in range(self.height):
            self.write_row(row_func(y))
            if progress_cb and y % 100 == 0:
                progress_cb(y, self.height)


# ---------------------------------------------------------------------------
# AnimationEngine
# ---------------------------------------------------------------------------

class AnimationEngine:
    """Generate BMP frame sequences."""

    def __init__(
        self,
        width:  int,
        height: int,
        bpp:    int = 24,
        fps:    Optional[int] = None,
        config: Optional[TTIConfig] = None,
    ) -> None:
        self._cfg   = config or get_config()
        self.width  = width
        self.height = height
        self.bpp    = bpp
        self.fps    = fps or self._cfg.art.animation_fps
        self._frame = 0

    def generate_frame(
        self,
        draw_func: Callable[[int, TTIImage], None],
    ) -> TTIImage:
        img = TTIImage(self.width, self.height, self.bpp, config=self._cfg)
        draw_func(self._frame, img)
        self._frame += 1
        return img

    def save_sequence(
        self,
        output_dir:  Union[str, Path],
        num_frames:  int,
        draw_func:   Callable[[int, TTIImage], None],
        prefix:      str = "frame",
        fmt:         str = "bmp",
    ) -> List[Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for _ in range(num_frames):
            img  = self.generate_frame(draw_func)
            path = out / f"{prefix}_{self._frame-1:04d}.{fmt}"
            img.save(path)
            paths.append(path)
        return paths

    @staticmethod
    def rotation_sequence(
        width:      int,
        height:     int,
        num_frames: int,
        output_dir: Union[str, Path],
        config:     Optional[TTIConfig] = None,
    ) -> List[Path]:
        cfg  = config or get_config()
        anim = AnimationEngine(width, height, config=cfg)

        def draw(frame: int, img: TTIImage) -> None:
            img.fill((0, 0, 0))
            angle  = 2 * math.pi * frame / num_frames
            cx, cy = width//2, height//2
            size   = min(width, height) // 4
            pts    = [
                (cx + int(size*math.cos(angle + i*math.pi/2)),
                 cy + int(size*math.sin(angle + i*math.pi/2)))
                for i in range(4)
            ]
            canvas = ImageCanvas(img)
            for i in range(4):
                canvas.line(*pts[i], *pts[(i+1)%4], (255, 220, 0), thickness=2)

        return anim.save_sequence(output_dir, num_frames, draw)


# ---------------------------------------------------------------------------
# CustomBitDepth
# ---------------------------------------------------------------------------

class CustomBitDepth:
    """
    Experimental N-channel / N-bit-per-channel image format.

    Supports arbitrary precision (e.g. 16-bit HDR, 48-bit, 50-bit, 10-channel).
    Stores as a plain bytearray with a custom binary header.
    """

    MAGIC = b'TTIBDEP\x00'   # 8 bytes
    VERSION = b'v2.0\x00\x00\x00\x00'

    def __init__(
        self,
        width:            int,
        height:           int,
        bits_per_channel: int,
        num_channels:     int,
    ) -> None:
        self.width            = width
        self.height           = height
        self.bits_per_channel = bits_per_channel
        self.num_channels     = num_channels
        self.bits_per_pixel   = bits_per_channel * num_channels
        self.max_value        = (1 << bits_per_channel) - 1
        self._pixels: List[List[List[int]]] = [
            [[0]*num_channels for _ in range(width)]
            for _ in range(height)
        ]

    # ------------------------------------------------------------------ #

    def set_pixel(self, x: int, y: int, values: List[int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self._pixels[y][x] = [
                max(0, min(self.max_value, v)) for v in values[:self.num_channels]
            ]

    def get_pixel(self, x: int, y: int) -> Optional[List[int]]:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self._pixels[y][x][:]
        return None

    # ------------------------------------------------------------------ #

    def save(self, filepath: Union[str, Path]) -> None:
        with open(filepath, 'wb') as f:
            f.write(self.MAGIC)
            f.write(self.VERSION)
            f.write(struct.pack('<IIHH',
                self.width, self.height,
                self.bits_per_channel, self.num_channels
            ))
            bpx = (self.bits_per_pixel + 7) // 8
            for y in range(self.height):
                for x in range(self.width):
                    buf  = bytearray(bpx)
                    offs = 0
                    for val in self._pixels[y][x]:
                        for b in range(self.bits_per_channel):
                            if val & (1 << b):
                                by, bi = divmod(offs, 8)
                                buf[by] |= (1 << bi)
                            offs += 1
                    f.write(buf)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CustomBitDepth":
        with open(filepath, 'rb') as f:
            magic = f.read(8)
            if magic not in (cls.MAGIC, b'CUSTIMG\x00'):
                raise TTIIOError("Not a valid CustomBitDepth file")
            f.read(8)  # version
            w, h, bpc, nch = struct.unpack('<IIHH', f.read(12))
            img = cls(w, h, bpc, nch)
            bpx = (bpc*nch + 7) // 8
            for y in range(h):
                for x in range(w):
                    buf  = f.read(bpx)
                    vals, offs = [], 0
                    for _ in range(nch):
                        v = 0
                        for b in range(bpc):
                            by, bi = divmod(offs, 8)
                            if buf[by] & (1 << bi):
                                v |= (1 << b)
                            offs += 1
                        vals.append(v)
                    img._pixels[y][x] = vals
        return img

    def to_tti_image(self, config: Optional[TTIConfig] = None) -> TTIImage:
        """Convert first 3 channels to a 24-bit TTIImage (lossy if >8bpc)."""
        img = TTIImage(self.width, self.height, 24, config=config)
        for y in range(self.height):
            for x in range(self.width):
                ch = self._pixels[y][x]
                def scale(v):
                    return int(v / self.max_value * 255) if self.max_value else 0
                rgb = tuple(scale(ch[i]) if i < len(ch) else 0 for i in range(3))
                img.set_pixel(x, y, rgb)
        return img

    def __repr__(self) -> str:
        return (
            f"CustomBitDepth({self.width}×{self.height}, "
            f"{self.bits_per_channel}bpc × {self.num_channels}ch, "
            f"max={self.max_value})"
        )
