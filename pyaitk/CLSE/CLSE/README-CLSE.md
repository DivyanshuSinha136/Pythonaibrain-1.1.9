# TTI — Text-To-Image System

A modular, pure-Python text-to-image framework built on **NumPy**, **NLTK**, and **scikit-learn**.  
Converts natural language prompts into images through an NLP pipeline, a VAE-style neural
architecture, and a multi-strategy procedural renderer — no GPU or internet connection required.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
  - [TTI\_config.py](#tti_configpy)
  - [TTI\_core.py](#tti_corepy)
  - [TTI\_art.py](#tti_artpy)
  - [TTI\_ai.py](#tti_aipy)
  - [TTI\_main.py](#tti_mainpy)
- [CLI Reference](#cli-reference)
- [Configuration Reference](#configuration-reference)
- [AI Pipeline Deep Dive](#ai-pipeline-deep-dive)
- [PyTorch Upgrade Path](#pytorch-upgrade-path)
- [Examples](#examples)
- [Demo](#demo)
- [License](#license)

---

## Overview

TTI takes a plain-English description and produces an image by:

1. **Parsing** the prompt with NLTK (tokenisation, PoS tagging, stopword filtering, stemming)
2. **Predicting** a colour palette with a TF-IDF + KNN classifier trained on 100+ colour/scene keywords
3. **Encoding** the analysis into a 128-dimensional latent code via a numpy VAE (526 k parameters)
4. **Decoding** the latent into scene parameters driving one of 14 specialised renderers
5. **Post-processing** with a configurable filter chain (blur, sharpen, vignette, sepia, noise …)

Everything runs on plain NumPy arrays; PyTorch is a drop-in swap (see
[PyTorch Upgrade Path](#pytorch-upgrade-path)).

---

## Architecture

```
  Prompt (str)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  NLPAnalyser   (TTI_ai.py)                          │
│  NLTK tokenise → PoS tag → colour/scene/modifier    │
│  extraction → PromptAnalysis                        │
└───────────────────┬─────────────────────────────────┘
                    │  PromptAnalysis
                    ▼
┌─────────────────────────────────────────────────────┐
│  ColourPredictor   (TTI_ai.py)                      │
│  TF-IDF vectoriser → PCA(32) → KNN → palette[6]    │
└───────────────────┬─────────────────────────────────┘
                    │  colour palette
                    ▼
┌─────────────────────────────────────────────────────┐
│  SceneComposer / VAE   (TTI_ai.py)                  │
│  embed(256) → Linear(512) → ReLU → μ,σ(128)        │
│  reparameterise → z(128)                            │
│  z → Linear(256) → ReLU → Linear(512) → params(64) │
└───────────────────┬─────────────────────────────────┘
                    │  SceneParameters
                    ▼
┌─────────────────────────────────────────────────────┐
│  ImageDecoder  (TTI_ai.py)                          │
│  14 scene-type renderers → TTIImage                 │
│  + post-processing filter chain                     │
└───────────────────┬─────────────────────────────────┘
                    │  TTIImage
                    ▼
              PNG / BMP / JPEG
```

### Scene types recognised

| Scene type   | Triggered by keywords                              |
|--------------|----------------------------------------------------|
| `gradient`   | sky, ocean, sea, sunset, sunrise, landscape        |
| `starfield`  | night, star, universe, space                       |
| `fractal`    | galaxy, forest, space, nebula                      |
| `mandelbrot` | mandelbrot                                         |
| `julia`      | julia                                              |
| `sierpinski` | sierpinski                                         |
| `plasma`     | fire, flame, lava                                  |
| `ripple`     | water, rain, wave                                  |
| `voronoi`    | voronoi                                            |
| `noise`      | noise                                              |
| `geometric`  | city, building, architecture, geometric            |
| `organic`    | tree, flower, garden, nature                       |
| `circles`    | circle                                             |
| `spiral`     | spiral                                             |
| `abstract`   | abstract, pattern, art *(fallback)*                |

---

## File Structure

```
TTI_config.py    200 lines   Settings & configuration singleton
TTI_core.py      847 lines   Pixel engine, drawing, I/O
TTI_art.py       840 lines   Procedural art, effects, streaming, animation
TTI_ai.py      1,194 lines   NLP, VAE model, renderers
TTI_main.py      501 lines   Pipeline façade & CLI
─────────────────────────────
Total          3,582 lines
```

---

## Requirements

| Package        | Version  | Role                                     |
|----------------|----------|------------------------------------------|
| `numpy`        | ≥ 1.24   | Tensor backend, all pixel operations     |
| `scikit-learn` | ≥ 1.3    | TF-IDF, PCA, KNN colour predictor        |
| `nltk`         | ≥ 3.8    | Tokenisation, PoS tagging, stemming      |
| `Pillow`       | ≥ 9.0    | PNG/JPEG read & write *(optional)*       |
| `torch`        | ≥ 2.0    | GPU acceleration *(optional — see below)*|
| `spacy`        | ≥ 3.6    | Alternative NLP backend *(optional)*     |

> **Pillow** is only required if reading/writing PNG or JPEG. BMP works with zero dependencies beyond numpy.

---

## Installation

```bash
# Minimum (BMP output only)
pip install numpy scikit-learn nltk

# Recommended (PNG/JPEG support)
pip install numpy scikit-learn nltk pillow

# Full (GPU support + alternative NLP)
pip install numpy scikit-learn nltk pillow torch torchvision spacy
python -m spacy download en_core_web_sm

# Download NLTK data (first run)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"
```

> The system degrades gracefully when optional packages are absent — NLTK data
> falling back to a built-in tokeniser and a 100+ keyword stopword list if
> offline.

---

## Quick Start

### Python API

```python
from TTI_main import TTIPipeline

pipe = TTIPipeline()

# Generate from a text prompt
img = pipe.generate("a calm blue ocean at sunset", output="ocean.png", seed=42)

# Procedural art (no prompt needed)
pipe.art("mandelbrot", output="fractal.png", width=800, height=600)

# Apply an effect to any existing image
pipe.effect("sepia", "photo.png", output="vintage.png")

# Generate 4 variations of one prompt
pipe.generate_variations("neon city at night", n=4, output_dir="vars/")

# Interpolate between two prompts (6-frame sequence)
pipe.interpolate("sunrise", "midnight", steps=6, output_dir="interp/")
```

### CLI

```bash
python TTI_main.py generate "red fire flame glowing" --output fire.png
python TTI_main.py art voronoi --n-cells 30 --output voronoi.png
python TTI_main.py effects blur photo.png --radius 4 --output blurred.png
python TTI_main.py demo
```

---

## Module Reference

### TTI\_config.py

Central settings singleton. All tuneable parameters live here; no magic constants are
scattered across the codebase.

```python
from TTI_config import get_config, update_config, reset_config, TTIConfig

cfg = get_config()                          # global singleton
cfg.image.default_width = 1024             # set directly
update_config(ai={"seed": 42},             # or bulk-update by section
              image={"default_width": 512})

cfg.save("my_project.json")               # persist to disk
cfg2 = TTIConfig.load("my_project.json")  # restore later
```

**Sub-configs**

| Class         | Key fields                                                          |
|---------------|---------------------------------------------------------------------|
| `ImageConfig` | `default_width`, `default_height`, `default_bpp`, `default_format`, `background_color`, `jpeg_quality` |
| `AIConfig`    | `nlp_backend`, `model_type`, `latent_dim`, `hidden_dim`, `text_embed_dim`, `num_inference_steps`, `seed` |
| `ArtConfig`   | `fractal_max_iter`, `blur_default_radius`, `noise_default_intensity`, `animation_fps` |
| `PathConfig`  | `output_dir`, `model_dir`, `cache_dir`, `log_dir`                  |
| `LogConfig`   | `level`, `log_to_file`, `log_filename`, `show_progress`            |

---

### TTI\_core.py

Low-level pixel engine. Everything operates on NumPy `uint8` arrays of shape `(H, W, C)`.

#### TTIImage

```python
from TTI_core import TTIImage

img = TTIImage(512, 512, bpp=24, background=(30, 30, 80))

img.set_pixel(100, 200, (255, 0, 0))
color = img.get_pixel(100, 200)          # → (255, 0, 0)
img.fill((0, 0, 0))
img.fill_region(10, 10, 200, 200, (255, 255, 0))

cropped  = img.crop(50, 50, 300, 300)
resized  = img.resize(256, 256)
flipped  = img.flip_horizontal()
rotated  = img.rotate_90(clockwise=True)
clone    = img.copy()

img.save("output.png")                   # BMP / PNG / JPEG auto-detected
img2 = TTIImage.load("photo.png")

print(img.size)    # (512, 512)
print(img.array)   # numpy array (H, W, C)
```

#### ImageCanvas

Drawing primitives on top of any `TTIImage`:

```python
from TTI_core import ImageCanvas

canvas = ImageCanvas(img)
canvas.line(0, 0, 511, 511, (255, 255, 0), thickness=2)
canvas.rect(50, 50, 200, 200, (0, 255, 0), filled=True)
canvas.circle(256, 256, 100, (255, 100, 0), filled=False)
canvas.ellipse(256, 256, 120, 60, (200, 0, 200), filled=True)
canvas.polygon([(100,50),(200,200),(50,200)], (0,200,255), filled=True)
canvas.text(10, 10, "Hello TTI", (255, 255, 255), scale=2)
canvas.linear_gradient((255, 0, 0), (0, 0, 255), direction="diagonal")
canvas.radial_gradient((255, 220, 0), (10, 0, 60))
```

#### ColorUtils

```python
from TTI_core import ColorUtils

rgb  = ColorUtils.hsv_to_rgb(0.6, 0.8, 0.9)    # → (46, 138, 230)
h,s,v = ColorUtils.rgb_to_hsv(46, 138, 230)
mid  = ColorUtils.lerp((255,0,0), (0,0,255), 0.5)
gray = ColorUtils.grayscale((120, 80, 200))
inv  = ColorUtils.invert((100, 150, 200))
lum  = ColorUtils.luminance((200, 100, 50))
```

#### ImageIO

```python
from TTI_core import ImageIO

img  = ImageIO.load("photo.jpg")
path = ImageIO.save(img, "output.png")
ImageIO.convert("photo.bmp", "photo.png")   # format conversion
```

---

### TTI\_art.py

#### ProceduralArt — 13 generators

```python
from TTI_art import ProceduralArt

# Fractals (vectorised with NumPy)
img = ProceduralArt.mandelbrot_set(800, 600, max_iterations=256, zoom=1.5)
img = ProceduralArt.julia_set(800, 600, c_real=-0.7, c_imag=0.27015)
img = ProceduralArt.sierpinski_triangle(600, 600, depth=7)

# Noise & patterns
img = ProceduralArt.plasma(512, 512, scale=0.05)
img = ProceduralArt.voronoi(512, 512, n_cells=30, seed=42)
img = ProceduralArt.perlin_noise_image(512, 512, scale=0.04, octaves=5)

# Custom pixel function
def checker(x, y, w, h):
    return (255, 255, 255) if (x//40 + y//40) % 2 == 0 else (0, 0, 0)
img = ProceduralArt.generate_pattern(512, 512, checker)

# Drawing primitives (delegate to ImageCanvas)
ProceduralArt.draw_circle(img, 256, 256, 100, (255, 0, 0), filled=True)
ProceduralArt.draw_triangle(img, 100, 400, 256, 50, 412, 400, (0,255,0))
ProceduralArt.draw_rect(img, 20, 20, 200, 200, (0, 0, 255), filled=False)
ProceduralArt.draw_ellipse(img, 256, 256, 150, 80, (200, 100, 0))
ProceduralArt.draw_polygon(img, [(x,y), ...], (255, 200, 0), filled=True)
```

#### VisualEffects — 13 filters

```python
from TTI_art import VisualEffects

blurred   = VisualEffects.apply_blur(img, radius=3)
gblurred  = VisualEffects.apply_gaussian_blur(img, sigma=2.0)
sharpened = VisualEffects.apply_sharpen(img)
edges     = VisualEffects.apply_edge_detect(img)       # Sobel
embossed  = VisualEffects.apply_emboss(img)
gray      = VisualEffects.apply_grayscale(img)
sepia     = VisualEffects.apply_sepia(img)
inverted  = VisualEffects.apply_invert(img)
noisy     = VisualEffects.add_noise(img, intensity=0.2, noise_type="gaussian")
pixel     = VisualEffects.pixelate(img, block_size=12)
vign      = VisualEffects.vignette(img, strength=0.6)
bright    = VisualEffects.adjust_brightness(img, factor=1.4)
contrast  = VisualEffects.adjust_contrast(img, factor=1.5)
blended   = VisualEffects.blend_images(img1, img2, alpha=0.5)
overlay   = VisualEffects.overlay(base, layer, x=50, y=50, alpha=0.8)
grad      = VisualEffects.create_linear_gradient(512, 512, (255,0,0), (0,0,255))
radial    = VisualEffects.create_radial_gradient(512, 512, (255,220,0), (0,0,80))
```

#### StreamingWriter — large images without full RAM allocation

```python
from TTI_art import StreamingWriter

def make_row(y):
    return [(int(x/8000*255), int(y/8000*255), 128) for x in range(8000)]

with StreamingWriter("huge.bmp", 8000, 8000) as writer:
    writer.generate_rows(make_row, progress_cb=lambda y,h: print(f"{y/h:.0%}"))
```

#### AnimationEngine

```python
from TTI_art import AnimationEngine
import math

anim = AnimationEngine(400, 400, fps=24)

def draw_frame(frame, img):
    img.fill((0, 0, 30))
    t = frame / 60
    cx = 200 + int(120 * math.sin(t * 2 * math.pi))
    from TTI_core import ImageCanvas
    ImageCanvas(img).circle(cx, 200, 30, (255, 200, 0), filled=True)

paths = anim.save_sequence("frames/", num_frames=60, draw_func=draw_frame, fmt="png")
```

#### CustomBitDepth — arbitrary precision images

```python
from TTI_art import CustomBitDepth

# 48-bit HDR (16 bits per channel × 3 channels)
img = CustomBitDepth(256, 256, bits_per_channel=16, num_channels=3)
img.set_pixel(128, 128, [65535, 32768, 0])
img.save("hdr.custimg")

# Load back
loaded = CustomBitDepth.load("hdr.custimg")
print(loaded)   # CustomBitDepth(256×256, 16bpc × 3ch, max=65535)

# Convert to viewable TTIImage (scales to 8bpc)
preview = loaded.to_tti_image()
preview.save("hdr_preview.png")
```

Supported configurations:

| Config | bits/ch | channels | bits/pixel | Use case                  |
|--------|---------|----------|------------|---------------------------|
| 24-bit | 8       | 3        | 24         | Standard RGB              |
| 48-bit | 16      | 3        | 48         | HDR photography           |
| 96-bit | 32      | 3        | 96         | Scientific imaging        |
| 80-bit | 16      | 5        | 80         | Multi-spectral            |
| Custom | 1–64    | 1–∞      | any        | Experimental              |

---

### TTI\_ai.py

#### NLPAnalyser

```python
from TTI_ai import NLPAnalyser

nlp      = NLPAnalyser()
analysis = nlp.analyse("a bright neon city at night glowing with vivid colours")

print(analysis.scene_type)        # "geometric"
print(analysis.colour_matches)    # [("bright", (255,255,200)), ("neon", (57,255,200)), ...]
print(analysis.modifiers)         # {"brightness": 1.4, "saturation": 1.5, ...}
print(analysis.nouns)             # ["city", "night", "colours"]
print(analysis.adjectives)        # ["bright", "vivid"]
print(analysis.complexity())      # 0.4  (0 – 1)
print(analysis.primary_colour())  # (255, 255, 200)
```

#### ColourPredictor

```python
from TTI_ai import ColourPredictor

predictor = ColourPredictor()
palette   = predictor.predict_palette("dark mysterious forest at night", n_colours=6)
# → [(25,25,112), (34,139,34), (0,100,0), (20,20,20), (75,0,130), (100,120,180)]
```

Trained automatically from 100+ built-in colour/scene keyword mappings. No dataset download needed.

#### SceneComposer (VAE)

```python
from TTI_ai import SceneComposer, NLPAnalyser

composer = SceneComposer()
analysis = NLPAnalyser().analyse("red fire glowing bright")

mu, log_var = composer.encode(analysis)
z           = composer.reparameterise(mu, log_var, seed=42)
params_vec  = composer.decode(z)
scene       = composer.compose(analysis, seed=42)   # full pipeline

print(scene.scene_type)       # "plasma"
print(scene.primary_color)    # (220, 50, 50)
print(scene.brightness)       # 1.4
print(scene.num_shapes)       # 18
print(scene.fractal_iter)     # 112
```

Model architecture:

```
Encoder:  embed(256) → Linear(512) → LayerNorm → ReLU
                     → Linear(256) → LayerNorm → ReLU
                     → μ(128),  log_σ(128)

Decoder:  z(128) → Linear(256) → ReLU
                 → Linear(512) → ReLU
                 → Linear(64)  → Sigmoid  →  SceneParameters
```

Total parameters: **526,144**

#### TTIGenerator

```python
from TTI_ai import TTIGenerator

gen = TTIGenerator()

# Single image
img = gen.generate("a mysterious purple galaxy", width=512, height=512, seed=7)

# Batch
imgs = gen.generate_batch(["red sunset", "blue ocean", "green forest"])

# Variations (same prompt, different seeds)
variations = gen.generate_variations("stormy night", n_variations=4)

# Prompt interpolation
frames = gen.interpolate("sunrise", "sunset", steps=8, width=512, height=512)

# Persist & reload weights
gen.save_model("weights/composer.pkl")
gen.load_model("weights/composer.pkl")

print(gen.get_model_info())
# {'model_type': 'vae_numpy', 'latent_dim': 128, 'hidden_dim': 512,
#  'text_embed_dim': 256, 'total_params': 526144, 'nlp_backend': 'nltk'}
```

---

### TTI\_main.py

#### TTIPipeline — high-level façade

```python
from TTI_main import TTIPipeline

pipe = TTIPipeline()                          # uses global config singleton
pipe = TTIPipeline(config=my_cfg)             # or pass a custom TTIConfig

# ── AI generation ─────────────────────────────────────────────────────────
img   = pipe.generate("a calm blue ocean", output="ocean.png", seed=42)
paths = pipe.generate_batch(["red sky", "green forest"], output_dir="batch/")
paths = pipe.generate_variations("neon city", n=4, output_dir="vars/")
paths = pipe.interpolate("sunrise", "midnight", steps=6, output_dir="interp/")

# ── Prompt analysis only ──────────────────────────────────────────────────
analysis = pipe.analyse("a bright glowing magical forest")

# ── Procedural art ────────────────────────────────────────────────────────
img = pipe.art("mandelbrot", width=800, height=600, output="fractal.png")
img = pipe.art("voronoi",    n_cells=25, seed=42)
img = pipe.art("julia",      c_real=-0.4, c_imag=0.6)
img = pipe.art("sierpinski", depth=7)

# ── Effects on existing images ────────────────────────────────────────────
img = pipe.effect("blur",   "photo.png", output="blurred.png", radius=4)
img = pipe.effect("sepia",  "photo.png", output="vintage.png")
img = pipe.effect("vignette","photo.png", strength=0.7)

# ── Animation ─────────────────────────────────────────────────────────────
pipe.animate("colourful abstract art", n_frames=30, output_dir="anim/")

# ── Streaming large images ────────────────────────────────────────────────
pipe.stream_large("giant.bmp", width=10000, height=10000, pattern="gradient")

# ── Custom bit-depth ──────────────────────────────────────────────────────
pipe.custom_bitdepth(256, 256, bits_per_channel=16, num_channels=3,
                     output="hdr.custimg", preview="hdr_preview.png")

# ── Config ────────────────────────────────────────────────────────────────
pipe.set_config(image={"default_width": 1024}, ai={"seed": 99})
pipe.show_config()
pipe.save_config("project.json")
print(pipe.model_info())
```

---

## CLI Reference

```
python TTI_main.py <command> [options]
```

### `generate` — AI text-to-image

```bash
python TTI_main.py generate "a stormy ocean at night" \
    --output storm.png \
    --width 512 --height 512 \
    --seed 42 \
    --format png          # png | bmp | jpeg
```

### `batch` — bulk generation from a file

```bash
# prompts.txt: one prompt per line
python TTI_main.py batch prompts.txt \
    --output-dir out/ \
    --prefix img \
    --format png \
    --seed 100
```

### `art` — procedural art (no prompt)

```bash
python TTI_main.py art mandelbrot  --output fractal.png --width 800 --height 600
python TTI_main.py art julia       --c-real -0.4 --c-imag 0.6 --output julia.png
python TTI_main.py art sierpinski  --depth 7 --output sierp.png
python TTI_main.py art voronoi     --n-cells 30 --seed 7 --output vor.png
python TTI_main.py art noise       --octaves 6 --output noise.png
python TTI_main.py art plasma      --output plasma.png
python TTI_main.py art gradient    --output grad.png
python TTI_main.py art radial      --output radial.png
python TTI_main.py art checkerboard --square-size 30 --output checker.png
python TTI_main.py art waves       --freq 0.04 --output waves.png
python TTI_main.py art circles     --output circles.png
python TTI_main.py art spiral      --turns 8 --output spiral.png
```

### `effects` — apply filter to existing image

```bash
python TTI_main.py effects blur         photo.png --radius 3   --output out.png
python TTI_main.py effects gaussian_blur photo.png --sigma 2.5 --output out.png
python TTI_main.py effects sharpen      photo.png              --output out.png
python TTI_main.py effects edge         photo.png              --output out.png
python TTI_main.py effects emboss       photo.png              --output out.png
python TTI_main.py effects grayscale    photo.png              --output out.png
python TTI_main.py effects sepia        photo.png              --output out.png
python TTI_main.py effects invert       photo.png              --output out.png
python TTI_main.py effects noise        photo.png --intensity 0.2  --output out.png
python TTI_main.py effects pixelate     photo.png --block-size 12  --output out.png
python TTI_main.py effects vignette     photo.png --strength 0.7   --output out.png
python TTI_main.py effects brightness   photo.png --factor 1.5     --output out.png
python TTI_main.py effects contrast     photo.png --factor 1.8     --output out.png
```

### `analyse` — NLP analysis without generating an image

```bash
python TTI_main.py analyse "a bright neon city at midnight"
# Output:
#   Prompt    : a bright neon city at midnight
#   Scene     : geometric
#   Colours   : ['bright', 'neon', 'night']
#   Nouns     : ['bright', 'neon', 'city', 'midnight']
#   Adjectives: []
#   Modifiers : ['brightness', 'saturation']
#   Complexity: 0.25
```

### `interpolate` — prompt-to-prompt frame sequence

```bash
python TTI_main.py interpolate "warm sunrise" "cold midnight" \
    --steps 8 \
    --output-dir interpolation/ \
    --width 512 --height 512
```

### `variations` — N seeds of one prompt

```bash
python TTI_main.py variations "magical glowing forest" \
    --n 6 \
    --output-dir variations/
```

### `animate` — frame sequence from a single prompt

```bash
python TTI_main.py animate "colourful abstract swirls" \
    --frames 30 \
    --output-dir animation/
```

### `stream` — write very large BMP without full RAM

```bash
python TTI_main.py stream huge.bmp \
    --width 10000 --height 10000 \
    --pattern gradient           # gradient | checkerboard | noise
```

### `config` — view or modify settings

```bash
python TTI_main.py config --show
python TTI_main.py config --set image.default_width=1024 ai.seed=42
python TTI_main.py config --set log.show_progress=false
python TTI_main.py config --save my_project.json
python TTI_main.py config --reset
```

### `demo` — full feature showcase

```bash
python TTI_main.py demo --output-dir tti_demo/
# Generates 41 images: 8 AI prompts, 12 art types, 13 effects,
# 4 variations, 5 interpolation frames, 3 bit-depth previews
```

---

## Configuration Reference

All defaults. Override via `update_config()`, `--set` flag, or a JSON file.

### `[image]`

| Key                | Default              | Description                          |
|--------------------|----------------------|--------------------------------------|
| `default_width`    | `512`                | Output image width in pixels         |
| `default_height`   | `512`                | Output image height in pixels        |
| `default_bpp`      | `24`                 | Bits per pixel (`24` or `32`)        |
| `default_format`   | `'png'`              | Default output format                |
| `background_color` | `(255, 255, 255)`    | Default background RGB               |
| `jpeg_quality`     | `92`                 | JPEG quality 1–95                    |

### `[ai]`

| Key                  | Default         | Description                                  |
|----------------------|-----------------|----------------------------------------------|
| `nlp_backend`        | `'nltk'`        | NLP engine (`'nltk'` or `'spacy'`)           |
| `max_prompt_tokens`  | `128`           | Max tokens to process from prompt            |
| `use_stopword_filter`| `True`          | Strip common stopwords before analysis       |
| `model_type`         | `'vae_numpy'`   | Backend (`'vae_numpy'` or `'torch_vae'`)     |
| `latent_dim`         | `128`           | VAE latent space dimensionality              |
| `text_embed_dim`     | `256`           | Text embedding vector size                   |
| `hidden_dim`         | `512`           | MLP hidden layer width                       |
| `num_inference_steps`| `50`            | Refinement iterations                        |
| `guidance_scale`     | `7.5`           | Prompt adherence weight                      |
| `seed`               | `None`          | Global RNG seed (`None` = random each time)  |

### `[art]`

| Key                       | Default | Description                          |
|---------------------------|---------|--------------------------------------|
| `fractal_max_iter`        | `256`   | Max Mandelbrot/Julia iterations      |
| `blur_default_radius`     | `2`     | Default box-blur kernel radius       |
| `noise_default_intensity` | `0.15`  | Default noise magnitude (0–1)        |
| `animation_fps`           | `24`    | Target frames per second             |
| `streaming_chunk_mb`      | `32`    | Max RAM per streaming chunk (MB)     |

### `[paths]`

| Key          | Default        | Description              |
|--------------|----------------|--------------------------|
| `output_dir` | `'tti_output'` | Default output directory |
| `model_dir`  | `'tti_models'` | Saved model weights      |
| `cache_dir`  | `'tti_cache'`  | Cached intermediate data |
| `log_dir`    | `'tti_logs'`   | Log files                |

### `[log]`

| Key             | Default  | Description                         |
|-----------------|----------|-------------------------------------|
| `level`         | `'INFO'` | `DEBUG` / `INFO` / `WARNING`        |
| `log_to_file`   | `False`  | Mirror output to `log_filename`     |
| `log_filename`  | `'tti.log'` | Log file name                    |
| `show_progress` | `True`   | Print progress to stdout            |

---

## AI Pipeline Deep Dive

### Colour Knowledge Base

Over **100 keyword → RGB** mappings cover environments, moods, objects, and named colours.

```python
from TTI_ai import COLOUR_KB, SCENE_KB, MODIFIER_KB

COLOUR_KB["ocean"]       # → (0, 105, 148)
COLOUR_KB["fire"]        # → (255, 69, 0)
COLOUR_KB["ethereal"]    # → (200, 180, 255)

SCENE_KB["galaxy"]       # → "fractal"
SCENE_KB["city"]         # → "geometric"

MODIFIER_KB["dreamy"]    # → {"blur": 2, "brightness": 1.1}
MODIFIER_KB["vintage"]   # → {"sepia": True, "brightness": 0.85}
```

### NumpyTensor — PyTorch-compatible Layer API

All neural layers implement the PyTorch naming convention:

```python
from TTI_ai import NumpyTensor, NumpyLinear, NumpyLayerNorm

x   = NumpyTensor.randn(256)
fc  = NumpyLinear(256, 512)
ln  = NumpyLayerNorm(512)
out = ln(fc(x).relu())             # forward pass

for name, param in fc.named_parameters():
    print(name, param.shape)       # "weight" (512, 256) / "bias" (512,)
```

---

## PyTorch Upgrade Path

`NumpyTensor`, `NumpyLinear`, and `NumpyLayerNorm` in `TTI_ai.py` match the PyTorch
`nn.Module` API. Switching to GPU acceleration requires only swapping the tensor backend:

```python
# Current (numpy)
from TTI_ai import NumpyTensor as Tensor, NumpyLinear as Linear

# After: install torch, then swap imports
import torch
Tensor = torch.Tensor
Linear = torch.nn.Linear
```

The `SceneComposer`, `ImageDecoder`, and `TTIGenerator` classes need no changes —
all tensor operations use the same method names (`relu()`, `sigmoid()`, `@`, `+` …).

To use `cfg.ai.model_type = "torch_vae"` and enable CUDA:

```python
from TTI_config import update_config
update_config(ai={"model_type": "torch_vae"})
```

---

## Examples

### Generate a series of variations and blend two together

```python
from TTI_main import TTIPipeline
from TTI_art  import VisualEffects

pipe = TTIPipeline()
vars_ = pipe._gen.generate_variations("misty mountain at dawn", n_variations=2, seed=10)
blend = VisualEffects.blend_images(vars_[0], vars_[1], alpha=0.5)
blend.save("blended.png")
```

### Custom pattern with procedural colour logic

```python
import math
from TTI_art import ProceduralArt

def rose_curve(x, y, w, h):
    cx, cy = w / 2, h / 2
    dx, dy = (x - cx) / cx, (y - cy) / cy
    r = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    k = 5
    pattern = abs(math.cos(k * theta)) - r
    t = max(0.0, min(1.0, pattern * 3 + 0.5))
    return (int(255 * t), int(80 * t), int(200 * (1 - t)))

img = ProceduralArt.generate_pattern(600, 600, rose_curve)
img.save("rose.png")
```

### Save and reload model weights for reproducibility

```python
from TTI_ai import TTIGenerator

gen = TTIGenerator()
img = gen.generate("a glowing nebula", seed=42)
gen.save_model("weights/run1.pkl")

# Later session
gen2 = TTIGenerator()
gen2.load_model("weights/run1.pkl")
img2 = gen2.generate("a glowing nebula", seed=42)   # identical result
```

### Write a 10,000 × 10,000 BMP using under 100 MB RAM

```python
from TTI_art import StreamingWriter

def plasma_row(y, width=10000):
    import math
    return [
        (
            int(127 + 127 * math.sin(x * 0.003 + y * 0.002)),
            int(127 + 127 * math.cos(x * 0.002 - y * 0.003)),
            int(127 + 127 * math.sin((x + y) * 0.002)),
        )
        for x in range(width)
    ]

with StreamingWriter("plasma_10k.bmp", 10000, 10000) as w:
    w.generate_rows(plasma_row)
```

### Pipe-chain effects

```python
from TTI_main import TTIPipeline
from TTI_art  import VisualEffects

pipe = TTIPipeline()
img  = pipe.generate("abstract digital art", seed=1)
img  = VisualEffects.apply_blur(img,       radius=2)
img  = VisualEffects.apply_sharpen(img)
img  = VisualEffects.vignette(img,         strength=0.5)
img  = VisualEffects.adjust_brightness(img, factor=1.2)
img.save("chained.png")
```

---

## Demo

Running `python TTI_main.py demo` (or `demo()` in Python) produces **41 outputs**:

| Category               | Count | Examples                                               |
|------------------------|-------|--------------------------------------------------------|
| AI text-to-image       | 8     | ocean at sunset, neon city, purple nebula, fire …      |
| Procedural art types   | 12    | Mandelbrot, Julia, Sierpiński, Voronoi, plasma, spiral… |
| Visual effects         | 13    | blur, edge, emboss, sepia, vignette, pixelate …        |
| Prompt variations      | 4     | 4 seeds of "magical glowing forest"                    |
| Interpolation frames   | 5     | "warm golden sunrise" → "cold dark night sky"          |
| Custom bit-depth       | 3     | 16bpc×3ch, 32bpc×3ch, 8bpc×6ch previews               |

---

## License

MIT License — free to use, modify, and distribute.
