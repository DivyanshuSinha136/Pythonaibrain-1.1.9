# CLSE - Compositional Latent Synthesis Engine

A complete, self-contained Compositional Latent Synthesis Engine pipeline built entirely on NumPy, NLTK, scikit-learn, and PyTorch — no Stable Diffusion or external model weights required. Converts natural-language prompts into images through a multi-stage NLP → neural model → renderer architecture, with full support for procedural art, visual effects, animation, streaming, and a rich CLI.

---

## What Is This?

The CLSE system is a nine-module package covering every layer of the Compositional Latent Synthesis Engine stack:

| Module | Class / Entry Point | Responsibility |
|---|---|---|
| `TTI_config.py` | `TTIConfig`, `get_config()` | Master config — all tuneable parameters |
| `TTI_core.py` | `TTIImage`, `ImageCanvas`, `ColorUtils`, `ImageIO` | Pixel-level image engine (BMP/PNG/JPEG I/O, drawing) |
| `TTI_art.py` | `ProceduralArt`, `VisualEffects`, `StreamingWriter`, `AnimationEngine` | Procedural art, effects, animation, large-image streaming |
| `TTI_ai.py` | `TTIGenerator`, `NLPAnalyser`, `PromptAnalysis` | NLP pipeline + numpy VAE renderer + colour predictor |
| `TTI_model.py` | `TTIModel`, `TTITrainer`, `TTILoss` | 4-layer transformer VAE (3.8M params), training loop |
| `TTI_dataset.py` | `TTIDataset`, `Vocabulary` | Synthetic 50k-sample dataset generator with caching |
| `TTI_pipeline.py` | `TTIPipeline` | Full end-to-end pipeline connecting all modules |
| `TTI_train.py` | CLI training script | Production training with checkpointing and early stopping |
| `TTI_main.py` | `TTIPipeline` façade + `main()` | Unified entry point + full CLI |

### Note

> These `*.py` files are inner file of CLSE (Compositional Latent Synthesis Engine).

---

## Installation

```bash
pip install numpy torch scikit-learn nltk pillow
python -m nltk.downloader punkt averaged_perceptron_tagger stopwords
```

Optional (better NLP accuracy):
```bash
pip install spacy && python -m spacy download en_core_web_sm
```

---

## Quick Start

```python
from TTI_main import TTIPipeline

pipe = TTIPipeline()

# Generate from a text prompt
img = pipe.generate("a calm blue ocean at sunset", output="ocean.png")

# Procedural art (no model needed)
img = pipe.art("mandelbrot", output="fractal.png")

# Apply an effect to an existing image
img = pipe.effect("sepia", "photo.png", output="photo_sepia.png")

# Analyse a prompt
analysis = pipe.analyse("dark gothic castle at midnight")
print(analysis.scene_type, analysis.colour_matches)
```

---

## Module Guide

---

### `TTI_config.py` — Configuration

```python
from TTI_config import TTIConfig, get_config, update_config, reset_config

# Global singleton (auto-discovers tti_config.json next to the module)
cfg = get_config()

# Read settings
print(cfg.image.default_width)       # 512
print(cfg.ai.model_type)             # "vae_numpy"
print(cfg.art.fractal_max_iter)      # 256
print(cfg.paths.output_dir)          # "tti_output"

# Bulk update
update_config(
    image={"default_width": 1024, "default_height": 1024},
    ai={"seed": 42, "num_inference_steps": 100},
)

# Save / load JSON snapshot
cfg.save("tti_config.json")
cfg2 = TTIConfig.load("tti_config.json")

# Create all output directories
cfg.ensure_dirs()

# Reset to factory defaults
reset_config()
```

**Config sections:**

| Section | Dataclass | Key fields |
|---|---|---|
| `cfg.image` | `ImageConfig` | `default_width`, `default_height`, `default_format`, `background_color`, `jpeg_quality` |
| `cfg.ai` | `AIConfig` | `model_type`, `latent_dim`, `vocab_size`, `num_inference_steps`, `guidance_scale`, `seed` |
| `cfg.art` | `ArtConfig` | `fractal_max_iter`, `blur_default_radius`, `noise_default_intensity`, `animation_fps` |
| `cfg.paths` | `PathConfig` | `output_dir`, `model_dir`, `cache_dir`, `log_dir` |
| `cfg.log` | `LogConfig` | `level`, `log_to_file`, `log_filename`, `show_progress` |

---

### `TTI_core.py` — Image Engine

```python
from TTI_core import TTIImage, ImageCanvas, ColorUtils, ImageIO, ImageValidator

# Create a blank image
img = TTIImage(width=512, height=512, bpp=24, background=(20, 30, 60))

# Pixel operations
img.set_pixel(100, 100, (255, 0, 0))
color = img.get_pixel(100, 100)   # (255, 0, 0)

# NumPy array interop
arr = img.to_array()              # shape (H, W, 3), dtype=uint8
img.from_array(arr)

# Save and load
img.save("output.png")
img.save("output.jpg", fmt="jpeg")
img.save("output.bmp", fmt="bmp")
img2 = TTIImage.load("output.png")

# Drawing (ImageCanvas)
canvas = ImageCanvas(img)
canvas.line(0, 0, 511, 511, (255, 255, 0))
canvas.rectangle(50, 50, 200, 200, (255, 100, 0), filled=True)
canvas.circle(256, 256, 100, (0, 200, 255), filled=False)
canvas.fill_background((10, 10, 30))

# Colour utilities
rgb = ColorUtils.hsv_to_rgb(0.6, 0.8, 0.9)
blended = ColorUtils.lerp((255, 0, 0), (0, 0, 255), t=0.5)
rgba = ColorUtils.to_rgba((100, 150, 200))         # adds alpha=255
clamped = ColorUtils.clamp(300)                    # → 255

# Multi-format I/O (static)
img = ImageIO.load("photo.png")
ImageIO.save(img, "photo.jpeg", quality=85)

# Integrity check
ImageValidator.validate(img)    # raises TTIImageError if corrupt
```

**Exception hierarchy:**

```
TTIError
├── TTIImageError   — pixel-level or dimension errors
└── TTIIOError      — file read/write failures
```

---

### `TTI_art.py` — Procedural Art & Effects

#### ProceduralArt

All methods are static and return `TTIImage` objects.

```python
from TTI_art import ProceduralArt, VisualEffects, StreamingWriter, AnimationEngine

# Fractals
img = ProceduralArt.mandelbrot_set(width=800, height=600)
img = ProceduralArt.julia_set(800, 600, c_real=-0.7, c_imag=0.27015)
img = ProceduralArt.sierpinski_triangle(512, 512, depth=7)

# Patterns
img = ProceduralArt.plasma(512, 512)
img = ProceduralArt.voronoi(512, 512, n_cells=25, seed=42)
img = ProceduralArt.perlin_noise_image(512, 512, octaves=4)

# Gradients
img = VisualEffects.create_linear_gradient(512, 512, (70, 130, 200), (200, 80, 120))
img = VisualEffects.create_radial_gradient(512, 512, (255, 220, 50), (30, 30, 120))
```

#### VisualEffects

Apply filters to existing `TTIImage` objects (all return the modified image):

```python
img = VisualEffects.blur(img, radius=3)
img = VisualEffects.gaussian_blur(img, sigma=2.0)
img = VisualEffects.sharpen(img, factor=1.5)
img = VisualEffects.edge_detect(img)
img = VisualEffects.emboss(img)
img = VisualEffects.grayscale(img)
img = VisualEffects.sepia(img, strength=0.8)
img = VisualEffects.invert(img)
img = VisualEffects.add_noise(img, intensity=0.15)
img = VisualEffects.pixelate(img, block_size=10)
img = VisualEffects.vignette(img, strength=0.6)
img = VisualEffects.adjust_brightness(img, factor=1.2)
img = VisualEffects.adjust_contrast(img, factor=1.3)
img = VisualEffects.blend(img1, img2, alpha=0.5)
```

#### StreamingWriter — large images without full RAM

```python
from TTI_art import StreamingWriter

# Write a 4000×4000 image row by row
with StreamingWriter("large.png", width=4000, height=4000, bpp=24) as sw:
    for y in range(4000):
        row = [(y % 255, 100, 200)] * 4000   # list of RGB tuples
        sw.write_row(row)
```

#### AnimationEngine — frame sequences

```python
from TTI_art import AnimationEngine

engine = AnimationEngine(fps=24)
frames = engine.generate_frames(base_img, n_frames=48, mode="zoom")
engine.save_frames(frames, output_dir="frames/", fmt="png")
```

#### CustomBitDepth — non-standard pixel formats

```python
from TTI_art import CustomBitDepth

# 16-bit per channel, 3 channels
cbd = CustomBitDepth(width=256, height=256, bits_per_channel=16, n_channels=3)
cbd.set_pixel(0, 0, [65535, 0, 32768])
cbd.save("high_depth.custimg")
cbd.save_preview("preview.png")   # downsample to 8-bit PNG for viewing
```

---

### `TTI_ai.py` — AI Engine

```python
from TTI_ai import TTIGenerator, NLPAnalyser

# NLP analysis
nlp = NLPAnalyser()
analysis = nlp.analyse("a mysterious purple galaxy with glowing stars")

print(analysis.scene_type)         # "starfield"
print(analysis.nouns)              # ["galaxy", "stars"]
print(analysis.adjectives)         # ["mysterious", "purple", "glowing"]
print(analysis.colour_matches)     # [("purple", (138,43,226)), ("gold", (255,215,0)), ...]
print(analysis.modifiers)          # {"mysterious": 0.8, "glowing": 0.6}
print(analysis.filtered_tokens)    # ["mysterious", "purple", "galaxy", "glowing", "stars"]
print(analysis.complexity())       # 0.72  (float 0–1)

# Full generation
from TTI_config import get_config
gen = TTIGenerator(get_config())
img = gen.generate("stormy sea at dusk", width=512, height=512, seed=99)
img.save("storm.png")

# Variations
imgs = gen.generate_variations("neon city at night", n_variations=4)
for i, img in enumerate(imgs):
    img.save(f"variation_{i}.png")

# Interpolation between two prompts
imgs = gen.interpolate("sunrise", "midnight", steps=6)
for i, img in enumerate(imgs):
    img.save(f"interp_{i:02d}.png")
```

**AI pipeline (internal stages):**

```
NLPAnalyser       → 256-d SemanticVector  (TF-IDF + PCA, NLTK tokeniser)
ColourPredictor   → PaletteSpec           (sklearn KNN on 170+ colour keywords)
SceneComposer     → 128-d LatentCode      (numpy VAE encoder)
ImageDecoder      → TTIImage              (14 scene-type renderers)
```

---

### `TTI_model.py` — Neural Architecture

```python
from TTI_model import TTIModel, TTIModelLarge, ModelConfig, TTITrainer, TTILoss

# Default model (4-layer transformer, 3.8M params)
cfg = ModelConfig(vocab_size=8192, embed_dim=256, n_layers=4, n_heads=8, latent_dim=128)
model = TTIModel(cfg)

# Large variant (6-layer, 512-d, ~14M params)
model_large = TTIModelLarge()

# Forward pass
import torch
tokens = torch.randint(0, 8192, (4, 32))   # batch=4, seq_len=32
out = model(tokens)
# out.scene_logits  : (4, 15)   — 15-class scene prediction
# out.colour_pred   : (4, 18)   — 6 colours × RGB
# out.param_pred    : (4, 64)   — scene renderer parameters
# out.mu, out.logvar: (4, 128)  — VAE latent distribution

# Multi-task loss
criterion = TTILoss(scene_weight=1.0, colour_weight=0.5, param_weight=0.3, kl_weight=0.01)
loss = criterion(out, scene_labels, colour_targets, param_targets)

# Training loop
trainer = TTITrainer(model, dataset, val_dataset, output_dir="tti_models/")
history = trainer.train(epochs=20, batch_size=64, lr=3e-4)
print(history["best_val_loss"])
```

**Model architecture:**

```
TokenEmbedding      — learned token + sinusoidal positional embeddings
TransformerEncoder  — 4× MultiHeadSelfAttention + FFN (BERT-style, pre-LN)
    ├── ColourHead      — 3-layer MLP → 18-d colour palette
    ├── SceneClassifier — 2-layer MLP → 15-class scene logit
    └── ParamDecoder    — VAE: μ/σ → z → 64-d parameter vector
```

---

### `TTI_dataset.py` — Dataset Generator

```python
from TTI_dataset import TTIDataset, Vocabulary, build_dataset

# Build a 50,000-sample dataset (cached to disk with SHA-256 integrity check)
dataset = build_dataset(
    n_samples=50_000,
    cache_dir="tti_models/",
    force_rebuild=False,     # use cache if available
)

# Train / val / test splits
train_ds, val_ds, test_ds = dataset.splits()
print(len(train_ds), len(val_ds), len(test_ds))   # 40000, 5000, 5000

# DataLoader-compatible access
sample = train_ds[0]
# sample["token_ids"]   : torch.LongTensor (32,)
# sample["scene_label"] : int  (0–14)
# sample["colour_vec"]  : torch.FloatTensor (18,)
# sample["param_vec"]   : torch.FloatTensor (64,)
# sample["prompt"]      : str

# Vocabulary
vocab = Vocabulary.load("tti_models/vocab.json")
ids = vocab.encode("a stormy sea at dusk")   # list of int
text = vocab.decode(ids)                     # str
print(vocab.size)                            # 8192
```

**Dataset statistics (default build):**

| Split | Samples |
|---|---|
| Train | 40,000 |
| Validation | 5,000 |
| Test | 5,000 |
| Vocab size | 8,192 |
| Scene classes | 15 |
| Colour dims | 18 (6 colours × RGB) |
| Parameter dims | 64 |

---

### `TTI_pipeline.py` — Unified Pipeline

```python
from TTI_pipeline import TTIPipeline

pipe = TTIPipeline()

# Generate
img = pipe.generate("a bright rainbow over a misty waterfall", output="rainbow.png")

# Variations
imgs = pipe.variations("neon city at night", n=4, output_dir="variations/")

# Interpolation
imgs = pipe.interpolate("sunrise", "midnight", steps=6, output_dir="interp/")

# Procedural art
img = pipe.art("julia", output="julia.png", c_real=-0.4, c_imag=0.6)

# Effects
img = pipe.effect("sepia", input_path="photo.png", output="photo_sepia.png")

# NLP analysis only
info = pipe.analyse("dark gothic castle at midnight")

# Animation
frames = pipe.animate("sunset over the ocean", n_frames=24, output_dir="frames/")

# Stream a very large image (memory-safe)
pipe.stream_large("huge.png", width=4000, height=4000, pattern="gradient")

# Model info
print(pipe.model_info())

# Config management
pipe.show_config()
pipe.save_config("snapshot.json")
```

---

### `TTI_train.py` — Training Script

```bash
# Build dataset then train
python TTI_train.py --build-dataset --n-samples 50000 --epochs 20

# Train on existing dataset
python TTI_train.py --epochs 20 --batch 64 --lr 3e-4

# Quick smoke-test (2 epochs, tiny model)
python TTI_train.py --smoke-test

# Resume from checkpoint
python TTI_train.py --resume tti_models/best_model.pt --epochs 10

# Large model variant
python TTI_train.py --model-size large --batch 128 --epochs 30
```

**Training flags:**

| Flag | Default | Description |
|---|---|---|
| `--build-dataset` | off | Generate synthetic dataset before training |
| `--n-samples N` | `50000` | Number of training samples to generate |
| `--epochs N` | `20` | Training epochs |
| `--batch N` | `64` | Batch size |
| `--lr FLOAT` | `3e-4` | Learning rate |
| `--model-size` | `default` | `default` (3.8M params) or `large` (14M params) |
| `--resume PATH` | — | Resume from a `.pt` checkpoint file |
| `--smoke-test` | off | 2-epoch sanity check with minimal config |
| `--output-dir DIR` | `tti_models/` | Directory for checkpoints and logs |

**Training features:** gradient checkpointing, cosine LR scheduling with linear warmup, early stopping, per-class accuracy tracking, best-model saving (`ModelCheckpoint`), metric logging (`MetricLogger`).

---

## CLI Reference (`TTI_main.py`)

```bash
python TTI_main.py <command> [options]
```

### Commands

#### `generate` — Text to image

```bash
python TTI_main.py generate "a calm ocean at sunset" -o ocean.png -W 512 -H 512
python TTI_main.py generate "neon city" --seed 42 --format jpeg
```

| Flag | Description |
|---|---|
| `prompt` | Input text prompt |
| `-o / --output` | Output file path |
| `-W / --width` | Image width in pixels |
| `-H / --height` | Image height in pixels |
| `-s / --seed` | Random seed |
| `-f / --format` | `png`, `bmp`, or `jpeg` |

#### `batch` — Multiple prompts from a file

```bash
python TTI_main.py batch prompts.txt -o tti_output/ --prefix scene --format png
```

One prompt per line in `prompts.txt`. Outputs `prefix_0000.png`, `prefix_0001.png`, …

#### `art` — Procedural art

```bash
python TTI_main.py art mandelbrot -o fractal.png -W 800 -H 600
python TTI_main.py art julia --c-real -0.4 --c-imag 0.6 -o julia.png
python TTI_main.py art voronoi --n-cells 30 --seed 7
python TTI_main.py art sierpinski --depth 8
python TTI_main.py art plasma -W 512 -H 512
python TTI_main.py art checkerboard --square-size 40
python TTI_main.py art waves --freq 0.08
python TTI_main.py art spiral --turns 8
python TTI_main.py art noise --octaves 6
python TTI_main.py art gradient
python TTI_main.py art radial
python TTI_main.py art circles
```

Available art types: `mandelbrot`, `julia`, `sierpinski`, `plasma`, `voronoi`, `noise`, `gradient`, `radial`, `checkerboard`, `waves`, `circles`, `spiral`

#### `effects` — Apply effect to an image

```bash
python TTI_main.py effects sepia photo.png -o photo_sepia.png
python TTI_main.py effects gaussian_blur photo.png --sigma 3.0
python TTI_main.py effects pixelate photo.png --block-size 15
python TTI_main.py effects vignette photo.png --strength 0.7
python TTI_main.py effects brightness photo.png --factor 1.3
```

Available effects: `blur`, `gaussian_blur`, `sharpen`, `edge`, `emboss`, `grayscale`, `sepia`, `invert`, `noise`, `pixelate`, `vignette`, `brightness`, `contrast`

#### `analyse` — NLP prompt analysis

```bash
python TTI_main.py analyse "mysterious purple galaxy with glowing stars"
# Prints: scene type, colour matches, nouns, adjectives, modifiers, complexity score
```

#### `interpolate` — Morph between two prompts

```bash
python TTI_main.py interpolate "sunrise" "midnight" --steps 8 -o interp/
```

#### `variations` — Multiple variations of one prompt

```bash
python TTI_main.py variations "dark forest" --n 6 -o variations/
```

#### `animate` — Frame sequence from a prompt

```bash
python TTI_main.py animate "waves on a beach" --frames 36 -o frames/
```

#### `stream` — Large image with bounded RAM

```bash
python TTI_main.py stream huge.png -W 4000 -H 4000 --pattern gradient
```

Patterns: `gradient`, `checkerboard`, `noise`

#### `config` — Inspect and modify settings

```bash
python TTI_main.py config --show
python TTI_main.py config --set image.default_width=1024 ai.seed=42
python TTI_main.py config --save my_config.json
python TTI_main.py config --reset
```

#### `demo` — Run everything

```bash
python TTI_main.py demo -o tti_demo/
```

Generates one image per art type, runs all effects, performs NLP analysis, interpolation, variations, and saves a config snapshot.

---

## Architecture Overview

```
TTI_main.py  /  TTI_pipeline.py   ←  unified façade
│
├── TTI_config.py    ←  all settings (reads from config.pbcfg / tti_config.json)
│
├── TTI_ai.py        ←  NLP + VAE renderer (no model weights needed)
│   ├── NLPAnalyser      NLTK tokenise → TF-IDF → PCA → SemanticVector
│   ├── ColourPredictor  KNN on 170+ colour keywords → PaletteSpec
│   ├── SceneComposer    numpy VAE encoder → 128-d LatentCode
│   └── ImageDecoder     14 scene-type renderers → TTIImage
│
├── TTI_model.py     ←  PyTorch transformer VAE (optional, boosts quality)
│   ├── TokenEmbedding   learnable + positional
│   ├── TransformerEncoder  4-layer BERT-style
│   ├── ColourHead       → 18-d palette
│   ├── SceneClassifier  → 15-class scene
│   └── ParamDecoder     → 64-d renderer params (VAE)
│
├── TTI_dataset.py   ←  50k-sample synthetic generator + Vocabulary + DataLoader
│
├── TTI_train.py     ←  training script (gradient ckpt, cosine LR, early stop)
│
├── TTI_core.py      ←  pixel engine (TTIImage, ImageCanvas, ColorUtils, I/O)
│
└── TTI_art.py       ←  procedural art, effects, streaming, animation
```

---

## Examples Summary

```python
from TTI_main import TTIPipeline
from TTI_art import ProceduralArt, VisualEffects
from TTI_ai import NLPAnalyser
from TTI_config import get_config, update_config

pipe = TTIPipeline()

# Generate
img = pipe.generate("stormy sea at midnight", output="storm.png")

# Batch
pipe.generate_batch(["sunrise", "sunset", "noon"], output_dir="batch/")

# Variations + interpolation
pipe.generate_variations("neon city", n=4, output_dir="vars/")
pipe.interpolate("calm lake", "raging ocean", steps=6, output_dir="interp/")

# Procedural art
pipe.art("mandelbrot", output="mandelbrot.png", width=800, height=600)
pipe.art("julia", output="julia.png", c_real=-0.4, c_imag=0.6)

# Effects
pipe.effect("sepia", "input.png", output="sepia.png")
pipe.effect("vignette", "input.png", strength=0.7, output="vig.png")

# Analysis
info = pipe.analyse("mysterious purple galaxy")
print(info.scene_type, info.complexity())

# Config override
update_config(image={"default_width": 1024}, ai={"seed": 7})

# Full demo
pipe.demo(output_dir="tti_demo/")
```