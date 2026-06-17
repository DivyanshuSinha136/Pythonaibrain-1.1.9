"""
TTI_pipeline.py
===============
Unified end-to-end pipeline connecting every TTI module:

    Prompt (str)
        │
        ▼
    NLPAnalyser          ← TTI_ai.py      (NLTK tokenisation, scene/colour/modifier extraction)
        │
        ├──→ Vocabulary.encode            ← TTI_dataset.py  (token IDs for transformer)
        │
        ▼
    TTIModel.generate    ← TTI_model.py   (trained 4-layer transformer VAE, 610k params)
        │  scene_logits (15-class)
        │  colour_pred  (18-d palette)
        │  param_pred   (64-d decoder params)
        │
        ▼
    ModelOutputBridge    ← this file      (merges model output + NLP analysis → SceneParameters)
        │
        ▼
    ImageDecoder         ← TTI_ai.py      (14 scene-type renderers)
        │
        ▼
    VisualEffects chain  ← TTI_art.py     (blur / sharpen / vignette / sepia / noise …)
        │
        ▼
    TTIImage             ← TTI_core.py    (PNG / BMP / JPEG save)

Public API
----------
    pipe = TTIPipeline()

    # Text to image (uses trained model)
    img  = pipe.generate("a calm blue ocean at sunset", output="ocean.png")

    # Variations and interpolation
    imgs = pipe.variations("neon city at night", n=4)
    imgs = pipe.interpolate("sunrise", "midnight", steps=6)

    # Pure procedural (no model needed)
    img  = pipe.art("mandelbrot", width=800, height=600)

    # Effects on any image
    img  = pipe.effect("sepia", input_path="photo.png")

    # NLP analysis only
    info = pipe.analyse("mysterious purple galaxy")

    # Run everything: full demo
    pipe.demo(output_dir="tti_output")
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

torch.set_num_threads(2)

# ── TTI modules ──────────────────────────────────────────────────────────────
from .TTI_config import TTIConfig, get_config, update_config, reset_config
from .TTI_core import (
    TTIImage, ImageCanvas, ImageIO, ColorUtils,
    TTIError, TTIIOError,
)
from .TTI_art import ProceduralArt, VisualEffects, StreamingWriter, AnimationEngine
from .TTI_ai import (
    NLPAnalyser, PromptAnalysis, SceneParameters,
    ImageDecoder, ColourPredictor,
    COLOUR_KB, SCENE_KB, MODIFIER_KB,
)
from .TTI_dataset import SCENE_CLASSES, Vocabulary
from .TTI_model import TTIModel, ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# ModelOutputBridge
# ─────────────────────────────────────────────────────────────────────────────

class ModelOutputBridge:
    """
    Converts raw TTIModel output tensors + NLP PromptAnalysis
    into a SceneParameters object the ImageDecoder can render.

    This is the critical join point between the trained neural model
    and the procedural rendering engine.

    Priority:
      1. Model's scene classification  (15-class softmax)
      2. Model's colour prediction      (6 × RGB, 0-1)
      3. Model's param vector           (64-d, 0-1)
      4. NLP modifiers                  (brightness / blur / sepia …)
      5. NLP colour matches             (fallback if model colours are flat)
    """

    # Minimum colour saturation before we fall back to NLP colours
    _COLOUR_SAT_THRESHOLD = 0.08

    @staticmethod
    def _colour_saturation(rgb_norm: np.ndarray) -> float:
        """Mean HSV saturation of 6 predicted colours (0-1 scale)."""
        sats = []
        for i in range(6):
            r, g, b = rgb_norm[i*3], rgb_norm[i*3+1], rgb_norm[i*3+2]
            mx, mn  = max(r, g, b), min(r, g, b)
            sats.append((mx - mn) / max(mx, 1e-6))
        return float(np.mean(sats))

    @staticmethod
    def _model_colours_to_palette(
        colour_pred: np.ndarray,              # (18,) float 0-1
        analysis:    PromptAnalysis,
        threshold:   float = 0.08,
    ) -> List[Tuple[int, int, int]]:
        """
        Convert model colour vector → list of 6 RGB tuples.
        Falls back to NLP-extracted colours if model output is too grey.
        """
        sat = ModelOutputBridge._colour_saturation(colour_pred)

        if sat >= threshold:
            # Model has learned confident colours — use them
            palette = []
            for i in range(6):
                r = int(np.clip(colour_pred[i*3]   * 255, 0, 255))
                g = int(np.clip(colour_pred[i*3+1] * 255, 0, 255))
                b = int(np.clip(colour_pred[i*3+2] * 255, 0, 255))
                palette.append((r, g, b))
            return palette

        # Model colours are flat — blend with NLP colours
        nlp_palette = [c for _, c in analysis.colour_matches[:6]]
        model_palette = []
        for i in range(6):
            r = int(np.clip(colour_pred[i*3]   * 255, 0, 255))
            g = int(np.clip(colour_pred[i*3+1] * 255, 0, 255))
            b = int(np.clip(colour_pred[i*3+2] * 255, 0, 255))
            model_palette.append((r, g, b))

        # Weighted blend: model 40% + NLP 60% when model is grey
        blended = []
        for i in range(6):
            mc = model_palette[i]
            nc = nlp_palette[i % len(nlp_palette)] if nlp_palette else mc
            blended.append(tuple(int(mc[j]*0.4 + nc[j]*0.6) for j in range(3)))

        # Pad with derived colours if NLP had fewer than 6
        while len(blended) < 6:
            base = blended[-1]
            h, s, v = ColorUtils.rgb_to_hsv(*base)
            blended.append(ColorUtils.hsv_to_rgb((h + 0.2) % 1.0, max(0.4, s), max(0.4, v)))

        return blended

    @classmethod
    def build(
        cls,
        model_out:  Any,             # TTIModelOutput
        analysis:   PromptAnalysis,
        width:      int,
        height:     int,
    ) -> SceneParameters:
        """
        Fuse model output + NLP analysis into SceneParameters.
        """
        # ── Scene type ────────────────────────────────────────────────────
        model_scene_idx = int(model_out.scene_class().item())
        model_scene     = SCENE_CLASSES[model_scene_idx]

        # Weight model scene vs NLP scene (model wins if high confidence)
        probs = model_out.scene_probs()[0].numpy()
        top_prob = float(probs[model_scene_idx])

        if top_prob >= 0.55:
            scene = model_scene           # model is confident
        elif analysis.scene_type in SCENE_CLASSES:
            # Blend: if NLP scene is in top-3 model predictions, use NLP
            top3 = np.argsort(probs)[-3:]
            nlp_idx = SCENE_CLASSES.index(analysis.scene_type)
            scene = analysis.scene_type if nlp_idx in top3 else model_scene
        else:
            scene = model_scene

        # ── Colours ───────────────────────────────────────────────────────
        colour_pred = model_out.colour_pred[0].numpy()
        palette     = cls._model_colours_to_palette(colour_pred, analysis)

        # ── Params ────────────────────────────────────────────────────────
        p     = model_out.param_pred[0].numpy()   # 64-d, 0-1
        mods  = analysis.modifiers

        brightness = float(mods.get("brightness", 0.7 + p[33] * 0.6))
        contrast   = float(mods.get("contrast",   0.8 + p[34] * 0.5))
        saturation = float(mods.get("saturation", 0.7 + p[35] * 0.6))
        noise_lv   = float(mods.get("noise",      p[36] * 0.25))
        blur_r     = int(mods.get("blur",         int(p[37] * 4)))
        sharpen    = bool(mods.get("sharpen",     p[38] > 0.75))
        use_sepia  = bool(mods.get("sepia",       p[39] > 0.80))
        vignette   = p[40] > 0.55
        detail     = float(mods.get("detail",     0.5 + p[41] * 1.5))
        n_shapes   = max(4, int(5 + p[48] * 20))
        frac_zoom  = 0.5 + p[45] * 2.5
        frac_iter  = max(64, int(64 + p[46] * 224))
        pat_scale  = 0.02 + p[47] * 0.12

        return SceneParameters(
            scene_type       = scene,
            primary_color    = palette[0],
            secondary_color  = palette[1],
            background_color = palette[2],
            accent_color     = palette[3],
            palette          = palette,
            brightness       = brightness,
            contrast         = contrast,
            saturation       = saturation,
            noise_level      = noise_lv,
            blur_radius      = blur_r,
            detail_scale     = detail,
            num_shapes       = n_shapes,
            fractal_zoom     = frac_zoom,
            fractal_iter     = frac_iter,
            pattern_scale    = pat_scale,
            apply_sepia      = use_sepia,
            apply_sharpen    = sharpen,
            apply_vignette   = vignette,
            modifiers        = mods,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TTIPipeline  — unified public API
# ─────────────────────────────────────────────────────────────────────────────

class TTIPipeline:
    """
    Unified Text-To-Image pipeline connecting:
      NLPAnalyser → TTIModel → ModelOutputBridge → ImageDecoder → TTIImage

    Parameters
    ----------
    model_path  : Path to trained .pt checkpoint.  If None and a checkpoint
                  exists at tti_models/best_model.pt it is loaded automatically.
    vocab_path  : Path to vocab.pkl.  Default: tti_cache/vocab.pkl
    config      : TTIConfig instance.  Default: global singleton.
    use_nlp_fallback : If True (default), falls back to pure-NLP generation
                  when the trained model is unavailable or not loaded.
    """

    _ART_TYPES = {
        "mandelbrot", "julia", "sierpinski", "plasma", "voronoi", "noise",
        "gradient", "radial", "checkerboard", "waves", "circles", "spiral",
    }
    _EFFECTS = {
        "blur", "gaussian_blur", "sharpen", "edge", "emboss",
        "grayscale", "sepia", "invert", "noise", "pixelate",
        "vignette", "brightness", "contrast",
    }

    def __init__(
        self,
        model_path:        Optional[str] = None,
        vocab_path:        Optional[str] = None,
        config:            Optional[TTIConfig] = None,
        use_nlp_fallback:  bool = True,
    ) -> None:
        self.cfg             = config or get_config()
        self.cfg.ensure_dirs()
        self._use_fallback   = use_nlp_fallback

        # ── NLP ──────────────────────────────────────────────────────────
        self._nlp      = NLPAnalyser(self.cfg)
        self._colour   = ColourPredictor(self.cfg)
        self._decoder  = ImageDecoder(self.cfg)

        # ── Vocabulary ───────────────────────────────────────────────────
        vp = vocab_path or "tti_cache/vocab.pkl"
        if Path(vp).exists():
            self._vocab: Optional[Vocabulary] = Vocabulary.load(vp)
            self._log(f"Vocabulary loaded  ({len(self._vocab)} tokens)")
        else:
            self._vocab = None
            self._log("Vocabulary not found — will use NLP fallback")

        # ── Trained model ─────────────────────────────────────────────────
        self._model: Optional[TTIModel] = None
        mp = model_path or "tti_models/best_model.pt"
        if Path(mp).exists():
            try:
                self._model = TTIModel.load(mp, map_location="cpu")
                self._model.eval()
                TTIModel.training = False
                self._log(
                    f"Model loaded  ({self._model.n_parameters():,} params)"
                    f"  ← {mp}"
                )
            except Exception as e:
                self._log(f"Model load failed ({e}) — using NLP fallback")
        else:
            self._log(f"No model at '{mp}' — using NLP fallback")

        self._bridge = ModelOutputBridge()

    # ── Tokenisation ──────────────────────────────────────────────────────

    def _tokenise(self, text: str) -> List[int]:
        """Convert prompt text → padded token ID list via trained vocabulary."""
        raw = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
        tokens = [t for t in raw if len(t) > 1]
        return self._vocab.encode(tokens, self._model.cfg.max_seq_len)

    # ── Core generation ───────────────────────────────────────────────────

    def _generate_scene_params(
        self,
        prompt:      str,
        seed:        Optional[int],
        width:       int,
        height:      int,
        temperature: float = 0.9,
    ) -> SceneParameters:
        """
        Run the full inference chain:
          NLP → (model | fallback) → ModelOutputBridge → SceneParameters
        """
        analysis = self._nlp.analyse(prompt)

        # ── Model path ────────────────────────────────────────────────────
        if self._model is not None and self._vocab is not None:
            token_ids = torch.tensor(
                [self._tokenise(prompt)], dtype=torch.long
            )
            with torch.no_grad():
                out = self._model.generate(
                    token_ids,
                    temperature=temperature,
                    seed=seed,
                )
            return self._bridge.build(out, analysis, width, height)

        # ── NLP fallback ──────────────────────────────────────────────────
        from TTI_ai import SceneComposer
        composer = SceneComposer(self.cfg)
        return composer.compose(analysis, seed=seed)

    def generate(
        self,
        prompt:      str,
        output:      Optional[str] = None,
        width:       Optional[int] = None,
        height:      Optional[int] = None,
        seed:        Optional[int] = None,
        temperature: float = 0.9,
        fmt:         Optional[str] = None,
    ) -> TTIImage:
        """
        Generate one image from *prompt* using the full pipeline.

        Parameters
        ----------
        prompt      : Natural-language image description
        output      : If given, save image here
        width/height: Output dimensions (defaults from config)
        seed        : RNG seed for reproducibility (None = random)
        temperature : Latent sampling temperature (0 = deterministic, 1 = varied)
        fmt         : Output format override ("png" | "bmp" | "jpeg")
        """
        w = width  or self.cfg.image.default_width
        h = height or self.cfg.image.default_height
        if seed is None:
            seed = int(time.time()) & 0xFFFF

        if self.cfg.log.show_progress:
            src = "model+NLP" if self._model else "NLP"
            print(f"[TTI] Generating ({src}) | '{prompt[:55]}'" +
                  ("…" if len(prompt) > 55 else ""))

        scene_params = self._generate_scene_params(prompt, seed, w, h, temperature)

        if self.cfg.log.show_progress:
            print(f"[TTI] Scene: {scene_params.scene_type:12s} | "
                  f"Primary: RGB{scene_params.primary_color} | "
                  f"Seed: {seed}")

        img = self._decoder.render(scene_params, w, h, seed=seed)

        if output:
            saved = img.save(output, fmt=fmt)
            if self.cfg.log.show_progress:
                print(f"[TTI] Saved → {saved}")

        return img

    def variations(
        self,
        prompt:     str,
        n:          int = 4,
        output_dir: Optional[str] = None,
        width:      Optional[int] = None,
        height:     Optional[int] = None,
        fmt:        str = "png",
        temperature: float = 1.0,
    ) -> List[TTIImage]:
        """Generate *n* visual variations of the same prompt."""
        base_seed = int(time.time()) & 0xFFFF
        imgs = []
        out_dir = Path(output_dir) if output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            img = self.generate(prompt, width=width, height=height,
                                seed=base_seed + i, temperature=temperature)
            if out_dir:
                img.save(out_dir / f"variation_{i:04d}.{fmt}")
            imgs.append(img)

        if self.cfg.log.show_progress:
            print(f"[TTI] {n} variations generated" +
                  (f" → {out_dir}" if out_dir else ""))
        return imgs

    def interpolate(
        self,
        prompt_a:   str,
        prompt_b:   str,
        steps:      int = 6,
        output_dir: Optional[str] = None,
        width:      Optional[int] = None,
        height:     Optional[int] = None,
        fmt:        str = "png",
        seed:       Optional[int] = None,
    ) -> List[TTIImage]:
        """Interpolate between two prompts, returning *steps* frames."""
        w = width  or self.cfg.image.default_width
        h = height or self.cfg.image.default_height
        sd = seed or (int(time.time()) & 0xFFFF)
        out_dir = Path(output_dir) if output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        sp_a = self._generate_scene_params(prompt_a, sd,     w, h)
        sp_b = self._generate_scene_params(prompt_b, sd + 1, w, h)

        imgs = []
        for i in range(steps):
            t    = i / max(1, steps - 1)
            sp_i = self._interpolate_params(sp_a, sp_b, t)
            img  = self._decoder.render(sp_i, w, h, seed=sd + i)
            if out_dir:
                img.save(out_dir / f"interp_{i:04d}.{fmt}")
            imgs.append(img)

        if self.cfg.log.show_progress:
            print(f"[TTI] Interpolation ({steps} frames)" +
                  (f" → {out_dir}" if out_dir else ""))
        return imgs

    @staticmethod
    def _interpolate_params(
        a: SceneParameters, b: SceneParameters, t: float
    ) -> SceneParameters:
        """Linear interpolation of two SceneParameters at position t ∈ [0,1]."""
        import copy, dataclasses
        sp = copy.copy(a)
        sp.primary_color    = ColorUtils.lerp(a.primary_color,    b.primary_color,    t)
        sp.secondary_color  = ColorUtils.lerp(a.secondary_color,  b.secondary_color,  t)
        sp.background_color = ColorUtils.lerp(a.background_color, b.background_color, t)
        sp.accent_color     = ColorUtils.lerp(a.accent_color,     b.accent_color,     t)
        sp.palette = [ColorUtils.lerp(ca, cb, t)
                      for ca, cb in zip(a.palette, b.palette)]
        sp.brightness   = a.brightness   * (1-t) + b.brightness   * t
        sp.noise_level  = a.noise_level  * (1-t) + b.noise_level  * t
        sp.blur_radius  = int(a.blur_radius * (1-t) + b.blur_radius * t)
        sp.fractal_zoom = a.fractal_zoom * (1-t) + b.fractal_zoom * t
        sp.fractal_iter = int(a.fractal_iter * (1-t) + b.fractal_iter * t)
        sp.scene_type   = a.scene_type if t < 0.5 else b.scene_type
        return sp

    # ── Analysis ──────────────────────────────────────────────────────────

    def analyse(self, prompt: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Full NLP + model analysis of a prompt.  No image is generated.

        Returns a dict with scene, colours, modifiers, token_ids, model_probs.
        """
        analysis   = self._nlp.analyse(prompt)
        colour_pal = self._colour.predict_palette(prompt, n_colours=6)

        result: Dict[str, Any] = {
            "prompt":     prompt,
            "scene_type": analysis.scene_type,
            "colours":    [k for k, _ in analysis.colour_matches],
            "colour_palette": colour_pal,
            "modifiers":  list(analysis.modifiers.keys()),
            "nouns":      analysis.nouns,
            "adjectives": analysis.adjectives,
            "complexity": round(analysis.complexity(), 3),
            "model_scene": None,
            "model_probs": None,
        }

        if self._model is not None and self._vocab is not None:
            token_ids = torch.tensor(
                [self._tokenise(prompt)], dtype=torch.long
            )
            with torch.no_grad():
                out = self._model.generate(token_ids, temperature=0.0, seed=0)
            probs = out.scene_probs()[0].numpy()
            top3  = np.argsort(probs)[-3:][::-1]
            result["model_scene"] = SCENE_CLASSES[out.scene_class().item()]
            result["model_probs"] = {
                SCENE_CLASSES[i]: round(float(probs[i]), 4)
                for i in top3
            }

        if verbose:
            print(f"\nPrompt     : {prompt}")
            print(f"NLP scene  : {result['scene_type']}")
            if result["model_scene"]:
                print(f"Model scene: {result['model_scene']}")
                for sc, p in result["model_probs"].items():
                    print(f"  {sc:14s}: {p:.4f}  {'█'*int(p*20)}")
            print(f"Colours    : {result['colours']}")
            print(f"Modifiers  : {result['modifiers']}")
            print(f"Nouns      : {result['nouns']}")
            print(f"Complexity : {result['complexity']}")

        return result

    # ── Procedural art (model-free) ───────────────────────────────────────

    def art(
        self,
        art_type:  str,
        output:    Optional[str] = None,
        width:     Optional[int] = None,
        height:    Optional[int] = None,
        seed:      Optional[int] = None,
        **kwargs,
    ) -> TTIImage:
        """Generate procedural art without any text prompt or model."""
        w, h, cfg = (width  or self.cfg.image.default_width,
                     height or self.cfg.image.default_height, self.cfg)
        rng = np.random.default_rng(seed)

        dispatch = {
            "mandelbrot":   lambda: ProceduralArt.mandelbrot_set(w, h, config=cfg, **kwargs),
            "julia":        lambda: ProceduralArt.julia_set(w, h, config=cfg,
                                       c_real=kwargs.get("c_real", -0.7),
                                       c_imag=kwargs.get("c_imag", 0.27015)),
            "sierpinski":   lambda: ProceduralArt.sierpinski_triangle(w, h, config=cfg,
                                       depth=kwargs.get("depth", 7)),
            "plasma":       lambda: ProceduralArt.plasma(w, h, config=cfg),
            "voronoi":      lambda: ProceduralArt.voronoi(w, h, seed=seed, config=cfg,
                                       n_cells=kwargs.get("n_cells", 20)),
            "noise":        lambda: ProceduralArt.perlin_noise_image(w, h, config=cfg,
                                       octaves=kwargs.get("octaves", 4)),
            "gradient":     lambda: VisualEffects.create_linear_gradient(w, h,
                                       kwargs.get("c1", (70, 130, 200)),
                                       kwargs.get("c2", (200, 80, 120)), config=cfg),
            "radial":       lambda: VisualEffects.create_radial_gradient(w, h,
                                       kwargs.get("center_color", (255, 220, 50)),
                                       kwargs.get("edge_color",   (30, 30, 120)), config=cfg),
            "checkerboard": lambda: self._checkerboard(w, h,
                                       kwargs.get("square_size", 40),
                                       kwargs.get("c1", (255,255,255)),
                                       kwargs.get("c2", (30,30,30))),
            "waves":        lambda: self._waves(w, h, kwargs.get("freq", 0.05)),
            "circles":      lambda: self._concentric_circles(w, h, kwargs.get("n", 12)),
            "spiral":       lambda: self._spiral(w, h, kwargs.get("turns", 6)),
        }
        if art_type not in dispatch:
            raise TTIError(f"Unknown art type '{art_type}'. Valid: {sorted(self._ART_TYPES)}")

        img = dispatch[art_type]()
        if output:
            img.save(output)
        return img

    def _checkerboard(self, w, h, sq, c1, c2):
        img = TTIImage(w, h, 24, config=self.cfg)
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        mask = ((xs//sq)+(ys//sq))%2==0
        arr  = np.where(mask[:,:,None], np.array(c1,dtype=np.uint8), np.array(c2,dtype=np.uint8))
        img.from_array(arr.astype(np.uint8)); return img

    def _waves(self, w, h, freq=0.05):
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        v = (np.sin(xs*freq)+np.sin(ys*freq*0.7))*0.5+0.5
        r = np.clip(v*100+50,  0,255).astype(np.uint8)
        g = np.clip(v*150+80,  0,255).astype(np.uint8)
        b = np.clip(v*200+100, 0,255).astype(np.uint8)
        img = TTIImage(w, h, 24, config=self.cfg)
        img.from_array(np.stack([r,g,b],axis=-1)); return img

    def _concentric_circles(self, w, h, n=12):
        img = TTIImage(w, h, 24, background=(10,10,40), config=self.cfg)
        canvas = ImageCanvas(img); cx, cy = w//2, h//2; max_r = min(w,h)//2-5
        for i in range(n):
            t = i/n; r = max(2, int(max_r*(1-t)))
            canvas.circle(cx, cy, r, ColorUtils.hsv_to_rgb(t,0.8,0.9), filled=False)
        return img

    def _spiral(self, w, h, turns=6):
        img = TTIImage(w, h, 24, background=(5,5,20), config=self.cfg)
        cx, cy = w//2, h//2; steps = turns*360
        for i in range(steps):
            t = i/steps; angle = t*turns*2*np.pi; radius = t*min(w,h)//2
            x = cx + int(radius*np.cos(angle)); y = cy + int(radius*np.sin(angle))
            col = ColorUtils.hsv_to_rgb(t, 0.9, 1.0)
            img.set_pixel(x, y, col); img.set_pixel(x+1, y, col); img.set_pixel(x, y+1, col)
        return img

    # ── Effects ───────────────────────────────────────────────────────────

    def effect(
        self,
        effect_name: str,
        input_path:  Optional[str]  = None,
        img:         Optional[TTIImage] = None,
        output:      Optional[str]  = None,
        **kwargs,
    ) -> TTIImage:
        """Apply a named visual effect to a file or TTIImage."""
        if img is None and input_path is None:
            raise TTIError("Provide either input_path or img")
        src = img if img is not None else ImageIO.load(input_path, config=self.cfg)
        out = self._apply_effect(src, effect_name, **kwargs)
        if output:
            out.save(output)
        return out

    def _apply_effect(self, img: TTIImage, name: str, **kw) -> TTIImage:
        dispatch = {
            "blur":          lambda: VisualEffects.apply_blur(img, kw.get("radius")),
            "gaussian_blur": lambda: VisualEffects.apply_gaussian_blur(img, kw.get("sigma", 2.0)),
            "sharpen":       lambda: VisualEffects.apply_sharpen(img),
            "edge":          lambda: VisualEffects.apply_edge_detect(img),
            "emboss":        lambda: VisualEffects.apply_emboss(img),
            "grayscale":     lambda: VisualEffects.apply_grayscale(img),
            "sepia":         lambda: VisualEffects.apply_sepia(img),
            "invert":        lambda: VisualEffects.apply_invert(img),
            "noise":         lambda: VisualEffects.add_noise(img, kw.get("intensity")),
            "pixelate":      lambda: VisualEffects.pixelate(img, kw.get("block_size", 10)),
            "vignette":      lambda: VisualEffects.vignette(img, kw.get("strength", 0.6)),
            "brightness":    lambda: VisualEffects.adjust_brightness(img, kw.get("factor", 1.3)),
            "contrast":      lambda: VisualEffects.adjust_contrast(img, kw.get("factor", 1.3)),
        }
        if name not in dispatch:
            raise TTIError(f"Unknown effect '{name}'. Valid: {sorted(self._EFFECTS)}")
        return dispatch[name]()

    # ── Model utilities ───────────────────────────────────────────────────

    def load_model(self, path: str) -> None:
        """Hot-swap the underlying model checkpoint."""
        self._model = TTIModel.load(path, map_location="cpu")
        self._model.eval(); TTIModel.training = False
        self._log(f"Model reloaded ← {path} ({self._model.n_parameters():,} params)")

    def model_info(self) -> Optional[Dict[str, Any]]:
        if self._model is None:
            return None
        return {
            "params":      self._model.n_parameters(),
            "embed_dim":   self._model.cfg.embed_dim,
            "n_layers":    self._model.cfg.n_layers,
            "n_heads":     self._model.cfg.n_heads,
            "latent_dim":  self._model.cfg.latent_dim,
            "vocab_size":  len(self._vocab) if self._vocab else 0,
            "scene_classes": len(SCENE_CLASSES),
            "model_loaded": True,
        }

    # ── Demo ─────────────────────────────────────────────────────────────

    def demo(self, output_dir: str = "tti_output") -> Dict[str, Any]:
        """
        Run a comprehensive demo of all pipeline features.
        Saves all outputs to *output_dir*.
        Returns a summary dict.
        """
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        results: List[Dict] = []
        t0 = time.time()

        self._log(f"Demo → {out}/")

        # 1. AI Text-to-Image (12 diverse prompts)
        ai_prompts = [
            ("a calm blue ocean at sunset",              "ai_ocean.png"),
            ("dark mysterious forest at night",          "ai_forest.png"),
            ("bright golden mandelbrot fractal",         "ai_mandelbrot.png"),
            ("neon geometric city lights reflection",    "ai_city.png"),
            ("red fire flame plasma energy explosion",   "ai_fire.png"),
            ("cold icy purple space nebula stars",       "ai_nebula.png"),
            ("warm vintage desert sand landscape",       "ai_desert.png"),
            ("voronoi mosaic stained glass pattern",     "ai_voronoi.png"),
            ("spiral vortex golden fibonacci",           "ai_spiral.png"),
            ("dreamy ethereal lavender abstract art",    "ai_abstract.png"),
            ("sierpinski triangle fractal geometry",     "ai_sierpinski.png"),
            ("ripple water reflection peaceful lake",    "ai_ripple.png"),
        ]
        print(f"\n── AI Generation ({len(ai_prompts)} prompts) ──")
        for prompt, fname in ai_prompts:
            try:
                img = self.generate(prompt, seed=42)
                img.save(out / fname)
                results.append({"type": "ai", "prompt": prompt, "file": fname})
                print(f"  ✓ {fname}")
            except Exception as e:
                print(f"  ✗ {fname}: {e}")

        # 2. Variations
        print(f"\n── Variations (4 seeds) ──")
        try:
            vars_dir = out / "variations"
            imgs = self.variations("glowing magical forest ethereal", n=4,
                                   output_dir=str(vars_dir), width=256, height=256)
            results.append({"type": "variations", "n": len(imgs)})
            print(f"  ✓ {len(imgs)} variations → {vars_dir}/")
        except Exception as e:
            print(f"  ✗ Variations: {e}")

        # 3. Interpolation
        print(f"\n── Interpolation (6 frames) ──")
        try:
            interp_dir = out / "interpolation"
            imgs = self.interpolate("warm golden sunrise", "cold dark midnight",
                                    steps=6, output_dir=str(interp_dir),
                                    width=256, height=256)
            results.append({"type": "interpolation", "n": len(imgs)})
            print(f"  ✓ {len(imgs)} frames → {interp_dir}/")
        except Exception as e:
            print(f"  ✗ Interpolation: {e}")

        # 4. Procedural art (all 12 types)
        print(f"\n── Procedural Art ({len(self._ART_TYPES)} types) ──")
        for art_type in sorted(self._ART_TYPES):
            try:
                img = self.art(art_type, width=256, height=256)
                img.save(out / f"art_{art_type}.png")
                results.append({"type": "art", "art_type": art_type})
                print(f"  ✓ art_{art_type}.png")
            except Exception as e:
                print(f"  ✗ {art_type}: {e}")

        # 5. Effects (all 13)
        print(f"\n── Visual Effects ({len(self._EFFECTS)}) ──")
        base = self.art("plasma", width=256, height=256)
        for eff in sorted(self._EFFECTS):
            try:
                img = self._apply_effect(base.copy(), eff)
                img.save(out / f"fx_{eff}.png")
                results.append({"type": "effect", "effect": eff})
                print(f"  ✓ fx_{eff}.png")
            except Exception as e:
                print(f"  ✗ {eff}: {e}")

        # 6. NLP + model analysis
        print(f"\n── NLP + Model Analysis ──")
        for p in ["a bright vivid rainbow over a misty waterfall",
                  "dark gothic castle at midnight"]:
            info = self.analyse(p, verbose=True)
            results.append({"type": "analysis", "prompt": p})

        # 7. Save model info
        info = self.model_info()
        if info:
            print(f"\n── Model Info ──")
            for k, v in info.items():
                print(f"  {k:16s}: {v}")

        elapsed = time.time() - t0
        summary = {
            "outputs":      len(results),
            "elapsed_s":    round(elapsed, 1),
            "model_loaded": self._model is not None,
            "output_dir":   str(out),
        }
        with open(out / "demo_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*55}")
        print(f"  Demo complete — {len(results)} outputs in {elapsed:.1f}s")
        print(f"  Output dir: {out}/")
        print(f"{'='*55}")
        return summary

    # ── Config ────────────────────────────────────────────────────────────

    def set_config(self, **kwargs) -> None:
        update_config(**kwargs)

    def show_config(self) -> None:
        print(self.cfg.to_json())

    # ── Internal ──────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.cfg.log.show_progress:
            print(f"[TTI] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse, sys

    p = argparse.ArgumentParser(
        prog="TTI_pipeline",
        description="TTI unified pipeline — trained model + NLP + procedural art",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python TTI_pipeline.py generate "a calm blue ocean at sunset" --output ocean.png
  python TTI_pipeline.py generate "neon city" --width 512 --height 512 --seed 42
  python TTI_pipeline.py analyse  "a mysterious purple galaxy"
  python TTI_pipeline.py art      mandelbrot --output fractal.png --width 800
  python TTI_pipeline.py effect   sepia photo.png --output vintage.png
  python TTI_pipeline.py variations "stormy ocean" --n 4 --output-dir vars/
  python TTI_pipeline.py interpolate "sunrise" "midnight" --steps 6
  python TTI_pipeline.py demo --output-dir tti_output/
        """,
    )
    sub = p.add_subparsers(dest="cmd")

    # generate
    g = sub.add_parser("generate")
    g.add_argument("prompt")
    g.add_argument("--output", "-o", default=None)
    g.add_argument("--width",  "-W", type=int, default=None)
    g.add_argument("--height", "-H", type=int, default=None)
    g.add_argument("--seed",   "-s", type=int, default=None)
    g.add_argument("--temp",   "-t", type=float, default=0.9)
    g.add_argument("--model",  "-m", default=None)

    # analyse
    a = sub.add_parser("analyse")
    a.add_argument("prompt")
    a.add_argument("--model", "-m", default=None)

    # art
    ar = sub.add_parser("art")
    ar.add_argument("art_type", choices=sorted(TTIPipeline._ART_TYPES))
    ar.add_argument("--output","-o", default=None)
    ar.add_argument("--width", "-W", type=int, default=None)
    ar.add_argument("--height","-H", type=int, default=None)
    ar.add_argument("--seed",  "-s", type=int, default=None)

    # effect
    e = sub.add_parser("effect")
    e.add_argument("effect", choices=sorted(TTIPipeline._EFFECTS))
    e.add_argument("input")
    e.add_argument("--output","-o", default=None)
    e.add_argument("--radius",     type=int,   default=None)
    e.add_argument("--factor",     type=float, default=1.3)
    e.add_argument("--strength",   type=float, default=0.6)
    e.add_argument("--intensity",  type=float, default=None)
    e.add_argument("--block-size", type=int,   default=10)

    # variations
    v = sub.add_parser("variations")
    v.add_argument("prompt")
    v.add_argument("--n",          type=int, default=4)
    v.add_argument("--output-dir", default="tti_output/variations")
    v.add_argument("--width",  "-W", type=int, default=None)
    v.add_argument("--height", "-H", type=int, default=None)

    # interpolate
    i = sub.add_parser("interpolate")
    i.add_argument("prompt_a")
    i.add_argument("prompt_b")
    i.add_argument("--steps",      type=int, default=6)
    i.add_argument("--output-dir", default="tti_output/interpolation")
    i.add_argument("--width",  "-W", type=int, default=None)
    i.add_argument("--height", "-H", type=int, default=None)

    # demo
    d = sub.add_parser("demo")
    d.add_argument("--output-dir", default="tti_output")
    d.add_argument("--model", "-m", default=None)

    args = p.parse_args()
    if args.cmd is None:
        p.print_help(); sys.exit(0)

    model_path = getattr(args, "model", None)
    pipe = TTIPipeline(model_path=model_path)

    if args.cmd == "generate":
        out = args.output or f"tti_output/generated.png"
        pipe.generate(args.prompt, output=out, width=args.width,
                      height=args.height, seed=args.seed, temperature=args.temp)

    elif args.cmd == "analyse":
        pipe.analyse(args.prompt, verbose=True)

    elif args.cmd == "art":
        out = args.output or f"tti_output/{args.art_type}.png"
        pipe.art(args.art_type, output=out, width=args.width,
                 height=args.height, seed=args.seed)

    elif args.cmd == "effect":
        out = args.output or f"tti_output/effect_{args.effect}.png"
        kw = {"factor": args.factor, "strength": args.strength,
              "block_size": args.block_size}
        if args.radius    is not None: kw["radius"]    = args.radius
        if args.intensity is not None: kw["intensity"] = args.intensity
        pipe.effect(args.effect, input_path=args.input, output=out, **kw)

    elif args.cmd == "variations":
        pipe.variations(args.prompt, n=args.n, output_dir=args.output_dir,
                        width=args.width, height=args.height)

    elif args.cmd == "interpolate":
        pipe.interpolate(args.prompt_a, args.prompt_b, steps=args.steps,
                         output_dir=args.output_dir,
                         width=args.width, height=args.height)

    elif args.cmd == "demo":
        pipe.demo(output_dir=args.output_dir)


if __name__ == "__main__":
    _cli()
