"""
TTI_ai.py
=========
AI engine for the Text-To-Image (TTI) system.

Architecture
------------
                        ┌─────────────────────────────────┐
  prompt text ─────────▶│   NLPAnalyser  (NLTK + sklearn) │
                        └──────────────┬──────────────────┘
                                       │  SemanticVector (256-d)
                        ┌──────────────▼──────────────────┐
                        │  ColourPredictor (sklearn KNN)   │
                        └──────────────┬──────────────────┘
                                       │  PaletteSpec
                        ┌──────────────▼──────────────────┐
                        │  SceneComposer  (numpy VAE-like) │
                        └──────────────┬──────────────────┘
                                       │  LatentCode (128-d)
                        ┌──────────────▼──────────────────┐
                        │  ImageDecoder  (numpy renderer)  │
                        └──────────────┬──────────────────┘
                                       │  TTIImage
                                       ▼

Torch compatibility
-------------------
All layers are implemented as numpy operations with a PyTorch-compatible
API (forward(), parameters(), named_parameters()).  If torch is installed,
swap NumpyTensor → torch.Tensor — the rest of the code is unchanged.

NLP stack: NLTK tokenisation + PoS tagging + stopword filtering.
           sklearn TF-IDF + KNN for colour-palette clustering.
           sklearn PCA for dimensionality reduction of text embeddings.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import PorterStemmer

from .TTI_config import TTIConfig, get_config
from .TTI_core import TTIImage, ImageCanvas, ColorUtils, Color
from .TTI_art import ProceduralArt, VisualEffects

# ---------------------------------------------------------------------------
# Safe NLTK resource loader (works offline)
# ---------------------------------------------------------------------------

_NLTK_READY = False


def _ensure_nltk() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    for resource in ("tokenizers/punkt_tab", "corpora/stopwords",
                     "taggers/averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(resource)
        except LookupError:
            pass          # offline – fall back to built-in tokeniser
    _NLTK_READY = True


# ---------------------------------------------------------------------------
# Built-in colour knowledge base
# ---------------------------------------------------------------------------

# Maps keywords → (R, G, B) colour
COLOUR_KB: Dict[str, Tuple[int,int,int]] = {
    # --- environments ---
    "sky":         (135, 206, 235), "ocean":    (0,  105, 148),
    "sea":         (0,  119, 182),  "water":    (64, 164, 223),
    "forest":      (34, 139,  34),  "tree":     (0,  100,   0),
    "grass":       (124, 252,  0),  "field":    (85, 170,  50),
    "desert":      (210, 180, 140), "sand":     (244, 214, 130),
    "mountain":    (105,  90,  75), "rock":     (128, 128, 128),
    "snow":        (255, 250, 250), "ice":      (176, 224, 230),
    "fire":        (255,  69,   0), "flame":    (255, 140,   0),
    "sun":         (255, 215,   0), "moon":     (245, 245, 210),
    "star":        (255, 255, 200), "space":    (10,  10,  40),
    "night":       (25,  25, 112),  "dawn":     (255, 160, 100),
    "sunset":      (255, 120,  60), "fog":      (200, 200, 200),
    "rain":        (100, 130, 180), "cloud":    (200, 210, 220),
    "rainbow":     (255, 100, 100), "lightning":(200, 200, 255),
    # --- moods ---
    "calm":        (173, 216, 230), "peaceful": (152, 251, 152),
    "dark":        (20,  20,  20),  "bright":   (255, 255, 200),
    "mysterious":  (75,  0, 130),   "magical":  (147, 112, 219),
    "scary":       (50,  0,   0),   "happy":    (255, 223,   0),
    "sad":         (100, 120, 180), "angry":    (180,  0,   0),
    "romantic":    (255, 105, 180), "warm":     (255, 160,  80),
    "cold":        (120, 180, 255), "hot":      (220,  50,   0),
    "ethereal":    (200, 180, 255), "dreamy":   (230, 190, 255),
    # --- objects ---
    "gold":        (255, 215,   0), "silver":   (192, 192, 192),
    "metal":       (160, 160, 160), "stone":    (128, 110,  90),
    "wood":        (139,  90,  43), "glass":    (200, 220, 255),
    "blood":       (160,   0,   0), "bone":     (240, 230, 210),
    "flower":      (255, 100, 200), "rose":     (220,  20,  60),
    "leaf":        (0,  160,  50),  "vine":     (0,  128,   0),
    # --- explicit colours ---
    "red":         (220,  50,  50), "green":    (50, 180,  50),
    "blue":        (50,  100, 220), "yellow":   (240, 220,  50),
    "orange":      (240, 140,  40), "purple":   (130,  50, 200),
    "pink":        (240, 130, 180), "brown":    (139,  90,  43),
    "black":       (20,  20,  20),  "white":    (240, 240, 240),
    "grey":        (160, 160, 160), "gray":     (160, 160, 160),
    "cyan":        (0,  200, 200),  "magenta":  (200,  0, 200),
    "teal":        (0,  128, 128),  "maroon":   (128,  0,   0),
    "navy":        (0,   0,  128),  "olive":    (128, 128,   0),
    "lime":        (0,  255,   0),  "violet":   (148,  0, 211),
    "indigo":      (75,  0, 130),   "crimson":  (220,  20,  60),
    "lavender":    (200, 162, 200), "turquoise":(64, 224, 208),
    # --- art styles ---
    "painting":    (200, 150, 100), "sketch":   (50,  50,  50),
    "neon":        (57, 255, 200),  "pastel":   (255, 200, 200),
    "vintage":     (180, 150, 100), "retro":    (200, 100,  50),
    "futuristic":  (50, 150, 255),  "glowing":  (200, 255, 100),
}

# Maps keywords → scene type
SCENE_KB: Dict[str, str] = {
    "sunset": "gradient", "sunrise": "gradient", "sky": "gradient",
    "ocean": "gradient",  "sea": "gradient",      "lake": "gradient",
    "forest": "fractal",  "tree": "organic",      "jungle": "organic",
    "galaxy": "fractal",  "space": "fractal",     "nebula": "fractal",
    "city": "geometric",  "building": "geometric","architecture": "geometric",
    "abstract": "abstract","pattern": "abstract", "art": "abstract",
    "landscape": "gradient","mountain": "organic","desert": "gradient",
    "fire": "plasma",     "flame": "plasma",      "lava": "plasma",
    "water": "ripple",    "rain": "ripple",       "wave": "ripple",
    "night": "starfield", "star": "starfield",    "universe": "starfield",
    "flower": "organic",  "garden": "organic",    "nature": "organic",
    "circle": "circles",  "spiral": "spiral",     "geometric": "geometric",
    "mandelbrot": "mandelbrot", "fractal": "fractal",
    "julia": "julia",     "sierpinski": "sierpinski",
    "voronoi": "voronoi", "noise": "noise",
}

# Maps modifiers → rendering adjustments
MODIFIER_KB: Dict[str, Dict] = {
    "bright":    {"brightness": 1.4, "saturation": 1.2},
    "dark":      {"brightness": 0.5, "saturation": 0.8},
    "vivid":     {"brightness": 1.1, "saturation": 1.5},
    "muted":     {"brightness": 0.9, "saturation": 0.6},
    "blurry":    {"blur": 3},
    "sharp":     {"sharpen": True},
    "noisy":     {"noise": 0.3},
    "vintage":   {"sepia": True,  "brightness": 0.85},
    "dreamy":    {"blur": 2, "brightness": 1.1},
    "neon":      {"brightness": 1.3, "saturation": 2.0},
    "foggy":     {"blur": 4, "brightness": 0.9},
    "glowing":   {"brightness": 1.5},
    "large":     {"scale": 1.5},
    "small":     {"scale": 0.7},
    "detailed":  {"detail": 2},
    "simple":    {"detail": 0.5},
}


# ---------------------------------------------------------------------------
# NumpyTensor — thin wrapper mimicking torch.Tensor API
# ---------------------------------------------------------------------------

class NumpyTensor:
    """
    A numpy-backed tensor with a PyTorch-compatible interface.
    Swap for torch.Tensor when PyTorch is available.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.array(data, dtype=np.float32)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"NumpyTensor({self.data.shape}, dtype=float32)"

    # ---- arithmetic ----
    def __add__(self, other):
        o = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data + o)
    def __mul__(self, other):
        o = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data * o)
    def __sub__(self, other):
        o = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data - o)
    def __truediv__(self, other):
        o = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data / o)
    def __matmul__(self, other):
        o = other.data if isinstance(other, NumpyTensor) else other
        return NumpyTensor(self.data @ o)

    # ---- activations ----
    def relu(self):      return NumpyTensor(np.maximum(0, self.data))
    def sigmoid(self):   return NumpyTensor(1 / (1 + np.exp(-self.data.clip(-500,500))))
    def tanh(self):      return NumpyTensor(np.tanh(self.data))
    def softmax(self, axis=-1):
        e = np.exp(self.data - self.data.max(axis=axis, keepdims=True))
        return NumpyTensor(e / e.sum(axis=axis, keepdims=True))
    def mean(self, axis=None): return NumpyTensor(self.data.mean(axis=axis))
    def sum(self, axis=None):  return NumpyTensor(self.data.sum(axis=axis))
    def flatten(self):   return NumpyTensor(self.data.flatten())
    def reshape(self, *shape): return NumpyTensor(self.data.reshape(*shape))
    def numpy(self):     return self.data

    @staticmethod
    def zeros(*shape):  return NumpyTensor(np.zeros(shape, dtype=np.float32))
    @staticmethod
    def ones(*shape):   return NumpyTensor(np.ones(shape,  dtype=np.float32))
    @staticmethod
    def randn(*shape):  return NumpyTensor(np.random.randn(*shape).astype(np.float32))
    @staticmethod
    def from_numpy(arr: np.ndarray): return NumpyTensor(arr)


# ---------------------------------------------------------------------------
# NumpyLinear — linear layer (nn.Linear equivalent)
# ---------------------------------------------------------------------------

class NumpyLinear:
    """Fully-connected linear layer: out = x @ W.T + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        scale = math.sqrt(2.0 / in_features)
        self.weight = NumpyTensor(
            np.random.randn(out_features, in_features).astype(np.float32) * scale
        )
        self.bias_: Optional[NumpyTensor] = (
            NumpyTensor(np.zeros(out_features, dtype=np.float32)) if bias else None
        )
        self.in_features  = in_features
        self.out_features = out_features

    def forward(self, x: NumpyTensor) -> NumpyTensor:
        out = x @ NumpyTensor(self.weight.data.T)
        if self.bias_ is not None:
            out = out + self.bias_
        return out

    def __call__(self, x: NumpyTensor) -> NumpyTensor:
        return self.forward(x)

    def parameters(self) -> List[NumpyTensor]:
        p = [self.weight]
        if self.bias_ is not None:
            p.append(self.bias_)
        return p

    def named_parameters(self):
        yield "weight", self.weight
        if self.bias_ is not None:
            yield "bias", self.bias_


class NumpyLayerNorm:
    """Layer normalisation."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.gamma = NumpyTensor(np.ones(dim,  dtype=np.float32))
        self.beta  = NumpyTensor(np.zeros(dim, dtype=np.float32))
        self.eps   = eps

    def forward(self, x: NumpyTensor) -> NumpyTensor:
        d = x.data
        mean = d.mean(axis=-1, keepdims=True)
        var  = d.var(axis=-1,  keepdims=True)
        norm = (d - mean) / np.sqrt(var + self.eps)
        return NumpyTensor(norm * self.gamma.data + self.beta.data)

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [self.gamma, self.beta]


# ---------------------------------------------------------------------------
# NLPAnalyser
# ---------------------------------------------------------------------------

class NLPAnalyser:
    """
    Analyses a text prompt and extracts:
    - tokens / stems
    - colour keywords
    - scene type
    - modifiers
    - a TF-IDF semantic embedding
    """

    _BUILTIN_STOPWORDS = {
        "a","an","the","of","in","on","at","to","is","it","be","and","or",
        "for","with","this","that","are","was","were","has","have","had",
        "do","does","did","by","from","as","i","me","my","you","your",
        "he","she","we","they","his","her","its","our","their","would",
        "could","should","may","might","will","shall","can","just","very",
        "so","then","than","but","not","no","some","any","all","more",
        "most","also","into","over","under","out","up","down","off","if",
        "about","between","through","after","before","during","make","makes",
        "made","create","creating","show","showing","depicts","featuring",
    }

    def __init__(self, config: Optional[TTIConfig] = None) -> None:
        self._cfg      = config or get_config()
        self._stemmer  = PorterStemmer()
        self._stopwords = self._BUILTIN_STOPWORDS.copy()
        _ensure_nltk()

    def tokenise(self, text: str) -> List[str]:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s\-']", " ", text)
        try:
            tokens = nltk.word_tokenize(text)
        except Exception:
            tokens = text.split()
        return [t for t in tokens if len(t) > 1]

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        if not self._cfg.ai.use_stopword_filter:
            return tokens
        try:
            sw = set(nltk.corpus.stopwords.words("english"))
            self._stopwords.update(sw)
        except Exception:
            pass
        return [t for t in tokens if t not in self._stopwords]

    def stem(self, tokens: List[str]) -> List[str]:
        return [self._stemmer.stem(t) for t in tokens]

    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        try:
            return nltk.pos_tag(tokens)
        except Exception:
            return [(t, "NN") for t in tokens]

    def extract_colours(self, tokens: List[str]) -> List[Tuple[str, Color]]:
        found = []
        for t in tokens:
            if t in COLOUR_KB:
                found.append((t, COLOUR_KB[t]))
        return found

    def extract_scene_type(self, tokens: List[str]) -> str:
        for t in tokens:
            if t in SCENE_KB:
                return SCENE_KB[t]
        return "abstract"

    def extract_modifiers(self, tokens: List[str]) -> Dict[str, Any]:
        mods: Dict[str, Any] = {}
        for t in tokens:
            if t in MODIFIER_KB:
                mods.update(MODIFIER_KB[t])
        return mods

    def analyse(self, prompt: str) -> "PromptAnalysis":
        tokens   = self.tokenise(prompt)
        filtered = self.filter_tokens(tokens)
        pos      = self.pos_tag(filtered)
        colours  = self.extract_colours(filtered)
        scene    = self.extract_scene_type(filtered)
        mods     = self.extract_modifiers(filtered)
        stems    = self.stem(filtered)
        nouns    = [w for w, t in pos if t.startswith("NN")]
        adjs     = [w for w, t in pos if t.startswith("JJ")]
        return PromptAnalysis(
            raw_prompt=prompt,
            tokens=tokens,
            filtered_tokens=filtered,
            stems=stems,
            pos_tags=pos,
            colour_matches=colours,
            scene_type=scene,
            modifiers=mods,
            nouns=nouns,
            adjectives=adjs,
        )


@dataclass
class PromptAnalysis:
    """Structured result of NLP analysis."""
    raw_prompt:     str
    tokens:         List[str]
    filtered_tokens: List[str]
    stems:          List[str]
    pos_tags:       List[Tuple[str, str]]
    colour_matches: List[Tuple[str, Color]]
    scene_type:     str
    modifiers:      Dict[str, Any]
    nouns:          List[str]
    adjectives:     List[str]

    def primary_colour(self) -> Color:
        return self.colour_matches[0][1] if self.colour_matches else (100, 120, 200)

    def secondary_colour(self) -> Color:
        return self.colour_matches[1][1] if len(self.colour_matches) > 1 else (
            ColorUtils.hsv_to_rgb(
                (ColorUtils.rgb_to_hsv(*self.primary_colour()[:3])[0] + 0.5) % 1.0,
                0.7, 0.8
            )
        )

    def background_colour(self) -> Color:
        return self.colour_matches[-1][1] if self.colour_matches else (20, 20, 40)

    def complexity(self) -> float:
        """0 – 1 normalised complexity hint from prompt length and keywords."""
        base = min(len(self.filtered_tokens) / 20.0, 1.0)
        if "detailed" in self.modifiers:
            base = min(base * self.modifiers["detail"], 1.0)
        return base

    def __repr__(self):
        return (
            f"PromptAnalysis(scene={self.scene_type!r}, "
            f"colours={[k for k,_ in self.colour_matches]}, "
            f"mods={list(self.modifiers.keys())})"
        )


# ---------------------------------------------------------------------------
# ColourPredictor  (sklearn KNN)
# ---------------------------------------------------------------------------

class ColourPredictor:
    """
    Predicts a colour palette from a TF-IDF text embedding using KNN.
    Self-trains from COLOUR_KB on first use (no external dataset needed).
    """

    def __init__(self, config: Optional[TTIConfig] = None) -> None:
        self._cfg  = config or get_config()
        self._vec  = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=512,
            sublinear_tf=True,
        )
        self._scaler = StandardScaler()
        self._pca    = PCA(n_components=min(32, len(COLOUR_KB)))
        self._knn    = KNeighborsClassifier(
            n_neighbors=min(5, len(COLOUR_KB)),
            metric="cosine",
            algorithm="brute",
        )
        self._trained = False
        self._train_self()

    def _train_self(self) -> None:
        """Bootstrap training from the built-in colour knowledge base."""
        # Build (sentence, colour_name) pairs
        sentences, labels = [], []
        colour_map = list(COLOUR_KB.keys())
        # Each keyword acts as its own training sentence
        for kw in colour_map:
            sentences.append(kw.replace("_", " "))
            labels.append(kw)
        # Augment with related pairs
        for kw in colour_map:
            if kw in SCENE_KB:
                sentences.append(f"{kw} scene landscape")
                labels.append(kw)

        X_raw = self._vec.fit_transform(sentences).toarray().astype(np.float32)
        # PCA to compress
        n_comp = min(32, X_raw.shape[0]-1, X_raw.shape[1])
        self._pca = PCA(n_components=n_comp)
        X_pca = self._pca.fit_transform(X_raw)
        X_sc  = self._scaler.fit_transform(X_pca)
        self._knn.fit(X_sc, labels)
        self._colour_map = colour_map
        self._trained = True

    def predict_palette(
        self,
        prompt:    str,
        n_colours: int = 5,
    ) -> List[Color]:
        """Return *n_colours* colours predicted for the *prompt*."""
        try:
            X = self._vec.transform([prompt]).toarray().astype(np.float32)
            X = self._pca.transform(X)
            X = self._scaler.transform(X)
            dists, idxs = self._knn.kneighbors(X, n_neighbors=min(n_colours*2, len(self._colour_map)))
            seen, palette = set(), []
            for idx in idxs[0]:
                label = self._knn.classes_[idx]
                if label not in seen and label in COLOUR_KB:
                    palette.append(COLOUR_KB[label])
                    seen.add(label)
                if len(palette) >= n_colours:
                    break
            # Pad with derived colours
            while len(palette) < n_colours:
                base = palette[0] if palette else (128, 128, 200)
                h, s, v = ColorUtils.rgb_to_hsv(*base)
                palette.append(ColorUtils.hsv_to_rgb((h + 0.15*len(palette)) % 1.0, max(0.3,s-0.1), v))
            return palette
        except Exception:
            return list(COLOUR_KB.values())[:n_colours]


# ---------------------------------------------------------------------------
# SceneComposer  — numpy VAE-like latent encoder
# ---------------------------------------------------------------------------

class SceneComposer:
    """
    Encodes a PromptAnalysis into a latent code via a small MLP,
    then decodes it to scene parameters used by the renderer.

    Architecture (numpy, PyTorch-compatible API):
        Encoder:  embed(256) → linear(512) → ReLU → linear(256) → ReLU → μ,σ(128)
        Decoder:  z(128)     → linear(256) → ReLU → linear(512) → ReLU → params(64)
    """

    def __init__(self, config: Optional[TTIConfig] = None) -> None:
        self._cfg    = config or get_config()
        dim_in  = self._cfg.ai.text_embed_dim     # 256
        dim_h   = self._cfg.ai.hidden_dim          # 512
        dim_lat = self._cfg.ai.latent_dim          # 128

        # Encoder
        self.enc1  = NumpyLinear(dim_in, dim_h)
        self.enc2  = NumpyLinear(dim_h,  dim_in)
        self.mu    = NumpyLinear(dim_in, dim_lat)
        self.log_v = NumpyLinear(dim_in, dim_lat)
        # Decoder
        self.dec1  = NumpyLinear(dim_lat, dim_in)
        self.dec2  = NumpyLinear(dim_in,  dim_h)
        self.dec3  = NumpyLinear(dim_h,   64)
        # Layer norms
        self.ln1   = NumpyLayerNorm(dim_h)
        self.ln2   = NumpyLayerNorm(dim_in)

    def _embed_analysis(self, analysis: PromptAnalysis) -> NumpyTensor:
        """Convert PromptAnalysis → fixed-dim float vector (text embedding)."""
        dim = self._cfg.ai.text_embed_dim
        vec = np.zeros(dim, dtype=np.float32)

        # Colour signals (first 96 dims)
        for i, (_, col) in enumerate(analysis.colour_matches[:8]):
            base = i * 12
            vec[base:base+3] = np.array(col[:3], dtype=np.float32) / 255.0

        # Scene type one-hot (next 20 dims)
        scene_types = list(dict.fromkeys(SCENE_KB.values()))
        try:
            idx = scene_types.index(analysis.scene_type)
            vec[96 + idx] = 1.0
        except ValueError:
            pass

        # Token hash fingerprint (next 64 dims)
        for i, tok in enumerate(analysis.filtered_tokens[:32]):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            dim_idx = 116 + (h % 64)
            vec[dim_idx] += 1.0 / (i + 1)

        # Modifier flags (next 32 dims)
        mods = list(MODIFIER_KB.keys())
        for i, m in enumerate(mods[:32]):
            if m in analysis.modifiers:
                vec[180 + i] = 1.0

        # Prompt length normalised (last 12 dims)
        vec[212] = min(len(analysis.tokens) / 50.0, 1.0)
        vec[213] = analysis.complexity()
        vec[214] = len(analysis.colour_matches) / 10.0
        vec[215] = float(analysis.scene_type == "fractal")
        vec[216] = float(analysis.scene_type == "gradient")
        vec[217] = float(analysis.scene_type == "geometric")

        # Normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return NumpyTensor(vec)

    def encode(self, analysis: PromptAnalysis) -> Tuple[NumpyTensor, NumpyTensor]:
        """Return (mu, log_var) latent parameters."""
        x = self._embed_analysis(analysis)
        h = self.ln1(self.enc1(x).relu())
        h = self.ln2(self.enc2(h).relu())
        return self.mu(h), self.log_v(h)

    def reparameterise(
        self,
        mu:      NumpyTensor,
        log_var: NumpyTensor,
        seed:    Optional[int] = None,
    ) -> NumpyTensor:
        """VAE reparameterisation trick."""
        rng = np.random.default_rng(seed)
        eps = NumpyTensor(rng.standard_normal(mu.shape).astype(np.float32))
        std = NumpyTensor(np.exp(0.5 * log_var.data))
        return mu + std * eps

    def decode(self, z: NumpyTensor) -> NumpyTensor:
        """Decode latent code → 64-dim scene parameter vector."""
        h = self.dec1(z).relu()
        h = self.dec2(h).relu()
        return self.dec3(h).sigmoid()

    def compose(
        self,
        analysis: PromptAnalysis,
        seed:     Optional[int] = None,
    ) -> "SceneParameters":
        """Full encode → reparameterise → decode → SceneParameters."""
        mu, lv = self.encode(analysis)
        z      = self.reparameterise(mu, lv, seed)
        params = self.decode(z).numpy()
        return SceneParameters.from_vector(params, analysis)

    def parameters(self):
        layers = [self.enc1, self.enc2, self.mu, self.log_v,
                  self.dec1, self.dec2, self.dec3, self.ln1, self.ln2]
        return [p for layer in layers for p in layer.parameters()]

    def named_parameters(self):
        for name, layer in [
            ("enc1", self.enc1), ("enc2", self.enc2),
            ("mu",   self.mu),   ("log_v", self.log_v),
            ("dec1", self.dec1), ("dec2", self.dec2), ("dec3", self.dec3),
        ]:
            for pname, p in layer.named_parameters():
                yield f"{name}.{pname}", p


@dataclass
class SceneParameters:
    """Decoded scene parameters ready for the renderer."""
    scene_type:       str
    primary_color:    Color
    secondary_color:  Color
    background_color: Color
    accent_color:     Color
    palette:          List[Color]
    brightness:       float
    contrast:         float
    saturation:       float
    noise_level:      float
    blur_radius:      int
    detail_scale:     float
    num_shapes:       int
    fractal_zoom:     float
    fractal_iter:     int
    pattern_scale:    float
    apply_sepia:      bool
    apply_sharpen:    bool
    apply_vignette:   bool
    modifiers:        Dict[str, Any]

    @classmethod
    def from_vector(
        cls,
        v:        np.ndarray,
        analysis: "PromptAnalysis",
    ) -> "SceneParameters":
        """Map 64-float parameter vector → SceneParameters."""
        palette = [analysis.primary_colour(),
                   analysis.secondary_colour(),
                   analysis.background_colour()]
        while len(palette) < 6:
            h, s, vv = ColorUtils.rgb_to_hsv(*palette[-1][:3])
            palette.append(
                ColorUtils.hsv_to_rgb((h + 0.17) % 1.0, max(0.2, s), max(0.2, vv))
            )

        mods = analysis.modifiers
        return cls(
            scene_type      = analysis.scene_type,
            primary_color   = palette[0],
            secondary_color = palette[1],
            background_color= palette[2],
            accent_color    = palette[3],
            palette         = palette,
            brightness      = float(mods.get("brightness",  0.8 + v[0]*0.6)),
            contrast        = float(mods.get("contrast",    0.8 + v[1]*0.5)),
            saturation      = float(mods.get("saturation",  0.7 + v[2]*0.6)),
            noise_level     = float(mods.get("noise",       v[3]*0.3)),
            blur_radius     = int(mods.get("blur",          int(v[4]*4))),
            detail_scale    = float(mods.get("detail",      0.5 + v[5])),
            num_shapes      = max(3, int(5 + v[6]*20)),
            fractal_zoom    = 0.5 + v[7]*2.0,
            fractal_iter    = max(64, int(64 + v[8]*192)),
            pattern_scale   = 0.02 + v[9]*0.12,
            apply_sepia     = bool(mods.get("sepia",        v[10] > 0.7)),
            apply_sharpen   = bool(mods.get("sharpen",      v[11] > 0.8)),
            apply_vignette  = v[12] > 0.6,
            modifiers       = mods,
        )


# ---------------------------------------------------------------------------
# ImageDecoder  (renderer)
# ---------------------------------------------------------------------------

class ImageDecoder:
    """
    Renders a TTIImage from SceneParameters.
    Each scene_type has its own rendering strategy.
    """

    _RENDERERS: Dict[str, str] = {}   # populated below

    def __init__(self, config: Optional[TTIConfig] = None) -> None:
        self._cfg = config or get_config()

    def render(
        self,
        params: SceneParameters,
        width:  int,
        height: int,
        seed:   Optional[int] = None,
    ) -> TTIImage:
        rng = np.random.default_rng(seed)
        fn  = getattr(self, f"_render_{params.scene_type}", self._render_abstract)
        img = fn(params, width, height, rng)
        img = self._apply_post(img, params)
        return img

    # ------------------------------------------------------------------ #
    # Post-processing
    # ------------------------------------------------------------------ #

    def _apply_post(self, img: TTIImage, p: SceneParameters) -> TTIImage:
        if p.blur_radius > 0:
            img = VisualEffects.apply_blur(img, p.blur_radius)
        if p.apply_sharpen:
            img = VisualEffects.apply_sharpen(img)
        if p.apply_sepia:
            img = VisualEffects.apply_sepia(img)
        if p.brightness != 1.0:
            img = VisualEffects.adjust_brightness(img, p.brightness)
        if p.contrast != 1.0:
            img = VisualEffects.adjust_contrast(img, p.contrast)
        if p.noise_level > 0.01:
            img = VisualEffects.add_noise(img, p.noise_level)
        if p.apply_vignette:
            img = VisualEffects.vignette(img, 0.6)
        return img

    # ------------------------------------------------------------------ #
    # Scene renderers
    # ------------------------------------------------------------------ #

    def _render_gradient(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        direction = rng.choice(["horizontal", "vertical", "diagonal"])
        img = VisualEffects.create_linear_gradient(
            w, h, p.primary_color, p.secondary_color,
            direction=direction, config=self._cfg
        )
        # Overlay a few accent circles
        canvas = ImageCanvas(img)
        for _ in range(p.num_shapes // 3):
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))
            r  = int(rng.integers(20, max(21, min(w,h)//4)))
            canvas.circle(cx, cy, r, p.accent_color, filled=False)
        return img

    def _render_fractal(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        return ProceduralArt.mandelbrot_set(
            w, h,
            max_iterations=p.fractal_iter,
            zoom=p.fractal_zoom,
            config=self._cfg,
        )

    def _render_mandelbrot(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        cx = float(rng.uniform(-1.5, 0.5))
        cy = float(rng.uniform(-1.0, 1.0))
        return ProceduralArt.mandelbrot_set(
            w, h, p.fractal_iter, p.fractal_zoom, cx, cy, self._cfg
        )

    def _render_julia(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        c_r = float(rng.uniform(-1.0, 0.5))
        c_i = float(rng.uniform(-0.5, 0.5))
        return ProceduralArt.julia_set(
            w, h, c_r, c_i, p.fractal_iter, p.fractal_zoom, self._cfg
        )

    def _render_sierpinski(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        return ProceduralArt.sierpinski_triangle(
            w, h,
            depth=max(4, int(p.detail_scale * 7)),
            color=p.primary_color,
            bg=p.background_color,
            config=self._cfg,
        )

    def _render_voronoi(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        n = max(5, p.num_shapes)
        return ProceduralArt.voronoi(w, h, n_cells=n, seed=int(rng.integers(0,9999)), config=self._cfg)

    def _render_noise(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        return ProceduralArt.perlin_noise_image(
            w, h,
            scale=p.pattern_scale,
            octaves=max(1, int(p.detail_scale * 6)),
            config=self._cfg,
        )

    def _render_plasma(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        img = ProceduralArt.plasma(w, h, scale=p.pattern_scale, config=self._cfg)
        # Colour-tint toward primary
        arr  = img._pixels.astype(np.float32)
        tint = np.array(p.primary_color[:3], dtype=np.float32) / 255.0
        arr[:,:,0] = np.clip(arr[:,:,0] * (0.5 + tint[0]*0.5), 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1] * (0.5 + tint[1]*0.5), 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2] * (0.5 + tint[2]*0.5), 0, 255)
        img.from_array(arr.astype(np.uint8))
        return img

    def _render_ripple(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        img  = TTIImage(w, h, 24, config=self._cfg)
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        cx, cy = w//2, h//2
        dist   = np.sqrt((xs-cx)**2 + (ys-cy)**2).astype(np.float32)
        freq   = p.pattern_scale * 30
        wave   = np.sin(dist * freq) * 0.5 + 0.5
        c1 = np.array(p.primary_color[:3],   dtype=np.float32)
        c2 = np.array(p.secondary_color[:3], dtype=np.float32)
        arr = (c1[None,None,:] * wave[:,:,None] +
               c2[None,None,:] * (1-wave[:,:,None])).astype(np.uint8)
        img.from_array(arr)
        return img

    def _render_starfield(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        img = TTIImage(w, h, 24, background=p.background_color, config=self._cfg)
        canvas = ImageCanvas(img)
        n_stars = max(50, p.num_shapes * 10)
        for _ in range(n_stars):
            sx = int(rng.integers(0, w))
            sy = int(rng.integers(0, h))
            br = int(rng.integers(150, 256))
            r  = int(rng.integers(1, 3))
            canvas.circle(sx, sy, r, (br, br, min(br+30,255)), filled=True)
        # Nebula blobs
        for _ in range(p.num_shapes // 2):
            nx = int(rng.integers(0, w))
            ny = int(rng.integers(0, h))
            nr = int(rng.integers(30, max(31, min(w,h)//3)))
            col = tuple(int(c) for c in rng.choice(p.palette)[:3])
            canvas.ellipse(nx, ny, nr, int(nr*0.6), col, filled=True)
        img = VisualEffects.apply_gaussian_blur(img, sigma=4.0)
        # Re-add sharp stars on top
        for _ in range(n_stars // 3):
            sx = int(rng.integers(0, w))
            sy = int(rng.integers(0, h))
            br = int(rng.integers(200, 256))
            canvas.circle(sx, sy, 1, (br, br, 255), filled=True)
        return img

    def _render_geometric(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        img    = TTIImage(w, h, 24, background=p.background_color, config=self._cfg)
        canvas = ImageCanvas(img)
        shapes = ["rect", "circle", "polygon", "ellipse"]
        for _ in range(p.num_shapes):
            shape = rng.choice(shapes)
            col   = tuple(int(c) for c in rng.choice(p.palette)[:3])
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            sz     = int(rng.integers(20, max(21, min(w,h)//3)))
            filled = bool(rng.integers(0, 2))
            if shape == "rect":
                canvas.rect(cx-sz, cy-sz, cx+sz, cy+sz, col, filled)
            elif shape == "circle":
                canvas.circle(cx, cy, sz, col, filled)
            elif shape == "ellipse":
                canvas.ellipse(cx, cy, sz, sz//2, col, filled)
            else:
                n_pts = int(rng.integers(3, 8))
                angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
                pts = [(cx + int(sz*np.cos(a)), cy + int(sz*np.sin(a))) for a in angles]
                canvas.polygon(pts, col, filled)
        return img

    def _render_organic(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        base = VisualEffects.create_linear_gradient(
            w, h, p.background_color, p.secondary_color, "diagonal", self._cfg
        )
        canvas = ImageCanvas(base)
        # Draw branch-like structures
        def branch(x, y, angle, length, depth, col):
            if depth == 0 or length < 2:
                return
            ex = x + int(length * math.cos(angle))
            ey = y + int(length * math.sin(angle))
            canvas.line(x, y, ex, ey, col, thickness=max(1, depth//2))
            spread = 0.4 + rng.random() * 0.3
            branch(ex, ey, angle - spread, length*0.7, depth-1, col)
            branch(ex, ey, angle + spread, length*0.7, depth-1, col)

        for _ in range(max(2, p.num_shapes//5)):
            sx = int(rng.integers(w//4, 3*w//4))
            sy = int(rng.integers(h//2, h))
            col = tuple(int(c) for c in rng.choice(p.palette)[:3])
            branch(sx, sy, -math.pi/2, min(w,h)//5, 7, col)
        return base

    def _render_circles(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        img    = TTIImage(w, h, 24, background=p.background_color, config=self._cfg)
        canvas = ImageCanvas(img)
        cx, cy = w//2, h//2
        max_r  = min(w, h)//2 - 5
        n      = max(5, p.num_shapes)
        for i in range(n):
            t   = i / n
            r   = max(2, int(max_r * (1 - t)))
            col = ColorUtils.lerp(p.primary_color, p.secondary_color, t)
            canvas.circle(
                cx + int(rng.integers(-10,11)),
                cy + int(rng.integers(-10,11)),
                r, col, filled=(i % 3 == 0)
            )
        return img

    def _render_spiral(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        img    = TTIImage(w, h, 24, background=p.background_color, config=self._cfg)
        canvas = ImageCanvas(img)
        cx, cy = w//2, h//2
        turns  = 5 + int(p.detail_scale * 5)
        steps  = int(turns * 360)
        for i in range(steps):
            t      = i / steps
            angle  = t * turns * 2 * math.pi
            radius = t * min(w, h) // 2
            x      = cx + int(radius * math.cos(angle))
            y      = cy + int(radius * math.sin(angle))
            col    = ColorUtils.lerp(p.primary_color, p.accent_color, t)
            img.set_pixel(x, y, col)
            if radius > 2:
                for dr in range(-1, 2):
                    img.set_pixel(x+dr, y, col)
                    img.set_pixel(x, y+dr, col)
        return img

    def _render_abstract(
        self, p: SceneParameters, w: int, h: int, rng
    ) -> TTIImage:
        img = ProceduralArt.plasma(w, h, scale=p.pattern_scale, config=self._cfg)
        canvas = ImageCanvas(img)
        for _ in range(p.num_shapes):
            col  = tuple(int(c) for c in rng.choice(p.palette)[:3])
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            r    = int(rng.integers(10, max(11, min(w,h)//4)))
            canvas.circle(cx, cy, r, col, filled=False)
        return img


# ---------------------------------------------------------------------------
# TTIGenerator  — public façade
# ---------------------------------------------------------------------------

class TTIGenerator:
    """
    Main public interface for Text-To-Image generation.

    Usage::

        gen = TTIGenerator()
        img = gen.generate("a calm blue ocean at sunset", width=512, height=512)
        img.save("output.png")
    """

    def __init__(self, config: Optional[TTIConfig] = None) -> None:
        self._cfg      = config or get_config()
        self._nlp      = NLPAnalyser(self._cfg)
        self._colour   = ColourPredictor(self._cfg)
        self._composer = SceneComposer(self._cfg)
        self._decoder  = ImageDecoder(self._cfg)

    def analyse(self, prompt: str) -> PromptAnalysis:
        """Run NLP analysis only — returns PromptAnalysis."""
        return self._nlp.analyse(prompt)

    def generate(
        self,
        prompt:  str,
        width:   Optional[int]  = None,
        height:  Optional[int]  = None,
        seed:    Optional[int]  = None,
        steps:   Optional[int]  = None,
    ) -> TTIImage:
        """
        Generate an image from *prompt*.

        Parameters
        ----------
        prompt  : natural language description
        width   : output width  (default from config)
        height  : output height (default from config)
        seed    : RNG seed for reproducibility (None = random)
        steps   : number of refinement iterations (default from config)
        """
        cfg  = self._cfg
        w    = width  or cfg.image.default_width
        h    = height or cfg.image.default_height
        sd   = seed   if seed is not None else cfg.ai.seed
        itr  = steps  or cfg.ai.num_inference_steps

        if cfg.log.show_progress:
            print(f"[TTI] Analysing prompt: '{prompt[:60]}…'" if len(prompt)>60 else f"[TTI] Analysing: '{prompt}'")

        # 1. NLP
        analysis = self._nlp.analyse(prompt)

        # 2. Colour prediction (augment analysis)
        predicted_palette = self._colour.predict_palette(prompt, n_colours=6)
        if not analysis.colour_matches:
            analysis.colour_matches = [
                (f"predicted_{i}", c) for i, c in enumerate(predicted_palette)
            ]

        if cfg.log.show_progress:
            print(f"[TTI] Scene type: {analysis.scene_type} | "
                  f"Colours: {[k for k,_ in analysis.colour_matches[:3]]}")

        # 3. Latent encode → scene params
        scene_params = self._composer.compose(analysis, seed=sd)

        if cfg.log.show_progress:
            print(f"[TTI] Rendering {w}×{h}  (seed={sd}) …")

        # 4. Render
        img = self._decoder.render(scene_params, w, h, seed=sd)

        # 5. Iterative refinement (diffusion-like)
        for step in range(min(itr, 3)):
            noise_amt = 0.05 * (1 - step / 3)
            if noise_amt > 0.01:
                img = VisualEffects.add_noise(img, noise_amt)
            if step == 1 and scene_params.apply_sharpen:
                img = VisualEffects.apply_sharpen(img)

        if cfg.log.show_progress:
            print(f"[TTI] Done → {img}")

        return img

    def generate_batch(
        self,
        prompts: List[str],
        width:   Optional[int] = None,
        height:  Optional[int] = None,
        seed:    Optional[int] = None,
    ) -> List[TTIImage]:
        """Generate multiple images from a list of prompts."""
        return [
            self.generate(p, width, height, seed=seed+i if seed else None)
            for i, p in enumerate(prompts)
        ]

    def generate_variations(
        self,
        prompt:      str,
        n_variations: int = 4,
        width:        Optional[int] = None,
        height:       Optional[int] = None,
    ) -> List[TTIImage]:
        """Generate N variations of the same prompt with different seeds."""
        base_seed = self._cfg.ai.seed or int(time.time()) & 0xFFFF
        return [
            self.generate(prompt, width, height, seed=base_seed + i)
            for i in range(n_variations)
        ]

    def interpolate(
        self,
        prompt_a:   str,
        prompt_b:   str,
        steps:      int = 5,
        width:      Optional[int] = None,
        height:     Optional[int] = None,
        seed:       Optional[int] = None,
    ) -> List[TTIImage]:
        """
        Generate a sequence of images interpolating between two prompts.
        Returns *steps* frames from prompt_a to prompt_b.
        """
        w  = width  or self._cfg.image.default_width
        h  = height or self._cfg.image.default_height
        sd = seed

        an_a = self._nlp.analyse(prompt_a)
        an_b = self._nlp.analyse(prompt_b)
        sp_a = self._composer.compose(an_a, sd)
        sp_b = self._composer.compose(an_b, sd)

        imgs = []
        for i in range(steps):
            t = i / max(1, steps-1)
            # Interpolate palettes
            blended_palette = [
                ColorUtils.lerp(ca, cb, t)
                for ca, cb in zip(sp_a.palette, sp_b.palette)
            ]
            # Build interpolated params (quick dataclass copy)
            import copy
            sp_i                 = copy.copy(sp_a)
            sp_i.primary_color   = ColorUtils.lerp(sp_a.primary_color,   sp_b.primary_color,   t)
            sp_i.secondary_color = ColorUtils.lerp(sp_a.secondary_color, sp_b.secondary_color, t)
            sp_i.palette         = blended_palette
            sp_i.brightness      = sp_a.brightness*(1-t) + sp_b.brightness*t
            sp_i.noise_level     = sp_a.noise_level*(1-t) + sp_b.noise_level*t
            img = self._decoder.render(sp_i, w, h, seed=sd)
            imgs.append(img)
        return imgs

    def save_model(self, path: str) -> None:
        """Persist composer weights (numpy arrays) to disk."""
        weights = {
            name: p.data for name, p in self._composer.named_parameters()
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(weights, f)
        print(f"[TTI] Model saved → {path}")

    def load_model(self, path: str) -> None:
        """Load composer weights from disk."""
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        for name, p in self._composer.named_parameters():
            if name in weights:
                p.data = weights[name]
        print(f"[TTI] Model loaded ← {path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the AI model."""
        total = sum(p.data.size for _, p in self._composer.named_parameters())
        return {
            "model_type":   self._cfg.ai.model_type,
            "latent_dim":   self._cfg.ai.latent_dim,
            "hidden_dim":   self._cfg.ai.hidden_dim,
            "text_embed_dim": self._cfg.ai.text_embed_dim,
            "total_params": total,
            "nlp_backend":  self._cfg.ai.nlp_backend,
        }
