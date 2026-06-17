"""
TTI_config.py
=============
Centralised configuration and settings for the Text-To-Image (TTI) system.

All tuneable parameters live here.  Import TTIConfig from this module to
access or override settings at runtime.

Usage
-----
    from TTI_config import TTIConfig, get_config, update_config

    cfg = get_config()                  # global singleton
    cfg.image.default_width = 512
    update_config(image={"default_width": 512})   # bulk update
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from ...config import get_config


# ---------------------------------------------------------------------------
# Sub-configs (each is its own dataclass so IDE auto-complete works)
# ---------------------------------------------------------------------------

@dataclass
class ImageConfig:
    """Output image settings."""
    config = get_config(path= "./config.pbcfg")
    image = config.tti_image
    default_width:       int   = image.default_width
    default_height:      int   = image.default_height
    default_bpp:         int   = image.default_bpp
    default_format:      str   = image.default_format       # "bmp" | "png" | "jpeg"
    background_color:    Tuple[int,int,int] = image.background_color
    jpeg_quality:        int   = image.jpeg_quality          # 1-95


@dataclass
class AIConfig:
    """AI / NLP model settings."""
    # --- NLP ---
    config = get_config(path= "./config.pbcfg")
    ai = config.tti_ai
    nlp_backend:         str   = ai.nlp_backend      # "nltk" | "spacy" (if installed)
    max_prompt_tokens:   int   = ai.max_prompt_tokens
    use_stopword_filter: bool  = ai.use_stopword_filter

    # --- Generative model ---
    model_type:          str   = ai.model_type   # "vae_numpy" | "torch_vae" (if torch)
    latent_dim:          int   = ai.latent_dim
    text_embed_dim:      int   = ai.text_embed_dim
    vocab_size:          int   = ai.vocab_size
    hidden_dim:          int   = ai.hidden_dim

    # --- Colour palette model (sklearn k-means) ---
    palette_clusters:    int   = ai.palette_clusters
    palette_model_path:  str   = ai.palette_model_path

    # --- Rendering ---
    num_inference_steps: int   = ai.num_inference_steps        # diffusion-like iteration count
    guidance_scale:      float = ai.guidance_scale
    seed:                Optional[int] = ai.seed   # None → random


@dataclass
class ArtConfig:
    """Procedural art & effects settings."""
    config = get_config(path= "./config.pbcfg")
    art = config.tti_art
    fractal_max_iter:    int   = art.fractal_max_iter
    blur_default_radius: int   = art.blur_default_radius
    noise_default_intensity: float = art.noise_default_intensity
    animation_fps:       int   = art.animation_fps
    streaming_chunk_mb:  int   = art.streaming_chunk_mb        # max RAM per streaming chunk


@dataclass
class PathConfig:
    """File-system paths."""
    config = get_config(path= "./config.pbcfg")
    paths = config.tti_paths
    output_dir:          str   = paths.output_dir
    model_dir:           str   = paths.model_dir
    cache_dir:           str   = paths.cache_dir
    log_dir:             str   = paths.log_dir


@dataclass
class LogConfig:
    """Logging settings."""
    level:               str   = "INFO"    # DEBUG | INFO | WARNING | ERROR
    log_to_file:         bool  = False
    log_filename:        str   = "tti.log"
    show_progress:       bool  = True


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class TTIConfig:
    """
    Master configuration object for the entire TTI system.

    Instantiate once and pass around, or use the module-level singleton
    via ``get_config()``.
    """
    image:  ImageConfig  = field(default_factory=ImageConfig)
    ai:     AIConfig     = field(default_factory=AIConfig)
    art:    ArtConfig    = field(default_factory=ArtConfig)
    paths:  PathConfig   = field(default_factory=PathConfig)
    log:    LogConfig    = field(default_factory=LogConfig)

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str | Path = "tti_config.json") -> None:
        """Persist settings to a JSON file."""
        Path(filepath).write_text(self.to_json())

    @classmethod
    def load(cls, filepath: str | Path = "tti_config.json") -> "TTIConfig":
        """Load settings from a JSON file (missing keys fall back to defaults)."""
        raw = json.loads(Path(filepath).read_text())
        cfg = cls()
        for section, values in raw.items():
            sub = getattr(cfg, section, None)
            if sub is not None:
                for k, v in values.items():
                    if hasattr(sub, k):
                        # Re-cast tuples that JSON round-trips as lists
                        orig = getattr(sub, k)
                        if isinstance(orig, tuple) and isinstance(v, list):
                            v = tuple(v)
                        setattr(sub, k, v)
        return cfg

    def ensure_dirs(self) -> None:
        """Create all configured output/model directories if missing."""
        for attr in ("output_dir", "model_dir", "cache_dir", "log_dir"):
            Path(getattr(self.paths, attr)).mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"TTIConfig(image={self.image}, ai={self.ai}, art={self.art})"


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_GLOBAL_CONFIG: Optional[TTIConfig] = None


def get_config() -> TTIConfig:
    """Return the global TTIConfig singleton (created on first call)."""
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        # Try to load from file next to this module, else use defaults
        cfg_file = Path(__file__).parent / "tti_config.json"
        if cfg_file.exists():
            _GLOBAL_CONFIG = TTIConfig.load(cfg_file)
        else:
            _GLOBAL_CONFIG = TTIConfig()
    return _GLOBAL_CONFIG


def update_config(**sections: Dict[str, Any]) -> TTIConfig:
    """
    Bulk-update the global config.

    Example
    -------
    update_config(image={"default_width": 1024}, ai={"seed": 42})
    """
    cfg = get_config()
    for section, values in sections.items():
        sub = getattr(cfg, section, None)
        if sub is None:
            raise ValueError(f"Unknown config section: '{section}'")
        for k, v in values.items():
            if not hasattr(sub, k):
                raise ValueError(f"Unknown config key: '{section}.{k}'")
            orig = getattr(sub, k)
            if isinstance(orig, tuple) and isinstance(v, list):
                v = tuple(v)
            setattr(sub, k, v)
    return cfg


def reset_config() -> TTIConfig:
    """Reset global config to factory defaults."""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = TTIConfig()
    return _GLOBAL_CONFIG
