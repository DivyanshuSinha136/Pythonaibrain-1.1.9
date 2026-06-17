"""
config.py  –  Master configuration.

All sections are available as typed dataclasses on the unified ``AppConfig`` object.
The ``PBConfig`` class handles reading/writing everything to a single .pbcfg file.

─────────────────────────────────────────────────────────────────────────────
File format  (*.pbcfg  — INI-style)
─────────────────────────────────────────────────────────────────────────────
    ; ── PythonAIBrain core ──────────────────────────────────────
    [brain]
    intents_path        = ./intents.json
    condition           = true
    smart_memory        = true
    memory_path         = memory.json
    memory_fit_interval = 20
    username            = user_name
    download            = false

    [model]
    model_path      = model.pth
    dimension_path  = dimensions.json
    batch_size      = 8
    learning_rate   = 0.001
    epochs          = 100

    [llm]
    n_ctx       = 2048
    n_threads   = 0
    max_tokens  = 512
    verbose     = false

    [tts]
    enabled = false

    [logging]
    level  = INFO
    format = %(asctime)s [%(levelname)s] %(name)s - %(message)s

    [weather]
    base_url = https://api.openweathermap.org/data/2.5/weather
    units    = metric

    [memory]
    auto_load = true
    auto_fit  = true

    [search]
    max_results = 5

    [webassistant]
    intents_path   = ./Webintents.json
    model_path     = WebAssistantModel.pth
    dimension_path = WebAssistantDimensions.json
    batch_size     = 8
    learning_rate  = 0.001
    epochs         = 100

    ; ── Memory Summarization System ─────────────────────────────
    [embedding]
    tfidf_max_features = 5000
    tfidf_ngram_range  = 1,3
    tfidf_sublinear_tf = true
    embed_dim          = 128
    vocab_size         = 10000

    [clustering]
    n_clusters              = 8
    kmeans_max_iter         = 300
    kmeans_random_state     = 42
    dbscan_eps              = 0.5
    dbscan_min_samples      = 2
    agglo_linkage           = ward
    agglo_distance_threshold =

    [classifier]
    lr_max_iter          = 500
    lr_c                 = 1.0
    lr_solver            = lbfgs
    lr_multi_class       = auto
    similarity_threshold = 0.65

    [summarizer]
    latent_dim              = 64
    hidden_dim              = 256
    ae_epochs               = 30
    ae_lr                   = 0.001
    ae_batch_size           = 16
    top_patterns_per_cluster = 3
    min_cluster_size        = 2

    ; ── Text-To-Image system ─────────────────────────────────────
    [tti_image]
    default_width    = 512
    default_height   = 512
    default_bpp      = 24
    default_format   = png
    background_color = 255,255,255
    jpeg_quality     = 92

    [tti_ai]
    nlp_backend          = nltk
    max_prompt_tokens    = 128
    use_stopword_filter  = true
    model_type           = vae_numpy
    latent_dim           = 128
    text_embed_dim       = 256
    vocab_size           = 4096
    hidden_dim           = 512
    palette_clusters     = 8
    palette_model_path   = tti_palette_model.pkl
    num_inference_steps  = 50
    guidance_scale       = 7.5
    seed                 =

    [tti_art]
    fractal_max_iter         = 256
    blur_default_radius      = 2
    noise_default_intensity  = 0.15
    animation_fps            = 24
    streaming_chunk_mb       = 32

    [tti_paths]
    output_dir = tti_output
    model_dir  = tti_models
    cache_dir  = tti_cache
    log_dir    = tti_logs

    ; ── Speech-To-Text system ─────────────────────────────────────
    [stt]
    energy_threshold           = 
    dynamic_energy_threshold   = true
    pause_threshold            = 0.8
    phrase_time_limit          =        ; hard cap per utterance (seconds)
    timeout                    = 5.0    ; seconds to wait for speech to start

    ; Ambient-noise calibration
    ambient_noise_duration     = 0.5    ; seconds spent sampling noise floor

    ; Connectivity check
    connectivity_host          = 8.8.8.8
    connectivity_port          = 53
    connectivity_timeout       = 2.0

    ; Engine preference override (None → auto-detect)
    preferred_engine           = None

    ; PocketSphinx keyword / language model (None → CMU US English default)
    sphinx_language            = en-US

    ; Google Speech API
    google_language            = en-US
    google_api_key             =        ; None → free tier

    ; Retry
    max_retries                = 3
    retry_delay                = 0.5    ; seconds between retries

    ; ── Text-To-Speech system ─────────────────────────────────────
    [tts]
    rate         = 150
    volume       = 1.0
    voice        = david
    default_text = Hello from PyAI
    output_path  =                   # if set, speech is saved here instead of played

    ; ── NER system ─────────────────────────────────────────────────
    [postprocessor]
    min_length       = 1
    max_length       =
    allowed_labels   =         ; None = allow all
    blocked_labels   =
    deduplicate      = true
    merge_adjacent   = false   ; merge back-to-back same-label spans
    lowercase_labels = false
    strip_punct      = true
    custom_label_map =         ; rename labels

    [preprocessor]
    lowercase            = false
    remove_urls          = true
    remove_emails        = false    ; keep: useful NER signal
    remove_html_tags     = true
    normalize_whitespace = true
    normalize_unicode    = true
    max_length           =          ; characters; None = no limit
    custom_patterns      =

Comments start with ; or #.  Inline comments are supported.
─────────────────────────────────────────────────────────────────────────────

Usage
-----
    from pyaitk.config import AppConfig, get_config, generate_default_config

    cfg = get_config()                    # module-level singleton (auto-discovers .pbcfg)
    cfg = AppConfig("my_app.pbcfg")       # explicit path
    cfg.load()                            # (re-)read from disk
    cfg.save()                            # write back

    # Typed attribute access
    brain_cfg    = cfg.brain              # BrainConfig
    embed_cfg    = cfg.embedding          # EmbeddingConfig
    tti_img_cfg  = cfg.tti_image          # TTIImageConfig

    # Generic typed getter with fallback
    cfg.get("brain", "smart_memory", fallback=True, cast=bool)

    # Mutation + persist
    cfg.set("tti_ai", "seed", 42)
    cfg.save()

    # Factories
    cfg = AppConfig.discover()
    cfg = AppConfig.from_dict({...})
    generate_default_config()             # writes ./config.pbcfg
"""

from __future__ import annotations

import configparser
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Set
from enum import Enum, auto

logger = logging.getLogger(__name__)

T = TypeVar("T")

_BOOL_TRUE  = {"1", "yes", "true",  "on"}
_BOOL_FALSE = {"0", "no",  "false", "off"}

PBCFG_EXTENSION     = ".pbcfg"
DEFAULT_CONFIG_NAME = "config.pbcfg"


# =============================================================================
# ── Section dataclasses: PythonAIBrain core ──────────────────────────────────
# =============================================================================

@dataclass
class BrainConfig:
    intents_path:        str  = r".\intents.json"
    condition:           bool = True
    smart_memory:        bool = True
    memory_path:         str  = "memory.json"
    memory_fit_interval: int  = 20
    username:            str  = "user_name"
    download:            bool = False


@dataclass
class ModelConfig:
    model_path:     str   = "model.pth"
    dimension_path: str   = "dimensions.json"
    batch_size:     int   = 8
    learning_rate:  float = 0.001
    epochs:         int   = 100


@dataclass
class LLMConfig:
    n_ctx:      int  = 2048
    n_threads:  int  = 0        # 0 → os.cpu_count()
    max_tokens: int  = 512
    verbose:    bool = False


@dataclass
class LoggingConfig:
    level:  str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


@dataclass
class WeatherConfig:
    base_url: str = "https://api.openweathermap.org/data/2.5/weather"
    units:    str = "metric"


@dataclass
class MemoryConfig:
    auto_load: bool = True
    auto_fit:  bool = True


@dataclass
class SearchConfig:
    max_results: int = 5


@dataclass
class WebAssistantConfig:
    intents_path:   str   = r".\Webintents.json"
    model_path:     str   = "WebAssistantModel.pth"
    dimension_path: str   = "WebAssistantDimensions.json"
    batch_size:     int   = 8
    learning_rate:  float = 0.001
    epochs:         int   = 100


# =============================================================================
# ── Section dataclasses: Memory Summarization System ─────────────────────────
# =============================================================================

@dataclass
class EmbeddingConfig:
    # TF-IDF parameters
    tfidf_max_features: int   = 5000
    tfidf_ngram_range:  tuple = (1, 3)   # stored as "1,3" in .pbcfg
    tfidf_sublinear_tf: bool  = True

    # PyTorch embedding dimensions (for learned embeddings)
    embed_dim:  int = 128
    vocab_size: int = 10000


@dataclass
class ClusteringConfig:
    # KMeans
    n_clusters:          int   = 8
    kmeans_max_iter:     int   = 300
    kmeans_random_state: int   = 42

    # DBSCAN
    dbscan_eps:         float = 0.5
    dbscan_min_samples: int   = 2

    # Agglomerative
    agglo_linkage:            str            = "ward"
    agglo_distance_threshold: Optional[float] = None


@dataclass
class ClassifierConfig:
    # Logistic Regression intent classifier
    lr_max_iter:   int   = 500
    lr_c:          float = 1.0    # note: lower-cased for INI compatibility
    lr_solver:     str   = "lbfgs"
    lr_multi_class: str  = "auto"

    # Pattern match threshold
    similarity_threshold: float = 0.65


@dataclass
class SummarizerConfig:
    # PyTorch autoencoder
    latent_dim:    int   = 64
    hidden_dim:    int   = 256
    ae_epochs:     int   = 30
    ae_lr:         float = 1e-3
    ae_batch_size: int   = 16

    # Summarization parameters
    top_patterns_per_cluster: int = 3
    min_cluster_size:         int = 2


# =============================================================================
# ── Section dataclasses: Text-To-Image system ────────────────────────────────
# =============================================================================

@dataclass
class TTIImageConfig:
    """Output image settings."""
    default_width:    int              = 512
    default_height:   int              = 512
    default_bpp:      int              = 24
    default_format:   str              = "png"          # "bmp" | "png" | "jpeg"
    background_color: tuple            = (255, 255, 255) # stored as "255,255,255"
    jpeg_quality:     int              = 92


@dataclass
class TTIAIConfig:
    """AI / NLP model settings for TTI."""
    nlp_backend:          str           = "nltk"        # "nltk" | "spacy"
    max_prompt_tokens:    int           = 128
    use_stopword_filter:  bool          = True
    model_type:           str           = "vae_numpy"   # "vae_numpy" | "torch_vae"
    latent_dim:           int           = 128
    text_embed_dim:       int           = 256
    vocab_size:           int           = 4096
    hidden_dim:           int           = 512
    palette_clusters:     int           = 8
    palette_model_path:   str           = "tti_palette_model.pkl"
    num_inference_steps:  int           = 50
    guidance_scale:       float         = 7.5
    seed:                 Optional[int] = None           # empty string in INI → None


@dataclass
class TTIArtConfig:
    """Procedural art & effects settings."""
    fractal_max_iter:         int   = 256
    blur_default_radius:      int   = 2
    noise_default_intensity:  float = 0.15
    animation_fps:            int   = 24
    streaming_chunk_mb:       int   = 32


@dataclass
class TTIPathConfig:
    """File-system paths for TTI outputs."""
    output_dir: str = "tti_output"
    model_dir:  str = "tti_models"
    cache_dir:  str = "tti_cache"
    log_dir:    str = "tti_logs"

    def ensure_dirs(self) -> None:
        """Create all output/model directories if missing."""
        for attr in ("output_dir", "model_dir", "cache_dir", "log_dir"):
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)

# =============================================================================
# ── Section dataclasses: Speech-To-Text system ───────────────────────────────
# =============================================================================

# ---------------------------------------------------------------------------
# Engine enum
# ---------------------------------------------------------------------------
class Engine(Enum):
    GOOGLE = auto()       # Online — Google Speech Recognition
    POCKETSPHINX = auto() # Offline — CMU PocketSphinx

@dataclass
class STTConfig:
    """Tunable knobs for the STT pipeline."""

    # Microphone / audio capture
    energy_threshold:         Optional[float] = None   # None → auto-calibrate
    dynamic_energy_threshold: bool = True
    pause_threshold:          float = 0.8               # seconds of silence = end of phrase
    phrase_time_limit:        Optional[float] = None  # hard cap per utterance (seconds)
    timeout:                  Optional[float] = 5.0            # seconds to wait for speech to start

    # Ambient-noise calibration
    ambient_noise_duration:   float = 0.5        # seconds spent sampling noise floor

    # Connectivity check
    connectivity_host:        str = "8.8.8.8"
    connectivity_port:        int = 53
    connectivity_timeout:     float = 2.0

    # Engine preference override (None → auto-detect)
    preferred_engine:         Optional[Engine] = None

    # PocketSphinx keyword / language model (None → CMU US English default)
    sphinx_language:          str = "en-US"

    # Google Speech API
    google_language:          str = "en-US"
    google_api_key:           Optional[str] = None       # None → free tier

    # Retry
    max_retries:              int = 3
    retry_delay:              float = 0.5                   # seconds between retries

# =============================================================================
# ── Section dataclasses: Text-To-Speech system ───────────────────────────────
# =============================================================================

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------
def _detect_platform() -> str:
    """Return one of ``'windows'``, ``'macos'``, ``'linux'``, or ``'unknown'``."""
    p = sys.platform
    if p.startswith("win"):
        return "windows"
    if p == "darwin":
        return "macos"
    if p.startswith("linux"):
        return "linux"
    return "unknown"

@dataclass
class TTSConfig:
    """Tunable parameters for the TTS pipeline."""

    rate:         int = 150                          # words per minute
    volume:       float = 1.0                      # 0.0 – 1.0
    voice:        str = "david"                     # fragment matched against installed voices
    default_text: str = "Hello from PyAI"

    # save_to_file
    output_path:  Optional[str] = None        # if set, speech is saved here instead of played

    # Internal — resolved at runtime
    _platform:    str = field(default_factory=_detect_platform, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError(f"volume must be between 0.0 and 1.0, got {self.volume}")
        if self.rate <= 0:
            raise ValueError(f"rate must be positive, got {self.rate}")

# =============================================================================
# ── Section dataclasses: NER system ──────────────────────────────────────────
# =============================================================================

@dataclass
class PostprocessorConfig:
    """Fine-grained control over post-processing behaviour."""

    min_length: int = 1
    max_length: Optional[int] = None
    allowed_labels: Optional[Set[str]] = None      # None = allow all
    blocked_labels: Set[str] = field(default_factory=set)
    deduplicate: bool = True
    merge_adjacent: bool = False                   # merge back-to-back same-label spans
    lowercase_labels: bool = False
    strip_punct: bool = True
    custom_label_map: Dict[str, str] = field(default_factory=dict)  # rename labels

@dataclass
class PreprocessorConfig:
    """Configuration for the text preprocessor."""

    lowercase: bool = False
    remove_urls: bool = True
    remove_emails: bool = False          # keep: useful NER signal
    remove_html_tags: bool = True
    normalize_whitespace: bool = True
    normalize_unicode: bool = True
    max_length: Optional[int] = None     # characters; None = no limit
    custom_patterns: List[str] = field(default_factory=list)

# =============================================================================
# ── Internal helpers ─────────────────────────────────────────────────────────
# =============================================================================

def _cast(value: str, target: Type[T]) -> T:
    """Cast a raw INI string to *target* type (str, int, float, bool)."""
    if target is bool:
        v = value.strip().lower()
        if v in _BOOL_TRUE:
            return True   # type: ignore[return-value]
        if v in _BOOL_FALSE:
            return False  # type: ignore[return-value]
        raise ValueError(
            f"Cannot cast {value!r} to bool — use one of: {_BOOL_TRUE | _BOOL_FALSE}"
        )
    return target(value)  # type: ignore[call-arg]


def _strip_inline_comment(value: str) -> str:
    """Remove trailing ; or # inline comments from a raw INI value."""
    return re.split(r"\s+[;#]", value, maxsplit=1)[0].strip()


def _tuple_to_str(t: tuple) -> str:
    return ",".join(str(x) for x in t)


def _str_to_int_tuple(s: str) -> tuple:
    return tuple(int(x.strip()) for x in s.split(","))


def _str_to_optional_float(s: str) -> Optional[float]:
    s = s.strip()
    return None if s == "" else float(s)


def _str_to_optional_int(s: str) -> Optional[int]:
    s = s.strip()
    return None if s == "" else int(s)


# =============================================================================
# ── AppConfig (unified PBConfig) ─────────────────────────────────────────────
# =============================================================================

class AppConfig:
    """
    Unified configuration manager for the entire Pythonaibrain application.

    Reads / writes a single *.pbcfg file that covers all three sub-systems:
      • PythonAIBrain core
      • AI Memory Summarization System
      • Text-To-Image (TTI) system

    Parameters
    ----------
    path : str | Path | None
        Path to the .pbcfg file. Defaults to ``config.pbcfg`` in cwd.
    auto_load : bool
        Load from disk automatically on construction if the file exists.
    """

    # Maps INI section name → (attribute name on self, dataclass type)
    _SECTION_MAP: Dict[str, Tuple[str, type]] = {
        # ── core ──────────────────────────────────────────────────────────────
        "brain":        ("brain",        BrainConfig),
        "model":        ("model",        ModelConfig),
        "llm":          ("llm",          LLMConfig),
        "logging":      ("logging",      LoggingConfig),
        "weather":      ("weather",      WeatherConfig),
        "memory":       ("memory",       MemoryConfig),
        "search":       ("search",       SearchConfig),
        "webassistant": ("webassistant", WebAssistantConfig),
        # ── memory summarizer ─────────────────────────────────────────────────
        "embedding":    ("embedding",    EmbeddingConfig),
        "clustering":   ("clustering",   ClusteringConfig),
        "classifier":   ("classifier",   ClassifierConfig),
        "summarizer":   ("summarizer",   SummarizerConfig),
        # ── TTI ───────────────────────────────────────────────────────────────
        "tti_image":    ("tti_image",    TTIImageConfig),
        "tti_ai":       ("tti_ai",       TTIAIConfig),
        "tti_art":      ("tti_art",      TTIArtConfig),
        "tti_paths":    ("tti_paths",    TTIPathConfig),
        # ── STT ───────────────────────────────────────────────────────────────
        "stt":       ("stt",       STTConfig),
        # ── TTS ───────────────────────────────────────────────────────────────
        "tts":       ("tts",       TTSConfig),
        # ── NER ───────────────────────────────────────────────────────────────
        "postprocessor":       ("postprocessor",       PostprocessorConfig),
        "preprocessor":       ("preprocessor",       PreprocessorConfig),
    }

    # Fields that need special (de)serialization beyond plain _cast
    _TUPLE_INT_FIELDS = {
        ("embedding",  "tfidf_ngram_range"),
        ("tti_image",  "background_color"),
    }
    _OPTIONAL_FLOAT_FIELDS = {
        ("clustering", "agglo_distance_threshold"),
        ("stt", "energy_threshold"),
        ("stt", "phrase_time_limit"),
        ("stt", "timeout"),
    }
    _OPTIONAL_INT_FIELDS = {
        ("tti_ai", "seed"),
        ("postprocessor", "max_length"),
        ("preprocessor",  "max_length"),
    }

    def __init__(
        self,
        path: Optional[str | Path] = None,
        auto_load: bool = True,
    ) -> None:
        self.path: Path = Path(path) if path else Path.cwd() / DEFAULT_CONFIG_NAME
        self._raw: configparser.RawConfigParser = configparser.RawConfigParser(
            comment_prefixes=("#", ";"),
            inline_comment_prefixes=(";", "#"),
            allow_no_value=True,
            strict=True,
        )

        # ── core ──────────────────────────────────────────────────────────────
        self.brain:        BrainConfig        = BrainConfig()
        self.model:        ModelConfig        = ModelConfig()
        self.llm:          LLMConfig          = LLMConfig()
        self.tts:          TTSConfig          = TTSConfig()
        self.logging:      LoggingConfig      = LoggingConfig()
        self.weather:      WeatherConfig      = WeatherConfig()
        self.memory:       MemoryConfig       = MemoryConfig()
        self.search:       SearchConfig       = SearchConfig()
        self.webassistant: WebAssistantConfig = WebAssistantConfig()

        # ── memory summarizer ─────────────────────────────────────────────────
        self.embedding:  EmbeddingConfig  = EmbeddingConfig()
        self.clustering: ClusteringConfig = ClusteringConfig()
        self.classifier: ClassifierConfig = ClassifierConfig()
        self.summarizer: SummarizerConfig = SummarizerConfig()

        # ── TTI ───────────────────────────────────────────────────────────────
        self.tti_image:  TTIImageConfig  = TTIImageConfig()
        self.tti_ai:     TTIAIConfig     = TTIAIConfig()
        self.tti_art:    TTIArtConfig    = TTIArtConfig()
        self.tti_paths:  TTIPathConfig   = TTIPathConfig()

        # ── STT ───────────────────────────────────────────────────────────────
        self.stt:  STTConfig   = STTConfig()

        # ── NER ───────────────────────────────────────────────────────────────
        self.postprocessor:  PostprocessorConfig   = PostprocessorConfig()
        self.preprocessor:  PreprocessorConfig   = PreprocessorConfig()

        if auto_load and self.path.exists():
            self.load()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load(self, path: Optional[str | Path] = None) -> "AppConfig":
        """
        Parse the .pbcfg file and populate all typed section objects.
        Missing keys fall back to dataclass defaults.
        """
        target = Path(path) if path else self.path
        if not target.exists():
            logger.warning("AppConfig.load: %s not found — using defaults.", target)
            return self
        if target.suffix != PBCFG_EXTENSION:
            logger.warning(
                "AppConfig.load: expected *%s file, got %s",
                PBCFG_EXTENSION, target.name,
            )
        try:
            self._raw.read(str(target), encoding="utf-8")
        except configparser.Error as exc:
            logger.error("AppConfig.load parse error: %s", exc)
            return self
        self._sync_from_raw()
        logger.info("AppConfig loaded from %s", target)
        return self

    def save(self, path: Optional[str | Path] = None) -> "AppConfig":
        """Write the current config to a .pbcfg file."""
        target = Path(path) if path else self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        self._sync_to_raw()
        header = (
            "; Unified PythonAIBrain configuration file\n"
            "; Generated by AppConfig — edit freely.\n"
            "; Covers: core brain | memory summarizer | TTI system\n"
            "; Format: INI sections with typed key = value pairs.\n"
            "; Inline comments start with ;\n\n"
        )
        try:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(header)
                self._raw.write(fh)
            logger.info("AppConfig saved to %s", target)
        except OSError as exc:
            logger.error("AppConfig.save: %s", exc)
        return self

    def to_json(self, indent: int = 2) -> str:
        """Serialise the entire config to a JSON string."""
        return json.dumps(self.as_dict(), indent=indent)

    def save_json(self, filepath: str | Path = "config.json") -> None:
        """Persist settings to a JSON file (alternative to .pbcfg)."""
        Path(filepath).write_text(self.to_json())

    def ensure_tti_dirs(self) -> None:
        """Create all TTI output/model directories if missing."""
        self.tti_paths.ensure_dirs()

    # ── generic get / set ────────────────────────────────────────────────────

    def get(
        self,
        section: str,
        key: str,
        fallback: Any = None,
        cast: Optional[Type] = None,
    ) -> Any:
        """
        Read a raw value from the underlying ConfigParser.

        Parameters
        ----------
        section  : INI section name (case-insensitive)
        key      : option key
        fallback : returned when section/key is absent or cast fails
        cast     : optional target type (str | int | float | bool)
        """
        raw_val = self._raw.get(section.lower(), key, fallback=None)
        if raw_val is None:
            return fallback
        raw_val = _strip_inline_comment(raw_val)
        if cast is None:
            return raw_val
        try:
            return _cast(raw_val, cast)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "AppConfig.get: cannot cast [%s] %s=%r to %s — using fallback. %s",
                section, key, raw_val, cast.__name__, exc,
            )
            return fallback

    def set(self, section: str, key: str, value: Any) -> None:
        """
        Write a typed value into the in-memory config.
        Call :meth:`save` to persist to disk.
        """
        sec = section.lower()
        if not self._raw.has_section(sec):
            self._raw.add_section(sec)
        if isinstance(value, tuple):
            str_val = _tuple_to_str(value)
        elif isinstance(value, bool):
            str_val = str(value).lower()
        elif value is None:
            str_val = ""
        else:
            str_val = str(value)
        self._raw.set(sec, key, str_val)
        self._sync_from_raw()

    # ── internal sync ────────────────────────────────────────────────────────

    def _sync_from_raw(self) -> None:
        """Populate typed dataclasses from the underlying ConfigParser."""
        for sec_name, (attr, _) in self._SECTION_MAP.items():
            if not self._raw.has_section(sec_name):
                continue
            dc_obj = getattr(self, attr)
            for field_name, field_val in asdict(dc_obj).items():
                raw_val = self._raw.get(sec_name, field_name, fallback=None)
                if raw_val is None:
                    continue
                raw_val = _strip_inline_comment(raw_val)
                try:
                    key = (sec_name, field_name)
                    if key in self._TUPLE_INT_FIELDS:
                        setattr(dc_obj, field_name, _str_to_int_tuple(raw_val))
                    elif key in self._OPTIONAL_FLOAT_FIELDS:
                        setattr(dc_obj, field_name, _str_to_optional_float(raw_val))
                    elif key in self._OPTIONAL_INT_FIELDS:
                        setattr(dc_obj, field_name, _str_to_optional_int(raw_val))
                    else:
                        target_type = type(field_val) if field_val is not None else str
                        setattr(dc_obj, field_name, _cast(raw_val, target_type))
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "AppConfig: cannot parse [%s] %s=%r — keeping default. %s",
                        sec_name, field_name, raw_val, exc,
                    )

    def _sync_to_raw(self) -> None:
        """Push typed dataclass values back into the ConfigParser for writing."""
        for sec_name, (attr, _) in self._SECTION_MAP.items():
            dc_obj = getattr(self, attr)
            if not self._raw.has_section(sec_name):
                self._raw.add_section(sec_name)
            for key, val in asdict(dc_obj).items():
                k = (sec_name, key)
                if k in self._TUPLE_INT_FIELDS:
                    str_val = _tuple_to_str(val) if val is not None else ""
                elif isinstance(val, bool):
                    str_val = str(val).lower()
                elif val is None:
                    str_val = ""
                else:
                    str_val = str(val)
                self._raw.set(sec_name, key, str_val)

    # ── factories ────────────────────────────────────────────────────────────

    @classmethod
    def discover(cls, start: Optional[str | Path] = None) -> "AppConfig":
        """
        Search upward from *start* (default cwd) for a config.pbcfg file.
        Returns an AppConfig with defaults if nothing is found.
        """
        search_dir = Path(start) if start else Path.cwd()
        home = Path.home()
        current = search_dir
        while True:
            candidate = current / DEFAULT_CONFIG_NAME
            if candidate.exists():
                logger.info("AppConfig.discover: found %s", candidate)
                return cls(candidate)
            if current == home or current == current.parent:
                break
            current = current.parent
        logger.info(
            "AppConfig.discover: no %s found — using built-in defaults.",
            DEFAULT_CONFIG_NAME,
        )
        return cls(search_dir / DEFAULT_CONFIG_NAME, auto_load=False)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Dict[str, Any]],
        path: Optional[str | Path] = None,
    ) -> "AppConfig":
        """Build an AppConfig from a nested dict ``{section: {key: value}}``."""
        cfg = cls(path, auto_load=False)
        for section, kv in data.items():
            for key, val in kv.items():
                cfg.set(section, key, val)
        return cfg

    # ── repr / introspection ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"AppConfig(path={self.path!r})"

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return the entire config as a plain nested dict."""
        return {
            sec_name: asdict(getattr(self, attr))
            for sec_name, (attr, _) in self._SECTION_MAP.items()
        }

    def dump(self) -> str:
        """Return a human-readable string of all settings (for debugging)."""
        lines = ["AppConfig dump:"]
        for sec_name, values in self.as_dict().items():
            lines.append(f"\n[{sec_name}]")
            for k, v in values.items():
                lines.append(f"  {k} = {v}")
        return "\n".join(lines)


# =============================================================================
# ── Module-level singleton helpers ───────────────────────────────────────────
# =============================================================================

_global_config: Optional[AppConfig] = None


def get_config(path: Optional[str | Path] = None) -> AppConfig:
    """
    Return the module-level singleton AppConfig.

    First call auto-discovers a config file (or starts with defaults).
    If *path* is supplied a fresh instance is created and stored.
    """
    global _global_config
    if path is not None:
        _global_config = AppConfig(path)
        return _global_config
    if _global_config is None:
        _global_config = AppConfig.discover()
    return _global_config


def reset_config() -> AppConfig:
    """Reset the global config to factory defaults."""
    global _global_config
    _global_config = AppConfig(auto_load=False)
    return _global_config


def generate_default_config(path: Optional[str | Path] = None) -> Path:
    """
    Write a fully-populated default .pbcfg file.

    Parameters
    ----------
    path : target path. Defaults to ``config.pbcfg`` in the current directory.

    Returns
    -------
    Path of the written file.

    Example
    -------
        from unified_config import generate_default_config
        generate_default_config()   # → ./config.pbcfg
    """
    target = Path(path) if path else Path.cwd() / DEFAULT_CONFIG_NAME
    cfg = AppConfig(target, auto_load=False)
    cfg.save()
    return target


# =============================================================================
# ── Public API ───────────────────────────────────────────────────────────────
# =============================================================================

__all__ = [
    # Main config class
    "AppConfig",
    # Core section dataclasses
    "BrainConfig",
    "ModelConfig",
    "LLMConfig",
    "TTSConfig",
    "LoggingConfig",
    "WeatherConfig",
    "MemoryConfig",
    "SearchConfig",
    "WebAssistantConfig",
    # Memory summarizer section dataclasses
    "EmbeddingConfig",
    "ClusteringConfig",
    "ClassifierConfig",
    "SummarizerConfig",
    # TTI section dataclasses
    "TTIImageConfig",
    "TTIAIConfig",
    "TTIArtConfig",
    "TTIPathConfig",
    # STT section dataclasses
    "STTConfig",
    # TTS section dataclasses
    "PostprocessorConfig",
    "PreprocessorConfig",
    # Helpers
    "get_config",
    "reset_config",
    "generate_default_config",
    "PBCFG_EXTENSION",
    "DEFAULT_CONFIG_NAME",
]