"""
Pythonaibrain - A versatile AI toolkit for building intelligent assistants.

Pythonaibrain is a plug-and-play Python package designed to help you build
offline intelligent AI assistants and applications effortlessly. With modules
covering text-to-speech, natural language understanding, and more.

Author: Divyanshu Sinha
Version: 1.1.9
License: LGPL-3.0-or-later
"""

from __future__ import annotations

import importlib
import logging
import sys
import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

__version__: str = "1.1.9"
__author__: str = "Divyanshu Sinha"
__license__: str = "LGPL-3.0-or-later"
__email__: str = "divyanshu.sinha631@gmail.com"
__url__: str = "https://pythonaibrain.readthedocs.io"

__doc__ = """
Pythonaibrain is a versatile, plug-and-play Python package designed to help you
build offline intelligent AI assistants and applications effortlessly.

Installation:
    pip install pythonaibrain==1.1.9

Usage:
    Import individual submodules directly:

        import pyaitk.Brain
        import pyaitk.TTS
        import pyaitk.STT
        import pyaitk.Camera
        import pyaitk.NER
        import pyaitk.CLSE
        import pyaitk.eye
        import pyaitk.Memory
        import pyaitk.MathAI
        import pyaitk.Search
        import pyaitk.PPTExtract
        import pyaitk.ITT
        import pyaitk.SummarizerAI
        import pyaitk.Context
        import pyaitk.config

Main Components:
    - Brain          : Basic AI brain using a JSON knowledge base
    - AdvanceBrain   : Advanced AI brain with richer understanding
    - TTS            : Text-to-speech conversion
    - STT            : Speech-to-text conversion
    - Camera         : Photo and video capture
    - MathAI         : Complex mathematical problem solver
    - Search         : Internet search functionality
    - Memory         : Long/short-term memory management
    - Contexts       : Context-based question answering
    - ITT            : Image-to-text extraction
    - PPTXExtractor  : PowerPoint text extraction
    - CLSE           : Compositional Latent Synthesis Engine — generate
                       high-resolution images on low-end hardware
    - NER            : Named-entity recognition pipeline
    - SummarizerAI   : Memory-aware text summarisation
    - EYE            : Real-time object detection

For detailed documentation, visit: https://pythonaibrain.readthedocs.io
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.getLogger(__name__).addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime Python version guard
# ---------------------------------------------------------------------------

if sys.version_info < (3, 9):
    raise RuntimeError(
        f"Pythonaibrain requires Python 3.9 or later. "
        f"You are running Python {sys.version}."
    )

# ---------------------------------------------------------------------------
# Submodule registry
# ---------------------------------------------------------------------------
# Maps every public symbol to the dotted submodule path it lives in.
# check_module_availability() uses this to probe availability without
# actually importing anything.

_SYMBOL_MAP: dict[str, str] = {
    # Core
    "Brain":            "pyaitk.core",
    "AdvanceBrain":     "pyaitk.core",
    "get_weather":      "pyaitk.core",
    "IntentsManager":   "pyaitk.core",
    "VectorizerMode":   "pyaitk.core",
    # TTS
    "TTS":              "pyaitk.TTS",
    "TTSConfig":        "pyaitk.TTS",
    "speak":            "pyaitk.TTS",
    # STT
    "STT":              "pyaitk.STT",
    # Camera
    "Camera":           "pyaitk.Camera",
    "Start":            "pyaitk.Camera",
    # Context
    "Contexts":         "pyaitk.Context",
    "contexts":         "pyaitk.Context",
    # Config
    "PBConfig":         "pyaitk.config",
    # CLSE
    "TTIGenerator":         "pyaitk.CLSE",
    "ProceduralArt":        "pyaitk.CLSE",
    "VisualEffects":        "pyaitk.CLSE",
    "StreamingWriter":      "pyaitk.CLSE",
    "AnimationEngine":      "pyaitk.CLSE",
    "CustomBitDepth":       "pyaitk.CLSE",
    "TTIImage":             "pyaitk.CLSE",
    "ImageCanvas":          "pyaitk.CLSE",
    "ColorUtils":           "pyaitk.CLSE",
    "ImageIO":              "pyaitk.CLSE",
    "ImageValidator":       "pyaitk.CLSE",
    "TTIDataset":           "pyaitk.CLSE",
    "TokenEmbedding":       "pyaitk.CLSE",
    "TransformerEncoder":   "pyaitk.CLSE",
    "ColourHead":           "pyaitk.CLSE",
    "SceneClassifier":      "pyaitk.CLSE",
    "ParamDecoder":         "pyaitk.CLSE",
    "TTIModel":             "pyaitk.CLSE",
    "TTIModelLarge":        "pyaitk.CLSE",
    "TTILoss":              "pyaitk.CLSE",
    "TTITrainer":           "pyaitk.CLSE",
    "ModelCheckpoint":      "pyaitk.CLSE",
    "TTIPipeline":          "pyaitk.CLSE",
    # EYE
    "EYE":              "pyaitk.eye",
    "OpenEYE":          "pyaitk.eye",
    "CameraManager":    "pyaitk.eye",
    "ModelLoader":      "pyaitk.eye",
    "ObjectDetector":   "pyaitk.eye",
    "FramePacket":      "pyaitk.eye",
    "DetectionApp":     "pyaitk.eye",
    "simple_detect":    "pyaitk.eye",
    "launch_gui":       "pyaitk.eye",
    # SummarizerAI
    "MemorySummarizer": "pyaitk.SummarizerAI",
    "DEFAULT_CONFIG":   "pyaitk.SummarizerAI",
    # NER
    "NERPipeline":          "pyaitk.NER",
    "NERTrainer":           "pyaitk.NER",
    "NEREvaluator":         "pyaitk.NER",
    "TextPreprocessor":     "pyaitk.NER",
    "EntityPostprocessor":  "pyaitk.NER",
    "EntityStore":          "pyaitk.NER",
    # Memory
    "Memory":           "pyaitk.Memory",
    # MathAI
    "MathAI":           "pyaitk.MathAI",
    # Search
    "Search":           "pyaitk.Search",
    # PPTXExtractor
    "PPTXExtractor":    "pyaitk.PPTExtract",
    # ITT
    "ITT":              "pyaitk.ITT",
}

# Groups used by check_module_availability() to represent whole subsystems.
_SUBSYSTEM_PROBE: dict[str, str] = {
    "Brain":        "Brain",
    "AdvanceBrain": "AdvanceBrain",
    "TTS":          "TTS",
    "STT":          "STT",
    "Camera":       "Camera",
    "Contexts":     "Contexts",
    "Memory":       "Memory",
    "MathAI":       "MathAI",
    "Search":       "Search",
    "PPTXExtractor":"PPTXExtractor",
    "ITT":          "ITT",
    "CLSE":         "TTIPipeline",
    "NER":          "NERPipeline",
    "EYE":          "EYE",
    "SummarizerAI": "MemorySummarizer",
    "PBConfig":     "PBConfig",
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def check_module_availability() -> dict[str, bool]:
    """
    Return a mapping of subsystem names to their availability status.

    Each subsystem is probed by attempting to import its representative symbol.
    No module is left imported as a side-effect of this call.

    Returns:
        dict[str, bool]: Keys are subsystem names; values are ``True`` when
        the underlying module can be imported without error.

    Example::

        >>> availability = check_module_availability()
        >>> if availability["Brain"]:
        ...     print("Brain module is available")
    """
    result: dict[str, bool] = {}
    for subsystem, symbol in _SUBSYSTEM_PROBE.items():
        module_path = _SYMBOL_MAP.get(symbol, "")
        try:
            mod = importlib.import_module(module_path)
            result[subsystem] = hasattr(mod, symbol)
        except Exception:
            result[subsystem] = False

    # NLTK is a standalone third-party package, not a pyaitk submodule.
    try:
        importlib.import_module("nltk")
        result["NLTK"] = True
    except Exception:
        result["NLTK"] = False

    return result


def get_version() -> str:
    """
    Return the current package version string.

    Returns:
        str: Version in ``MAJOR.MINOR.PATCH`` format.

    Example::

        >>> get_version()
        '1.1.9'
    """
    return __version__


def get_info() -> dict[str, Any]:
    """
    Return package metadata and module availability.

    Returns:
        dict: Contains the following keys:

        - ``version`` (str): Package version.
        - ``author`` (str): Package author.
        - ``license`` (str): SPDX license identifier.
        - ``url`` (str): Documentation URL.
        - ``python`` (str): Active Python version.
        - ``modules`` (dict[str, bool]): Per-subsystem availability map.

    Example::

        >>> info = get_info()
        >>> print(info["version"])
        1.1.9
    """
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "url": __url__,
        "python": sys.version,
        "modules": check_module_availability(),
    }


def InstallNLTKData() -> bool:
    """
    Download required NLTK data packages.

    Downloads the following datasets:

    - ``wordnet``                    — lemmatisation
    - ``punkt`` / ``punkt_tab``      — tokenisation
    - ``stopwords``                  — stop-word lists
    - ``averaged_perceptron_tagger`` — POS tagging
    - ``maxent_ne_chunker``          — named-entity recognition
    - ``words``                      — word corpus

    Returns:
        bool: ``True`` if every package downloaded successfully, ``False``
        if one or more failed (partial installs are still applied).

    Raises:
        RuntimeError: If NLTK is not installed in the current environment.

    Example::

        >>> if InstallNLTKData():
        ...     print("NLTK data installed successfully")
    """
    try:
        import nltk  # noqa: PLC0415 — intentional deferred import
    except ImportError as exc:
        raise RuntimeError(
            "NLTK is not installed. Add it with: pip install nltk"
        ) from exc

    packages: list[str] = [
        "wordnet",
        "maxent_ne_chunker",
        "words",
        "punkt",
        "punkt_tab",
        "stopwords",
        "averaged_perceptron_tagger",
    ]

    failed: list[str] = []

    for package in packages:
        try:
            logger.debug("Downloading NLTK package: %s", package)
            nltk.download(package, quiet=True)
            logger.debug("Downloaded NLTK package: %s", package)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to download NLTK package %r: %s", package, exc)
            failed.append(package)

    if failed:
        warnings.warn(
            f"The following NLTK packages could not be downloaded: "
            f"{', '.join(failed)}. Some NLP features may be unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    logger.info("All NLTK data packages downloaded successfully.")
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # Core
    "Brain",
    "AdvanceBrain",
    "get_weather",
    "IntentsManager",
    "VectorizerMode",
    # Audio
    "TTS",
    "TTSConfig",
    "STT",
    "speak",
    # Vision
    "Camera",
    "Start",
    "ITT",
    # EYE / object detection
    "EYE",
    "OpenEYE",
    "CameraManager",
    "ModelLoader",
    "ObjectDetector",
    "FramePacket",
    "DetectionApp",
    "simple_detect",
    "launch_gui",
    # Text processing
    "Contexts",
    "contexts",
    "PPTXExtractor",
    # AI modules
    "MathAI",
    "Search",
    # Memory
    "Memory",
    # NER
    "NERPipeline",
    "NERTrainer",
    "NEREvaluator",
    "TextPreprocessor",
    "EntityPostprocessor",
    "EntityStore",
    # SummarizerAI
    "MemorySummarizer",
    "DEFAULT_CONFIG",
    # CLSE
    "TTIGenerator",
    "ProceduralArt",
    "VisualEffects",
    "StreamingWriter",
    "AnimationEngine",
    "CustomBitDepth",
    "TTIImage",
    "ImageCanvas",
    "ColorUtils",
    "ImageIO",
    "ImageValidator",
    "TTIDataset",
    "TokenEmbedding",
    "TransformerEncoder",
    "ColourHead",
    "SceneClassifier",
    "ParamDecoder",
    "TTIModel",
    "TTIModelLarge",
    "TTILoss",
    "TTITrainer",
    "ModelCheckpoint",
    "TTIPipeline",
    # Config
    "PBConfig",
    # Utilities
    "InstallNLTKData",
    "check_module_availability",
    "get_version",
    "get_info",
    # Metadata
    "__version__",
    "__author__",
    "__license__",
    "__email__",
    "__url__",
    "__doc__",
]

# ---------------------------------------------------------------------------
# Package initialisation
# ---------------------------------------------------------------------------

logger.debug("Pythonaibrain v%s ready (Python %s)", __version__, sys.version.split()[0])
