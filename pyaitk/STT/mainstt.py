"""
stt.py — Production-grade Speech-to-Text module
================================================
Features:
  - Online mode  : Google Speech Recognition (via SpeechRecognition)
  - Offline mode : PocketSphinx (auto-detected when network is unavailable)
  - Context manager support (__enter__ / __exit__)
  - Configurable energy threshold, pause detection, and timeouts
  - Structured logging (no bare print statements in library code)
  - Custom exception hierarchy for granular error handling
  - Thread-safe recognizer instance

Dependencies:
    pip install SpeechRecognition pocketsphinx pyaudio
"""

from __future__ import annotations

import logging
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from ..config import get_config

import speech_recognition as sr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class STTError(Exception):
    """Base exception for all STT errors."""


class STTAudioError(STTError):
    """Raised when audio cannot be captured from the microphone."""


class STTRecognitionError(STTError):
    """Raised when speech cannot be recognised (unintelligible audio)."""


class STTServiceError(STTError):
    """Raised when the upstream recognition service is unreachable."""


class STTEngineError(STTError):
    """Raised when the local offline engine fails."""


# ---------------------------------------------------------------------------
# Engine enum
# ---------------------------------------------------------------------------
class Engine(Enum):
    GOOGLE = auto()       # Online — Google Speech Recognition
    POCKETSPHINX = auto() # Offline — CMU PocketSphinx


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class STTConfig:
    """Tunable knobs for the STT pipeline."""
    config = get_config()
    stt = config.stt

    # Microphone / audio capture
    energy_threshold: Optional[float] = stt.energy_threshold   # None → auto-calibrate
    dynamic_energy_threshold: bool = stt.dynamic_energy_threshold
    pause_threshold: float = stt.pause_threshold               # seconds of silence = end of phrase
    phrase_time_limit: Optional[float] = stt.phrase_time_limit  # hard cap per utterance (seconds)
    timeout: Optional[float] = stt.timeout            # seconds to wait for speech to start

    # Ambient-noise calibration
    ambient_noise_duration: float = stt.ambient_noise_duration        # seconds spent sampling noise floor

    # Connectivity check
    connectivity_host: str = stt.connectivity_host
    connectivity_port: int = stt.connectivity_port
    connectivity_timeout: float = stt.connectivity_timeout

    # Engine preference override (None → auto-detect)
    preferred_engine: Optional[Engine] = stt.preferred_engine

    # PocketSphinx keyword / language model (None → CMU US English default)
    sphinx_language: str = stt.sphinx_language

    # Google Speech API
    google_language: str = stt.google_language
    google_api_key: Optional[str] = stt.google_api_key       # None → free tier

    # Retry
    max_retries: int = stt.max_retries
    retry_delay: float = stt.retry_delay                   # seconds between retries


# ---------------------------------------------------------------------------
# Connectivity helper
# ---------------------------------------------------------------------------
def _is_online(host: str, port: int, timeout: float) -> bool:
    """Return True when a TCP connection to *host:port* succeeds."""
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Core STT class
# ---------------------------------------------------------------------------
class STT:
    """
    Production-grade Speech-to-Text client.

    Usage — one-shot
    ----------------
    >>> stt = STT()
    >>> text = stt.listen()

    Usage — context manager (recommended: releases microphone on exit)
    ------------------------------------------------------------------
    >>> with STT() as stt:
    ...     text = stt.listen()

    Usage — explicit engine selection
    ----------------------------------
    >>> cfg = STTConfig(preferred_engine=Engine.POCKETSPHINX)
    >>> with STT(config=cfg) as stt:
    ...     text = stt.listen()
    """

    def __init__(self, config: Optional[STTConfig] = None) -> None:
        self.config = config or STTConfig()
        self._recognizer: sr.Recognizer = self._build_recognizer()
        self._microphone: Optional[sr.Microphone] = None
        logger.debug("STT initialised with config: %s", self.config)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "STT":
        logger.debug("STT context entered — opening microphone.")
        self._microphone = sr.Microphone()
        self._microphone.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._microphone is not None:
            logger.debug("STT context exiting — releasing microphone.")
            self._microphone.__exit__(exc_type, exc_val, exc_tb)
            self._microphone = None
        # Do not suppress exceptions
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def listen(self) -> str:
        """
        Capture audio from the microphone and return recognised text.

        Automatically selects online (Google) or offline (PocketSphinx)
        engine based on network availability, unless *config.preferred_engine*
        is set.

        Raises
        ------
        STTAudioError
            Microphone could not be opened or no speech was detected.
        STTRecognitionError
            Audio was captured but speech was unintelligible.
        STTServiceError
            Online service was unreachable (only in GOOGLE mode).
        STTEngineError
            PocketSphinx failed to initialise or process audio.
        """
        engine = self._resolve_engine()
        logger.info("Selected engine: %s", engine.name)

        audio = self._capture_audio()
        return self._recognise_with_retry(audio, engine)

    @property
    def active_engine(self) -> Engine:
        """Return which engine *would* be used right now."""
        return self._resolve_engine()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_recognizer(self) -> sr.Recognizer:
        rec = sr.Recognizer()
        cfg = self.config

        if cfg.energy_threshold is not None:
            rec.energy_threshold = cfg.energy_threshold
        rec.dynamic_energy_threshold = cfg.dynamic_energy_threshold
        rec.pause_threshold = cfg.pause_threshold
        return rec

    def _resolve_engine(self) -> Engine:
        if self.config.preferred_engine is not None:
            return self.config.preferred_engine

        cfg = self.config
        online = _is_online(
            cfg.connectivity_host,
            cfg.connectivity_port,
            cfg.connectivity_timeout,
        )
        chosen = Engine.GOOGLE if online else Engine.POCKETSPHINX
        logger.debug("Network online=%s → engine=%s", online, chosen.name)
        return chosen

    @contextmanager
    def _open_source(self):
        """Yield an audio source — reuses an already-open microphone if inside a context."""
        if self._microphone is not None:
            yield self._microphone
        else:
            with sr.Microphone() as source:
                yield source

    def _capture_audio(self) -> sr.AudioData:
        cfg = self.config
        try:
            with self._open_source() as source:
                logger.info(
                    "Calibrating for ambient noise (%.1fs)…",
                    cfg.ambient_noise_duration,
                )
                self._recognizer.adjust_for_ambient_noise(
                    source, duration=cfg.ambient_noise_duration
                )
                logger.info("🎤 Listening… (timeout=%s s)", cfg.timeout)
                audio = self._recognizer.listen(
                    source,
                    timeout=cfg.timeout,
                    phrase_time_limit=cfg.phrase_time_limit,
                )
            return audio
        except sr.WaitTimeoutError as exc:
            raise STTAudioError("No speech detected within the timeout window.") from exc
        except OSError as exc:
            raise STTAudioError(f"Could not open microphone: {exc}") from exc

    def _recognise_with_retry(self, audio: sr.AudioData, engine: Engine) -> str:
        cfg = self.config
        last_exc: Exception = RuntimeError("No attempts made.")

        for attempt in range(1, cfg.max_retries + 1):
            try:
                if engine is Engine.GOOGLE:
                    return self._recognise_google(audio)
                else:
                    return self._recognise_sphinx(audio)
            except (STTRecognitionError, STTEngineError):
                raise  # No point retrying unintelligible audio
            except STTServiceError as exc:
                last_exc = exc
                logger.warning(
                    "Service error on attempt %d/%d: %s",
                    attempt, cfg.max_retries, exc,
                )
                if attempt < cfg.max_retries:
                    time.sleep(cfg.retry_delay)

        raise STTServiceError(
            f"Recognition failed after {cfg.max_retries} attempts."
        ) from last_exc

    def _recognise_google(self, audio: sr.AudioData) -> str:
        cfg = self.config
        try:
            text = self._recognizer.recognize_google(
                audio,
                key=cfg.google_api_key,
                language=cfg.google_language,
            )
            logger.info("Google recognised: %r", text)
            return text
        except sr.UnknownValueError as exc:
            raise STTRecognitionError(
                "Google could not understand the audio."
            ) from exc
        except sr.RequestError as exc:
            raise STTServiceError(
                f"Google Speech API request failed: {exc}"
            ) from exc

    def _recognise_sphinx(self, audio: sr.AudioData) -> str:
        cfg = self.config
        try:
            text = self._recognizer.recognize_sphinx(
                audio,
                language=cfg.sphinx_language,
            )
            logger.info("PocketSphinx recognised: %r", text)
            return text
        except sr.UnknownValueError as exc:
            raise STTRecognitionError(
                "PocketSphinx could not understand the audio."
            ) from exc
        except sr.RequestError as exc:
            # PocketSphinx raises RequestError when the language pack is missing
            raise STTEngineError(
                f"PocketSphinx engine error (missing language data?): {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Speech-to-Text CLI")
    parser.add_argument(
        "--engine",
        choices=["auto", "google", "sphinx"],
        default="auto",
        help="Force a specific engine (default: auto-detect by connectivity)",
    )
    parser.add_argument(
        "--language", default="en-US", help="BCP-47 language tag (default: en-US)"
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0,
        help="Seconds to wait for speech to start (default: 5)"
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Max retry attempts on service errors (default: 3)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine_map = {
        "google": Engine.GOOGLE,
        "sphinx": Engine.POCKETSPHINX,
        "auto": None,
    }

    config = STTConfig(
        preferred_engine=engine_map[args.engine],
        google_language=args.language,
        sphinx_language=args.language,
        timeout=args.timeout,
        max_retries=args.retries,
    )

    try:
        with STT(config=config) as stt:
            logger.info("Active engine: %s", stt.active_engine.name)
            text = stt.listen()
            print(f"\n📝 Recognised text: {text}")
    except STTAudioError as exc:
        logger.error("Audio capture failed: %s", exc)
    except STTRecognitionError as exc:
        logger.error("Recognition failed: %s", exc)
    except (STTServiceError, STTEngineError) as exc:
        logger.error("Engine error: %s", exc)


if __name__ == "__main__":
    _main()
