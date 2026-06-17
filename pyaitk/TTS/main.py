"""
tts.py — Production-grade Text-to-Speech module
================================================
Features:
  - pyttsx3 backend (fully offline, cross-platform)
  - Context manager support (__enter__ / __exit__)
  - Engine lifecycle managed explicitly (init once, reuse, stop cleanly)
  - Voice selection with fuzzy name matching + ranked fallback
  - save_to_file() support
  - Structured logging (no bare print statements in library code)
  - Custom exception hierarchy
  - Module-level speak() convenience function
  - CLI entry-point with --help, --voice, --rate, --save flags

Dependencies:
    pip install pyttsx3
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
from ..config import get_config

import pyttsx3
from pyttsx3.engine import Engine as Pyttsx3Engine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class TTSError(Exception):
    """Base exception for all TTS errors."""


class TTSEngineError(TTSError):
    """Raised when pyttsx3 cannot be initialised."""


class TTSVoiceError(TTSError):
    """Raised when a requested voice cannot be resolved."""


class TTSSpeakError(TTSError):
    """Raised when speech synthesis fails at runtime."""


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


# Known voice name fragments per platform — used for matching and help text.
_PLATFORM_VOICES: dict[str, list[str]] = {
    "windows": [
        "david",   # Microsoft David — Male, en-US
        "zira",    # Microsoft Zira  — Female, en-US
        "mark",    # Microsoft Mark  — Male, en-US
    ],
    "macos": [
        "alex",       # Male,   en-US
        "samantha",   # Female, en-US
        "victoria",   # Female, en-US
        "fred",       # Male,   robotic
        "daniel",     # Male,   en-GB
        "fiona",      # Female, en-GB
    ],
    "linux": [
        "english",     # espeak default
        "english-us",  # espeak American
        "english-uk",  # espeak British
        "mb-en1",      # MBROLA English 1
        "mb-fr1",      # MBROLA French 1
    ],
    "unknown": [],
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class TTSConfig:
    """Tunable parameters for the TTS pipeline."""
    config = get_config()
    tts = config.tts

    rate: int = tts.rate                          # words per minute
    volume: float = tts.volume                     # 0.0 – 1.0
    voice: str = tts.voice                     # fragment matched against installed voices
    default_text: str = tts.default_text

    # save_to_file
    output_path: Optional[str] = tts.output_path        # if set, speech is saved here instead of played

    # Internal — resolved at runtime
    _platform: str = tts._platform

    def __post_init__(self) -> None:
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError(f"volume must be between 0.0 and 1.0, got {self.volume}")
        if self.rate <= 0:
            raise ValueError(f"rate must be positive, got {self.rate}")


# ---------------------------------------------------------------------------
# Core TTS class
# ---------------------------------------------------------------------------
class TTS:
    """
    Production-grade Text-to-Speech client.

    Usage — one-shot
    ----------------
    >>> TTS().say("Hello!")

    Usage — context manager (recommended: engine is initialised once and
    torn down cleanly on exit, even on exception)
    -----------------------------------------------------------------------
    >>> with TTS(TTSConfig(voice="zira", rate=160)) as tts:
    ...     tts.say("Hello!")
    ...     tts.say("How are you?")

    Usage — save to file
    --------------------
    >>> cfg = TTSConfig(output_path="greeting.wav")
    >>> with TTS(cfg) as tts:
    ...     tts.say("Hello!", save=True)
    """

    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        self.config = config or TTSConfig()
        self._engine: Optional[Pyttsx3Engine] = None
        self._owns_engine: bool = False   # True only when we created the engine ourselves
        logger.debug("TTS created — config: %s", self.config)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "TTS":
        self._start_engine()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._stop_engine()
        return False   # never suppress exceptions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def say(self, text: Optional[str] = None, *, voice: Optional[str] = None) -> None:
        """
        Synthesise *text* through the speaker.

        Parameters
        ----------
        text:
            Words to speak. Falls back to ``config.default_text`` when omitted.
        voice:
            Voice name fragment (e.g. ``"zira"``). Overrides ``config.voice``
            for this call only.

        Raises
        ------
        TTSEngineError   – engine failed to initialise.
        TTSVoiceError    – no matching voice found and no fallback available.
        TTSSpeakError    – pyttsx3 raised an unexpected error during synthesis.
        """
        utterance = text or self.config.default_text
        voice_name = voice or self.config.voice

        engine = self._ensure_engine()
        selected = self._resolve_voice(engine, voice_name)
        engine.setProperty("voice", selected.id)

        logger.info("Speaking with voice=%r | rate=%d | text=%r",
                    selected.name, self.config.rate, utterance)
        try:
            engine.say(utterance)
            engine.runAndWait()
        except Exception as exc:
            raise TTSSpeakError(f"Speech synthesis failed: {exc}") from exc

    def save(
        self,
        text: Optional[str] = None,
        path: Optional[str] = None,
        *,
        voice: Optional[str] = None,
    ) -> str:
        """
        Save synthesised speech to a WAV file instead of playing it.

        Parameters
        ----------
        text:
            Words to synthesise. Falls back to ``config.default_text``.
        path:
            Output file path. Falls back to ``config.output_path``, then
            ``"output.wav"``.
        voice:
            Voice name fragment. Overrides ``config.voice`` for this call.

        Returns
        -------
        str
            Resolved output path.

        Raises
        ------
        TTSEngineError  – engine failed to initialise.
        TTSVoiceError   – no matching voice found.
        TTSSpeakError   – pyttsx3 raised an error during file save.
        """
        utterance = text or self.config.default_text
        out_path = _normalise_wav_path(path or self.config.output_path or "output.wav")
        voice_name = voice or self.config.voice

        engine = self._ensure_engine()
        selected = self._resolve_voice(engine, voice_name)
        engine.setProperty("voice", selected.id)

        logger.info("Saving speech → %r | voice=%r | text=%r",
                    out_path, selected.name, utterance)
        try:
            engine.save_to_file(utterance, out_path)
            engine.runAndWait()
        except Exception as exc:
            raise TTSSpeakError(f"Failed to save speech to '{out_path}': {exc}") from exc

        logger.info("Saved: %s", out_path)
        return out_path

    def available_voices(self) -> list[str]:
        """Return display names of all voices installed on this system."""
        engine = self._ensure_engine()
        return [v.name for v in engine.getProperty("voices")]

    def platform_voice_hints(self) -> list[str]:
        """Return the fragment strings that are likely to work on this platform."""
        hints = _PLATFORM_VOICES.get(self.config._platform, [])
        logger.info("Platform voice hints (%s): %s", self.config._platform, hints)
        return hints

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _start_engine(self) -> None:
        """Initialise pyttsx3 and apply config properties."""
        if self._engine is not None:
            return
        try:
            self._engine = pyttsx3.init()
        except Exception as exc:
            raise TTSEngineError(f"Failed to initialise pyttsx3: {exc}") from exc

        self._apply_config(self._engine)
        self._owns_engine = True
        logger.debug("pyttsx3 engine started.")

    def _stop_engine(self) -> None:
        """Tear down pyttsx3 cleanly."""
        if self._engine is not None and self._owns_engine:
            try:
                self._engine.stop()
            except Exception:  # noqa: BLE001
                logger.debug("Engine stop raised (ignored during teardown).")
            self._engine = None
            self._owns_engine = False
            logger.debug("pyttsx3 engine stopped.")

    def _ensure_engine(self) -> Pyttsx3Engine:
        """Return the running engine, initialising it on-demand if needed."""
        if self._engine is None:
            self._start_engine()
        return self._engine  # type: ignore[return-value]

    def _apply_config(self, engine: Pyttsx3Engine) -> None:
        engine.setProperty("rate", self.config.rate)
        engine.setProperty("volume", self.config.volume)

    def _resolve_voice(self, engine: Pyttsx3Engine, voice_fragment: str):
        """
        Find the best installed voice matching *voice_fragment*.

        Matching priority
        -----------------
        1. Exact substring match against v.name.lower()
        2. Any platform hint that itself contains the fragment AND is a
           substring of v.name.lower()
        3. First installed voice (fallback with warning)

        Raises
        ------
        TTSVoiceError – if no voices are installed at all.
        """
        voices = engine.getProperty("voices")
        if not voices:
            raise TTSVoiceError("No TTS voices are installed on this system.")

        needle = voice_fragment.lower()
        hints = _PLATFORM_VOICES.get(self.config._platform, [])

        for v in voices:
            vname = v.name.lower()
            if needle in vname:
                logger.debug("Voice matched (direct): %s", v.name)
                return v

        for v in voices:
            vname = v.name.lower()
            if any(hint in vname for hint in hints if needle in hint):
                logger.debug("Voice matched (via hint): %s", v.name)
                return v

        fallback = voices[0]
        logger.warning(
            "Voice %r not found — falling back to %r. "
            "Available hints for this platform: %s",
            voice_fragment, fallback.name, hints,
        )
        return fallback


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _normalise_wav_path(path: str) -> str:
    """Ensure the path ends with ``.wav``."""
    stem = path.rsplit(".", 1)[0] if "." in path else path
    return f"{stem}.wav"


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------
def _default_voice() -> str:
    platform = _detect_platform()
    hints = _PLATFORM_VOICES.get(platform, [])
    return hints[0] if hints else "david"


def speak(
    text: str = "Hi",
    *,
    voice: Optional[str] = None,
    rate: int = 150,
    volume: float = 1.0,
) -> None:
    """
    One-liner convenience wrapper.

    >>> speak("Hello world!")
    >>> speak("Bonjour", voice="fiona", rate=140)
    """
    cfg = TTSConfig(
        voice=voice or _default_voice(),
        rate=rate,
        volume=volume,
    )
    with TTS(cfg) as tts:
        tts.say(text)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tts",
        description="Text-to-Speech CLI — pyttsx3 backend",
    )
    parser.add_argument("text", nargs="?", default="Hello from PyAI", help="Text to speak")
    parser.add_argument("--voice", default=None, help="Voice name fragment (e.g. 'zira', 'alex')")
    parser.add_argument("--rate", type=int, default=150, help="Speech rate in WPM (default: 150)")
    parser.add_argument("--volume", type=float, default=1.0, help="Volume 0.0–1.0 (default: 1.0)")
    parser.add_argument("--save", metavar="FILE", default=None, help="Save output to WAV file instead of playing")
    parser.add_argument("--list-voices", action="store_true", help="List all installed voices and exit")
    parser.add_argument("--hints", action="store_true", help="Show platform voice hints and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = TTSConfig(
        voice=args.voice or _default_voice(),
        rate=args.rate,
        volume=args.volume,
    )

    try:
        with TTS(cfg) as tts:
            if args.list_voices:
                print("Installed voices:")
                for name in tts.available_voices():
                    print(f"  • {name}")
                return

            if args.hints:
                print(f"Platform voice hints ({cfg._platform}):")
                for hint in tts.platform_voice_hints():
                    print(f"  • {hint}")
                return

            if args.save:
                out = tts.save(args.text, path=args.save)
                print(f"Saved → {out}")
            else:
                tts.say(args.text)

    except TTSEngineError as exc:
        logger.error("Engine error: %s", exc)
        sys.exit(1)
    except TTSVoiceError as exc:
        logger.error("Voice error: %s", exc)
        sys.exit(1)
    except TTSSpeakError as exc:
        logger.error("Speak error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    _main()
