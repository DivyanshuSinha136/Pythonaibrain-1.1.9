"""
TextPreprocessor — Cleans and normalises raw text before NER inference.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional
from ..config import get_config

logger = logging.getLogger(__name__)


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


class TextPreprocessor:
    """
    Stateless text pre-processing layer.

    Each step is a standalone method so callers can cherry-pick
    transformations without running the full pipeline.
    """

    # Pre-compiled patterns for performance
    _URL_RE = re.compile(
        r"https?://\S+|www\.\S+",
        re.IGNORECASE,
    )
    _HTML_RE = re.compile(r"<[^>]+>")
    _WHITESPACE_RE = re.compile(r"\s+")
    _EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")

    def __init__(self, config: Optional[PreprocessorConfig] = None) -> None:
        self.config = config or PreprocessorConfig()
        self._custom_res = [
            re.compile(p) for p in self.config.custom_patterns
        ]
        logger.debug("TextPreprocessor initialised with config: %s", self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> str:
        """Run all configured preprocessing steps in order."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)
        if self.config.remove_html_tags:
            text = self._strip_html(text)
        if self.config.remove_urls:
            text = self._remove_urls(text)
        if self.config.remove_emails:
            text = self._remove_emails(text)
        for pattern in self._custom_res:
            text = pattern.sub(" ", text)
        if self.config.lowercase:
            text = text.lower()
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)
        if self.config.max_length and len(text) > self.config.max_length:
            text = text[: self.config.max_length]
            logger.warning(
                "Text truncated to %d characters.", self.config.max_length
            )

        return text.strip()

    def process_batch(self, texts: List[str]) -> List[str]:
        """Vectorised preprocessing over a list of texts."""
        return [self.process(t) for t in texts]

    # ------------------------------------------------------------------
    # Individual transformation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    @classmethod
    def _strip_html(cls, text: str) -> str:
        return cls._HTML_RE.sub(" ", text)

    @classmethod
    def _remove_urls(cls, text: str) -> str:
        return cls._URL_RE.sub(" ", text)

    @classmethod
    def _remove_emails(cls, text: str) -> str:
        return cls._EMAIL_RE.sub(" ", text)

    @classmethod
    def _normalize_whitespace(cls, text: str) -> str:
        return cls._WHITESPACE_RE.sub(" ", text)