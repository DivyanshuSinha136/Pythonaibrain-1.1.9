"""
Data layer: loading, cleaning, and preprocessing memory patterns.
"""

import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Domain objects ───────────────────────────────────────────────────────────

@dataclass
class MemoryPattern:
    """A single input→response memory pattern with derived metadata."""
    input_text: str
    response_text: str
    input_clean: str = ""
    response_clean: str = ""
    intent_tag: str = ""          # inferred broad intent
    response_type: str = ""       # "greeting", "joke", "cmd", "open", "error", "info"
    word_count: int = 0
    has_error: bool = False
    is_command: bool = False
    is_open_action: bool = False


# ─── Cleaning helpers ─────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def infer_response_type(response: str) -> str:
    r = response.lower()
    if any(g in r for g in ["hello", "hey", "hi!", "greet"]):
        return "greeting"
    if "joke" in r or "parrot" in r or "vim" in r or "recursion" in r or "capitals" in r:
        return "joke"
    if r.startswith("done:") or "cmd_start" in r:
        return "command"
    if r.startswith("['open'"):
        return "open_action"
    if r.startswith("[error]") or r.startswith("error:"):
        return "error"
    if len(r) > 120:
        return "info"
    if any(w in r for w in ["goodbye", "bye", "come back"]):
        return "farewell"
    if any(w in r for w in ["wait", "moment", "checking"]):
        return "processing"
    return "generic_reply"


def infer_intent(input_text: str, response_type: str) -> str:
    t = input_text.lower()
    if any(w in t for w in ["hi", "hello", "hey", "yo", "greet"]):
        return "greeting"
    if any(w in t for w in ["bye", "goodbye", "ok bye"]):
        return "farewell"
    if any(w in t for w in ["joke", "jock", "funny"]):
        return "humor"
    if any(w in t for w in ["who", "name", "created", "founder", "age", "you"]):
        return "identity_query"
    if any(w in t for w in ["cmd", "start", "command"]):
        return "command_trigger"
    if any(w in t for w in ["what is", "tell me about", "python", "pythonaibrain"]):
        return "knowledge_query"
    if any(w in t for w in ["time", "date"]):
        return "temporal_query"
    if any(w in t for w in ["thanks", "great", "done", "ok", "well done"]):
        return "acknowledgment"
    return "general"


# ─── Loader ───────────────────────────────────────────────────────────────────

class MemoryLoader:
    """Loads and preprocesses raw memory dicts into MemoryPattern objects."""

    def __init__(self, skip_empty_keys: bool = True, skip_errors: bool = False):
        self.skip_empty_keys = skip_empty_keys
        self.skip_errors = skip_errors

    def from_dict(self, raw: Dict[str, str]) -> List[MemoryPattern]:
        patterns: List[MemoryPattern] = []
        for k, v in raw.items():
            if self.skip_empty_keys and k.strip() == "":
                logger.debug("Skipping empty key.")
                continue

            response_type = infer_response_type(v)

            if self.skip_errors and response_type == "error":
                logger.debug(f"Skipping error pattern for key: {k!r}")
                continue

            mp = MemoryPattern(
                input_text=k,
                response_text=v,
                input_clean=clean_text(k),
                response_clean=clean_text(v),
                intent_tag=infer_intent(k, response_type),
                response_type=response_type,
                word_count=len(k.split()),
                has_error=(response_type == "error"),
                is_command=(response_type == "command"),
                is_open_action=(response_type == "open_action"),
            )
            patterns.append(mp)

        logger.info(f"Loaded {len(patterns)} memory patterns.")
        return patterns

    def from_json_file(self, path: str) -> List[MemoryPattern]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return self.from_dict(data)

    def to_dataframe(self, patterns: List[MemoryPattern]) -> pd.DataFrame:
        return pd.DataFrame([p.__dict__ for p in patterns])


# ─── Feature extraction ───────────────────────────────────────────────────────

class TextFeatureExtractor:
    """
    Extracts TF-IDF feature matrix from MemoryPatterns.
    Combines input + response text for richer representation.
    """

    def __init__(self, config=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        cfg = config or {}
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.get("max_features", 5000),
            ngram_range=cfg.get("ngram_range", (1, 3)),
            sublinear_tf=cfg.get("sublinear_tf", True),
            analyzer="char_wb",   # char n-grams handle typos better
        )
        self._fitted = False

    def fit(self, patterns: List[MemoryPattern]) -> "TextFeatureExtractor":
        corpus = self._build_corpus(patterns)
        self.vectorizer.fit(corpus)
        self._fitted = True
        return self

    def transform(self, patterns: List[MemoryPattern]) -> np.ndarray:
        assert self._fitted, "Call fit() first."
        corpus = self._build_corpus(patterns)
        mat = self.vectorizer.transform(corpus)
        from sklearn.preprocessing import normalize
        return normalize(mat, norm="l2")

    def fit_transform(self, patterns: List[MemoryPattern]) -> np.ndarray:
        return self.fit(patterns).transform(patterns)

    @staticmethod
    def _build_corpus(patterns: List[MemoryPattern]) -> List[str]:
        # Concatenate input + response for richer context
        return [f"{p.input_clean} [SEP] {p.response_clean}" for p in patterns]
