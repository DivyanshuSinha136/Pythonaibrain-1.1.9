"""
EntityPostprocessor — Filters, deduplicates, and enriches NER outputs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from ..config import get_config

logger = logging.getLogger(__name__)


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


class EntityPostprocessor:
    """
    Applies a configurable chain of transformations to a list of Entity objects.

    Design rationale: each step is a pure function (list-in / list-out)
    so they can be unit-tested and reordered independently.
    """

    def __init__(self, config: Optional[PostprocessorConfig] = None) -> None:
        self.config = config or PostprocessorConfig()
        logger.debug("EntityPostprocessor initialised: %s", self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, entities: list) -> list:
        """Apply the full post-processing chain."""
        entities = self._filter_by_length(entities)
        entities = self._filter_by_label(entities)
        if self.config.strip_punct:
            entities = self._strip_punct(entities)
        if self.config.custom_label_map:
            entities = self._remap_labels(entities)
        if self.config.lowercase_labels:
            entities = self._lowercase_labels(entities)
        if self.config.deduplicate:
            entities = self._deduplicate(entities)
        if self.config.merge_adjacent:
            entities = self._merge_adjacent(entities)
        return entities

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _filter_by_length(self, entities: list) -> list:
        out = []
        for e in entities:
            n = len(e.text)
            if n < self.config.min_length:
                continue
            if self.config.max_length and n > self.config.max_length:
                continue
            out.append(e)
        return out

    def _filter_by_label(self, entities: list) -> list:
        out = []
        for e in entities:
            if self.config.allowed_labels is not None:
                if e.label not in self.config.allowed_labels:
                    continue
            if e.label in self.config.blocked_labels:
                continue
            out.append(e)
        return out

    @staticmethod
    def _strip_punct(entities: list) -> list:
        _punct = re.compile(r"^[\W_]+|[\W_]+$")
        for e in entities:
            e.text = _punct.sub("", e.text).strip()
        return [e for e in entities if e.text]

    def _remap_labels(self, entities: list) -> list:
        for e in entities:
            e.label = self.config.custom_label_map.get(e.label, e.label)
        return entities

    @staticmethod
    def _lowercase_labels(entities: list) -> list:
        for e in entities:
            e.label = e.label.lower()
        return entities

    @staticmethod
    def _deduplicate(entities: list) -> list:
        seen: Set[tuple] = set()
        out = []
        for e in entities:
            key = (e.text, e.label, e.start_char, e.end_char)
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    @staticmethod
    def _merge_adjacent(entities: list) -> list:
        """Merge back-to-back spans with the same label (gap ≤ 1 char)."""
        if not entities:
            return entities
        merged = [entities[0]]
        for current in entities[1:]:
            prev = merged[-1]
            if (
                current.label == prev.label
                and current.start_char - prev.end_char <= 1
            ):
                prev.text = prev.text + " " + current.text
                prev.end_char = current.end_char
                prev.score = min(prev.score, current.score)
            else:
                merged.append(current)
        return merged