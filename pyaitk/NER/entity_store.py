"""
EntityStore — Persists extracted entities and provides query/analytics APIs.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Generator, List, Optional

from .pipeline import Entity, NERResult

logger = logging.getLogger(__name__)


class EntityStore:
    """
    Lightweight in-memory entity store with optional JSONL persistence.

    Typical usage
    -------------
    >>> store = EntityStore(persist_path="entities.jsonl")
    >>> store.add(result)
    >>> top_orgs = store.top_entities("ORG", n=10)
    """

    def __init__(self, persist_path: Optional[str | Path] = None) -> None:
        self._records: List[Dict] = []
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path and self._persist_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, result: NERResult) -> None:
        """Store all entities from a single NERResult."""
        for entity in result.entities:
            record = {
                "text": entity.text,
                "label": entity.label,
                "start_char": entity.start_char,
                "end_char": entity.end_char,
                "score": entity.score,
                "source_text": result.text,
            }
            self._records.append(record)

        if self._persist_path:
            self._append(result)

    def add_batch(self, results: List[NERResult]) -> None:
        for result in results:
            self.add(result)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        label: Optional[str] = None,
        text_contains: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """Filter stored entity records."""
        results = []
        for rec in self._records:
            if label and rec["label"] != label:
                continue
            if text_contains and text_contains.lower() not in rec["text"].lower():
                continue
            if rec["score"] < min_score:
                continue
            results.append(rec)
        return results

    def unique_entities(self, label: Optional[str] = None) -> List[str]:
        """Return sorted unique entity surface forms."""
        records = self.query(label=label)
        return sorted({r["text"] for r in records})

    def top_entities(
        self, label: Optional[str] = None, n: int = 10
    ) -> List[tuple]:
        """Return (entity_text, count) pairs, most frequent first."""
        records = self.query(label=label)
        counts = Counter(r["text"] for r in records)
        return counts.most_common(n)

    def label_distribution(self) -> Dict[str, int]:
        """Count of entities per label."""
        counts: Counter = Counter(r["label"] for r in self._records)
        return dict(counts.most_common())

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self._records, fh, ensure_ascii=False, indent=2)
        logger.info("Exported %d entity records to %s.", len(self._records), path)

    def iter_records(self) -> Generator[Dict, None, None]:
        yield from self._records

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        dist = self.label_distribution()
        return f"EntityStore(total={len(self)}, labels={dist})"

    # ------------------------------------------------------------------
    # Persistence (JSONL)
    # ------------------------------------------------------------------

    def _append(self, result: NERResult) -> None:
        with open(self._persist_path, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
            for entity in result.entities:
                record = {
                    "text": entity.text,
                    "label": entity.label,
                    "start_char": entity.start_char,
                    "end_char": entity.end_char,
                    "score": entity.score,
                    "source_text": result.text,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        with open(self._persist_path, encoding="utf-8") as fh:  # type: ignore[arg-type]
            for line in fh:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
        logger.info(
            "Loaded %d entity records from %s.", len(self._records), self._persist_path
        )