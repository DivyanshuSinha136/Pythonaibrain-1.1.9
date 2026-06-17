"""
NERPipeline — Core inference pipeline wrapping a spaCy NER model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

from .preprocessor import TextPreprocessor, PreprocessorConfig
from .postprocessor import EntityPostprocessor, PostprocessorConfig

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Normalised representation of a detected entity."""

    text: str
    label: str
    start_char: int
    end_char: int
    score: float = 1.0          # confidence (0–1); populated when available
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "label": self.label,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class NERResult:
    """Full result for a single piece of text."""

    text: str
    entities: List[Entity]
    processing_time_ms: float
    doc: Optional[Doc] = None   # raw spaCy Doc, opt-in

    @property
    def entity_types(self) -> List[str]:
        return sorted({e.label for e in self.entities})

    def filter_by_label(self, label: str) -> List[Entity]:
        return [e for e in self.entities if e.label == label]

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": self.entity_types,
            "processing_time_ms": round(self.processing_time_ms, 3),
        }


class NERPipeline:
    """
    Production NER inference pipeline.

    Features
    --------
    - Lazy model loading (loads only when first used)
    - Pre/post processing hooks
    - Batch inference with configurable chunk size
    - Optional raw ``Doc`` passthrough for downstream tasks
    - Thread-safe (``nlp`` objects are read-only after load)

    Usage
    -----
    >>> pipeline = NERPipeline.from_model_path("models/my_ner")
    >>> result = pipeline.predict("Apple was founded by Steve Jobs in California.")
    >>> for entity in result.entities:
    ...     print(entity.label, entity.text)
    """

    def __init__(
        self,
        nlp: Language,
        preprocessor: Optional[TextPreprocessor] = None,
        postprocessor: Optional[EntityPostprocessor] = None,
        return_doc: bool = False,
        batch_size: int = 64,
    ) -> None:
        self._nlp = nlp
        self._preprocessor = preprocessor or TextPreprocessor()
        self._postprocessor = postprocessor or EntityPostprocessor()
        self._return_doc = return_doc
        self._batch_size = batch_size
        logger.info(
            "NERPipeline ready | components: %s | batch_size: %d",
            self._nlp.pipe_names,
            self._batch_size,
        )

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        *,
        preprocessor_config: Optional[PreprocessorConfig] = None,
        postprocessor_config: Optional[PostprocessorConfig] = None,
        **kwargs,
    ) -> "NERPipeline":
        """Load a saved spaCy model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        nlp = spacy.load(model_path)
        return cls(
            nlp,
            preprocessor=TextPreprocessor(preprocessor_config),
            postprocessor=EntityPostprocessor(postprocessor_config),
            **kwargs,
        )

    @classmethod
    def from_blank(
        cls,
        lang: str = "en",
        labels: Optional[List[str]] = None,
        **kwargs,
    ) -> "NERPipeline":
        """Create a blank model (for training or rule-based use)."""
        nlp = spacy.blank(lang)
        if not nlp.has_pipe("ner"):
            ner = nlp.add_pipe("ner")
            for label in (labels or []):
                ner.add_label(label)
        return cls(nlp, **kwargs)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> NERResult:
        """Run NER on a single text string."""
        cleaned = self._preprocessor.process(text)
        t0 = time.perf_counter()
        doc = self._nlp(cleaned)
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        entities = self._spans_to_entities(doc.ents)
        entities = self._postprocessor.process(entities)

        return NERResult(
            text=text,
            entities=entities,
            processing_time_ms=elapsed_ms,
            doc=doc if self._return_doc else None,
        )

    def predict_batch(
        self,
        texts: Iterable[str],
    ) -> Generator[NERResult, None, None]:
        """
        Run NER over an iterable of texts using spaCy's efficient
        ``nlp.pipe`` method. Yields results lazily.
        """
        text_list = list(texts)
        cleaned = self._preprocessor.process_batch(text_list)

        docs = self._nlp.pipe(cleaned, batch_size=self._batch_size)
        for original_text, doc in zip(text_list, docs):
            entities = self._spans_to_entities(doc.ents)
            entities = self._postprocessor.process(entities)
            yield NERResult(
                text=original_text,
                entities=entities,
                processing_time_ms=0.0,  # aggregate timing not meaningful here
                doc=doc if self._return_doc else None,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spans_to_entities(spans: tuple[Span, ...]) -> List[Entity]:
        entities = []
        for span in spans:
            score = (
                float(span._.get("score", 1.0))
                if span._.has("score")
                else 1.0
            )
            entities.append(
                Entity(
                    text=span.text,
                    label=span.label_,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    score=score,
                )
            )
        return entities

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def save(self, output_path: str | Path) -> None:
        """Persist the underlying spaCy model."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self._nlp.to_disk(output_path)
        logger.info("Model saved to %s", output_path)

    @property
    def labels(self) -> List[str]:
        if "ner" in self._nlp.pipe_names:
            return list(self._nlp.get_pipe("ner").labels)
        return []

    @property
    def nlp(self) -> Language:
        return self._nlp