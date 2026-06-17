"""
NERTrainer — Production training loop for custom spaCy NER models.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for the canonical training-data format expected by this module
# ---------------------------------------------------------------------------
# [
#   ("Apple was founded in 1976.", {"entities": [(0, 5, "ORG"), (21, 25, "DATE")]}),
#   ...
# ]
TrainingSample = Tuple[str, Dict[str, List[Tuple[int, int, str]]]]


@dataclass
class TrainerConfig:
    """Hyper-parameters and runtime settings for the training loop."""

    lang: str = "en"
    n_iter: int = 30
    dropout: float = 0.35
    batch_start: int = 4
    batch_compound: float = 1.001
    eval_every: int = 5                   # evaluate on dev set every N epochs
    patience: int = 5                     # early stopping patience (epochs)
    min_delta: float = 0.001              # min improvement to reset patience counter
    seed: int = 42


class NERTrainer:
    """
    Encapsulates the full spaCy NER training lifecycle:
      1. Model initialisation
      2. Label registration
      3. Training loop with early stopping
      4. Model serialisation

    Example
    -------
    >>> trainer = NERTrainer(config=TrainerConfig(n_iter=20))
    >>> trainer.prepare(TRAIN_DATA)
    >>> trainer.train(TRAIN_DATA, dev_data=DEV_DATA, output_dir="models/v1")
    """

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        base_model: Optional[str] = None,   # e.g. "en_core_web_sm"
    ) -> None:
        self.config = config or TrainerConfig()
        self._base_model = base_model
        self._nlp: Optional[Language] = None
        self._best_score: float = 0.0
        self._no_improve: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def prepare(self, train_data: List[TrainingSample]) -> None:
        """
        Build the spaCy pipeline and register all entity labels
        found in *train_data*.
        """
        random.seed(self.config.seed)

        if self._base_model:
            self._nlp = spacy.load(self._base_model, disable=["ner"])
        else:
            self._nlp = spacy.blank(self.config.lang)

        if not self._nlp.has_pipe("ner"):
            ner = self._nlp.add_pipe("ner", last=True)
        else:
            ner = self._nlp.get_pipe("ner")

        labels = self._extract_labels(train_data)
        for label in labels:
            ner.add_label(label)
        logger.info("Registered %d NER labels: %s", len(labels), sorted(labels))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_data: List[TrainingSample],
        dev_data: Optional[List[TrainingSample]] = None,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Run the full training loop.

        Returns a dict with training history and best dev scores.
        """
        if self._nlp is None:
            self.prepare(train_data)

        assert self._nlp is not None
        optimizer = self._nlp.initialize()

        # Disable all components except NER during training
        other_pipes = [p for p in self._nlp.pipe_names if p != "ner"]

        history: List[Dict] = []
        best_model_bytes: Optional[bytes] = None

        with self._nlp.select_pipes(disable=other_pipes):
            for epoch in range(1, self.config.n_iter + 1):
                random.shuffle(train_data)
                losses: Dict[str, float] = {}

                batches = minibatch(
                    train_data,
                    size=spacy.util.compounding(
                        self.config.batch_start,
                        32,
                        self.config.batch_compound,
                    ),
                )
                for batch in batches:
                    examples = self._to_examples(batch)
                    self._nlp.update(
                        examples,
                        drop=self.config.dropout,
                        losses=losses,
                    )

                record: Dict[str, Any] = {
                    "epoch": epoch,
                    "ner_loss": round(losses.get("ner", 0.0), 4),
                }

                # Dev evaluation
                if dev_data and epoch % self.config.eval_every == 0:
                    scores = self._evaluate(dev_data)
                    record.update(scores)
                    f1 = scores.get("ents_f", 0.0)
                    logger.info(
                        "Epoch %03d | loss=%.4f | dev_F1=%.4f | dev_P=%.4f | dev_R=%.4f",
                        epoch,
                        record["ner_loss"],
                        f1,
                        scores.get("ents_p", 0.0),
                        scores.get("ents_r", 0.0),
                    )
                    # Early stopping
                    if f1 > self._best_score + self.config.min_delta:
                        self._best_score = f1
                        self._no_improve = 0
                        best_model_bytes = self._nlp.to_bytes()
                    else:
                        self._no_improve += 1
                        if self._no_improve >= self.config.patience:
                            logger.info(
                                "Early stopping at epoch %d (patience=%d).",
                                epoch,
                                self.config.patience,
                            )
                            break
                else:
                    logger.info(
                        "Epoch %03d | loss=%.4f", epoch, record["ner_loss"]
                    )

                history.append(record)

        # Restore best weights if available
        if best_model_bytes:
            self._nlp.from_bytes(best_model_bytes)
            logger.info("Best model restored (F1=%.4f).", self._best_score)

        if output_dir:
            self._save(output_dir)

        return {"history": history, "best_dev_f1": self._best_score}

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def _evaluate(
        self, dev_data: List[TrainingSample]
    ) -> Dict[str, float]:
        examples = self._to_examples(dev_data)
        scores = self._nlp.evaluate(examples)  # type: ignore[arg-type]
        return {
            "ents_p": round(scores.get("ents_p", 0.0), 4),
            "ents_r": round(scores.get("ents_r", 0.0), 4),
            "ents_f": round(scores.get("ents_f", 0.0), 4),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_examples(self, data: List[TrainingSample]) -> List[Example]:
        examples = []
        for text, annotations in data:
            doc = self._nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        return examples

    @staticmethod
    def _extract_labels(data: List[TrainingSample]) -> List[str]:
        labels: set = set()
        for _, ann in data:
            for _, _, label in ann.get("entities", []):
                labels.add(label)
        return sorted(labels)

    def _save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._nlp.to_disk(output_dir)
        logger.info("Model saved to %s", output_dir)

    # ------------------------------------------------------------------
    # DocBin helpers (for large-scale / spaCy v3 config-based training)
    # ------------------------------------------------------------------

    @staticmethod
    def data_to_docbin(
        data: List[TrainingSample],
        nlp: Language,
        output_path: str | Path,
    ) -> None:
        """Serialise training data to the spaCy v3 ``DocBin`` format."""
        db = DocBin()
        for text, annotations in data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            db.add(example.reference)
        db.to_disk(output_path)
        logger.info("DocBin saved to %s (%d docs).", output_path, len(data))