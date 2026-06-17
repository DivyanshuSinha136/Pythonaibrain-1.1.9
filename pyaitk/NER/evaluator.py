"""
NEREvaluator — Precision / Recall / F1 evaluation for NER models.

Supports both exact-match and partial (token-overlap) evaluation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# (start_char, end_char, label)
GoldSpan = Tuple[int, int, str]
PredSpan = Tuple[int, int, str]


@dataclass
class LabelScore:
    """Per-label metric container."""

    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation results."""

    per_label: Dict[str, LabelScore] = field(default_factory=dict)
    overall_tp: int = 0
    overall_fp: int = 0
    overall_fn: int = 0

    @property
    def micro_precision(self) -> float:
        denom = self.overall_tp + self.overall_fp
        return self.overall_tp / denom if denom else 0.0

    @property
    def micro_recall(self) -> float:
        denom = self.overall_tp + self.overall_fn
        return self.overall_tp / denom if denom else 0.0

    @property
    def micro_f1(self) -> float:
        p, r = self.micro_precision, self.micro_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def macro_f1(self) -> float:
        scores = [ls.f1 for ls in self.per_label.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> Dict:
        return {
            "micro": {
                "precision": round(self.micro_precision, 4),
                "recall": round(self.micro_recall, 4),
                "f1": round(self.micro_f1, 4),
            },
            "macro_f1": round(self.macro_f1, 4),
            "per_label": {
                lbl: ls.to_dict() for lbl, ls in self.per_label.items()
            },
        }

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"  Micro  P={self.micro_precision:.4f}  R={self.micro_recall:.4f}  F1={self.micro_f1:.4f}",
            f"  Macro F1 = {self.macro_f1:.4f}",
            "-" * 60,
            f"{'Label':<20} {'P':>8} {'R':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}",
            "-" * 60,
        ]
        for ls in sorted(self.per_label.values(), key=lambda x: -x.f1):
            lines.append(
                f"{ls.label:<20} {ls.precision:>8.4f} {ls.recall:>8.4f}"
                f" {ls.f1:>8.4f} {ls.tp:>6} {ls.fp:>6} {ls.fn:>6}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class NEREvaluator:
    """
    Computes entity-level evaluation metrics.

    Supports:
    - Exact span match (default)
    - Partial / boundary-lenient match (``partial=True``)

    Usage
    -----
    >>> evaluator = NEREvaluator()
    >>> report = evaluator.evaluate(gold_dataset, pipeline)
    >>> print(report)
    """

    def __init__(self, partial: bool = False) -> None:
        self._partial = partial

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        gold_data: List[Tuple[str, Dict]],
        pipeline,                         # NERPipeline; avoid circular import
    ) -> EvaluationReport:
        """
        Parameters
        ----------
        gold_data:
            List of (text, {"entities": [(start, end, label), ...]}) tuples.
        pipeline:
            An initialised NERPipeline.
        """
        label_scores: Dict[str, LabelScore] = defaultdict(
            lambda: LabelScore(label="")
        )

        for text, annotations in gold_data:
            gold_spans: List[GoldSpan] = annotations.get("entities", [])
            result = pipeline.predict(text)
            pred_spans: List[PredSpan] = [
                (e.start_char, e.end_char, e.label) for e in result.entities
            ]
            self._score_document(gold_spans, pred_spans, label_scores)

        # Ensure label field is set (was overwritten by defaultdict)
        for lbl, ls in label_scores.items():
            ls.label = lbl

        report = EvaluationReport(per_label=dict(label_scores))
        for ls in label_scores.values():
            report.overall_tp += ls.tp
            report.overall_fp += ls.fp
            report.overall_fn += ls.fn

        logger.info("Evaluation complete:\n%s", report)
        return report

    def evaluate_from_predictions(
        self,
        gold_spans_list: List[List[GoldSpan]],
        pred_spans_list: List[List[PredSpan]],
    ) -> EvaluationReport:
        """
        Evaluate pre-computed predictions — useful when the pipeline
        has already run and results are cached.
        """
        label_scores: Dict[str, LabelScore] = defaultdict(
            lambda: LabelScore(label="")
        )
        for gold, pred in zip(gold_spans_list, pred_spans_list):
            self._score_document(gold, pred, label_scores)

        for lbl, ls in label_scores.items():
            ls.label = lbl

        report = EvaluationReport(per_label=dict(label_scores))
        for ls in label_scores.values():
            report.overall_tp += ls.tp
            report.overall_fp += ls.fp
            report.overall_fn += ls.fn
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_document(
        self,
        gold: List[GoldSpan],
        pred: List[PredSpan],
        scores: Dict[str, LabelScore],
    ) -> None:
        gold_set = set(gold)
        pred_set = set(pred)

        matched_gold: set = set()
        for ps in pred_set:
            ps_start, ps_end, ps_label = ps
            hit = self._find_match(ps, gold_set)
            if hit:
                scores[ps_label].tp += 1
                matched_gold.add(hit)
            else:
                scores[ps_label].fp += 1

        for gs in gold_set:
            _, _, gs_label = gs
            if gs not in matched_gold:
                scores[gs_label].fn += 1

    def _find_match(
        self, pred: PredSpan, gold_set: set
    ) -> Optional[GoldSpan]:
        ps, pe, pl = pred
        for gs, ge, gl in gold_set:
            if gl != pl:
                continue
            if self._partial:
                if ps < ge and pe > gs:   # any overlap
                    return (gs, ge, gl)
            else:
                if ps == gs and pe == ge:  # exact
                    return (gs, ge, gl)
        return None