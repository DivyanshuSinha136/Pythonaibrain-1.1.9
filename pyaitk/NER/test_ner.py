"""
Test suite for the production NER system.

Run with: python -m pytest tests/ -v
"""

from __future__ import annotations

import pytest
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ner.preprocessor import TextPreprocessor, PreprocessorConfig
from ner.postprocessor import EntityPostprocessor, PostprocessorConfig
from ner.pipeline import Entity, NERPipeline, NERResult
from ner.evaluator import NEREvaluator, LabelScore, EvaluationReport
from ner.entity_store import EntityStore
from ner.trainer import NERTrainer, TrainerConfig


# ==========================================================================
# Fixtures
# ==========================================================================


def make_entity(text="Apple", label="ORG", start=0, end=5, score=1.0):
    return Entity(text=text, label=label, start_char=start, end_char=end, score=score)


def make_result(text="Apple was founded by Steve Jobs.", entities=None):
    if entities is None:
        entities = [
            make_entity("Apple", "ORG", 0, 5),
            make_entity("Steve Jobs", "PERSON", 21, 31),
        ]
    return NERResult(text=text, entities=entities, processing_time_ms=1.5)


# ==========================================================================
# TextPreprocessor
# ==========================================================================


class TestTextPreprocessor:
    def test_removes_html(self):
        pp = TextPreprocessor(PreprocessorConfig(remove_html_tags=True))
        assert "<b>" not in pp.process("<b>Hello</b>")

    def test_removes_urls(self):
        pp = TextPreprocessor(PreprocessorConfig(remove_urls=True))
        result = pp.process("Visit https://example.com for more info.")
        assert "https" not in result

    def test_lowercase(self):
        pp = TextPreprocessor(PreprocessorConfig(lowercase=True))
        assert pp.process("Hello World") == "hello world"

    def test_normalise_whitespace(self):
        pp = TextPreprocessor(PreprocessorConfig(normalize_whitespace=True))
        assert pp.process("a   b\tc") == "a b c"

    def test_max_length_truncation(self):
        pp = TextPreprocessor(PreprocessorConfig(max_length=5))
        assert len(pp.process("Hello World")) == 5

    def test_batch_processing(self):
        pp = TextPreprocessor()
        results = pp.process_batch(["Hello", "World"])
        assert len(results) == 2

    def test_raises_on_non_string(self):
        pp = TextPreprocessor()
        with pytest.raises(TypeError):
            pp.process(123)  # type: ignore


# ==========================================================================
# EntityPostprocessor
# ==========================================================================


class TestEntityPostprocessor:
    def test_filter_by_min_length(self):
        pp = EntityPostprocessor(PostprocessorConfig(min_length=6))
        entities = [make_entity("AB", "ORG", 0, 2)]
        assert pp.process(entities) == []

    def test_filter_by_allowed_labels(self):
        pp = EntityPostprocessor(PostprocessorConfig(allowed_labels={"PERSON"}))
        entities = [
            make_entity("Apple", "ORG"),
            make_entity("Steve Jobs", "PERSON"),
        ]
        result = pp.process(entities)
        assert all(e.label == "PERSON" for e in result)

    def test_filter_by_blocked_labels(self):
        pp = EntityPostprocessor(PostprocessorConfig(blocked_labels={"ORG"}))
        entities = [make_entity("Apple", "ORG"), make_entity("Steve", "PERSON")]
        result = pp.process(entities)
        assert all(e.label != "ORG" for e in result)

    def test_deduplicate(self):
        pp = EntityPostprocessor(PostprocessorConfig(deduplicate=True))
        entities = [make_entity(), make_entity()]  # identical
        assert len(pp.process(entities)) == 1

    def test_remap_labels(self):
        pp = EntityPostprocessor(PostprocessorConfig(custom_label_map={"ORG": "ORGANIZATION"}))
        entities = [make_entity("Apple", "ORG")]
        result = pp.process(entities)
        assert result[0].label == "ORGANIZATION"

    def test_lowercase_labels(self):
        pp = EntityPostprocessor(PostprocessorConfig(lowercase_labels=True))
        entities = [make_entity("Apple", "ORG")]
        result = pp.process(entities)
        assert result[0].label == "org"

    def test_strip_punct(self):
        pp = EntityPostprocessor(PostprocessorConfig(strip_punct=True))
        entities = [make_entity(",Apple.", "ORG")]
        result = pp.process(entities)
        assert result[0].text == "Apple"


# ==========================================================================
# NERPipeline
# ==========================================================================


class TestNERPipeline:
    def test_from_blank(self):
        pipeline = NERPipeline.from_blank(lang="en", labels=["ORG", "PERSON"])
        assert "ORG" in pipeline.labels
        assert "PERSON" in pipeline.labels

    def test_predict_returns_result(self):
        """A trained (or loaded) pipeline returns NERResult correctly."""
        # Use a trained pipeline from a mini training run
        import spacy
        from ner.trainer import NERTrainer, TrainerConfig
        trainer = NERTrainer(TrainerConfig(n_iter=2))
        trainer.train([
            ("Apple is an ORG.", {"entities": [(0, 5, "ORG")]}),
        ])
        pipeline = NERPipeline(trainer._nlp)
        result = pipeline.predict("Some text here.")
        assert isinstance(result, NERResult)
        assert result.text == "Some text here."

    def test_predict_batch_yields_results(self):
        """Batch inference returns one result per input text."""
        from ner.trainer import NERTrainer, TrainerConfig
        trainer = NERTrainer(TrainerConfig(n_iter=2))
        trainer.train([
            ("Apple is an ORG.", {"entities": [(0, 5, "ORG")]}),
        ])
        pipeline = NERPipeline(trainer._nlp)
        texts = ["Text one.", "Text two."]
        results = list(pipeline.predict_batch(texts))
        assert len(results) == 2

    def test_entity_to_dict(self):
        e = make_entity()
        d = e.to_dict()
        assert d["text"] == "Apple"
        assert d["label"] == "ORG"

    def test_result_filter_by_label(self):
        result = make_result()
        orgs = result.filter_by_label("ORG")
        assert all(e.label == "ORG" for e in orgs)

    def test_result_entity_types(self):
        result = make_result()
        assert set(result.entity_types) == {"ORG", "PERSON"}

    def test_result_to_dict(self):
        result = make_result()
        d = result.to_dict()
        assert "entities" in d
        assert "processing_time_ms" in d


# ==========================================================================
# NEREvaluator
# ==========================================================================


class TestNEREvaluator:
    def _run(self, gold_list, pred_list):
        evaluator = NEREvaluator()
        return evaluator.evaluate_from_predictions(gold_list, pred_list)

    def test_perfect_prediction(self):
        gold = [[(0, 5, "ORG")]]
        pred = [[(0, 5, "ORG")]]
        report = self._run(gold, pred)
        assert report.micro_f1 == 1.0

    def test_all_false_positives(self):
        gold = [[]]
        pred = [[(0, 5, "ORG")]]
        report = self._run(gold, pred)
        assert report.micro_f1 == 0.0
        assert report.micro_precision == 0.0

    def test_all_false_negatives(self):
        gold = [[(0, 5, "ORG")]]
        pred = [[]]
        report = self._run(gold, pred)
        assert report.micro_f1 == 0.0
        assert report.micro_recall == 0.0

    def test_partial_match(self):
        evaluator = NEREvaluator(partial=True)
        gold = [[(0, 10, "ORG")]]
        pred = [[(0, 5, "ORG")]]   # overlapping but not exact
        report = evaluator.evaluate_from_predictions(gold, pred)
        assert report.micro_f1 > 0.0

    def test_macro_f1(self):
        gold = [[(0, 5, "ORG"), (10, 20, "PERSON")]]
        pred = [[(0, 5, "ORG"), (10, 20, "PERSON")]]
        report = self._run(gold, pred)
        assert report.macro_f1 == 1.0

    def test_report_str(self):
        gold = [[(0, 5, "ORG")]]
        pred = [[(0, 5, "ORG")]]
        report = self._run(gold, pred)
        assert "ORG" in str(report)

    def test_label_score(self):
        ls = LabelScore(label="ORG", tp=8, fp=2, fn=2)
        assert ls.precision == pytest.approx(0.8)
        assert ls.recall == pytest.approx(0.8)
        assert ls.f1 == pytest.approx(0.8)


# ==========================================================================
# EntityStore
# ==========================================================================


class TestEntityStore:
    def test_add_and_len(self):
        store = EntityStore()
        store.add(make_result())
        assert len(store) == 2

    def test_add_batch(self):
        store = EntityStore()
        store.add_batch([make_result(), make_result()])
        assert len(store) == 4

    def test_query_by_label(self):
        store = EntityStore()
        store.add(make_result())
        orgs = store.query(label="ORG")
        assert all(r["label"] == "ORG" for r in orgs)

    def test_query_by_text(self):
        store = EntityStore()
        store.add(make_result())
        results = store.query(text_contains="apple")
        assert any(r["text"] == "Apple" for r in results)

    def test_query_by_score(self):
        store = EntityStore()
        low_score_entity = make_entity(score=0.1)
        result = NERResult(text="test", entities=[low_score_entity], processing_time_ms=0)
        store.add(result)
        high = store.query(min_score=0.9)
        assert len(high) == 0

    def test_unique_entities(self):
        store = EntityStore()
        store.add(make_result())
        store.add(make_result())  # duplicates
        uniq = store.unique_entities(label="ORG")
        assert len(uniq) == 1

    def test_top_entities(self):
        store = EntityStore()
        store.add(make_result())
        store.add(make_result())
        top = store.top_entities("ORG", n=5)
        assert top[0][0] == "Apple"
        assert top[0][1] == 2

    def test_label_distribution(self):
        store = EntityStore()
        store.add(make_result())
        dist = store.label_distribution()
        assert "ORG" in dist and "PERSON" in dist

    def test_to_json(self, tmp_path):
        store = EntityStore()
        store.add(make_result())
        out = tmp_path / "entities.json"
        store.to_json(out)
        assert out.exists()

    def test_persist_and_reload(self, tmp_path):
        path = tmp_path / "entities.jsonl"
        store1 = EntityStore(persist_path=path)
        store1.add(make_result())
        store2 = EntityStore(persist_path=path)
        assert len(store2) == len(store1)


# ==========================================================================
# NERTrainer (smoke tests — no GPU/large model needed)
# ==========================================================================

TRAIN_DATA = [
    ("Apple was founded by Steve Jobs in California.", {
        "entities": [(0, 5, "ORG"), (21, 31, "PERSON"), (35, 45, "GPE")],
    }),
    ("Google is headquartered in Mountain View.", {
        "entities": [(0, 6, "ORG"), (27, 40, "GPE")],
    }),
    ("Elon Musk leads Tesla.", {
        "entities": [(0, 9, "PERSON"), (16, 21, "ORG")],
    }),
]


class TestNERTrainer:
    def test_prepare_registers_labels(self):
        trainer = NERTrainer(TrainerConfig(n_iter=1))
        trainer.prepare(TRAIN_DATA)
        assert "ORG" in trainer._nlp.get_pipe("ner").labels

    def test_train_returns_history(self):
        trainer = NERTrainer(TrainerConfig(n_iter=2))
        report = trainer.train(TRAIN_DATA)
        assert "history" in report
        assert len(report["history"]) == 2

    def test_train_saves_model(self, tmp_path):
        trainer = NERTrainer(TrainerConfig(n_iter=2))
        trainer.train(TRAIN_DATA, output_dir=tmp_path / "model")
        assert (tmp_path / "model" / "config.cfg").exists()

    def test_data_to_docbin(self, tmp_path):
        import spacy
        trainer = NERTrainer(TrainerConfig(n_iter=1))
        trainer.prepare(TRAIN_DATA)
        out = tmp_path / "train.spacy"
        NERTrainer.data_to_docbin(TRAIN_DATA, trainer._nlp, out)
        assert out.exists()