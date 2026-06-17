"""
demo.py — End-to-end walkthrough of the NER system.

Demonstrates:
  1. Text preprocessing
  2. Model training on toy data
  3. Inference (single & batch)
  4. Post-processing
  5. Evaluation metrics
  6. Entity store usage
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ner.logging_config import setup_logging

setup_logging(level="INFO")

from ner import (
    NERPipeline,
    NERTrainer,
    NEREvaluator,
    TextPreprocessor,
    EntityPostprocessor,
    EntityStore,
)
from ner.preprocessor import PreprocessorConfig
from ner.postprocessor import PostprocessorConfig
from ner.trainer import TrainerConfig

# ---------------------------------------------------------------------------
# Training data (toy dataset — replace with real annotations in production)
# ---------------------------------------------------------------------------

TRAIN_DATA = [
    ("Apple was founded by Steve Jobs in California.", {
        "entities": [(0, 5, "ORG"), (21, 31, "PERSON"), (35, 45, "GPE")],
    }),
    ("Google is headquartered in Mountain View, California.", {
        "entities": [(0, 6, "ORG"), (27, 40, "GPE"), (42, 52, "GPE")],
    }),
    ("Elon Musk leads Tesla and SpaceX.", {
        "entities": [(0, 9, "PERSON"), (16, 21, "ORG"), (26, 32, "ORG")],
    }),
    ("Barack Obama served as President of the United States.", {
        "entities": [(0, 12, "PERSON"), (40, 53, "GPE")],
    }),
    ("Amazon was founded by Jeff Bezos in Seattle.", {
        "entities": [(0, 6, "ORG"), (21, 31, "PERSON"), (35, 42, "GPE")],
    }),
    ("Microsoft is based in Redmond, Washington.", {
        "entities": [(0, 9, "ORG"), (22, 29, "GPE"), (31, 41, "GPE")],
    }),
]

DEV_DATA = TRAIN_DATA[:2]   # reuse a slice; use a real held-out set in production

# ---------------------------------------------------------------------------
# Section 1 — Preprocessing
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  SECTION 1: TEXT PREPROCESSING")
print("=" * 60)

preprocessor = TextPreprocessor(
    PreprocessorConfig(remove_html_tags=True, remove_urls=True, normalize_whitespace=True)
)
raw = "<p>Visit https://apple.com — Apple  is a  great  company.</p>"
cleaned = preprocessor.process(raw)
print(f"Raw:     {raw!r}")
print(f"Cleaned: {cleaned!r}")

# ---------------------------------------------------------------------------
# Section 2 — Training
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  SECTION 2: TRAINING")
print("=" * 60)

trainer = NERTrainer(config=TrainerConfig(n_iter=15, eval_every=5, dropout=0.35))
training_report = trainer.train(TRAIN_DATA, dev_data=DEV_DATA, output_dir="models/demo_ner")
print(f"\nTraining complete. Best dev F1: {training_report['best_dev_f1']:.4f}")

# ---------------------------------------------------------------------------
# Section 3 — Inference
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  SECTION 3: INFERENCE")
print("=" * 60)

pipeline = NERPipeline.from_model_path("models/demo_ner")

single_text = "Satya Nadella leads Microsoft, which is based in Redmond."
result = pipeline.predict(single_text)
print(f"\nText: {result.text}")
print(f"Entities detected ({len(result.entities)}):")
for entity in result.entities:
    print(f"  [{entity.label:>10}]  {entity.text!r}  (chars {entity.start_char}–{entity.end_char})")

# Batch inference
print("\nBatch inference:")
batch_texts = [
    "Jeff Bezos founded Amazon in Seattle.",
    "Tesla was co-founded by Elon Musk.",
    "The United Nations is based in New York.",
]
for batch_result in pipeline.predict_batch(batch_texts):
    labels = [(e.label, e.text) for e in batch_result.entities]
    print(f"  {batch_result.text!r}  →  {labels}")

# ---------------------------------------------------------------------------
# Section 4 — Post-processing
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  SECTION 4: POST-PROCESSING")
print("=" * 60)

pipeline_filtered = NERPipeline.from_model_path(
    "models/demo_ner",
    postprocessor_config=PostprocessorConfig(
        allowed_labels={"ORG", "PERSON"},
        deduplicate=True,
    ),
)
filtered_result = pipeline_filtered.predict(single_text)
print(f"Filtered (ORG + PERSON only): {[(e.label, e.text) for e in filtered_result.entities]}")

# ---------------------------------------------------------------------------
# Section 5 — Evaluation
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  SECTION 5: EVALUATION")
print("=" * 60)

evaluator = NEREvaluator()
eval_report = evaluator.evaluate(DEV_DATA, pipeline)
print(eval_report)

# ---------------------------------------------------------------------------
# Section 6 — Entity Store
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  SECTION 6: ENTITY STORE")
print("=" * 60)

store = EntityStore(persist_path="entities.jsonl")

for text, _ in TRAIN_DATA:
    store.add(pipeline.predict(text))

print(f"\nStore: {store}")
print("\nTop ORGs:")
for text, count in store.top_entities("ORG", n=5):
    print(f"  {text}: {count}")
print("\nLabel distribution:", store.label_distribution())
store.to_json("entities_export.json")
print("\nEntities exported to entities_export.json")
print("\n✅  Demo complete.")