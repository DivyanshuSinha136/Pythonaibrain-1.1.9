# NER System — Production-grade Named Entity Recognition

A complete, modular Named Entity Recognition system built on spaCy. Covers the full ML lifecycle: text preprocessing, inference, postprocessing, training with early stopping, evaluation with precision/recall/F1, and persistent entity storage.

---

## What Is This?

This package is a production-quality NER system with six cooperating modules:

| Module               | Class / Function       | Responsibility                                          |
|----------------------|------------------------|---------------------------------------------------------|
| `pipeline.py`        | `NERPipeline`          | Core inference — single and batch prediction            |
| `trainer.py`         | `NERTrainer`           | Training loop with early stopping and model saving      |
| `evaluator.py`       | `NEREvaluator`         | Precision / Recall / F1 metrics (exact + partial match) |
| `preprocessor.py`    | `TextPreprocessor`     | Cleans and normalises raw text before inference         |
| `postprocessor.py`   | `EntityPostprocessor`  | Filters, deduplicates, and enriches entity outputs      |
| `entity_store.py`    | `EntityStore`          | In-memory store with JSONL persistence and analytics    |
| `logging_config.py`  | `setup_logging()`      | Configures root logger (console + optional file)        |

---

## Installation

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

## Quick Start

```python
from ner import NERPipeline, EntityStore

# Load a trained model
pipeline = NERPipeline.from_model_path("models/my_ner")

# Predict
result = pipeline.predict("Apple was founded by Steve Jobs in California.")
for entity in result.entities:
    print(entity.label, entity.text)
# ORG   Apple
# PERSON Steve Jobs
# GPE   California

# Store results
store = EntityStore(persist_path="entities.jsonl")
store.add(result)
print(store.top_entities("ORG", n=5))
```

---

## Module Guide

### NERPipeline — Inference

```python
from ner import NERPipeline

# From a saved model on disk
pipeline = NERPipeline.from_model_path("models/my_ner")

# From a blank model (for training or rule-based use)
pipeline = NERPipeline.from_blank(lang="en", labels=["ORG", "PERSON", "GPE"])

# Single prediction
result = pipeline.predict("Google acquired DeepMind in 2014.")
print(result.entities)                    # list of Entity objects
print(result.entity_types)               # ['DATE', 'ORG']
print(result.filter_by_label("ORG"))     # [Entity(text='Google', ...)]
print(result.processing_time_ms)         # e.g. 12.4

# Batch prediction (lazy generator, memory-efficient)
texts = ["Text one.", "Text two.", "Text three."]
for result in pipeline.predict_batch(texts):
    print(result.entities)

# Save model to disk
pipeline.save("models/my_ner_v2")

# Inspect labels
print(pipeline.labels)   # ['DATE', 'GPE', 'ORG', 'PERSON']
```

**Constructor options:**

| Parameter       | Type                      | Default | Description                                        |
|-----------------|---------------------------|---------|----------------------------------------------------|
| `nlp`           | `spacy.Language`          | *(required)* | Loaded spaCy model                            |
| `preprocessor`  | `TextPreprocessor`        | `None`  | Custom preprocessor; defaults to standard config   |
| `postprocessor` | `EntityPostprocessor`     | `None`  | Custom postprocessor; defaults to standard config  |
| `return_doc`    | `bool`                    | `False` | Include raw spaCy `Doc` in results                 |
| `batch_size`    | `int`                     | `64`    | Chunk size for `nlp.pipe` in batch mode            |

---

### NERTrainer — Training

```python
from ner import NERTrainer
from ner.trainer import TrainerConfig

TRAIN_DATA = [
    ("Apple was founded by Steve Jobs.", {"entities": [(0, 5, "ORG"), (21, 31, "PERSON")]}),
    ("Google is based in Mountain View.", {"entities": [(0, 6, "ORG"), (19, 32, "GPE")]}),
]
DEV_DATA = [...]

# Train from scratch
trainer = NERTrainer(config=TrainerConfig(n_iter=30, dropout=0.35))
trainer.prepare(TRAIN_DATA)
results = trainer.train(TRAIN_DATA, dev_data=DEV_DATA, output_dir="models/v1")

print(results["best_dev_f1"])   # e.g. 0.8712
print(results["history"])       # list of per-epoch loss + dev scores

# Fine-tune from an existing spaCy model
trainer = NERTrainer(base_model="en_core_web_sm")
trainer.train(TRAIN_DATA, output_dir="models/v1_finetuned")

# Export training data to spaCy v3 DocBin format
import spacy
nlp = spacy.blank("en")
NERTrainer.data_to_docbin(TRAIN_DATA, nlp, "train.spacy")
```

**`TrainerConfig` fields:**

| Field            | Default | Description                                              |
|------------------|---------|----------------------------------------------------------|
| `lang`           | `"en"`  | spaCy language code for blank model                      |
| `n_iter`         | `30`    | Maximum training epochs                                  |
| `dropout`        | `0.35`  | Dropout rate during training                             |
| `batch_start`    | `4`     | Starting batch size for compounding scheduler            |
| `batch_compound` | `1.001` | Compounding factor for batch size growth                 |
| `eval_every`     | `5`     | Evaluate on dev set every N epochs                       |
| `patience`       | `5`     | Early stopping patience (epochs without improvement)     |
| `min_delta`      | `0.001` | Minimum F1 improvement to reset patience counter         |
| `seed`           | `42`    | Random seed for reproducibility                          |

---

### NEREvaluator — Metrics

```python
from ner import NEREvaluator, NERPipeline

pipeline = NERPipeline.from_model_path("models/my_ner")
evaluator = NEREvaluator()          # exact match (default)
evaluator_partial = NEREvaluator(partial=True)   # any span overlap counts

gold_data = [
    ("Apple was founded by Steve Jobs.", {"entities": [(0, 5, "ORG"), (21, 31, "PERSON")]}),
]

report = evaluator.evaluate(gold_data, pipeline)
print(report)
# ============================================================
#   Micro  P=0.9200  R=0.8750  F1=0.8970
#   Macro F1 = 0.8800
# ------------------------------------------------------------
# Label                P        R       F1     TP     FP     FN
# ------------------------------------------------------------
# ORG             0.9500   0.9000   0.9245    ...
# PERSON          0.8100   0.8500   0.8295    ...
# ============================================================

print(report.to_dict())   # structured dict for logging / serialisation

# Evaluate pre-computed predictions (no pipeline needed)
gold_spans_list = [[(0, 5, "ORG"), (21, 31, "PERSON")]]
pred_spans_list = [[(0, 5, "ORG"), (21, 31, "PERSON")]]
report = evaluator.evaluate_from_predictions(gold_spans_list, pred_spans_list)
```

**`EvaluationReport` properties:**

| Property          | Description                                     |
|-------------------|-------------------------------------------------|
| `micro_precision` | TP / (TP + FP) across all labels               |
| `micro_recall`    | TP / (TP + FN) across all labels               |
| `micro_f1`        | Harmonic mean of micro precision and recall     |
| `macro_f1`        | Unweighted average F1 across all labels         |
| `per_label`       | Dict of `LabelScore` objects keyed by label     |

---

### TextPreprocessor — Input Cleaning

```python
from ner import TextPreprocessor
from ner.preprocessor import PreprocessorConfig

# Default config
preprocessor = TextPreprocessor()
clean = preprocessor.process("<b>Visit https://example.com for info.</b>")
# → "Visit for info."

# Custom config
config = PreprocessorConfig(
    lowercase=True,
    remove_urls=True,
    remove_emails=True,
    remove_html_tags=True,
    normalize_whitespace=True,
    normalize_unicode=True,
    max_length=512,
    custom_patterns=[r"\d{4}-\d{2}-\d{2}"],   # strip ISO dates
)
preprocessor = TextPreprocessor(config)

# Batch
cleaned_texts = preprocessor.process_batch(["Text one.", "Text two."])
```

**`PreprocessorConfig` fields:**

| Field                | Default | Description                                         |
|----------------------|---------|-----------------------------------------------------|
| `lowercase`          | `False` | Lowercase the entire text                           |
| `remove_urls`        | `True`  | Remove `http://`, `https://`, and `www.` URLs       |
| `remove_emails`      | `False` | Remove email addresses (kept by default as NER signal) |
| `remove_html_tags`   | `True`  | Strip HTML tags                                     |
| `normalize_whitespace` | `True` | Collapse all whitespace to single spaces           |
| `normalize_unicode`  | `True`  | NFC-normalise Unicode                               |
| `max_length`         | `None`  | Truncate text to this many characters               |
| `custom_patterns`    | `[]`    | List of regex strings to remove                     |

---

### EntityPostprocessor — Output Cleaning

```python
from ner import EntityPostprocessor
from ner.postprocessor import PostprocessorConfig

config = PostprocessorConfig(
    min_length=2,
    max_length=100,
    allowed_labels={"ORG", "PERSON", "GPE"},
    blocked_labels={"MISC"},
    deduplicate=True,
    merge_adjacent=True,
    strip_punct=True,
    custom_label_map={"PERSON": "PER"},   # rename labels
)
postprocessor = EntityPostprocessor(config)
entities = postprocessor.process(entities)
```

**Processing chain (in order):**

1. Filter by text length (`min_length` / `max_length`)
2. Filter by label (`allowed_labels` / `blocked_labels`)
3. Strip leading/trailing punctuation from entity text (`strip_punct`)
4. Remap label names (`custom_label_map`)
5. Lowercase labels (`lowercase_labels`)
6. Deduplicate identical spans (`deduplicate`)
7. Merge back-to-back same-label spans (`merge_adjacent`, gap ≤ 1 char)

---

### EntityStore — Persistence & Analytics

```python
from ner import EntityStore

# In-memory only
store = EntityStore()

# With JSONL persistence (appends on each add; loads on init if file exists)
store = EntityStore(persist_path="entities.jsonl")

store.add(result)                        # add a single NERResult
store.add_batch([result1, result2])      # add multiple results

# Query
store.query(label="ORG")                             # all ORG records
store.query(text_contains="apple", min_score=0.8)    # filtered
store.unique_entities(label="PERSON")                # sorted unique names
store.top_entities("ORG", n=10)                      # [(entity, count), ...]
store.label_distribution()                           # {"ORG": 42, "PERSON": 31}

# Export
store.to_json("all_entities.json")

# Iterate
for record in store.iter_records():
    print(record["text"], record["label"])

print(len(store))   # total entity count
print(store)        # EntityStore(total=132, labels={'ORG': 42, 'PERSON': 31})
```

**Each stored record contains:**

| Field         | Description                            |
|---------------|----------------------------------------|
| `text`        | Entity surface form                    |
| `label`       | Entity type (e.g. `ORG`, `PERSON`)     |
| `start_char`  | Start character offset in source text  |
| `end_char`    | End character offset in source text    |
| `score`       | Detection confidence (0.0–1.0)         |
| `source_text` | The original text the entity came from |

---

### Logging Setup

```python
from ner.logging_config import setup_logging

setup_logging(level="DEBUG", log_file="ner.log")
```

| Parameter | Default                                      | Description                           |
|-----------|----------------------------------------------|---------------------------------------|
| `level`   | `"INFO"`                                     | Logging level                         |
| `log_file`| `None`                                       | Optional path for file output         |
| `fmt`     | `"%(asctime)s [%(levelname)s] %(name)s — …"` | Log record format string              |

---

## Full End-to-End Example

```python
from ner import NERPipeline, NERTrainer, NEREvaluator, EntityStore
from ner.trainer import TrainerConfig
from ner.logging_config import setup_logging

setup_logging(level="INFO", log_file="ner.log")

# 1. Prepare data
TRAIN_DATA = [
    ("Apple was founded by Steve Jobs in California.", {
        "entities": [(0, 5, "ORG"), (21, 31, "PERSON"), (35, 45, "GPE")]
    }),
]
DEV_DATA = TRAIN_DATA   # use your actual dev split

# 2. Train
trainer = NERTrainer(config=TrainerConfig(n_iter=20, eval_every=5))
results = trainer.train(TRAIN_DATA, dev_data=DEV_DATA, output_dir="models/v1")
print("Best dev F1:", results["best_dev_f1"])

# 3. Load and predict
pipeline = NERPipeline.from_model_path("models/v1")
result = pipeline.predict("Microsoft acquired GitHub in 2018.")
print(result.entities)

# 4. Evaluate
evaluator = NEREvaluator()
report = evaluator.evaluate(DEV_DATA, pipeline)
print(report)

# 5. Store
store = EntityStore(persist_path="entities.jsonl")
store.add(result)
print(store.top_entities("ORG"))
print(store.label_distribution())
```

---

## Data Format

Training and evaluation data uses spaCy's standard annotation format:

```python
[
    ("Text to annotate.", {"entities": [(start, end, "LABEL"), ...]}),
    ("Google is in Mountain View.", {"entities": [(0, 6, "ORG"), (13, 26, "GPE")]}),
]
```

- `start` / `end` are character offsets (not token indices)
- Spans must not overlap
- Labels are arbitrary strings registered via `NERTrainer.prepare()` or `NERPipeline.from_blank()`
