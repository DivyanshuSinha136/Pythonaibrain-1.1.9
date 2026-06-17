# GrammarCorrector — Production-grade Grammar Correction Pipeline

A three-tier grammar correction system combining SpaCy rule-based morphology, a scikit-learn statistical token corrector, and a PyTorch LSTM Seq2Seq neural model — all unified behind a single `GrammarCorrector` facade with a JSON intents system for training on real-world error pairs.

---

## What Is This?

`grammar.py` provides a complete grammar correction pipeline with three escalating tiers:

| Tier | Engine | Requires fitting? | Description |
|------|--------|-------------------|-------------|
| **1** | SpaCy + regex rules | No | Morphological rules, subject-verb agreement, capitalisation, confused-word pairs |
| **2** | Scikit-learn (Logistic Regression + DictVectorizer) | Yes | Statistical token-level correction trained on (corrupted, correct) pairs |
| **3** | PyTorch LSTM Seq2Seq | Yes | Neural fallback for out-of-vocabulary and complex patterns |

Tier 1 always runs. Tiers 2 and 3 activate after `fit()` and each falls back gracefully if unavailable.

**Additional features:**

- **JSON intents system** — load structured `{incorrect, correct, tag}` pairs from a file; two supported layouts (flat and grouped)
- **Auto-corruption engine** — 100+ deterministic corruption rules (homophones, slang, typos, contractions) to generate training data from plain correct sentences
- **BPE tokenizer** — trained from scratch on the training corpus via HuggingFace `tokenizers`
- **Context manager interface** — loads SpaCy on enter, releases all GPU/CPU resources on exit
- **`quick_correct()`** — one-liner convenience function for rule-only or full pipeline correction
- **CLI entry-point** — correct text from terminal with optional training, intents files, model save/load

---

## Installation

```bash
pip install torch spacy scikit-learn tokenizers nltk
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt averaged_perceptron_tagger
```

---

## How to Use

### 1. Rule-only correction (no fitting needed)

Tier 1 (SpaCy + rules) always runs — no training data required.

```python
from grammar import GrammarCorrector

with GrammarCorrector() as gc:
    print(gc.correct("i wants to here more about this"))
    # → "I wants to hear more about this."

    print(gc.correct("she go to the market"))
    # → "She goes to the market."
```

### 2. Full three-tier pipeline from a sentence list

```python
from grammar import GrammarCorrector

sentences = [
    "I am going to the market.",
    "She has a beautiful garden.",
    "They were playing football yesterday.",
]

with GrammarCorrector() as gc:
    gc.fit(sentences)
    print(gc.correct("i wants to here more"))
    print(gc.correct("she go to market"))
```

### 3. Train from a JSON intents file (recommended)

```python
from grammar import GrammarCorrector

with GrammarCorrector() as gc:
    gc.fit_from_intents_file("intents.json")
    print(gc.correct("i wants to here you"))
```

### 4. Train from intents with a tag filter

Only use specific error categories for training:

```python
from grammar import GrammarCorrector, load_intents

examples = load_intents("intents.json", tags=["verb_agreement", "capitalisation"])

with GrammarCorrector() as gc:
    gc.fit_from_intents(examples)
    print(gc.correct("he go home"))
```

### 5. Merge intents + plain sentences

```python
from grammar import GrammarCorrector, load_intents

sentences = ["The cat sat on the mat.", "I enjoy reading books."]
examples = load_intents("intents.json")

with GrammarCorrector() as gc:
    gc.fit(sentences, intents=examples)
    print(gc.correct("u r gr8"))
```

### 6. Save and load a trained model

```python
from grammar import GrammarCorrector

# Train and save
with GrammarCorrector() as gc:
    gc.fit_from_intents_file("intents.json")
    gc.save("models/grammar_v1")

# Load and use
with GrammarCorrector() as gc:
    gc.load("models/grammar_v1")
    print(gc.correct("she go to the market"))
```

### 7. One-liner convenience function

```python
from grammar import quick_correct

# Rule-only (no training data)
print(quick_correct("i go to school"))

# With training
sentences = ["She goes to school.", "I am happy."]
print(quick_correct("i go to school", sentences=sentences))
```

### 8. Custom configuration

```python
from grammar import GrammarCorrector, CorrectorConfig

cfg = CorrectorConfig(
    epochs=20,
    hidden_dim=512,
    num_layers=3,
    dropout=0.4,
    learning_rate=5e-4,
    vocab_size=3000,
    device="cuda",
)

with GrammarCorrector(cfg) as gc:
    gc.fit_from_intents_file("intents.json")
    print(gc.correct("they was at home"))
```

---

## JSON Intents File Format

Two supported layouts — freely mixable in the same file.

### Layout A — flat pair

```json
{
  "intents": [
    {
      "tag": "subject_verb_agreement",
      "incorrect": "she go to the market",
      "correct":   "she goes to the market"
    },
    {
      "tag": "pronoun_capitalisation",
      "incorrect": "i am here",
      "correct":   "I am here"
    }
  ]
}
```

### Layout B — grouped examples

```json
{
  "intents": [
    {
      "tag": "capitalisation",
      "examples": [
        { "incorrect": "i am here",  "correct": "I am here" },
        { "incorrect": "hi my name", "correct": "Hi my name" }
      ]
    }
  ]
}
```

**Rules:**
- `"tag"` is required; defaults to `"general"` if omitted
- Pairs where `incorrect == correct` are silently skipped
- Duplicate pairs (case-insensitive) are deduplicated automatically
- Both layouts may be mixed freely in the same file

### Loading and inspecting intents

```python
from grammar import load_intents, intents_summary, intents_to_pairs

# Load all
examples = load_intents("intents.json")

# Load with tag filter
examples = load_intents("intents.json", tags=["verb_agreement"])

# Load from a dict (useful for testing)
examples = load_intents({
    "intents": [{"tag": "test", "incorrect": "i go", "correct": "I go"}]
})

# Summary: {tag: count}
summary = intents_summary(examples)
print(summary)  # {"capitalisation": 5, "verb_agreement": 8}

# Convert to raw pairs
pairs = intents_to_pairs(examples)  # [("i go", "I go"), ...]
```

---

## API Reference

### `GrammarCorrector(config?)`

| Parameter | Type                      | Default | Description                              |
|-----------|---------------------------|---------|------------------------------------------|
| `config`  | `CorrectorConfig` or `None` | `None`  | Uses `CorrectorConfig()` defaults if `None` |

#### Methods

| Method | Returns | Description |
|---|---|---|
| `fit(sentences, intents?)` | `self` | Train all three tiers; auto-corrupts `sentences`; merges `intents` pairs |
| `correct(text)` | `str` | Apply all available tiers in order |
| `fit_from_intents(intents, sentences?, tags?)` | `self` | Train primarily from `IntentExample` list |
| `fit_from_intents_file(path, sentences?, tags?)` | `self` | Load JSON file and train in one call |
| `save(path)` | `None` | Save Seq2Seq weights + BPE tokenizer to a directory |
| `load(path)` | `self` | Restore weights + tokenizer from a directory |

All `fit*` methods return `self` for chaining:
```python
gc.fit_from_intents_file("intents.json").correct("she go to market")
```

---

### `CorrectorConfig` fields

| Field                   | Type    | Default | Description |
|-------------------------|---------|---------|-------------|
| `vocab_size`            | `int`   | `2000`  | BPE tokenizer vocabulary size |
| `special_tokens`        | `list`  | `["<s>", "</s>", "<unk>", "<pad>"]` | Special tokens |
| `emb_dim`               | `int`   | `128`   | Token embedding dimension |
| `hidden_dim`            | `int`   | `256`   | LSTM hidden layer size |
| `num_layers`            | `int`   | `2`     | Number of LSTM layers |
| `dropout`               | `float` | `0.3`   | Dropout rate |
| `epochs`                | `int`   | `10`    | Seq2Seq training epochs |
| `learning_rate`         | `float` | `1e-3`  | Adam optimizer learning rate |
| `teacher_forcing_ratio` | `float` | `0.5`   | Probability of using teacher forcing per step |
| `max_decode_len`        | `int`   | `120`   | Max tokens to generate during inference |
| `spacy_model`           | `str`   | `en_core_web_sm` | SpaCy model name |
| `device`                | `str`   | `"cuda"` if available else `"cpu"` | PyTorch device |

---

### Exception Hierarchy

```
GrammarCorrectorError
├── NotFittedError          — correct() called before fit()
├── TokenizerError          — BPE tokenizer missing special tokens
└── IntentsValidationError  — JSON intents file fails schema validation
```

---

### Data Helpers

```python
from grammar import corrupt_sentence, build_dataset

# Apply deterministic corruption rules to one sentence
noisy = corrupt_sentence("She goes to the market.")
# → "She go 2 the market."

# Build (corrupted, correct) pairs from a list of correct sentences
dataset = build_dataset(["I am happy.", "She goes to school."])
# → [("i'm happy.", "I am happy."), ...]
```

The corruption engine covers 100+ rules across five categories: homophones (`their/there/they're`), internet slang (`u/r/gr8/lol`), informal contractions (`wanna/gonna/gotta`), common typos (`teh/freind/recieve`), and punctuation corruption.

---

### Low-level API

These are available for advanced use when you want direct access to individual tiers or the training loop:

```python
from grammar import (
    rule_based_corrector,       # Tier 1 standalone
    train_tokenizer,            # BPE tokenizer training
    Seq2Seq, Encoder, Decoder,  # Tier 3 PyTorch models
    train_model,                # Tier 3 training loop
    correct_sentence_neural,    # Tier 3 greedy inference
)
import spacy

nlp = spacy.load("en_core_web_sm")
corrected = rule_based_corrector("she go to the market", nlp)
print(corrected)  # "She goes to the market."
```

---

## CLI Usage

```bash
python grammar.py [text] [options]
```

### Flags

| Flag | Description |
|---|---|
| `text` | Text to correct (reads stdin if omitted) |
| `--train-file FILE` | Plain-text file with one correct sentence per line |
| `--intents-file JSON` | JSON intents file with incorrect/correct pairs |
| `--tags TAG [TAG ...]` | Filter intents to specific tags |
| `--show-intents-summary` | Print `{tag: count}` breakdown and exit |
| `--epochs N` | Training epochs (default: `10`) |
| `--device DEVICE` | PyTorch device: `cpu` or `cuda` (default: `cpu`) |
| `--save-model DIR` | Save trained model to directory after correction |
| `--load-model DIR` | Load a pre-trained model instead of training |
| `--verbose` / `-v` | Enable debug logging |

### CLI Examples

```bash
# Rule-only correction (no training)
python grammar.py "i wants to here you"

# Train from intents file and correct
python grammar.py "she go to market" --intents-file intents.json

# Tag-filtered training
python grammar.py "u r gr8" --intents-file intents.json --tags capitalisation verb_agreement

# Train from plain sentences file
python grammar.py "i go home" --train-file sentences.txt --epochs 20

# Load saved model and correct
python grammar.py "he dont know" --load-model models/grammar_v1

# Train, correct, and save
python grammar.py "she go school" --intents-file intents.json --save-model models/v1

# Show intents tag summary
python grammar.py --intents-file intents.json --show-intents-summary

# Pipe from stdin
echo "i wants to here you" | python grammar.py --intents-file intents.json
```

---

## Architecture Overview

```
GrammarCorrector.correct(text)
        │
        ▼
┌───────────────────────────────┐
│  Tier 1 — SpaCy + Regex       │  Always runs
│  • Lowercase "i" → "I"        │
│  • Sentence capitalisation    │
│  • Subject-verb agreement     │
│  • Confused-word pairs        │
└───────────────┬───────────────┘
                │
                ▼ (if fitted)
┌───────────────────────────────┐
│  Tier 2 — Sklearn Pipeline    │  Runs after fit()
│  • DictVectorizer features    │
│  • OneVsRest LogisticReg.     │
│  • Token-level sequence fix   │
└───────────────┬───────────────┘
                │
                ▼ (if fitted)
┌───────────────────────────────┐
│  Tier 3 — LSTM Seq2Seq        │  Runs after fit()
│  • BPE tokenizer              │
│  • Encoder (embedding + LSTM) │
│  • Decoder (teacher forcing)  │
│  • Greedy inference           │
└───────────────────────────────┘
```

---

## Examples Summary

```python
from grammar import GrammarCorrector, load_intents, quick_correct

# Rule-only (no fitting)
with GrammarCorrector() as gc:
    print(gc.correct("i go to school"))

# Full pipeline from sentences
with GrammarCorrector() as gc:
    gc.fit(["She goes to school.", "I am happy."])
    print(gc.correct("she go to school"))

# From intents file
with GrammarCorrector() as gc:
    gc.fit_from_intents_file("intents.json", tags=["verb_agreement"])
    print(gc.correct("he dont know"))

# Save and reload
with GrammarCorrector() as gc:
    gc.fit_from_intents_file("intents.json")
    gc.save("models/v1")

with GrammarCorrector() as gc:
    gc.load("models/v1")
    print(gc.correct("they was playing"))

# One-liner
print(quick_correct("i go home"))
```