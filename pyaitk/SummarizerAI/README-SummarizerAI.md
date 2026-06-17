# SummarizerAI — AI Memory Summarization System

A complete ML pipeline that analyses, clusters, and generates structured reports from Brain conversation memory. Converts a raw `{input → response}` memory dictionary into semantic clusters, intent labels, coherence scores, and actionable recommendations — all in a single `fit()` call.

---

## What Is This?

The SummarizerAI package is a five-module ML system used internally by `SmartMemory` to understand and organise conversation history:

| Module | Class / Function | Responsibility |
|---|---|---|
| `summarizer.py` | `MemorySummarizer` | Orchestrates the full 7-step pipeline; produces `MemorySummaryReport` |
| `data_pipeline.py` | `MemoryLoader`, `TextFeatureExtractor` | Cleans patterns, infers intent/response type, extracts TF-IDF features |
| `autoencoder.py` | `MemoryAutoencoder`, `AutoencoderTrainer` | PyTorch AE: compresses TF-IDF → dense latent embeddings |
| `clustering.py` | `MemoryClusterer`, `IntentClassifier`, `PatternMatcher` | Clusters embeddings; trains LR classifier; fuzzy fallback matcher |
| `settings.py` | `SystemConfig`, `DEFAULT_CONFIG` | All tuneable hyperparameters, reads from `config.pbcfg` |

---

## Installation

```bash
pip install torch scikit-learn numpy pandas scipy
```

---

## Quick Start

```python
from SummarizerAI import MemorySummarizer

# Raw memory: {input_text: response_text}
memory = {
    "Hello": "Hi! How can I help you?",
    "Hey there": "Hello! What can I do for you?",
    "Tell me a joke": "Why do programmers prefer dark mode? Because light attracts bugs!",
    "What is Python?": "Python is a high-level programming language known for simplicity.",
    "Goodbye": "See you later! Come back soon.",
    # … more entries …
}

summarizer = MemorySummarizer()
summarizer.fit(memory)

# Print the full report to stdout
summarizer.print_report()

# Export as JSON
summarizer.export_report("memory_report.json")

# Query routing
result = summarizer.query("what does python do")
print(result["predicted_intent"])      # "knowledge_query"
print(result["intent_confidence"])     # 0.912
print(result["cluster_id"])            # 2
print(result["top_matches"])           # [{"score": 0.87, "input": "...", "response": "..."}]
```

---

## Pipeline Overview

`MemorySummarizer.fit()` runs seven sequential steps:

```
Step 1  MemoryLoader          raw dict → List[MemoryPattern]
           │  clean_text(), infer_intent(), infer_response_type()
           ▼
Step 2  TextFeatureExtractor  patterns → TF-IDF sparse matrix
           │  char-level n-grams (1-3), L2-normalised
           ▼
Step 3  AutoencoderTrainer    TF-IDF → dense latent embeddings
           │  PyTorch: Encoder (TF-IDF dim → latent_dim) + Decoder
           │  AdamW + CosineAnnealingLR + gradient clipping
           ▼
Step 4  MemoryClusterer       embeddings → cluster labels
           │  auto k-selection via silhouette score
           │  KMeans | Agglomerative | DBSCAN
           ▼
Step 5  IntentClassifier      TF-IDF + intent labels → LR classifier
           │  sklearn LogisticRegression (OvR, lbfgs)
           ▼
Step 6  PatternMatcher        TF-IDF matrix → cosine similarity index
           │  fuzzy fallback for low-confidence queries
           ▼
Step 7  _build_report()       → MemorySummaryReport
              ClusterSummary × n_clusters + SystemStats + Recommendations
```

---

## Module Guide

### `MemorySummarizer` — Orchestrator

```python
from SummarizerAI import MemorySummarizer

# Default config (reads from config.pbcfg / DEFAULT_CONFIG)
summarizer = MemorySummarizer()

# Custom config overrides
summarizer = MemorySummarizer(config={
    "latent_dim": 64,
    "hidden_dim": 256,
    "ae_epochs": 50,
    "ae_lr": 5e-4,
    "ae_batch_size": 16,
    "device": "cuda",          # "cpu" or "cuda"
    "embedding": {
        "max_features": 8000,
        "ngram_range": (1, 3),
        "sublinear_tf": True,
    },
})

# Fit the full pipeline
summarizer.fit(memory_dict)

# Query a new input through the pipeline
result = summarizer.query("tell me a joke")
# {
#   "input": "tell me a joke",
#   "predicted_intent": "humor",
#   "intent_confidence": 0.934,
#   "cluster_id": 1,
#   "top_matches": [
#     {"score": 0.91, "input": "say a joke", "response": "Why do ..."},
#     {"score": 0.82, "input": "tell me something funny", "response": "..."},
#   ]
# }

# Print full report
summarizer.print_report()

# Export report to JSON
summarizer.export_report("report.json")

# Access report object
report = summarizer.report_
print(report.system_stats.n_clusters)
print(report.system_stats.intent_distribution)
print(report.recommendations)
```

#### State after `fit()`

| Attribute | Type | Description |
|---|---|---|
| `patterns_` | `List[MemoryPattern]` | All cleaned and annotated patterns |
| `embeddings_` | `np.ndarray` | Dense latent embeddings, shape `(n, latent_dim)` |
| `tfidf_matrix_` | sparse matrix | Raw TF-IDF feature matrix |
| `report_` | `MemorySummaryReport` | Full structured report |

---

### `data_pipeline.py` — Data Layer

#### `MemoryLoader`

Converts a raw `{str: str}` dict into typed `MemoryPattern` objects.

```python
from SummarizerAI.core.data_pipeline import MemoryLoader

loader = MemoryLoader(
    skip_empty_keys=True,   # skip patterns with empty input
    skip_errors=False,      # include error-type responses
)

# From a dict
patterns = loader.from_dict({"Hello": "Hi!", "Bye": "Goodbye!"})

# From a JSON file
patterns = loader.from_json_file("memory.json")

# As a DataFrame
df = loader.to_dataframe(patterns)
print(df.columns.tolist())
# ['input_text', 'response_text', 'input_clean', 'response_clean',
#  'intent_tag', 'response_type', 'word_count', 'has_error',
#  'is_command', 'is_open_action']
```

#### `MemoryPattern` fields

| Field | Type | Description |
|---|---|---|
| `input_text` | `str` | Raw input key from memory |
| `response_text` | `str` | Raw response value from memory |
| `input_clean` | `str` | Lowercased, punctuation-stripped input |
| `response_clean` | `str` | Lowercased, punctuation-stripped response |
| `intent_tag` | `str` | Inferred broad intent (see Intent Types table) |
| `response_type` | `str` | Inferred response category (see Response Types table) |
| `word_count` | `int` | Number of words in `input_text` |
| `has_error` | `bool` | `True` if response is classified as an error |
| `is_command` | `bool` | `True` if response triggers a system command |
| `is_open_action` | `bool` | `True` if response opens an external resource |

#### Intent types (auto-inferred)

| Label | Trigger signals |
|---|---|
| `greeting` | hi, hello, hey, yo, greet |
| `farewell` | bye, goodbye, ok bye |
| `humor` | joke, jock, funny |
| `identity_query` | who, name, created, founder, age, you |
| `command_trigger` | cmd, start, command |
| `knowledge_query` | what is, tell me about, python, pythonaibrain |
| `temporal_query` | time, date |
| `acknowledgment` | thanks, great, done, ok, well done |
| `general` | *(fallback)* |

#### Response types (auto-inferred)

| Label | Detection rule |
|---|---|
| `greeting` | response contains hello, hey, hi!, greet |
| `joke` | response contains joke, parrot, vim, recursion, capitals |
| `command` | response starts with `done:` or contains `cmd_start` |
| `open_action` | response starts with `['open'` |
| `error` | response starts with `[error]` or `error:` |
| `info` | response length > 120 characters |
| `farewell` | response contains goodbye, bye, come back |
| `processing` | response contains wait, moment, checking |
| `generic_reply` | *(fallback)* |

#### `TextFeatureExtractor`

```python
from SummarizerAI.core.data_pipeline import TextFeatureExtractor

extractor = TextFeatureExtractor(config={
    "max_features": 5000,
    "ngram_range": (1, 3),
    "sublinear_tf": True,
})

matrix = extractor.fit_transform(patterns)  # sparse matrix, L2-normalised
matrix = extractor.transform(new_patterns)  # after fit, for new data
```

Text representation: `"{input_clean} [SEP] {response_clean}"` — combines both sides for richer character n-gram features. Uses `analyzer="char_wb"` to handle typos and informal text.

---

### `autoencoder.py` — Neural Compression

#### `MemoryAutoencoder` architecture

```
Input (TF-IDF dim)
    │
MemoryEncoder
    Linear(input_dim → hidden_dim) → LayerNorm → GELU → Dropout(0.1)
    Linear(hidden_dim → hidden_dim // 2) → LayerNorm → GELU
    Linear(hidden_dim // 2 → latent_dim)
    │
  z  (dense embedding, shape: latent_dim)
    │
MemoryDecoder
    Linear(latent_dim → hidden_dim // 2) → GELU
    Linear(hidden_dim // 2 → hidden_dim) → GELU
    Linear(hidden_dim → input_dim) → Sigmoid
    │
Reconstructed TF-IDF
```

#### `AutoencoderTrainer`

```python
from SummarizerAI.models.autoencoder import AutoencoderTrainer

trainer = AutoencoderTrainer(
    latent_dim=64,
    hidden_dim=256,
    epochs=30,
    lr=1e-3,
    batch_size=16,
    device="cpu",    # or "cuda"
)

trainer.fit(X_dense)                          # numpy array, shape (n, features)
embeddings = trainer.get_embeddings(X_dense)  # shape (n, latent_dim)

# Save and load weights
trainer.save("models/checkpoints/ae.pt")
trainer.load("models/checkpoints/ae.pt")

# Inspect training curve
print(trainer.train_losses)   # list of avg MSE loss per epoch
```

**Training configuration:** AdamW optimiser, CosineAnnealingLR scheduler, gradient clipping at 1.0, MSELoss reconstruction objective.

---

### `clustering.py` — Clustering, Classification, Matching

#### `MemoryClusterer`

```python
from SummarizerAI.core.clustering import MemoryClusterer

# Auto-select best k via silhouette score (default, recommended)
clusterer = MemoryClusterer(method="auto", n_clusters=8)

# Or force a specific method
clusterer = MemoryClusterer(method="kmeans",        n_clusters=6)
clusterer = MemoryClusterer(method="agglomerative", n_clusters=6)
clusterer = MemoryClusterer(method="dbscan")        # DBSCAN (no k needed)

clusterer.fit(embeddings)          # embeddings: np.ndarray (n, latent_dim)
labels = clusterer.labels_         # np.ndarray of int cluster assignments
print(clusterer.n_clusters_found)  # number of clusters (excl. DBSCAN noise -1)

# Assign new points to nearest centroid
new_labels = clusterer.predict(new_embeddings)
```

**Auto k-selection:** grid searches `k ∈ [3, min(12, n-1)]`, picks highest silhouette score. Falls back to KMeans with the best k found.

#### `IntentClassifier`

```python
from SummarizerAI.core.clustering import IntentClassifier

clf = IntentClassifier(C=1.0, max_iter=500)
clf.fit(tfidf_matrix, intent_labels)      # List[str] of intent tags

# Single prediction
predictions = clf.predict(tfidf_matrix)   # List[str]

# Probability distribution for one sample
probs = clf.predict_proba(vec)            # {"greeting": 0.82, "farewell": 0.03, …}
top = max(probs, key=probs.get)

# Accuracy on labelled data
acc = clf.score(tfidf_matrix, intent_labels)
print(clf.classes_)   # list of all known intent labels
```

#### `PatternMatcher` (fuzzy fallback)

```python
from SummarizerAI.core.clustering import PatternMatcher

matcher = PatternMatcher(threshold=0.65)
matcher.index(tfidf_matrix, patterns)   # index the corpus

# Query with a TF-IDF vector
results = matcher.query(query_vec, top_k=3)
# results: [(score: float, pattern: MemoryPattern), ...]
for score, pattern in results:
    print(f"{score:.3f}  {pattern.input_text!r}  →  {pattern.response_text!r}")
```

Results only include patterns where cosine similarity ≥ `threshold`. Used as a fallback when the classifier confidence is low.

---

### `settings.py` — Configuration

```python
from SummarizerAI.settings import SystemConfig, DEFAULT_CONFIG

# Use the singleton
cfg = DEFAULT_CONFIG
print(cfg.summarizer.latent_dim)        # 64
print(cfg.clustering.n_clusters)        # 8
print(cfg.embedding.tfidf_max_features) # 5000
print(cfg.device)                       # "cpu"

# Build a custom config
from SummarizerAI.settings import (
    EmbeddingConfig, ClusteringConfig,
    ClassifierConfig, SummarizerConfig, SystemConfig,
)

cfg = SystemConfig(
    embedding=EmbeddingConfig(tfidf_max_features=8000, tfidf_ngram_range=(1, 3)),
    clustering=ClusteringConfig(n_clusters=10, dbscan_eps=0.4),
    classifier=ClassifierConfig(lr_C=0.5, similarity_threshold=0.70),
    summarizer=SummarizerConfig(latent_dim=128, hidden_dim=512, ae_epochs=60),
    device="cuda",
    model_save_dir="models/v2",
    log_level="DEBUG",
)
```

**All settings cascade from `config.pbcfg` via `get_config()`** — the same `.pbcfg` file used by the rest of the pyaitk framework. See `README-Config.md` for the full field reference.

#### Config section reference

| Section | Class | Key fields |
|---|---|---|
| `cfg.embedding` | `EmbeddingConfig` | `tfidf_max_features`, `tfidf_ngram_range`, `tfidf_sublinear_tf`, `embed_dim`, `vocab_size` |
| `cfg.clustering` | `ClusteringConfig` | `n_clusters`, `kmeans_max_iter`, `dbscan_eps`, `dbscan_min_samples`, `agglo_linkage` |
| `cfg.classifier` | `ClassifierConfig` | `lr_C`, `lr_max_iter`, `lr_solver`, `similarity_threshold` |
| `cfg.summarizer` | `SummarizerConfig` | `latent_dim`, `hidden_dim`, `ae_epochs`, `ae_lr`, `ae_batch_size`, `top_patterns_per_cluster`, `min_cluster_size` |

---

## Report Reference

### `MemorySummaryReport`

```python
report = summarizer.report_

print(report.title)                          # "Memory Summary Report"
print(report.recommendations)                # List[str] of actionable insights
print(report.raw_cluster_map)                # {cluster_id: [input_key, ...]}

for cs in report.cluster_summaries:
    print(cs.cluster_id)                     # int
    print(cs.dominant_intent)                # e.g. "knowledge_query"
    print(cs.pattern_count)                  # int
    print(cs.coherence_score)                # float 0.0–1.0
    print(cs.response_types)                 # {"info": 5, "greeting": 2}
    print(cs.key_tokens)                     # ["python", "what", "tell", ...]
    print(cs.representative_inputs)          # 3 shortest/most typical inputs
    print(cs.representative_responses)       # corresponding responses
    print(cs.description)                    # human-readable paragraph

s = report.system_stats
print(s.total_patterns)                      # int
print(s.unique_intents)                      # List[str]
print(s.intent_distribution)                 # {"greeting": 4, "humor": 2, ...}
print(s.response_type_distribution)          # {"generic_reply": 8, "info": 5, ...}
print(s.error_count)                         # int
print(s.command_count)                       # int
print(s.open_action_count)                   # int
print(s.avg_input_length)                    # float (characters)
print(s.n_clusters)                          # int
print(s.autoencoder_final_loss)              # float (last epoch MSE)
```

### Automatic recommendations

The report always ends with a `recommendations` list. Rules:

| Condition | Recommendation |
|---|---|
| `error_count > 0` | Add fallback responses for offline/API failure patterns |
| `coherence_score < 0.2` for any cluster | Split cluster or add varied training examples |
| `pattern_count ≤ 2` for any cluster | Expand with more input variations |
| `open_action_count > 0` | Verify external links/documents are still accessible |
| Humor cluster has > 5 patterns | Add randomisation to avoid repetitive jokes |
| `n_clusters > 10` | Consolidate similar intents for a leaner memory footprint |
| None of the above | "✅ Memory patterns look well-structured and coherent!" |

---

## Print Report Output

```
══════════════════════════════════════════════════════════════════════
  Memory Summary Report
══════════════════════════════════════════════════════════════════════

📊  SYSTEM STATS
   Total patterns : 42
   Unique intents : greeting, farewell, humor, knowledge_query, general
   Clusters found : 5
   AE final loss  : 0.002341
   Errors/Cmds    : 0 / 3
   Intent dist    : {"greeting": 8, "knowledge_query": 14, ...}

🗂️   CLUSTER SUMMARIES (5 clusters)

  ┌─ Cluster #0  [greeting]  (8 patterns, coherence=0.7231)
  │  This cluster handles greeting and opening exchanges. ...
  │  Response types: {"greeting": 8}
  │  Key tokens: ['hello', 'hey', 'greet', 'hi', 'morning']
  │    • 'Hello'  →  'Hi! How can I help you?'
  └────────────────────────────────────────────────────────────────

💡  RECOMMENDATIONS
   ✅  Memory patterns look well-structured and coherent!
══════════════════════════════════════════════════════════════════════
```

---

## Integration with `SmartMemory`

`MemorySummarizer` is the ML engine behind `SmartMemory`. You do not need to call it directly when using `Brain` or `SmartMemory` — it is invoked automatically:

```python
from Memory import SmartMemory

mem = SmartMemory(path="memory.json", auto_fit=True, fit_interval=20)

# Every 20 remember() calls → MemorySummarizer.fit() runs in background thread
mem.remember("What is Python?", "Python is a programming language.")

# Semantic search uses autoencoder embeddings + PatternMatcher
results = mem.semantic_search("tell me about python", top_k=3)

# Intent prediction uses IntentClassifier
intent = mem.predict_intent("tell me a joke")

# Get / export the report
mem.print_report()
mem.export_report("report.json")
```

For direct use, import from the package root:

```python
from SummarizerAI import MemorySummarizer
```

---

## Architecture Overview

```
SummarizerAI
│
├── settings.py          ← SystemConfig (EmbeddingConfig + ClusteringConfig +
│                           ClassifierConfig + SummarizerConfig), reads config.pbcfg
│
├── summarizer.py        ← MemorySummarizer (7-step orchestrator)
│   ├── report objects   ← MemorySummaryReport, ClusterSummary, SystemStats
│   └── query()         ← routes new input through all fitted components
│
├── core/
│   ├── data_pipeline.py
│   │   ├── MemoryPattern     ← typed domain object per input/response pair
│   │   ├── MemoryLoader      ← dict/JSON → List[MemoryPattern] with auto-labelling
│   │   └── TextFeatureExtractor  ← char n-gram TF-IDF → L2-normalised sparse matrix
│   │
│   └── clustering.py
│       ├── MemoryClusterer   ← auto k-select silhouette → KMeans/Agglo/DBSCAN
│       ├── IntentClassifier  ← sklearn LR (OvR, lbfgs) → intent label
│       └── PatternMatcher    ← cosine similarity index → fuzzy top-k lookup
│
└── models/
    └── autoencoder.py
        ├── MemoryEncoder     ← Linear → LayerNorm → GELU (3 layers)
        ├── MemoryDecoder     ← Linear → GELU → Sigmoid (3 layers)
        └── AutoencoderTrainer ← AdamW + CosineAnnealingLR + grad clip + save/load
```

---

## Examples Summary

```python
from SummarizerAI import MemorySummarizer

# Minimal
summarizer = MemorySummarizer()
summarizer.fit({"Hello": "Hi!", "Tell me a joke": "Why do programmers..."})
summarizer.print_report()

# Custom config
summarizer = MemorySummarizer(config={
    "latent_dim": 128, "ae_epochs": 60, "device": "cuda",
    "embedding": {"max_features": 8000},
})
summarizer.fit(memory_dict)

# Query routing
result = summarizer.query("what is python")
print(result["predicted_intent"], result["intent_confidence"])
print(result["top_matches"])

# Export
summarizer.export_report("report.json")

# Access components directly after fit
from SummarizerAI.core.data_pipeline import MemoryLoader, TextFeatureExtractor
from SummarizerAI.core.clustering import MemoryClusterer, IntentClassifier, PatternMatcher
from SummarizerAI.models.autoencoder import AutoencoderTrainer

loader = MemoryLoader()
patterns = loader.from_dict(memory_dict)

extractor = TextFeatureExtractor()
X = extractor.fit_transform(patterns)

trainer = AutoencoderTrainer(latent_dim=64, epochs=30)
trainer.fit(X.toarray())
embeddings = trainer.get_embeddings(X.toarray())

clusterer = MemoryClusterer(method="auto")
clusterer.fit(embeddings)
print(clusterer.n_clusters_found, clusterer.labels_)

clf = IntentClassifier()
clf.fit(X, [p.intent_tag for p in patterns])
print(clf.predict(X))

matcher = PatternMatcher(threshold=0.65)
matcher.index(X, patterns)
print(matcher.query(X[0], top_k=3))
```