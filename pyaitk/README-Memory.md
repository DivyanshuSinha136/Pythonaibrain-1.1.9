# Memory — Production-grade Memory Module for PythonAIBrain

A two-tier episodic memory system for Brain and AdvanceBrain. `Memory` provides a thread-safe, JSON-backed key/value store. `SmartMemory` extends it transparently with a full ML pipeline — TF-IDF → Autoencoder → Clustering → IntentClassifier — enabling semantic search over conversation history without changing any call sites.

---

## What Is This?

`Memory.py` provides two public classes and a factory function:

| Class / Function | Description |
|---|---|
| `Memory` | Thread-safe, JSON-backed key/value episodic store with LRU eviction |
| `SmartMemory` | Drop-in superset of `Memory` with semantic search, intent prediction, and cluster analytics |
| `build_memory()` | Factory that returns `SmartMemory` when available, `Memory` otherwise |
| `MemoryEntry` | Dataclass holding a single episodic record with timestamps and access metadata |

**Key properties shared by both classes:**

- **Thread-safe** — all public methods acquire an `RLock`; the background ML fit thread takes only a snapshot so reads/writes are never blocked
- **Atomic writes** — `save_memory()` writes to a `.tmp` file then renames it, preventing corruption on crash
- **LRU eviction** — when the store reaches `max_entries`, the least-recently-used key is evicted
- **Backward-compatible format** — transparently migrates legacy flat `{key: value}` JSON files to the v2 rich format
- **Graceful degradation** — if `SummarizerAI` is absent, `SmartMemory` silently falls back to plain `Memory` behaviour

---

## Installation

No extra dependencies for `Memory`. `SmartMemory` additionally requires:

```bash
pip install scikit-learn torch  # used by SummarizerAI internally
```

`SummarizerAI` must be importable either as a package-level module or from the same directory.

---

## How to Use

### 1. Plain `Memory` (classic key/value)

```python
from Memory import Memory

mem = Memory(path="memory.json")

mem.remember("user_name", "Divyanshu")
mem.remember("last_topic", "Python programming")

print(mem.recall("user_name"))    # "Divyanshu"
print(mem.recall("missing_key", default="N/A"))  # "N/A"

mem.save_memory()
mem.load_memory()   # reload from disk
```

### 2. `SmartMemory` — drop-in with ML extras

```python
from Memory import SmartMemory

mem = SmartMemory(
    path="memory.json",
    auto_fit=True,        # refit every fit_interval calls to remember()
    fit_interval=20,      # default: 20
)

mem.remember("Hello! How are you?", "I'm doing great, thanks!")
mem.remember("Tell me a joke", "Why did the chicken cross the road?")
# … add more entries …

# Semantic search
results = mem.semantic_search("a funny story", top_k=3)
for r in results:
    print(r["score"], r["key"], r["value"])

# Intent prediction
intent = mem.predict_intent("What's the weather today?")
print(intent)   # e.g. "weather_query"

# Export cluster/intent analytics report
mem.export_report("memory_report.json")
mem.print_report()

mem.save_memory()
```

### 3. `build_memory()` factory (recommended for `Brain`)

```python
from Memory import build_memory

# Returns SmartMemory if SummarizerAI is available, Memory otherwise
mem = build_memory("memory.json", smart=True, fit_interval=50)
mem.remember("greeting", "Hello!")
mem.save_memory()
```

### 4. Checking and controlling the summarizer

```python
from Memory import SmartMemory

mem = SmartMemory(path="memory.json", auto_fit=False)

# Manually trigger a (blocking) fit
success = mem.fit_summarizer()
print(success)   # True / False

# Check fit state
print(mem.is_summarizer_fitted())   # True after fit

# Get the report object
report = mem.get_report()   # MemorySummaryReport dataclass or None
```

### 5. Extended `Memory` API

```python
from Memory import Memory

mem = Memory()

mem.remember("a", "alpha")
mem.remember("b", "beta")

# Delete one entry
mem.forget("a")

# Check membership
print("b" in mem)       # True
print("a" in mem)       # False

# Iterate
for key in mem.keys():
    print(key)

for key, value in mem.items():
    print(key, "→", value)

# Snapshot as plain dict
snapshot = mem.to_dict()  # {"b": "beta"}

print(mem.size)   # 1
print(len(mem))   # 1

# Clear everything (in-memory only; disk unchanged until save_memory)
mem.wipe()
```

### 6. Inspecting `MemoryEntry` metadata

```python
from Memory import Memory

mem = Memory()
mem.remember("topic", "AI and robotics")

# Access internal entry (advanced)
entry = mem._entries["topic"]
print(entry.key)            # "topic"
print(entry.value)          # "AI and robotics"
print(entry.timestamp)      # Unix timestamp of creation
print(entry.access_count)   # Number of recalls
print(entry.last_accessed)  # Unix timestamp of last recall
print(entry.to_dict())      # Full dict for serialization
```

---

## API Reference

### `Memory(path, max_entries, auto_load)`

| Parameter     | Type   | Default        | Description                                         |
|---------------|--------|----------------|-----------------------------------------------------|
| `path`        | `str`  | `memory.json`  | Path to the JSON persistence file                   |
| `max_entries` | `int`  | `10_000`       | Max entries before LRU eviction                     |
| `auto_load`   | `bool` | `True`         | Load from disk automatically if file exists on init |

#### Core contract methods (used by Brain / AdvanceBrain)

| Method                           | Returns | Description                                                  |
|----------------------------------|---------|--------------------------------------------------------------|
| `load_memory()`                  | `None`  | Load (or reload) from disk; silent no-op if file absent      |
| `remember(key, value)`           | `None`  | Store or update; evicts LRU entry when at `max_entries`      |
| `recall(key, default="")`        | `str`   | Retrieve value for key; returns `default` if not found       |
| `save_memory()`                  | `None`  | Atomically persist current state to disk                     |

#### Extended methods

| Method              | Returns                   | Description                                         |
|---------------------|---------------------------|-----------------------------------------------------|
| `forget(key)`       | `bool`                    | Delete one entry; returns `True` if key existed     |
| `wipe()`            | `None`                    | Clear all in-memory entries (disk unchanged)        |
| `keys()`            | `Iterator[str]`           | Iterate over all stored keys                        |
| `items()`           | `Iterator[tuple[str,str]]`| Iterate over `(key, value)` pairs                   |
| `to_dict()`         | `dict[str, str]`          | Plain `{key: value}` snapshot without metadata      |

#### Properties

| Property | Type   | Description                           |
|----------|--------|---------------------------------------|
| `path`   | `Path` | Resolved path to the backing JSON file|
| `size`   | `int`  | Number of currently stored entries    |

---

### `SmartMemory(path, max_entries, auto_load, auto_fit, fit_interval, summarizer_config)`

Inherits all `Memory` methods. Additional constructor parameters:

| Parameter            | Type             | Default | Description                                                   |
|----------------------|------------------|---------|---------------------------------------------------------------|
| `auto_fit`           | `bool`           | `True`  | Auto-refit summarizer every `fit_interval` calls to `remember()`|
| `fit_interval`       | `int`            | `20`    | Number of `remember()` calls between automatic refits         |
| `summarizer_config`  | `dict` or `None` | `None`  | Pass custom config to `MemorySummarizer`; `None` = from `.pbcfg`|

#### SmartMemory-only methods

| Method                              | Returns              | Description                                                        |
|-------------------------------------|----------------------|--------------------------------------------------------------------|
| `fit_summarizer(force=False)`       | `bool`               | (Re-)train ML pipeline; blocks caller; returns `True` on success   |
| `semantic_search(text, top_k=3)`    | `list[dict]`         | Ranked semantic matches; falls back to substring search if unfit   |
| `predict_intent(text)`              | `str`                | Predict intent of query from trained classifier                     |
| `get_report()`                      | `MemorySummaryReport` or `None` | Return cluster/intent analysis report            |
| `export_report(path)`               | `bool`               | Write report as JSON; returns `True` on success                    |
| `print_report()`                    | `None`               | Pretty-print cluster report to stdout                              |
| `is_summarizer_fitted()`            | `bool`               | `True` if summarizer is trained and not dirty                      |

#### `semantic_search()` result format

Each item in the returned list:

| Field     | Type    | Description                                         |
|-----------|---------|-----------------------------------------------------|
| `score`   | `float` | Cosine similarity score 0.0–1.0                     |
| `key`     | `str`   | Original input key stored in memory                 |
| `value`   | `str`   | Stored response value                               |
| `intent`  | `str`   | Predicted intent label                              |
| `cluster` | `int`   | Cluster assignment index (`-1` if unassigned)       |

---

### `build_memory(path, smart, **kwargs)` → `Memory`

Factory function. Returns `SmartMemory` when `smart=True` and `SummarizerAI` is importable; returns plain `Memory` otherwise.

| Parameter   | Type   | Default        | Description                                        |
|-------------|--------|----------------|----------------------------------------------------|
| `path`      | `str`  | `memory.json`  | Path for JSON persistence                          |
| `smart`     | `bool` | `True`         | Attempt to use `SmartMemory`                       |
| `**kwargs`  | —      | —              | Forwarded to `SmartMemory` or `Memory` constructor |

---

### `MemoryEntry` fields

| Field           | Type    | Description                                  |
|-----------------|---------|----------------------------------------------|
| `key`           | `str`   | Entry key                                    |
| `value`         | `str`   | Stored value                                 |
| `timestamp`     | `float` | Unix timestamp of creation                   |
| `access_count`  | `int`   | Number of times `recall()` accessed this key |
| `last_accessed` | `float` | Unix timestamp of most recent recall         |

---

## Architecture

```
Memory
├── _entries : OrderedDict[str, MemoryEntry]  ← LRU-ordered episodic store
├── _lock : threading.RLock                   ← protects all mutations
└── _path : Path                              ← JSON backing file

SmartMemory(Memory)
├── _summarizer : MemorySummarizer | None     ← ML pipeline
├── _dirty : bool                             ← needs refit flag
├── _auto_fit : bool                          ← enable background refits
├── _fit_interval : int                       ← refit every N remembers
├── _fit_lock : threading.Lock                ← serializes fit calls
└── _fit_thread : Thread | None               ← background training thread

MemorySummarizer (optional external module)
└── TF-IDF → Autoencoder → KMeans/DBSCAN → IntentClassifier → PatternMatcher
```

**Background fit behaviour:**
- Every `fit_interval` calls to `remember()`, `_fit_background()` spawns a daemon thread
- The thread takes a snapshot of `_store` under the lock, then trains outside it — memory reads/writes are never blocked by training
- If a fit is already running, additional triggers are skipped
- `semantic_search()` triggers a blocking `fit_summarizer()` automatically if the summarizer is unfit (`_dirty=True`)

---

## File Format

Memory is persisted as a JSON file with a version marker:

```json
{
  "__version__": 2,
  "saved_at": 1718000000.0,
  "entry_count": 3,
  "entries": [
    {
      "key": "user_name",
      "value": "Divyanshu",
      "timestamp": 1718000000.0,
      "access_count": 4,
      "last_accessed": 1718001000.0
    }
  ]
}
```

Legacy flat `{key: value}` files from v1 are automatically migrated on load.

---

## Examples Summary

```python
# Plain Memory
from Memory import Memory
mem = Memory("memory.json")
mem.remember("name", "Divyanshu")
print(mem.recall("name"))
mem.save_memory()

# SmartMemory
from Memory import SmartMemory
mem = SmartMemory("memory.json", auto_fit=True, fit_interval=10)
mem.remember("How are you?", "I'm great!")
results = mem.semantic_search("how are things", top_k=3)
intent = mem.predict_intent("What's the weather?")
mem.export_report("report.json")
mem.save_memory()

# Factory (recommended)
from Memory import build_memory
mem = build_memory("memory.json", smart=True, fit_interval=50)
mem.remember("greeting", "Hello!")
mem.save_memory()

# Extended API
mem.forget("greeting")
print("greeting" in mem)   # False
print(mem.size)
print(mem.to_dict())
mem.wipe()
```