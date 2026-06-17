# PythonAIBrain Config — `config.py`

The master configuration layer for the entire Pythonaibrain / pyaitk framework. All subsystems — Brain, STT, TTS, NER, TTI, Memory Summarizer, LLM — read their settings from a single `.pbcfg` file via typed dataclass section objects on one unified `AppConfig` instance.

---

## What Is This?

`config.py` provides:

- **`.pbcfg` file format** — INI-style config file with `; #` comment support and inline comments
- **`AppConfig`** — unified config manager that reads/writes all sections and exposes them as typed dataclasses
- **22 section dataclasses** — one per subsystem, each with typed fields and sensible defaults
- **Auto-discovery** — searches upward from cwd to find `config.pbcfg`, falls back to built-in defaults
- **Typed `get()` / `set()`** — generic read/write with optional type casting and fallback
- **`generate_default_config()`** — writes a fully-populated default `.pbcfg` to disk
- **JSON export** — serialize the entire config to JSON string or file
- **Module-level singleton** — `get_config()` returns the same instance across the entire application

---

## Installation

No extra dependencies — uses Python stdlib `configparser`, `dataclasses`, and `pathlib` only.

---

## File Format (`.pbcfg`)

The config file uses standard INI format. Comments start with `;` or `#`. Inline comments are supported.

```ini
; PythonAIBrain configuration file

[brain]
intents_path        = ./intents.json
condition           = true
smart_memory        = true
memory_path         = memory.json
memory_fit_interval = 20
username            = user_name
download            = false

[model]
model_path      = model.pth
dimension_path  = dimensions.json
batch_size      = 8
learning_rate   = 0.001
epochs          = 100

[llm]
n_ctx       = 2048
n_threads   = 0          ; 0 → use os.cpu_count()
max_tokens  = 512
verbose     = false

[tts]
rate         = 150
volume       = 1.0
voice        = david
default_text = Hello from PyAI
output_path  =           ; if set, saves audio here instead of playing

[stt]
energy_threshold         =        ; empty → auto-calibrate
dynamic_energy_threshold = true
pause_threshold          = 0.8
phrase_time_limit        =        ; hard cap per utterance (seconds)
timeout                  = 5.0
ambient_noise_duration   = 0.5
connectivity_host        = 8.8.8.8
connectivity_port        = 53
connectivity_timeout     = 2.0
preferred_engine         = None   ; None | google | pocketsphinx
sphinx_language          = en-US
google_language          = en-US
google_api_key           =        ; empty → free tier
max_retries              = 3
retry_delay              = 0.5

[logging]
level  = INFO
format = %(asctime)s [%(levelname)s] %(name)s - %(message)s

[weather]
base_url = https://api.openweathermap.org/data/2.5/weather
units    = metric

[memory]
auto_load = true
auto_fit  = true

[search]
max_results = 5

[webassistant]
intents_path   = ./Webintents.json
model_path     = WebAssistantModel.pth
dimension_path = WebAssistantDimensions.json
batch_size     = 8
learning_rate  = 0.001
epochs         = 100

[embedding]
tfidf_max_features = 5000
tfidf_ngram_range  = 1,3
tfidf_sublinear_tf = true
embed_dim          = 128
vocab_size         = 10000

[clustering]
n_clusters              = 8
kmeans_max_iter         = 300
kmeans_random_state     = 42
dbscan_eps              = 0.5
dbscan_min_samples      = 2
agglo_linkage           = ward
agglo_distance_threshold =

[classifier]
lr_max_iter          = 500
lr_c                 = 1.0
lr_solver            = lbfgs
lr_multi_class       = auto
similarity_threshold = 0.65

[summarizer]
latent_dim               = 64
hidden_dim               = 256
ae_epochs                = 30
ae_lr                    = 0.001
ae_batch_size            = 16
top_patterns_per_cluster = 3
min_cluster_size         = 2

[tti_image]
default_width    = 512
default_height   = 512
default_bpp      = 24
default_format   = png
background_color = 255,255,255
jpeg_quality     = 92

[tti_ai]
nlp_backend         = nltk
max_prompt_tokens   = 128
use_stopword_filter = true
model_type          = vae_numpy
latent_dim          = 128
text_embed_dim      = 256
vocab_size          = 4096
hidden_dim          = 512
palette_clusters    = 8
num_inference_steps = 50
guidance_scale      = 7.5
seed                =

[tti_art]
fractal_max_iter        = 256
blur_default_radius     = 2
noise_default_intensity = 0.15
animation_fps           = 24
streaming_chunk_mb      = 32

[tti_paths]
output_dir = tti_output
model_dir  = tti_models
cache_dir  = tti_cache
log_dir    = tti_logs

[postprocessor]
min_length       = 1
max_length       =
allowed_labels   =
blocked_labels   =
deduplicate      = true
merge_adjacent   = false
lowercase_labels = false
strip_punct      = true
custom_label_map =

[preprocessor]
lowercase            = false
remove_urls          = true
remove_emails        = false
remove_html_tags     = true
normalize_whitespace = true
normalize_unicode    = true
max_length           =
custom_patterns      =
```

Boolean values accept: `true/false`, `yes/no`, `on/off`, `1/0`.
Empty values (e.g. `timeout =`) resolve to `None` for optional fields.
Tuple fields (e.g. `tfidf_ngram_range`, `background_color`) are stored as comma-separated integers.

---

## How to Use

### 1. Module-level singleton (recommended)

```python
from pythonaibrain.config import get_config

cfg = get_config()   # auto-discovers config.pbcfg from cwd upward
print(cfg.brain.smart_memory)    # True
print(cfg.model.epochs)          # 100
print(cfg.stt.pause_threshold)   # 0.8
```

### 2. Load a specific file

```python
from pythonaibrain.config import AppConfig

cfg = AppConfig("myproject.pbcfg")
print(cfg.tts.voice)    # "david"
print(cfg.llm.n_ctx)    # 2048
```

### 3. Auto-discover

```python
cfg = AppConfig.discover()            # search cwd → home
cfg = AppConfig.discover("/my/dir")   # search from a custom start path
```

### 4. Build from a dict

```python
cfg = AppConfig.from_dict({
    "brain": {"smart_memory": True, "username": "divyanshu"},
    "model": {"epochs": 50, "learning_rate": 0.0005},
})
```

### 5. Read, mutate, and save

```python
cfg = get_config()

# Read with typed access
print(cfg.brain.intents_path)
print(cfg.embedding.tfidf_max_features)

# Generic get with cast and fallback
val = cfg.get("tti_ai", "latent_dim", fallback=128, cast=int)

# Mutate in memory
cfg.set("tti_ai", "seed", 42)
cfg.set("tts", "rate", 180)

# Persist to disk
cfg.save()                         # saves to original path
cfg.save("backup.pbcfg")           # save to alternate path
```

### 6. Reload from disk

```python
cfg.load()                        # reload from current path
cfg.load("other.pbcfg")           # reload from different file
```

### 7. Generate a default config file

```python
from pythonaibrain.config import generate_default_config

path = generate_default_config()              # → ./config.pbcfg
path = generate_default_config("myapp.pbcfg")
```

### 8. Reset to factory defaults

```python
from pythonaibrain.config import reset_config

cfg = reset_config()   # clears singleton, returns AppConfig with all defaults
```

### 9. Export to JSON

```python
json_str = cfg.to_json(indent=2)
cfg.save_json("config.json")
```

### 10. Inspect all settings

```python
print(cfg.dump())           # human-readable dump of all sections
d = cfg.as_dict()           # nested dict {section: {key: value}}
```

---

## Section Reference

### `[brain]` → `cfg.brain` (`BrainConfig`)

| Key                  | Type   | Default              | Description                                      |
|----------------------|--------|----------------------|--------------------------------------------------|
| `intents_path`       | `str`  | `./intents.json`     | Path to intents JSON file                        |
| `condition`          | `bool` | `True`               | Enable dynamic intent learning from web search   |
| `smart_memory`       | `bool` | `True`               | Use `SmartMemory` (semantic search + clustering) |
| `memory_path`        | `str`  | `memory.json`        | Path for memory persistence                      |
| `memory_fit_interval`| `int`  | `20`                 | Auto-fit SmartMemory every N stored memories     |
| `username`           | `str`  | `user_name`          | Key used for user name storage in memory         |
| `download`           | `bool` | `False`              | Auto-download NLTK data on Brain init            |

### `[model]` → `cfg.model` (`ModelConfig`)

| Key              | Type    | Default           | Description                        |
|------------------|---------|-------------------|------------------------------------|
| `model_path`     | `str`   | `model.pth`       | Path to save/load model weights    |
| `dimension_path` | `str`   | `dimensions.json` | Path to save/load vocab/intent map |
| `batch_size`     | `int`   | `8`               | Training batch size                |
| `learning_rate`  | `float` | `0.001`           | Adam optimizer learning rate       |
| `epochs`         | `int`   | `100`             | Training epochs                    |

### `[llm]` → `cfg.llm` (`LLMConfig`)

| Key          | Type   | Default | Description                                 |
|--------------|--------|---------|---------------------------------------------|
| `n_ctx`      | `int`  | `2048`  | LLM context window size                     |
| `n_threads`  | `int`  | `0`     | CPU threads; `0` = `os.cpu_count()`         |
| `max_tokens` | `int`  | `512`   | Max tokens to generate per response         |
| `verbose`    | `bool` | `False` | Enable llama.cpp verbose logging            |

### `[tts]` → `cfg.tts` (`TTSConfig`)

| Key            | Type            | Default           | Description                                   |
|----------------|-----------------|-------------------|-----------------------------------------------|
| `rate`         | `int`           | `150`             | Words per minute                              |
| `volume`       | `float`         | `1.0`             | Volume 0.0–1.0 (validated in `__post_init__`) |
| `voice`        | `str`           | `david`           | Voice name fragment (fuzzy matched)           |
| `default_text` | `str`           | `Hello from PyAI` | Fallback text for empty `say()` calls         |
| `output_path`  | `str` or `None` | `None`            | Save to WAV file instead of playing           |

### `[stt]` → `cfg.stt` (`STTConfig`)

| Key                        | Type                  | Default   | Description                                      |
|----------------------------|-----------------------|-----------|--------------------------------------------------|
| `energy_threshold`         | `float` or `None`     | `None`    | Mic sensitivity; `None` = auto-calibrate         |
| `dynamic_energy_threshold` | `bool`                | `True`    | Continuously adjust threshold                    |
| `pause_threshold`          | `float`               | `0.8`     | Silence seconds marking end of phrase            |
| `phrase_time_limit`        | `float` or `None`     | `None`    | Hard cap per utterance in seconds                |
| `timeout`                  | `float` or `None`     | `5.0`     | Seconds to wait for speech to start              |
| `ambient_noise_duration`   | `float`               | `0.5`     | Seconds to sample noise floor before listening   |
| `connectivity_host`        | `str`                 | `8.8.8.8` | Host used for the network connectivity probe     |
| `connectivity_port`        | `int`                 | `53`      | Port for the connectivity probe                  |
| `connectivity_timeout`     | `float`               | `2.0`     | Timeout for the connectivity probe               |
| `preferred_engine`         | `Engine` or `None`    | `None`    | Force `GOOGLE` or `POCKETSPHINX`; `None` = auto  |
| `sphinx_language`          | `str`                 | `en-US`   | PocketSphinx language code                       |
| `google_language`          | `str`                 | `en-US`   | Google Speech API BCP-47 language tag            |
| `google_api_key`           | `str` or `None`       | `None`    | Google API key; `None` = free tier               |
| `max_retries`              | `int`                 | `3`       | Max retry attempts on service errors             |
| `retry_delay`              | `float`               | `0.5`     | Seconds between retries                          |

### `[logging]` → `cfg.logging` (`LoggingConfig`)

| Key      | Type  | Default                                              | Description         |
|----------|-------|------------------------------------------------------|---------------------|
| `level`  | `str` | `INFO`                                               | Root logging level  |
| `format` | `str` | `%(asctime)s [%(levelname)s] %(name)s - %(message)s` | Log record format   |

### `[memory]` → `cfg.memory` (`MemoryConfig`)

| Key         | Type   | Default | Description                             |
|-------------|--------|---------|-----------------------------------------|
| `auto_load` | `bool` | `True`  | Load memory from disk on init           |
| `auto_fit`  | `bool` | `True`  | Auto-fit SmartMemory summarizer on load |

### `[embedding]` → `cfg.embedding` (`EmbeddingConfig`)

| Key                  | Type    | Default  | Description                                     |
|----------------------|---------|----------|-------------------------------------------------|
| `tfidf_max_features` | `int`   | `5000`   | Max vocabulary size for TF-IDF                  |
| `tfidf_ngram_range`  | `tuple` | `(1, 3)` | N-gram range (stored as `1,3` in file)          |
| `tfidf_sublinear_tf` | `bool`  | `True`   | Apply sublinear TF scaling                      |
| `embed_dim`          | `int`   | `128`    | Learned embedding dimension                     |
| `vocab_size`         | `int`   | `10000`  | Vocabulary size for learned embeddings          |

### `[clustering]` → `cfg.clustering` (`ClusteringConfig`)

| Key                       | Type               | Default | Description                           |
|---------------------------|--------------------|---------|---------------------------------------|
| `n_clusters`              | `int`              | `8`     | KMeans cluster count                  |
| `kmeans_max_iter`         | `int`              | `300`   | KMeans max iterations                 |
| `kmeans_random_state`     | `int`              | `42`    | KMeans random seed                    |
| `dbscan_eps`              | `float`            | `0.5`   | DBSCAN epsilon radius                 |
| `dbscan_min_samples`      | `int`              | `2`     | DBSCAN minimum samples per cluster    |
| `agglo_linkage`           | `str`              | `ward`  | Agglomerative linkage criterion       |
| `agglo_distance_threshold`| `float` or `None`  | `None`  | Agglomerative distance threshold      |

### `[classifier]` → `cfg.classifier` (`ClassifierConfig`)

| Key                    | Type    | Default  | Description                                    |
|------------------------|---------|----------|------------------------------------------------|
| `lr_max_iter`          | `int`   | `500`    | Logistic Regression max iterations             |
| `lr_c`                 | `float` | `1.0`    | LR regularization strength (lower = stronger)  |
| `lr_solver`            | `str`   | `lbfgs`  | LR solver                                      |
| `lr_multi_class`       | `str`   | `auto`   | LR multi-class strategy                        |
| `similarity_threshold` | `float` | `0.65`   | Min cosine similarity for pattern matching     |

### `[summarizer]` → `cfg.summarizer` (`SummarizerConfig`)

| Key                       | Type    | Default | Description                                 |
|---------------------------|---------|---------|---------------------------------------------|
| `latent_dim`              | `int`   | `64`    | Autoencoder latent space dimension          |
| `hidden_dim`              | `int`   | `256`   | Autoencoder hidden layer size               |
| `ae_epochs`               | `int`   | `30`    | Autoencoder training epochs                 |
| `ae_lr`                   | `float` | `0.001` | Autoencoder learning rate                   |
| `ae_batch_size`           | `int`   | `16`    | Autoencoder training batch size             |
| `top_patterns_per_cluster`| `int`   | `3`     | Top patterns to extract per cluster         |
| `min_cluster_size`        | `int`   | `2`     | Minimum cluster size for summarization      |

### `[tti_image]` → `cfg.tti_image` (`TTIImageConfig`)

| Key               | Type    | Default           | Description                                  |
|-------------------|---------|-------------------|----------------------------------------------|
| `default_width`   | `int`   | `512`             | Output image width in pixels                 |
| `default_height`  | `int`   | `512`             | Output image height in pixels                |
| `default_bpp`     | `int`   | `24`              | Bits per pixel                               |
| `default_format`  | `str`   | `png`             | Output format: `png`, `bmp`, `jpeg`          |
| `background_color`| `tuple` | `(255, 255, 255)` | RGB background color (stored as `255,255,255`) |
| `jpeg_quality`    | `int`   | `92`              | JPEG compression quality (1–95)              |

### `[tti_ai]` → `cfg.tti_ai` (`TTIAIConfig`)

| Key                   | Type            | Default                | Description                                 |
|-----------------------|-----------------|------------------------|---------------------------------------------|
| `nlp_backend`         | `str`           | `nltk`                 | NLP tokenizer: `nltk` or `spacy`            |
| `max_prompt_tokens`   | `int`           | `128`                  | Max tokens per prompt                       |
| `use_stopword_filter` | `bool`          | `True`                 | Filter stopwords from prompts               |
| `model_type`          | `str`           | `vae_numpy`            | AI model type: `vae_numpy` or `torch_vae`   |
| `latent_dim`          | `int`           | `128`                  | VAE latent space dimension                  |
| `text_embed_dim`      | `int`           | `256`                  | Text embedding dimension                    |
| `vocab_size`          | `int`           | `4096`                 | Vocabulary size for text encoding           |
| `hidden_dim`          | `int`           | `512`                  | Hidden layer size                           |
| `palette_clusters`    | `int`           | `8`                    | KMeans clusters for palette extraction      |
| `palette_model_path`  | `str`           | `tti_palette_model.pkl`| Path for saved palette model                |
| `num_inference_steps` | `int`           | `50`                   | Diffusion inference steps                   |
| `guidance_scale`      | `float`         | `7.5`                  | Classifier-free guidance scale              |
| `seed`                | `int` or `None` | `None`                 | Random seed; empty in file resolves to `None`|

### `[tti_art]` → `cfg.tti_art` (`TTIArtConfig`)

| Key                       | Type    | Default | Description                                |
|---------------------------|---------|---------|--------------------------------------------|
| `fractal_max_iter`        | `int`   | `256`   | Max iterations for fractal generation      |
| `blur_default_radius`     | `int`   | `2`     | Default Gaussian blur radius               |
| `noise_default_intensity` | `float` | `0.15`  | Default noise overlay intensity            |
| `animation_fps`           | `int`   | `24`    | Frames per second for animations           |
| `streaming_chunk_mb`      | `int`   | `32`    | Chunk size for streaming output (MB)       |

### `[tti_paths]` → `cfg.tti_paths` (`TTIPathConfig`)

| Key          | Type  | Default      | Description                      |
|--------------|-------|--------------|----------------------------------|
| `output_dir` | `str` | `tti_output` | Directory for generated images   |
| `model_dir`  | `str` | `tti_models` | Directory for saved models       |
| `cache_dir`  | `str` | `tti_cache`  | Directory for cached data        |
| `log_dir`    | `str` | `tti_logs`   | Directory for TTI log files      |

Call `cfg.ensure_tti_dirs()` (or `cfg.tti_paths.ensure_dirs()`) to create all four directories.

### `[postprocessor]` → `cfg.postprocessor` (`PostprocessorConfig`)

| Key                | Type                    | Default | Description                                      |
|--------------------|-------------------------|---------|--------------------------------------------------|
| `min_length`       | `int`                   | `1`     | Minimum entity text length                       |
| `max_length`       | `int` or `None`         | `None`  | Maximum entity text length                       |
| `allowed_labels`   | `set[str]` or `None`    | `None`  | Allow only these labels; `None` = allow all      |
| `blocked_labels`   | `set[str]`              | `{}`    | Always exclude these labels                      |
| `deduplicate`      | `bool`                  | `True`  | Remove duplicate spans                           |
| `merge_adjacent`   | `bool`                  | `False` | Merge consecutive same-label spans               |
| `lowercase_labels` | `bool`                  | `False` | Lowercase all label names                        |
| `strip_punct`      | `bool`                  | `True`  | Strip leading/trailing punctuation from text     |
| `custom_label_map` | `dict[str, str]`        | `{}`    | Rename labels e.g. `{"PERSON": "PER"}`          |

### `[preprocessor]` → `cfg.preprocessor` (`PreprocessorConfig`)

| Key                    | Type            | Default | Description                                     |
|------------------------|-----------------|---------|-------------------------------------------------|
| `lowercase`            | `bool`          | `False` | Lowercase the text                              |
| `remove_urls`          | `bool`          | `True`  | Strip `http://`, `https://`, `www.` URLs        |
| `remove_emails`        | `bool`          | `False` | Strip email addresses                           |
| `remove_html_tags`     | `bool`          | `True`  | Strip HTML tags                                 |
| `normalize_whitespace` | `bool`          | `True`  | Collapse whitespace to single spaces            |
| `normalize_unicode`    | `bool`          | `True`  | NFC Unicode normalization                       |
| `max_length`           | `int` or `None` | `None`  | Truncate text to this many characters           |
| `custom_patterns`      | `list[str]`     | `[]`    | Regex patterns to strip                         |

---

## `AppConfig` API Reference

| Method / Property           | Returns       | Description                                                      |
|-----------------------------|---------------|------------------------------------------------------------------|
| `load(path?)`               | `AppConfig`   | Parse `.pbcfg` file and populate all sections                    |
| `save(path?)`               | `AppConfig`   | Write current config to `.pbcfg`                                 |
| `get(section, key, fallback, cast)` | `Any` | Read a raw value with optional type casting                      |
| `set(section, key, value)`  | `None`        | Write a value into the in-memory config (call `save()` to persist)|
| `to_json(indent?)`          | `str`         | Serialize all sections to a JSON string                          |
| `save_json(filepath?)`      | `None`        | Write config to a JSON file                                      |
| `as_dict()`                 | `dict`        | Return entire config as a nested `{section: {key: value}}` dict  |
| `dump()`                    | `str`         | Human-readable summary of all settings                           |
| `ensure_tti_dirs()`         | `None`        | Create TTI output/model/cache/log directories if missing         |
| `AppConfig.discover(start?)`| `AppConfig`   | Search cwd → home for `config.pbcfg`; use defaults if not found  |
| `AppConfig.from_dict(data, path?)` | `AppConfig` | Build config from a nested dict                             |

---

## Module-Level Helpers

```python
from pythonaibrain.config import get_config, reset_config, generate_default_config

# Get or create the global singleton
cfg = get_config()

# Force a specific file into the singleton
cfg = get_config("custom.pbcfg")

# Reset singleton to factory defaults
cfg = reset_config()

# Write a default config file to disk
path = generate_default_config()            # → ./config.pbcfg
path = generate_default_config("myapp.pbcfg")
```

---

## Examples Summary

```python
from pythonaibrain.config import AppConfig, get_config, generate_default_config

# Generate a default file
generate_default_config("myproject.pbcfg")

# Load and read
cfg = AppConfig("myproject.pbcfg")
print(cfg.brain.smart_memory)        # True
print(cfg.model.epochs)              # 100
print(cfg.stt.google_language)       # "en-US"
print(cfg.tti_image.default_format)  # "png"
print(cfg.embedding.tfidf_ngram_range)  # (1, 3)

# Mutate and save
cfg.set("model", "epochs", 200)
cfg.set("tts", "voice", "zira")
cfg.set("tti_ai", "seed", 1337)
cfg.save()

# Generic typed get
val = cfg.get("clustering", "n_clusters", fallback=8, cast=int)

# JSON export
cfg.save_json("config_backup.json")

# Factory methods
cfg2 = AppConfig.discover()
cfg3 = AppConfig.from_dict({
    "brain": {"username": "divyanshu", "smart_memory": True},
    "llm": {"max_tokens": 256},
})

# Create TTI directories
cfg.ensure_tti_dirs()

# Inspect
print(cfg.dump())
```