# PythonAIBrain Core — `core.py`

The central orchestration layer of the PythonAIBrain (pyaitk) framework. Wires together intent classification, neural chatbot training, memory, NER, translation, frame analysis, weather, and an optional quantized LLM — all under two primary entry points: `Brain` and `AdvanceBrain`.

---

## What Is This?

`core.py` is the heart of the pyaitk package, providing:

- **`Brain`** — intent-based chatbot with memory, NER, translation, frame classification, grammar correction, and TTS
- **`AdvanceBrain`** — routes through a local quantized LLM (`pythonaibrain-llm`) for open-ended generation, with the same interface as `Brain`
- **`VectorizerMode`** — switchable feature extraction: Binary BoW (NumPy), TF-IDF (scikit-learn), or Gensim TF-IDF
- **`IntentsManager`** — load, save, and extend `intents.json` files dynamically
- **Frame classification** — lightweight neural sentence-type classifier (Statement / Question / Command / Name / …)
- **NER integration** — entity extraction via the NER subsystem, with optional in-call training
- **Translation** — GRU seq-to-seq translator (multilingual → English), trained once per process
- **Language detection** — keyword-based classifier for English / Hindi / French / Spanish
- **Weather API** — city-level weather lookup via OpenWeatherMap
- **`configure()`** — load a `.pbcfg` file to override all settings before constructing any Brain
- **Structured logging** — all subsystems use Python's `logging` module, config-driven level and format

---

## Installation

```bash
pip install torch nltk scikit-learn psutil pyjokes python-dotenv
python -m nltk.downloader punkt wordnet

# Optional: Gensim vectorizer
pip install gensim

# Optional: AdvanceBrain LLM support
pip install pythonaibrain-llm
```

Set your OpenWeatherMap API key in a `.env` file:

```
weather_api_key=YOUR_KEY_HERE
```

---

## Quick Start

```python
from pythonaibrain.core import Brain

with Brain() as brain:
    brain.load()
    print(brain.ask("Hello"))
```

---

## Configuration

Load a `.pbcfg` file before constructing any Brain to override all settings:

```python
import pythonaibrain.core as core
core.configure("project.pbcfg")
brain = core.Brain()
```

`configure()` updates the global config singleton, applies logging settings, and returns the loaded `AppConfig`.

---

## `VectorizerMode` — Feature Extraction

Controls how text is converted to feature vectors for the neural intent classifier.

| Mode             | Backend       | Notes                                      |
|------------------|---------------|--------------------------------------------|
| `BOW` (default)  | Pure NumPy    | Binary Bag-of-Words; zero extra deps       |
| `TFIDF`          | scikit-learn  | TF-IDF weighting via `TfidfVectorizer`     |
| `GENSIM`         | Gensim        | Requires `pip install gensim`              |

```python
from pythonaibrain.core import Brain, VectorizerMode

brain = Brain(vectorizer_mode=VectorizerMode.TFIDF)
brain = Brain(vectorizer_mode=VectorizerMode.GENSIM)
brain = Brain()   # default BOW
```

---

## `Brain` — Intent-Based Chatbot

### Basic usage

```python
from pythonaibrain.core import Brain

# Train from scratch
with Brain() as brain:
    brain.train()
    brain.save()
    print(brain.ask("What is the weather?"))

# Load a saved model
with Brain() as brain:
    brain.load()
    response = brain.process_messages("Tell me a joke", grammar=True)
    print(response)
```

### Constructor parameters

| Parameter            | Type              | Default | Description                                              |
|----------------------|-------------------|---------|----------------------------------------------------------|
| `intents_path`       | `str` or `None`   | config  | Path to `intents.json`                                   |
| `condition`          | `bool` or `None`  | config  | Enable dynamic intent learning from search results       |
| `download`           | `bool` or `None`  | config  | Auto-download NLTK data on init                          |
| `memory_path`        | `str` or `None`   | config  | Path for memory persistence file                         |
| `smart_memory`       | `bool` or `None`  | config  | Use `SmartMemory` (semantic search + clustering)         |
| `memory_fit_interval`| `int` or `None`   | config  | Auto-fit SmartMemory every N memories                    |
| `config`             | `AppConfig` or `None` | global | Override global config for this instance            |
| `vectorizer_mode`    | `VectorizerMode`  | `BOW`   | Feature extraction strategy                              |
| `**function_mapping` | `Any`             | —       | Map intent tags to Python callables                      |

### Methods

| Method                                      | Returns   | Description                                                   |
|---------------------------------------------|-----------|---------------------------------------------------------------|
| `train()`                                   | `None`    | Parse intents, build features, train neural model             |
| `load()`                                    | `None`    | Load saved model from paths in config                         |
| `save()`                                    | `None`    | Save model weights and dimension file                         |
| `process_messages(message, grammar, TTS)`   | `str`     | Classify intent, respond, remember, optionally speak          |
| `talk(message, grammar, TTS)`               | `str`     | Attempt web search first, fall back to `process_messages`     |
| `ask(query, TTS)`                           | `str`     | `talk()` + typewriter-style `write()` output                  |
| `write(message, set_timer, TTS)`            | `None`    | Print message character by character with optional TTS        |
| `translator(message)`                       | `str`     | Translate input to English via GRU seq-to-seq model           |
| `classify_language(message)`                | `str`     | Detect language: `english`, `hindi`, `french`, `spanish`      |
| `predict_message_type(message)`             | `str`     | Classify sentence type: Statement / Question / Command / …    |
| `predict_entitie(message, train)`           | `list`    | Extract named entities via NER pipeline                       |
| `memorize_user_name(message)`               | `None`    | Detect and store user name from message                       |
| `recall_user_name()`                        | `str`     | Retrieve stored user name from memory                         |
| `search_memory(query, top_k)`               | `list`    | Semantic or substring search over conversation history        |
| `memory_intent(query)`                      | `str`     | Predict intent of query from stored patterns                  |
| `memory_report()`                           | `Any`     | Return SmartMemory cluster/intent analysis report             |
| `export_memory_report(path)`                | `bool`    | Export memory report as JSON                                  |
| `fit_memory()`                              | `bool`    | Manually retrain SmartMemory summarizer                       |
| `is_loaded()` / `is_saved()` / `is_trained()` | `bool`  | State flags                                                   |
| `count_query()`                             | `int`     | Total number of queries processed                             |
| `pyai_say(*msg)`                            | `None`    | Print with `PYAI :` prefix                                    |

### Function mapping

Map intent tags to Python callables — called automatically when that intent is predicted:

```python
def open_calculator():
    import subprocess; subprocess.Popen("calc.exe")

with Brain(calculator=open_calculator) as brain:
    brain.load()
    brain.process_messages("open calculator")
```

### Memory

`Brain` uses `SmartMemory` by default (falls back to plain `Memory` if the summarizer module is absent). Every `process_messages()` call automatically stores the exchange.

```python
# Semantic search over history
results = brain.search_memory("weather", top_k=3)

# Export memory analytics
brain.export_memory_report("memory_report.json")

# Manually trigger SmartMemory refit
brain.fit_memory()
```

---

## `AdvanceBrain` — LLM-powered Brain

Routes responses through a local quantized LLM from `pythonaibrain-llm`. Falls back to the intent classifier when `advance=False`.

```python
from pythonaibrain.core import AdvanceBrain

with AdvanceBrain() as brain:
    brain.load()
    print(brain.process_messages("What is Python?"))

# Skip LLM, use intent classifier only
print(brain.process_messages("Hello", advance=False))
```

`AdvanceBrain` has the same `train()`, `load()`, `save()`, `translator()`, `classify_language()`, `predict_message_type()`, and `predict_entitie()` methods as `Brain`.

> **Requires:** `pip install pythonaibrain-llm`
> The LLM is lazy-loaded on first `process_messages(advance=True)` call.

---

## `IntentsManager` — Dynamic Intent Management

```python
from pythonaibrain.core import IntentsManager

im = IntentsManager("intents.json")

# Add or extend an intent
im.add_intent(
    tag="greeting",
    patterns=["Hi", "Hey there"],
    responses=["Hello!", "Hey! How can I help?"]
)

# Add a search-derived intent
im.add_search_intent("best Python books", ["Python Crash Course", "Fluent Python"])
```

---

## Utility Functions

### Weather

```python
from pythonaibrain.core import get_weather

weather = get_weather("London")   # e.g. "Clouds"
```

Requires `weather_api_key` in `.env`. Also available: `longitude(city)`, `latitude(city)`, `humidity(city)`, `temperature(city)`.

### Frame classification

```python
from pythonaibrain.core import predict_frame, VectorizerMode

frame = predict_frame("What time is it?")           # → "Question"
frame = predict_frame("Open the door")              # → "Command"
frame = predict_frame("The sky is blue")            # → "Statement"
frame = predict_frame("My name is Divyanshu")       # → "Name"
```

Supported frame types: `Statement`, `Question`, `Command`, `Answer`, `Name`, `Know`, `Shutdown`, `Make Dir`, `Start`.

### Translation

```python
from pythonaibrain.core import translate_to_en

text = translate_to_en("tum kaise ho")   # → "how are you"
text = translate_to_en("hola como estas")
```

Model trains once per process on a built-in multilingual corpus (Hindi, French, Spanish, English).

### Language detection

```python
from pythonaibrain.core import language_classifier

lang = language_classifier("tum kaise ho")   # → "hindi"
lang = language_classifier("je m'appelle")  # → "french"
lang = language_classifier("hola como estas")  # → "spanish"
lang = language_classifier("Hello there")   # → "english"
```

### NER

```python
from pythonaibrain.core import Brain

with Brain() as brain:
    brain.load()
    entities = brain.predict_entitie("Apple was founded by Steve Jobs.")
    print(entities)   # list of Entity objects
```

Or directly:

```python
from pythonaibrain.core import predictNER

entities = predictNER("NASA launched Voyager 1.")
entities = predictNER("LeBron James plays for the Lakers.", train=True)  # retrain first
```

---

## Full Example

```python
from pythonaibrain.core import Brain, VectorizerMode, configure

# Optional: load a custom config before anything else
configure("myproject.pbcfg")

# Train and save
with Brain(vectorizer_mode=VectorizerMode.TFIDF) as brain:
    brain.train()
    brain.save()

# Load and interact
with Brain() as brain:
    brain.load()

    # Chat
    print(brain.ask("Tell me a joke"))
    print(brain.process_messages("What is the weather in Delhi?"))

    # Language tools
    print(brain.classifier_language("tum kaise ho"))   # hindi
    print(brain.translator("mera naam ravi hai"))       # my name is ravi
    print(brain.predict_message_type("Shutdown /s"))    # Shutdown

    # NER
    entities = brain.predict_entitie("Elon Musk founded SpaceX.")
    print(entities)

    # Memory
    brain.memorize_user_name("My name is Divyanshu")
    print(brain.recall_user_name())
    results = brain.search_memory("joke", top_k=3)
    brain.export_memory_report("report.json")

    # Stats
    print(brain.count_query())
    print(brain.is_loaded(), brain.is_saved())
```

---

## Architecture Overview

```
Brain / AdvanceBrain
├── ChatbotAssistant          ← intent parse → feature matrix → PyTorch neural classifier
│   ├── VectorizerMode        ← BOW (NumPy) | TF-IDF (sklearn) | Gensim TF-IDF
│   └── IntentsManager        ← load/save/extend intents.json
├── Memory / SmartMemory      ← conversation persistence + semantic search
├── NERPipeline               ← named entity recognition (spaCy-based)
├── GrammarCorrector          ← optional post-processing on responses
├── TTS (speak)               ← optional audio output
├── Search                    ← fallback web search for unknown intents
├── FrameClassifier           ← sentence-type classifier (singleton, trains once)
├── GRU Translator            ← multilingual → English (singleton, trains once)
├── _PyAILLM                  ← lazy-loaded quantized LLM (AdvanceBrain only)
└── WeatherAPI                ← OpenWeatherMap lookup
```