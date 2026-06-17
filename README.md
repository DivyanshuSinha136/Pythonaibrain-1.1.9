![Pythonaibrain](https://github.com/DivyanshuSinha136/Pythonaibrain-1.1.9/blob/main/Pythonaibrain-Logo.png)

# Pythonaibrain

Pythonaibrain is a versatile, plug-and-play Python package designed to help you build offline intelligent AI assistants and applications effortlessly. With modules covering speech recognition, text-to-speech, natural language understanding, and more, Pythonaibrain lets you create powerful AI solutions without deep expertise or complex setup. Whether you’re a beginner or an experienced developer, get ready to bring your AI ideas to life quickly and efficiently.

---

## Requirements

- Python **3.9** or later
- pip 23+

---

## Installation

### Minimal install *(core package only)*

```bash
pip install pythonaibrain==1.1.9
```

### Install with specific modules

Pick only what you need:

```bash
# Text-to-Speech
pip install "pythonaibrain[tts]==1.1.9"

# Speech-to-Text
pip install "pythonaibrain[stt]==1.1.9"

# Camera + QR/barcode
pip install "pythonaibrain[camera]==1.1.9"

# Image-to-Text (OCR)
pip install "pythonaibrain[itt]==1.1.9"

# AI Brain + NLP
pip install "pythonaibrain[core]==1.1.9"

# Named-Entity Recognition
pip install "pythonaibrain[ner]==1.1.9"

# Memory management
pip install "pythonaibrain[memory]==1.1.9"

# Mathematical AI + CAS
pip install "pythonaibrain[math]==1.1.9"

# Internet Search
pip install "pythonaibrain[search]==1.1.9"

# PowerPoint extraction
pip install "pythonaibrain[pptx]==1.1.9"

# PDF extraction
pip install "pythonaibrain[pdf]==1.1.9"

# Real-time object detection (YOLOv8)
pip install "pythonaibrain[eye]==1.1.9"

# Text summarisation
pip install "pythonaibrain[summarizer]==1.1.9"

# ZENTRAA encrypted chat server + clients
pip install "pythonaibrain[zentraa]==1.1.9"

# CLSE — image generation  ⚠ AGPL-3.0-or-later
pip install "pythonaibrain[clse]==1.1.9"
```

### Install multiple modules at once

```bash
pip install "pythonaibrain[core,tts,stt]==1.1.9"
pip install "pythonaibrain[core,ner,memory,summarizer]==1.1.9"
```

### Install everything

```bash
pip install "pythonaibrain[all]==1.1.9"
```

---

## Linux — System Dependencies

Some modules require native system libraries. Install them **before** running pip:

```bash
# STT (Speech-to-Text) — PortAudio
sudo apt install portaudio19-dev python3-pyaudio

# Camera / QR barcode — zbar
sudo apt install libzbar0
```

---

## Post-install Steps

### NLTK data *(required by Brain, Context, NER, SummarizerAI)*

```python
import pyaitk
pyaitk.InstallNLTKData()
```

Or manually:

```bash
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger stopwords wordnet words maxent_ne_chunker
```

### spaCy model *(required by NER, optional for Core)*

```bash
python -m spacy download en_core_web_sm
```

### STT offline engine *(optional — pocketsphinx)*

```bash
pip install pocketsphinx
```

---

## Verify Installation

```python
import pyaitk

print(pyaitk.get_version())   # 1.1.9

info = pyaitk.get_info()
print(info)

availability = pyaitk.check_module_availability()
for module, ok in availability.items():
    status = "✔" if ok else "✘"
    print(f"  {status}  {module}")
```

Or from the terminal:

```bash
pythonaibrain --version
pythonaibrain --modules
pythonaibrain --info
```

---

## Import Usage

Each module is imported directly by its submodule path:

```python
import pyaitk.core
import pyaitk.TTS
import pyaitk.STT
import pyaitk.Camera
import pyaitk.NER
import pyaitk.Memory
import pyaitk.MathAI
import pyaitk.Search
import pyaitk.SummarizerAI
import pyaitk.CLSE
import pyaitk.eye
import pyaitk.ITT
import pyaitk.PPTExtract
import pyaitk.Grammar
```

---

## License Notice

| Component | License |
|---|---|
| Pythonaibrain (all modules) | LGPL-3.0-or-later |
| `pyaitk.CLSE` | **AGPL-3.0-or-later** |

If you use `pyaitk.CLSE` in your application, the AGPL requires you to release your application's source code. See `pyaitk/CLSE/LICENSE.txt` for full terms.

---

## About Pythonaibrain Package

Pythonaibrain package consists of `pyaitk` (which means `Python AI Toolkit`) module and `PyAgent` modules, `pyaitk` provides you methods to create your AI models/chatbots where as `PyAgent` provides best `GUI` and `Web` supports for intrection with models/chatbots. It also provide best contorl on your device. In this package you got default pre-trained models.

---

### pyaitk (`Python AI ToolKit`)

`pyaitk` provides various type of methods and functions to create an advance AI.

---

#### How to import pyaitk

After `pythonaibrain` installations run this code for importing pyaitk

```python
import pyaitk
```

---


## Core Component

The central orchestration layer of the Pythonaibrain (pyaitk) framework. Wires together intent classification, neural chatbot training, memory, NER, translation, frame analysis, weather, and an optional quantized LLM — all under two primary entry points: `Brain` and `AdvanceBrain`.

---

### What Is This?

`pyaitk.core` is the heart of the pyaitk package, providing:

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

### Installation

```bash
# Optional: AdvanceBrain LLM support
pip install pythonaibrain-llm
```

Set your OpenWeatherMap API key in a `.env` file:

```
weather_api_key=YOUR_KEY_HERE
```

---

### Quick Start

```python
from pyaitk.core import Brain

with Brain() as brain:
    brain.load()
    print(brain.process_messages("Hello"))
```

### Configuration

Load a `.pbcfg` file before constructing any Brain to override all settings:

```python
import pyaitk.core as core
core.configure("project.pbcfg")
brain = core.Brain()
```

`configure()` updates the global config singleton, applies logging settings, and returns the loaded `AppConfig`.

---

### `VectorizerMode` — Feature Extraction

Controls how text is converted to feature vectors for the neural intent classifier.

| Mode             | Backend       | Notes                                      |
|------------------|---------------|--------------------------------------------|
| `BOW` (default)  | Pure NumPy    | Binary Bag-of-Words; zero extra deps       |
| `TFIDF`          | scikit-learn  | TF-IDF weighting via `TfidfVectorizer`     |
| `GENSIM`         | Gensim        | Requires `pip install gensim`              |

```python
from pyaitk.core import Brain, VectorizerMode

brain = Brain(vectorizer_mode=VectorizerMode.TFIDF)
brain = Brain(vectorizer_mode=VectorizerMode.GENSIM)
brain = Brain()   # default BOW
```

---

### `Brain` — Intent-Based Chatbot

#### Basic usage

```python
from pyaitk.core import Brain

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

#### Constructor parameters

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

#### Methods

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

#### Function mapping

Map intent tags to Python callables — called automatically when that intent is predicted:

```python
def open_calculator():
    import subprocess; subprocess.Popen("calc.exe")

with Brain(calculator=open_calculator) as brain:
    brain.load()
    brain.process_messages("open calculator")
```

#### Memory

`Brain` uses `SmartMemory` by default (falls back to plain `Memory` if the summarizer module is absent). Every `process_messages()` call automatically stores the exchange.

```python
# Semantic search over history
results = brain.search_memory("weather", top_k=3)

# Export memory analytics
brain.export_memory_report("memory_report.json")

# Manually trigger SmartMemory refit
brain.fit_memory()
```

### `AdvanceBrain` — LLM-powered Brain

Routes responses through a local quantized LLM from `pythonaibrain-llm`. Falls back to the intent classifier when `advance=False`.

```python
from pyaitk.core import AdvanceBrain

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

### `IntentsManager` — Dynamic Intent Management

```python
from pyaitk.core import IntentsManager

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

### Utility Functions

#### Weather

```python
from pyaitk.core import get_weather

weather = get_weather("London")   # e.g. "Clouds"
```

Requires `weather_api_key` in `.env`. Also available: `longitude(city)`, `latitude(city)`, `humidity(city)`, `temperature(city)`.

#### Frame classification

```python
from pyaitk.core import predict_frame, VectorizerMode

frame = predict_frame("What time is it?")           # → "Question"
frame = predict_frame("Open the door")              # → "Command"
frame = predict_frame("The sky is blue")            # → "Statement"
frame = predict_frame("My name is Divyanshu")       # → "Name"
```

Supported frame types: `Statement`, `Question`, `Command`, `Answer`, `Name`, `Know`, `Shutdown`, `Make Dir`, `Start`.

#### Translation

```python
from pyaitk.core import translate_to_en

text = translate_to_en("tum kaise ho")   # → "how are you"
text = translate_to_en("hola como estas")
```

Model trains once per process on a built-in multilingual corpus (Hindi, French, Spanish, English).

#### Language detection

```python
from pyaitk.core import language_classifier

lang = language_classifier("tum kaise ho")   # → "hindi"
lang = language_classifier("je m'appelle")  # → "french"
lang = language_classifier("hola como estas")  # → "spanish"
lang = language_classifier("Hello there")   # → "english"
```

#### NER

```python
from pyaitk.core import Brain

with Brain() as brain:
    brain.load()
    entities = brain.predict_entitie("Apple was founded by Steve Jobs.")
    print(entities)   # list of Entity objects
```

Or directly:

```python
from pyaitk.core import predictNER

entities = predictNER("NASA launched Voyager 1.")
entities = predictNER("LeBron James plays for the Lakers.", train=True)  # retrain first
```

---

### Full Example

```python
from pyaitk.core import Brain, VectorizerMode, configure

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

### Architecture Overview

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

---

## TTS - Text To Speech

A fully offline, cross-platform Text-to-Speech module built on the `pyttsx3` backend. Works on Windows (SAPI5), macOS (NSSpeechSynthesizer), and Linux (eSpeak) without any network calls.

### How to Use

#### 1. One-shot (simplest)

```python
from pyaitk.TTS import TTS

TTS().say("Hello, world!")
```

#### 2. Context manager (recommended)

The engine is initialised once and shut down cleanly on exit — even if an exception is raised.

```python
from pyaitk.TTS import TTS, TTSConfig

with TTS(TTSConfig(voice="zira", rate=160)) as tts:
    tts.say("Hello!")
    tts.say("How are you?")
```

#### 3. Module-level convenience function

```python
from pyaitk.TTS import speak

speak("Hello world!")
speak("Bonjour", voice="fiona", rate=140)
```

#### 4. Save speech to a WAV file

```python
from pyaitk.TTS import TTS, TTSConfig

cfg = TTSConfig(output_path="greeting.wav")
with TTS(cfg) as tts:
    saved_path = tts.save("Hello, this will be saved to a file.")
    print(f"Saved to: {saved_path}")
```

You can also override the path at call time:

```python
with TTS() as tts:
    tts.save("Custom path example", path="custom_output.wav")
```

#### 5. List available voices

```python
from pyaitk.TTS import TTS

with TTS() as tts:
    for name in tts.available_voices():
        print(name)
```

#### 6. Get platform-appropriate voice hints

```python
from pyaitk.TTS import TTS

with TTS() as tts:
    print(tts.platform_voice_hints())
# e.g. ['david', 'zira', 'mark'] on Windows
```

---

### Configuration (`TTSConfig`)

All parameters are optional and fall back to values from your project config (`config.tts.*`).

| Parameter      | Type            | Description                                      |
|----------------|-----------------|--------------------------------------------------|
| `rate`         | `int`           | Speech rate in words per minute                  |
| `volume`       | `float`         | Volume level from `0.0` (silent) to `1.0` (full) |
| `voice`        | `str`           | Voice name fragment (fuzzy matched)              |
| `default_text` | `str`           | Fallback text when `say()` is called with no argument |
| `output_path`  | `str` or `None` | Default WAV output path for `save()`             |

```python
cfg = TTSConfig(rate=180, volume=0.8, voice="samantha")
```

---

### Voice Matching

Voices are resolved from a short fragment string using a ranked strategy:

1. **Direct substring match** — `"zira"` matches `"Microsoft Zira Desktop"`
2. **Platform-hint-assisted match** — uses known fragments for the current OS
3. **First installed voice** — fallback with a warning logged

**Platform voice hints:**

| Voice      | Gender | Style               | OS      |
|:----------:|:------:|:-------------------:|:-------:|
| David      | Male   | en-US               | Windows |
| Mark       | Male   | en-US               | Windows |
| Zira       | Female | en-US               | Windows |
| Alex       | Male   | en-US               | macOS   |
| Samantha   | Female | en-US               | macOS   |
| Victoria   | Female | en-US               | macOS   |
| Fred       | Male   | Robotic             | macOS   |
| Daniel     | Male   | en-GB               | macOS   |
| Fiona      | Female | en-GB               | macOS   |
| English    | —      | Default eSpeak      | Linux   |
| English-US | —      | American English    | Linux   |
| English-UK | —      | British English     | Linux   |
| MB-EN1     | —      | MBROLA English 1    | Linux   |
| MB-FR1     | —      | MBROLA French 1     | Linux   |

---

### Examples Summary

```python
# Quickest usage
from pyaitk.TTS import speak
speak("Hello!")

# Full control with context manager
from pyaitk.TTS import TTS, TTSConfig
with TTS(TTSConfig(voice="david", rate=175, volume=0.9)) as tts:
    tts.say("Line one.")
    tts.say("Line two.")

# Save to file
with TTS(TTSConfig(voice="samantha")) as tts:
    tts.save("Saved audio example.", path="demo.wav")

# List voices
with TTS() as tts:
    voices = tts.available_voices()
    print(voices)
```

---

## STT —  Speech-to-Text Module

A fully cross-platform Speech-to-Text library supporting both online (Google Speech Recognition) and offline (PocketSphinx) backends, with automatic engine selection based on network availability.

### What Is This?

- **Dual engine support** — Google Speech Recognition (online) and PocketSphinx (offline), selected automatically
- **Auto engine detection** — checks network connectivity at runtime and picks the best available engine
- **Context manager interface** — opens the microphone once and releases it cleanly on exit, even on exception
- **Ambient noise calibration** — samples the noise floor before each listen to improve accuracy
- **Configurable capture parameters** — energy threshold, pause detection, phrase time limits, and timeouts
- **Retry logic** — automatically retries on transient service errors with configurable delay
- **Structured logging** — all internal events use Python's `logging` module; no bare `print` statements in library code
- **Custom exception hierarchy** — fine-grained errors (`STTAudioError`, `STTRecognitionError`, `STTServiceError`, `STTEngineError`) for clean error handling

---

### Installation

> On Linux, you may also need:
> ```bash
> sudo apt install portaudio19-dev python3-pyaudio
> ```

---

### How to Use

#### 1. One-shot (simplest)

```python
from pyaitk.stt import STT

stt = STT()
text = stt.listen()
print(text)
```

#### 2. Context manager (recommended)

The microphone is opened once and released cleanly on exit — even if an exception is raised.

```python
from pyaitk.stt import STT

with STT() as stt:
    text = stt.listen()
    print(text)
```

#### 3. Force a specific engine

```python
from pyaitk.stt import STT, STTConfig, Engine

# Force offline mode
cfg = STTConfig(preferred_engine=Engine.POCKETSPHINX)
with STT(config=cfg) as stt:
    text = stt.listen()
    print(text)

# Force online Google mode
cfg = STTConfig(preferred_engine=Engine.GOOGLE)
with STT(config=cfg) as stt:
    text = stt.listen()
    print(text)
```

#### 4. Check which engine is active

```python
from pyaitk.stt import STT

with STT() as stt:
    print(stt.active_engine)  # Engine.GOOGLE or Engine.POCKETSPHINX
    text = stt.listen()
```

#### 5. Custom configuration

```python
from pyaitk.stt import STT, STTConfig

cfg = STTConfig(
    timeout=8.0,               # wait up to 8 seconds for speech to start
    phrase_time_limit=15.0,    # cap each utterance at 15 seconds
    pause_threshold=1.0,       # 1 second of silence = end of phrase
    ambient_noise_duration=2.0, # spend 2 seconds calibrating noise floor
    google_language="en-GB",   # use British English
    max_retries=5,             # retry up to 5 times on service errors
)

with STT(config=cfg) as stt:
    text = stt.listen()
    print(text)
```

#### 6. Multi-turn listening loop

```python
from pyaitk.stt import STT, STTAudioError, STTRecognitionError

with STT() as stt:
    while True:
        try:
            text = stt.listen()
            print(f"You said: {text}")
            if "stop" in text.lower():
                break
        except STTAudioError:
            print("No speech detected, trying again…")
        except STTRecognitionError:
            print("Couldn't understand that, please repeat.")
```

---

### Configuration (`STTConfig`)

All parameters are optional and fall back to values from your project config (`config.stt.*`).

| Parameter                | Type                    | Description                                                |
|--------------------------|-------------------------|------------------------------------------------------------|
| `energy_threshold`       | `float` or `None`       | Mic sensitivity; `None` = auto-calibrate                   |
| `dynamic_energy_threshold` | `bool`               | Continuously adjust threshold for ambient noise            |
| `pause_threshold`        | `float`                 | Seconds of silence that mark end of phrase                 |
| `phrase_time_limit`      | `float` or `None`       | Hard cap per utterance in seconds                          |
| `timeout`                | `float` or `None`       | Seconds to wait for speech to begin                        |
| `ambient_noise_duration` | `float`                 | Seconds spent sampling the noise floor before listening    |
| `preferred_engine`       | `Engine` or `None`      | Force a specific engine; `None` = auto-detect              |
| `google_language`        | `str`                   | BCP-47 language tag for Google (e.g. `"en-US"`, `"fr-FR"`) |
| `google_api_key`         | `str` or `None`         | Google API key; `None` = free tier                         |
| `sphinx_language`        | `str`                   | Language/model name for PocketSphinx                       |
| `max_retries`            | `int`                   | Max retry attempts on transient service errors             |
| `retry_delay`            | `float`                 | Seconds to wait between retries                            |
| `connectivity_host`      | `str`                   | Host used for the network connectivity check               |
| `connectivity_port`      | `int`                   | Port used for the connectivity check                       |
| `connectivity_timeout`   | `float`                 | Timeout for the connectivity probe                         |

---

### Engine Selection

The engine is resolved in this order:

1. **`config.preferred_engine`** — if explicitly set, always used
2. **Network probe** — a TCP connection to `connectivity_host:connectivity_port` is attempted
   - Success → `Engine.GOOGLE` (online)
   - Failure → `Engine.POCKETSPHINX` (offline)

| Engine          | Mode    | Requires         | Best for                     |
|-----------------|---------|------------------|------------------------------|
| `GOOGLE`        | Online  | Internet access  | High accuracy, broad language support |
| `POCKETSPHINX`  | Offline | `pocketsphinx`   | Air-gapped / low-latency use |

---

### Exception Hierarchy

```
STTError
├── STTAudioError        — mic could not be opened, or no speech detected in time
├── STTRecognitionError  — audio captured but speech was unintelligible
├── STTServiceError      — online service was unreachable or returned an error
└── STTEngineError       — PocketSphinx failed to init or missing language data
```

Example error handling:

```python
from pyaitk.STT import STT, STTAudioError, STTRecognitionError, STTServiceError, STTEngineError

try:
    with STT() as stt:
        text = stt.listen()
        print(text)
except STTAudioError as e:
    print(f"Mic issue: {e}")
except STTRecognitionError as e:
    print(f"Could not understand: {e}")
except STTServiceError as e:
    print(f"Service unreachable: {e}")
except STTEngineError as e:
    print(f"Offline engine error: {e}")
```

---

### Examples Summary

```python
# Quickest usage
from pyaitk.STT import STT
text = STT().listen()

# Context manager with config
from pyaitk.STT import STT, STTConfig, Engine
cfg = STTConfig(preferred_engine=Engine.GOOGLE, google_language="fr-FR", timeout=6.0)
with STT(config=cfg) as stt:
    text = stt.listen()
    print(text)

# Resilient loop with error recovery
from pyaitk.STT import STT, STTAudioError, STTRecognitionError
with STT() as stt:
    while True:
        try:
            print(stt.listen())
        except STTAudioError:
            pass   # timed out, try again
        except STTRecognitionError:
            print("Please repeat.")
```

---

## PTT (PDF To Text) Function

### What Is This?


- **Full document extraction** — reads all pages and joins them into a single string
- **Per-page fault tolerance** — if a single page fails, extraction continues for the rest
- **Configurable page separator** — control how pages are joined in the output string
- **Structured logging** — all warnings and errors go through Python's `logging` module
- **Custom exception** — `PDFExtractionError` wraps PyMuPDF errors for clean upstream handling
- **Input validation** — checks for empty paths, missing files, and non-file paths before opening

---

### How to Use

#### 1. Basic extraction

```python
from pyaitk.PTT import extract_text_from_pdf

text = extract_text_from_pdf("document.pdf")
print(text)
```

#### 2. Custom page separator

```python
text = extract_text_from_pdf("report.pdf", page_separator="\n---\n")
print(text)
```

#### 3. Custom encoding

```python
text = extract_text_from_pdf("document.pdf", encoding="latin-1")
```

#### 4. With error handling

```python
from pyaitk.PTT import extract_text_from_pdf, PDFExtractionError

try:
    text = extract_text_from_pdf("document.pdf")
    print(f"Extracted {len(text)} characters.")
except FileNotFoundError:
    print("File not found.")
except ValueError as e:
    print(f"Invalid input: {e}")
except PDFExtractionError as e:
    print(f"Could not read PDF: {e}")
```

#### 5. Processing multiple files

```python
from pathlib import Path
from pyaitk.PTT import extract_text_from_pdf, PDFExtractionError

results = {}
for pdf_path in Path("./docs").glob("*.pdf"):
    try:
        results[pdf_path.name] = extract_text_from_pdf(str(pdf_path))
    except PDFExtractionError as e:
        print(f"Skipping {pdf_path.name}: {e}")

for name, text in results.items():
    print(f"{name}: {len(text)} characters")
```

---

### API Reference

#### `extract_text_from_pdf(pdf_path, encoding, page_separator)`

Extracts all text from a PDF and returns it as a single string.

| Parameter        | Type  | Default       | Description                                      |
|------------------|-------|---------------|--------------------------------------------------|
| `pdf_path`       | `str` | *(required)*  | Path to the PDF file                             |
| `encoding`       | `str` | `"utf-8"`     | Text encoding for extraction                     |
| `page_separator` | `str` | `"\n\n"`      | String inserted between pages in the output     |

**Returns:** `str` — full extracted text from all pages.

**Raises:**

| Exception             | When                                             |
|-----------------------|--------------------------------------------------|
| `ValueError`          | `pdf_path` is `None`, empty, or not a file path |
| `FileNotFoundError`   | The specified file does not exist on disk        |
| `PDFExtractionError`  | PDF is corrupted, invalid, or unreadable         |

---

### Exception Reference

```
PDFExtractionError
└── Wraps fitz.FileDataError and other PyMuPDF runtime errors
```

`PDFExtractionError` is the single catch-all for PDF-level failures. `FileNotFoundError` and `ValueError` are raised directly for path/input issues and do not need to be caught as `PDFExtractionError`.

---

### Behaviour Notes

- **Empty PDF** — if the document has zero pages, an empty string `""` is returned and a warning is logged.
- **Page-level errors** — if one page fails to extract, that page contributes an empty string and extraction continues for remaining pages. The error is logged.
- **No partial results lost** — all successfully extracted pages are still returned even if some pages failed.

---

### Examples Summary

```python
# Minimal usage
from pyaitk.PTT import extract_text_from_pdf
text = extract_text_from_pdf("file.pdf")

# Custom separator between pages
text = extract_text_from_pdf("file.pdf", page_separator="\n--- PAGE BREAK ---\n")

# Full error handling
from pyaitk.PTT import extract_text_from_pdf, PDFExtractionError
try:
    text = extract_text_from_pdf("file.pdf")
except FileNotFoundError:
    print("File not found.")
except PDFExtractionError as e:
    print(f"Extraction failed: {e}")

# Batch processing a folder
from pathlib import Path
from pyaitk.PTT import extract_text_from_pdf, PDFExtractionError
for f in Path("./docs").glob("*.pdf"):
    try:
        print(f"{f.name}: {len(extract_text_from_pdf(str(f)))} chars")
    except PDFExtractionError:
        print(f"{f.name}: failed")
```

---

## PPTXExtractor — PowerPoint Content Extraction Utility

A straightforward utility for extracting text, images, and tables from `.pptx` PowerPoint files using `python-pptx`. Processes all slides and organises extracted content by slide number.

---

### What Is This?

`PPTXExtractor` is a class-based PPTX extraction tool that provides:

- **Text extraction** — pulls all non-empty text from every shape across all slides
- **Image extraction** — saves embedded images to disk, preserving their original format (PNG, JPEG, etc.)
- **Table extraction** — reads all table shapes and returns row/cell data as nested lists
- **All-in-one extraction** — a single `extract_all()` call returns text, images, and tables together
- **Auto output directory** — creates the image output folder automatically if it doesn't exist
- **Slide-indexed results** — all output is keyed by slide number (1-based) for easy lookup

---

### How to Use

#### 1. Extract everything at once

```python
from pyaitk.PPTExtract import PPTXExtractor

extractor = PPTXExtractor("presentation.pptx")
data = extractor.extract_all()

# data["texts"]  → {slide_num: [str, ...]}
# data["images"] → {slide_num: [image_path, ...]}
# data["tables"] → {slide_num: [[[cell, ...], ...], ...]}
```

#### 2. Extract text only

```python
extractor = PPTXExtractor("presentation.pptx")
texts = extractor.extract_text()

for slide_num, lines in texts.items():
    print(f"Slide {slide_num}:")
    for line in lines:
        print(f"  {line}")
```

#### 3. Extract and save images

```python
extractor = PPTXExtractor("presentation.pptx", image_output_dir="my_images")
images = extractor.extract_images()

for slide_num, paths in images.items():
    for path in paths:
        print(f"Slide {slide_num}: saved → {path}")
```

Images are saved as `slide{N}_image{M}.{ext}` inside the output directory.

#### 4. Extract tables

```python
extractor = PPTXExtractor("presentation.pptx")
tables = extractor.extract_tables()

for slide_num, slide_tables in tables.items():
    for table in slide_tables:
        for row in table:
            print("\t".join(row))
```

#### 5. Iterate all content by slide

```python
extractor = PPTXExtractor("presentation.pptx")
data = extractor.extract_all()

for slide_num in data["texts"]:
    print(f"\n--- Slide {slide_num} ---")

    for text in data["texts"][slide_num]:
        print(f"  Text: {text}")

    for img_path in data["images"].get(slide_num, []):
        print(f"  Image: {img_path}")

    for table in data["tables"].get(slide_num, []):
        for row in table:
            print("  Row:", "\t".join(row))
```

---

### API Reference

#### `PPTXExtractor(pptx_path, image_output_dir)`

| Parameter          | Type  | Default               | Description                                      |
|--------------------|-------|-----------------------|--------------------------------------------------|
| `pptx_path`        | `str` | *(required)*          | Path to the `.pptx` file                         |
| `image_output_dir` | `str` | `"extracted_images"`  | Directory where extracted images will be saved   |

The image output directory is created automatically if it does not exist.

---

#### `extract_text()` → `dict[int, list[str]]`

Returns all non-empty text from every shape across all slides.

```python
{
    1: ["Title of Slide One", "Bullet point A", "Bullet point B"],
    2: ["Slide Two heading", "Some body text"],
}
```

---

#### `extract_images()` → `dict[int, list[str]]`

Saves all embedded images to `image_output_dir` and returns their file paths.

```python
{
    1: ["extracted_images/slide1_image1.png"],
    3: ["extracted_images/slide3_image1.jpeg", "extracted_images/slide3_image2.png"],
}
```

Filenames follow the pattern: `slide{slide_num}_image{shape_num}.{ext}`

---

#### `extract_tables()` → `dict[int, list[list[list[str]]]]`

Returns table data as nested lists: slide → list of tables → list of rows → list of cell strings.

```python
{
    2: [
        [["Header A", "Header B"], ["Row 1A", "Row 1B"], ["Row 2A", "Row 2B"]],
    ]
}
```

---

#### `extract_all()` → `dict`

Runs all three extractors and returns a combined dictionary:

```python
{
    "texts":  { 1: [...], 2: [...] },
    "images": { 1: [...], 3: [...] },
    "tables": { 2: [...] },
}
```

---

### Examples Summary

```python
from pyaitk.PPTExtract import PPTXExtractor

# Extract everything
data = PPTXExtractor("deck.pptx").extract_all()

# Text only
texts = PPTXExtractor("deck.pptx").extract_text()

# Images saved to a custom folder
images = PPTXExtractor("deck.pptx", image_output_dir="assets").extract_images()

# Tables only
tables = PPTXExtractor("deck.pptx").extract_tables()

# Iterate slide by slide
extractor = PPTXExtractor("deck.pptx")
data = extractor.extract_all()
for slide_num in data["texts"]:
    print(f"Slide {slide_num}:", data["texts"][slide_num])
```

---

## NER System — Named Entity Recognition

A complete, modular Named Entity Recognition system built on spaCy. Covers the full ML lifecycle: text preprocessing, inference, postprocessing, training with early stopping, evaluation with precision/recall/F1, and persistent entity storage.

---

### What Is This?

This package is a NER system with six cooperating modules:

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

### Quick Start

```python
from pyaitk.NER import NERPipeline, EntityStore

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

### Module Guide

#### NERPipeline — Inference

```python
from pyaitk.NER import NERPipeline

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

#### NERTrainer — Training

```python
from pyaitk.NER import NERTrainer
from pyaitk.NER.trainer import TrainerConfig

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

#### NEREvaluator — Metrics

```python
from pyaitk.NER import NEREvaluator, NERPipeline

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

#### TextPreprocessor — Input Cleaning

```python
from pyaitk.NER import TextPreprocessor
from pyaitk.NER.preprocessor import PreprocessorConfig

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

#### EntityPostprocessor — Output Cleaning

```python
from pyaitk.NER import EntityPostprocessor
from pyaitk.NER.postprocessor import PostprocessorConfig

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

#### EntityStore — Persistence & Analytics

```python
from pyaitk.NER import EntityStore

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

#### Logging Setup

```python
from pyaitk.NER.logging_config import setup_logging

setup_logging(level="DEBUG", log_file="ner.log")
```

| Parameter | Default                                      | Description                           |
|-----------|----------------------------------------------|---------------------------------------|
| `level`   | `"INFO"`                                     | Logging level                         |
| `log_file`| `None`                                       | Optional path for file output         |
| `fmt`     | `"%(asctime)s [%(levelname)s] %(name)s — …"` | Log record format string              |

---

### Full End-to-End Example

```python
from pyaitk.NER import NERPipeline, NERTrainer, NEREvaluator, EntityStore
from pyaitk.NER.trainer import TrainerConfig
from pyaitk.NER.logging_config import setup_logging

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

### Data Format

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

---

## MathAI — Production-grade Mathematical Expression Solver

A robust symbolic mathematics solver built on SymPy, supporting simplification, equation solving, differentiation, integration, matrix analysis, and Taylor series expansion — with automatic operation detection, input validation, and structured result objects.

---

### What Is This?

`MathAI` is a symbolic math engine that provides:

- **Auto-detection** — `process()` and `MathAI()` infer the operation type from the query automatically
- **Symbolic computation** — exact symbolic results powered by SymPy (no floating-point approximation unless requested)
- **Numeric evaluation** — automatically computes a decimal approximation when the result is a pure number
- **Equation solving** — single equations and systems of equations (via `x + y = 5 and x - y = 1` syntax)
- **Calculus** — differentiation (any order) and definite/indefinite integration
- **Matrix analysis** — determinant, trace, rank, inverse, and eigenvalues in one call
- **Taylor/Laurent series** — configurable expansion point and order
- **Input validation** — rejects empty input, oversized expressions, and dangerous code patterns
- **Structured results** — every operation returns a `MathResult` dataclass with `success`, `result`, `simplified`, `numeric`, and `metadata` fields
- **Structured logging** — all internal events go through Python's `logging` module
- **Convenience API** — single `MathAI(query, operation)` function for quick one-liner use

---

### How to Use

#### 1. One-liner convenience function (simplest)

```python
from pyaitk.MathAI import MathAI

print(MathAI("x^2 + 2*x + 1"))
print(MathAI("x^2 - 4 = 0", operation="solve"))
print(MathAI("sin(x)", operation="differentiate"))
```

#### 2. Auto-detect operation

The `MathAI()` function (and `MathSolver.process()`) detect the operation from the query:

- Contains `=` → solve
- Starts with `Matrix` → matrix operations
- Starts with `diff` or `derivative(...)` → differentiate
- Starts with `int` or `integrate(...)` → integrate
- Anything else → simplify

```python
from pyaitk.MathAI import MathAI

print(MathAI("x^2 - 9 = 0"))                         # → solve
print(MathAI("Matrix([[2, 1], [5, 3]])"))             # → matrix
print(MathAI("sin(x)^2 + cos(x)^2"))                 # → simplify
```

#### 3. Using `MathSolver` directly

```python
from pyaitk.mathai import MathSolver

solver = MathSolver()

result = solver.simplify("sin(x)^2 + cos(x)^2")
print(result)
```

#### 4. Simplification

```python
result = solver.simplify("x^3 + 3*x^2 + 3*x + 1")
print(result.simplified)   # (x + 1)**3
print(result.numeric)      # set if result is a pure number
```

#### 5. Solving equations

```python
# Single equation
result = solver.solve_equation("x^2 - 4 = 0")
print(result.result)   # [{x: -2}, {x: 2}]

# System of equations (separate with "and")
result = solver.solve_equation("x + y = 10 and x - y = 2")
print(result.result)   # [{x: 6, y: 4}]
```

#### 6. Differentiation

```python
# First derivative (default)
result = solver.differentiate("x^3 + sin(x)")
print(result.result)      # 3*x**2 + cos(x)
print(result.simplified)  # simplified form

# Higher-order derivative
result = solver.differentiate("x^5", var="x", order=3)
print(result.result)   # 60*x**2
```

#### 7. Integration

```python
# Indefinite integral
result = solver.integrate("x^2 + 3*x")
print(result.result)   # x**3/3 + 3*x**2/2

# Definite integral
result = solver.integrate("x^2", var="x", limits=(0, 1))
print(result.result)   # 1/3
print(result.numeric)  # 0.333333333333333
```

#### 8. Matrix operations

```python
result = solver.matrix_operations("Matrix([[1, 2], [3, 4]])")
print(result.metadata["determinant"])   # -2
print(result.metadata["inverse"])       # Matrix([[-2, 1], [3/2, -1/2]])
print(result.metadata["eigenvalues"])   # {-sqrt(33)/2 + 5/2: 1, sqrt(33)/2 + 5/2: 1}
print(result.metadata["rank"])          # 2
print(result.metadata["trace"])         # 5
```

#### 9. Taylor series expansion

```python
# Default: expand around x=0, 6 terms
result = solver.series_expansion("sin(x)")
print(result.result)   # x - x**3/6 + x**5/120 + O(x**6)

# Custom point and order
result = solver.series_expansion("exp(x)", var="x", point=0, order=4)
print(result.result)   # 1 + x + x**2/2 + x**3/6 + O(x**4)
```

---

### API Reference

#### `MathAI(query, operation)` → `str`

Top-level convenience function. Returns a formatted string of the result.

| Parameter   | Type  | Default  | Description                                                                 |
|-------------|-------|----------|-----------------------------------------------------------------------------|
| `query`     | `str` | `'1*x + 2*x - 3*x'` | Mathematical expression or equation                          |
| `operation` | `str` | `'auto'` | One of: `auto`, `simplify`, `solve`, `differentiate`, `integrate`, `matrix`, `series` |

---

#### `MathSolver` methods

| Method                                              | Description                                          |
|-----------------------------------------------------|------------------------------------------------------|
| `process(query)`                                    | Auto-detect and dispatch to the right operation      |
| `simplify(expr)`                                    | Simplify a symbolic expression                       |
| `solve_equation(expr)`                              | Solve one or more equations                          |
| `differentiate(expr, var, order)`                   | Differentiate to any order                           |
| `integrate(expr, var, limits)`                      | Indefinite or definite integral                      |
| `matrix_operations(expr)`                           | Det, trace, rank, inverse, eigenvalues               |
| `series_expansion(expr, var, point, order)`         | Taylor/Laurent series around a point                 |

---

#### `MathResult` fields

Every method returns a `MathResult` dataclass:

| Field        | Type            | Description                                              |
|--------------|-----------------|----------------------------------------------------------|
| `success`    | `bool`          | `True` if the operation completed without error          |
| `operation`  | `str`           | Name of the operation performed                          |
| `input_expr` | `str`           | The original input string                                |
| `result`     | `str` or `None` | Primary result of the operation                          |
| `simplified` | `str` or `None` | Simplified form (where applicable)                       |
| `numeric`    | `str` or `None` | Decimal evaluation (when result is a pure number)        |
| `error`      | `str` or `None` | Error message if `success` is `False`                    |
| `metadata`   | `dict` or `None`| Extra details (variable, limits, matrix properties, etc.)|

Calling `str(result)` produces a human-readable summary of all populated fields.

---

### Supported Symbols and Functions

The solver pre-loads a wide set of SymPy functions accessible directly in expressions:

| Category          | Available                                                      |
|-------------------|----------------------------------------------------------------|
| Trigonometric     | `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh` |
| Exponential / Log | `exp`, `log`, `ln`                                             |
| Constants         | `e`, `pi`, `I` (imaginary unit), `oo` (infinity)              |
| Algebra           | `sqrt`, `abs`, `floor`, `ceil`, `factorial`, `binomial`        |
| Calculus          | `diff`, `integrate`, `limit`, `Sum`, `Product`                 |
| Linear Algebra    | `Matrix`, `det`, `transpose`                                   |
| Simplification    | `factor`, `expand`, `simplify`, `cancel`, `apart`, `together`  |
| Special Functions | `gamma`, `erf`                                                 |
| Variables         | `x y z a b c … theta phi alpha beta gamma delta epsilon`       |

---

### Input Validation

Before any operation, expressions are checked for:

- Empty or whitespace-only input
- Expressions exceeding 10,000 characters
- Forbidden patterns: `__`, `import`, `eval`, `exec`, `compile`, `open`, `file`

Rejected inputs return a `MathResult` with `success=False` and a descriptive `error` message — no exceptions are raised to the caller.

---

### Examples Summary

```python
from pyaitk.mathai import MathAI, MathSolver

# One-liner API
print(MathAI("x^2 + 2*x + 1"))
print(MathAI("x^2 - 4 = 0", operation="solve"))
print(MathAI("sin(x)", operation="differentiate"))
print(MathAI("x^2", operation="integrate"))
print(MathAI("Matrix([[1, 2], [3, 4]])", operation="matrix"))
print(MathAI("cos(x)", operation="series"))

# Using MathSolver directly
solver = MathSolver()
r = solver.solve_equation("x + y = 10 and x - y = 2")
print(r.result)

r = solver.differentiate("x^5", order=3)
print(r.simplified)

r = solver.integrate("x^2", limits=(0, 1))
print(r.numeric)

r = solver.matrix_operations("Matrix([[1, 2], [3, 4]])")
print(r.metadata)
```

---
### Prompts for MathAI

#### Solve normal numeric problems.

```c
1 + 2 + 3 + 4 - 5 (1 - 55) + 10
```

#### Solve symbolic methamatic.

```c
X + 2Y - X + 10Z * (10 - 100)X
```

#### Matrix

```c
Matrix([[1, 0], [0, 1]])

Matrix([[1,2,3], [2, 3, 10]])
```

#### Trigonometric Functions

##### sin

```c
sin(0)
sin(30)
sin(45)
sin(60)
sin(90)
...
sin(180)
...
```

###### Syntax

```c
sin(<value of theata>)
```

##### cos

```c
cos(0)
cos(30)
cos(45)
cos(60)
cos(90)
...
cos(180)
...
```

###### Syntax

```c
cos(<value of theata>)
```

##### tan

```c
tan(0)
tan(30)
tan(45)
tan(60)
tan(90)
...
tan(180)
...
```

###### Syntax

```c
tan(<value of theata>)
```

##### cosec

```cpp
csc(0)
csc(30)
csc(45)
csc(60)
csc(90)
...
csc(180)
...
```

###### Syntax

```cpp
csc(<value of theata>)
```

##### sec

```cpp
sec(0)
sec(30)
sec(45)
sec(60)
sec(90)
...
sec(180)
...
```

###### Syntax

```cpp
sec(<value of theata>)
```

##### cot

```cpp
cot(0)
cot(30)
cot(45)
cot(60)
cot(90)
...
cot(180)
...
```

###### Syntax

```cpp
cot(<value of theata>)
```

#### Limit

```cpp
limit('2X')
limit('2X + 3Y')
```

##### Syntax

```cpp
limit(<symbolic and trigonometric values in string formate>)
```

#### Determinants

```cpp
det(Matrix([[10]]))
det(Matrix([[10], [20]]))
det(Matrix([[10], [30], [40]]))
det(Matrix([[10], [100], [0], [50]]))
det(Matrix([[10], [97], [95], [1], [99]]))
det(Matrix([[10], [11], [100], [3], [2], [150]]))
det(Matrix([[10, 20]]))
det(Matrix([[10, 20], [20, 30]]))
det(Matrix([[10, 20], [60, 70], [80, 100]]))
det(Matrix([[10, 20, 30]]))
det(Matrix([[10, 20, 30], [40, 50, 60]]))
det(Matrix([[10, 20, 30], [40, 50, 60], [90, 100, 102]]))
det(Matrix([[10, 20, 30], [40, 50, 60], [90, 100, 102], [95, 101, 1000]]))
...
```

#### Log

```cpp
log(10)
log(20)
log(0)
log(100)
...
```

#### Ln

```cpp
ln(0)
ln(1)
ln(10)
ln(100)
ln(1000)
ln(102)
ln(3)
ln(90)
ln(893)
ln(9)
...
```

#### E

```cpp
e()
```

`↑` Get the value of `e`.

```cpp
e(10)
e(3)
e(21)
e(38)
e(0)
...
```

#### ⊼ (`pi`)

```cpp
pi()
```

`↑` Get the value of `⊼`

#### Square

Get all square of the value.

```cpp
sqrt(10)
sqrt(2)
sqrt(30)
sqrt(28039)
sqrt(19)
sqrt(289843190)
...
```

#### Differential

```cpp
diff('2x')
diff('x')
diff('y')
...
```

Give all the values in string formate.

#### Integration (`∫`)

Give all the values in string formate.

```cpp
integrate('dx')
integrate('x dx')
integrate('2x dx')
integrate('2x dy')
integrate('x dy')
integrate('2x + 3y - 3y dx')
...
```

#### Factor

Get all the factors of the numbers

```cpp
factor(10)
factor(213)
factor(389)
factor(1983)
factor(0)
factor(12)
...
```

---

## ITT — Image-to-Text (OCR) Utility

A minimal, dependency-light OCR utility built on EasyOCR. Extracts text from images in a single function call, with the recognition model loaded once at startup for efficient repeated use.

---

### What Is This?

`ITT` is a lightweight image-to-text module that provides:

- **Single function API** — `ITT(image_path)` returns all recognised text as a plain string
- **EasyOCR backend** — deep learning-based OCR supporting 80+ languages out of the box
- **Model loaded once** — the `reader` is initialised at module level so repeated calls pay no reload cost
- **Multi-language support** — pass any EasyOCR-supported language codes at call time
- **Zero boilerplate** — no configuration classes, no context managers; just import and call

---

#### Note
> EasyOCR downloads model weights on first use (~100 MB). An internet connection is required for the initial download; subsequent runs work fully offline.

---

### How to Use

#### 1. Basic usage

```python
from pyaitk.ITT import ITT

text = ITT("screenshot.png")
print(text)
```

#### 2. Extract text from any image format

EasyOCR supports JPEG, PNG, BMP, TIFF, and more.

```python
text = ITT("photo.jpg")
text = ITT("scan.bmp")
text = ITT("document.tiff")
```

#### 3. Multi-language recognition

Pass a list of BCP-47 / EasyOCR language codes as the second argument.

```python
# English + French
text = ITT("menu.jpg", languages=["en", "fr"])

# English + Hindi
text = ITT("sign.png", languages=["en", "hi"])
```

> Note: The module-level `reader` is initialised with `['en']` only. To use other languages reliably, re-initialise `easyocr.Reader` with the desired codes before calling `readtext`.

#### 4. Batch processing multiple images

```python
from pyaitk.ITT import ITT
from pathlib import Path

results = {}
for img in Path("./images").glob("*.png"):
    results[img.name] = ITT(str(img))

for name, text in results.items():
    print(f"{name}: {text[:80]}")
```

#### 5. Using the result downstream

```python
text = ITT("invoice.png")

# Search for keywords
if "TOTAL" in text.upper():
    print("Invoice contains a total amount.")

# Write extracted text to file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)
```

---

### API Reference

#### `ITT(image_path, languages)` → `str`

Runs OCR on the given image and returns all detected text joined into a single space-separated string.

| Parameter    | Type        | Default    | Description                                      |
|--------------|-------------|------------|--------------------------------------------------|
| `image_path` | `str`       | *(required)* | Path to the image file                         |
| `languages`  | `list[str]` | `['en']`   | EasyOCR language codes to use for recognition   |

**Returns:** `str` — all detected text regions joined by spaces, in detection order.

---

### Module-level `reader`

```python
reader = easyocr.Reader(['en'])
```

The reader is created once when the module is imported. This means:

- Model files are downloaded on the **first import** only
- All subsequent `ITT()` calls reuse the same loaded model — no per-call overhead
- If you need languages beyond English, create your own `easyocr.Reader` instance with the required codes

---

### Supported Languages (selected)

EasyOCR supports 80+ languages. Common codes:

| Code  | Language   | Code  | Language    |
|-------|------------|-------|-------------|
| `en`  | English    | `fr`  | French      |
| `hi`  | Hindi      | `de`  | German      |
| `zh`  | Chinese    | `ja`  | Japanese    |
| `ko`  | Korean     | `ar`  | Arabic      |
| `es`  | Spanish    | `pt`  | Portuguese  |

Full list: [https://www.jaided.ai/easyocr](https://www.jaided.ai/easyocr)

---

### Notes

- **Detection order** — text regions are returned in the order EasyOCR detects them, which follows a rough top-to-bottom, left-to-right reading order but may vary for complex layouts.
- **Low-quality images** — blurry, low-contrast, or heavily compressed images will reduce accuracy. Pre-processing with a library like Pillow (resize, sharpen, greyscale) can improve results.
- **GPU acceleration** — EasyOCR uses the GPU automatically if a CUDA-capable device is available. On CPU, recognition is slower but fully functional.

---

### Examples Summary

```python
from pyaitk.ITT import ITT

# Basic
text = ITT("image.png")

# Multi-language
text = ITT("document.jpg", languages=["en", "fr"])

# Batch
from pathlib import Path
for img in Path("./scans").glob("*.jpg"):
    print(img.name, ITT(str(img)))

# Save to file
with open("extracted.txt", "w") as f:
    f.write(ITT("receipt.png"))
```

---

## GrammarCorrector — Grammar Correction Pipeline

A three-tier grammar correction system combining SpaCy rule-based morphology, a scikit-learn statistical token corrector, and a PyTorch LSTM Seq2Seq neural model — all unified behind a single `GrammarCorrector` facade with a JSON intents system for training on real-world error pairs.

---

### What Is This?

`Grammar` provides a complete grammar correction pipeline with three escalating tiers:

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

---

### How to Use

#### 1. Rule-only correction (no fitting needed)

Tier 1 (SpaCy + rules) always runs — no training data required.

```python
from pyaitk.Grammar import GrammarCorrector

with GrammarCorrector() as gc:
    print(gc.correct("i wants to here more about this"))
    # → "I wants to hear more about this."

    print(gc.correct("she go to the market"))
    # → "She goes to the market."
```

#### 2. Full three-tier pipeline from a sentence list

```python
from pyaitk.Grammar import GrammarCorrector

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

#### 3. Train from a JSON intents file (recommended)

```python
from pyaitk.Grammar import GrammarCorrector

with GrammarCorrector() as gc:
    gc.fit_from_intents_file("intents.json")
    print(gc.correct("i wants to here you"))
```

#### 4. Train from intents with a tag filter

Only use specific error categories for training:

```python
from pyaitk.Grammar import GrammarCorrector, load_intents

examples = load_intents("intents.json", tags=["verb_agreement", "capitalisation"])

with GrammarCorrector() as gc:
    gc.fit_from_intents(examples)
    print(gc.correct("he go home"))
```

#### 5. Merge intents + plain sentences

```python
from pyaitk.Grammar import GrammarCorrector, load_intents

sentences = ["The cat sat on the mat.", "I enjoy reading books."]
examples = load_intents("intents.json")

with GrammarCorrector() as gc:
    gc.fit(sentences, intents=examples)
    print(gc.correct("u r gr8"))
```

#### 6. Save and load a trained model

```python
from pyaitk.Grammar import GrammarCorrector

# Train and save
with GrammarCorrector() as gc:
    gc.fit_from_intents_file("intents.json")
    gc.save("models/grammar_v1")

# Load and use
with GrammarCorrector() as gc:
    gc.load("models/grammar_v1")
    print(gc.correct("she go to the market"))
```

#### 7. One-liner convenience function

```python
from pyaitk.Grammar import quick_correct

# Rule-only (no training data)
print(quick_correct("i go to school"))

# With training
sentences = ["She goes to school.", "I am happy."]
print(quick_correct("i go to school", sentences=sentences))
```

#### 8. Custom configuration

```python
from pyaitk.Grammar import GrammarCorrector, CorrectorConfig

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

### JSON Intents File Format

Two supported layouts — freely mixable in the same file.

#### Layout A — flat pair

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

#### Layout B — grouped examples

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

#### Loading and inspecting intents

```python
from pyaitk.Grammar import load_intents, intents_summary, intents_to_pairs

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

### API Reference

#### `GrammarCorrector(config?)`

| Parameter | Type                      | Default | Description                              |
|-----------|---------------------------|---------|------------------------------------------|
| `config`  | `CorrectorConfig` or `None` | `None`  | Uses `CorrectorConfig()` defaults if `None` |

##### Methods

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

#### `CorrectorConfig` fields

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

#### Exception Hierarchy

```
GrammarCorrectorError
├── NotFittedError          — correct() called before fit()
├── TokenizerError          — BPE tokenizer missing special tokens
└── IntentsValidationError  — JSON intents file fails schema validation
```

---

#### Data Helpers

```python
from pyaitk.Grammar import corrupt_sentence, build_dataset

# Apply deterministic corruption rules to one sentence
noisy = corrupt_sentence("She goes to the market.")
# → "She go 2 the market."

# Build (corrupted, correct) pairs from a list of correct sentences
dataset = build_dataset(["I am happy.", "She goes to school."])
# → [("i'm happy.", "I am happy."), ...]
```

The corruption engine covers 100+ rules across five categories: homophones (`their/there/they're`), internet slang (`u/r/gr8/lol`), informal contractions (`wanna/gonna/gotta`), common typos (`teh/freind/recieve`), and punctuation corruption.

---

#### Low-level API

These are available for advanced use when you want direct access to individual tiers or the training loop:

```python
from pyaitk.Grammar import (
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

### Architecture Overview

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

### Examples Summary

```python
from pyaitk.Grammar import GrammarCorrector, load_intents, quick_correct

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

---

## EYE — Real-Time Object Detection

A production-ready webcam object detection application built on YOLOv8 (Ultralytics) and CustomTkinter. Supports a full GUI with live controls, headless CLI detection, a unified `EyeSession` facade, and full context-manager support on every public class — with extensive threading fixes and feature additions over a naive implementation.

---

### What Is This?

`EYE` is a fully-featured real-time object detection module that provides:

- **Full context-manager protocol** — every public class (`CameraManager`, `ObjectDetector`, `DetectionApp`, `EyeSession`) supports `with` blocks with guaranteed resource cleanup
- **`EyeSession` facade** — unified entry point with `.gui()` and `.headless()` factory context managers covering the complete lifecycle
- **Live webcam detection** — reads frames from any connected camera and runs YOLOv8 inference at a capped FPS
- **Full GUI application** — dark-mode CustomTkinter UI with video feed, control panel, and detection log
- **Headless CLI mode** — `simple_detect()` uses context managers internally for safe camera/detector teardown
- **Hot-swappable model variants** — switch between `yolov8n/s/m/l/x.pt` at runtime without restarting
- **Class filter** — show bounding boxes only for specific classes (e.g. `person`, `car`)
- **Confidence threshold** — adjustable slider (0.1–0.95), persisted to `~/.yolov8_app.json`
- **Detection heat-map** — alpha-blended overlay that accumulates and decays detection positions
- **Video recording** — save annotated footage to `.mp4` at camera resolution
- **Screenshot** — save the current annotated frame as a timestamped JPG
- **Live FPS counter** — exponential moving average over the last 30 frames
- **Detection log** — scrollable timestamped list of the last 200 detection events
- **Keyboard shortcuts** — Space (pause), S (screenshot), R (record), Q (quit)
- **Serialisable config** — `DetectionConfig` saves/loads all settings as JSON
- **Thread-safe design** — all widget mutations dispatched via `self.after()`, frame buffer protected with `threading.Lock`
- **Legacy aliases** — `EYE()` and `OpenEYE()` for backwards compatibility with `core.py` / pyaitk

---

#### Note

> YOLOv8 model weights are downloaded automatically on first use (~6 MB for `yolov8n.pt`).
> An internet connection is required for the initial download; subsequent runs work offline.
---

### Context-Manager Patterns

All four public classes now support the full context-manager protocol. Resources are always released — even when exceptions are raised.

#### `EyeSession` — recommended top-level API

```python
from pyaitk.eye import EyeSession

# GUI mode (blocks until window closed)
with EyeSession.gui() as session:
    session.run()
    print(session.last_detections)

# Headless mode (blocks until target seen or 'q' pressed)
with EyeSession.headless(target_class="person") as session:
    detected = session.run()
    print(detected)

# GUI with custom config
from pyaitk.eye import DetectionConfig
cfg = DetectionConfig(model_name="yolov8s.pt", show_heatmap=True, confidence_threshold=0.6)
with EyeSession.gui(config=cfg) as session:
    session.run()
```

#### `CameraManager` — camera resource

```python
from pyaitk.eye import CameraManager

# __enter__ / __exit__
with CameraManager(camera_index=0, width=640, height=480) as cam:
    ret, frame = cam.read()

# Factory context manager
with CameraManager.open_device(0, width=1280, height=720) as cam:
    ret, frame = cam.read()
    print(cam.is_opened)   # True
```

#### `ObjectDetector` — inference resource

```python
from pyaitk.eye import ObjectDetector, DetectionConfig

cfg = DetectionConfig()

# __enter__ / __exit__ (provide your own model)
from pyaitk.eye import ModelLoader
model = ModelLoader.load_model("yolov8n.pt")
with ObjectDetector(model, cfg) as det:
    classes, annotated, count, confs = det.detect(frame)

# Factory context manager (loads model internally)
with ObjectDetector.from_model("yolov8n.pt", cfg) as det:
    classes, annotated, count, confs = det.detect(frame)
```

#### `DetectionApp` — GUI application

```python
from pyaitk.eye import DetectionApp, DetectionConfig
import customtkinter as ctk

# __enter__ / __exit__
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
with DetectionApp(DetectionConfig()) as app:
    app.mainloop()

# Factory context manager (sets up CTk theme automatically)
with DetectionApp.run_app(DetectionConfig()) as app:
    app.mainloop()
```

#### Composing context managers (low-level pipeline)

```python
from pyaitk.eye import CameraManager, ObjectDetector
import cv2

with CameraManager(0) as cam:
    with ObjectDetector.from_model("yolov8n.pt") as det:
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            classes, annotated, count, confs = det.detect(frame)
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
cv2.destroyAllWindows()
# Camera and detector released automatically on exit
```

---

### How to Use

#### 1. Launch the GUI (simplest)

```python
from pyaitk.eye import launch_gui

launch_gui()
```

#### 2. GUI with custom config

```python
from pyaitk.eye import launch_gui, DetectionConfig

config = DetectionConfig(
    model_name="yolov8s.pt",
    confidence_threshold=0.6,
    target_fps=25,
    show_heatmap=True,
    filter_classes=["person", "car"],
)
launch_gui(config)
```

#### 3. Headless detection (no GUI)

`simple_detect()` opens an OpenCV window and runs until the target class is seen or `q` is pressed. Uses `CameraManager` and `ObjectDetector` context managers internally — camera is always released.

```python
from pyaitk.eye import simple_detect

detected = simple_detect(target_class="person")
print("Detected:", detected)
```

> **macOS note:** `cv2.imshow` must be called from the main thread. Only call `simple_detect()` from the main thread.

#### 4. Headless with custom config

```python
from pyaitk.eye import simple_detect, DetectionConfig

config = DetectionConfig(
    model_name="yolov8m.pt",
    confidence_threshold=0.45,
    camera_index=1,
    target_fps=20,
)
detected = simple_detect(target_class="car", config=config)
print(detected)
```

#### 5. Legacy aliases (pyaitk / core.py compatibility)

```python
from pyaitk.eye import EYE, OpenEYE

detected = EYE()      # → simple_detect()
OpenEYE()             # → launch_gui()
```

#### 6. Low-level components with context managers

```python
from pyaitk.eye import CameraManager, ObjectDetector, DetectionConfig
import cv2

config = DetectionConfig()

with CameraManager(camera_index=0, width=640, height=480) as cam:
    with ObjectDetector.from_model("yolov8n.pt", config) as det:
        ret, frame = cam.read()
        if ret:
            detected_classes, annotated, count, confidences = det.detect(frame)
            print(f"Detected {count} objects: {detected_classes}")
            cv2.imshow("Frame", annotated)
            cv2.waitKey(0)
cv2.destroyAllWindows()
# No explicit release needed — context managers handle it
```

---

### GUI Controls Reference

| Control | Description |
|---|---|
| Model selector | Switch between `yolov8n/s/m/l/x.pt` (hot-swap, no restart needed) |
| Confidence slider | Detection threshold from 0.10 to 0.95 |
| Class filter field | Type a class name and press Enter or "Add Filter" |
| Clear Filter button | Remove all filters (show all classes) |
| Heatmap toggle | Enable/disable the alpha-blended detection heat-map |
| Pause / Resume button | Freeze/unfreeze the video loop |
| Screenshot button | Save annotated frame as `screenshot_YYYYMMDD_HHMMSS.jpg` |
| Record button | Start/stop saving annotated video as `recording_*.mp4` |
| Detection log | Scrollable list of last 200 timestamped events |
| Clear Log button | Empty the detection log |
| Status bar | Camera info, resolution, active model name |

#### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | Pause / Resume |
| `S` | Screenshot |
| `R` | Start / Stop recording |
| `Q` | Quit |

---

### API Reference

#### `EyeSession`

Unified high-level facade with two factory context managers.

##### `EyeSession.gui(config?)` → context manager

Launches the full CustomTkinter GUI. Blocks on `session.run()` until the window is closed. Tears down the app on exit.

```python
with EyeSession.gui(config=DetectionConfig(model_name="yolov8s.pt")) as session:
    session.run()
    print(session.last_detections)
```

##### `EyeSession.headless(target_class, config?)` → context manager

Headless detection. Blocks on `session.run()` until target is detected or `q` is pressed. Calls `cv2.destroyAllWindows()` on exit.

```python
with EyeSession.headless("car") as session:
    detected = session.run()
```

| Property | Type | Description |
|---|---|---|
| `last_detections` | `list[str]` | Class names detected at the time of exit |

---

#### `launch_gui(config)` → `None`

Functional launcher. Uses `DetectionApp.run_app()` context manager internally.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `DetectionConfig` or `None` | `None` | Load from `~/.yolov8_app.json` if `None` |

---

#### `simple_detect(target_class, config)` → `list[str]`

Headless OpenCV detection loop. Uses `CameraManager` and `ObjectDetector` context managers internally.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_class` | `str` | `"person"` | Class name to trigger exit on |
| `config` | `DetectionConfig` or `None` | `None` | Uses defaults if `None` |

**Returns:** sorted list of all class names detected at the time of exit.

---

#### `CameraManager(camera_index, width, height)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `camera_index` | `int` | `0` | OpenCV device index |
| `width` | `int` | `640` | Requested frame width |
| `height` | `int` | `480` | Requested frame height |

| Method / Property | Returns | Description |
|---|---|---|
| `open()` | `bool` | Open the device; returns `True` on success |
| `read()` | `tuple[bool, ndarray\|None]` | Read one frame |
| `release()` | `None` | Release the capture device |
| `is_opened` | `bool` | `True` if device is currently open |
| `__enter__` / `__exit__` | — | Opens on enter, releases on exit |
| `CameraManager.open_device(idx, w, h)` | context manager | Factory `@contextmanager` |

---

#### `ObjectDetector(model, config)`

| Method | Returns | Description |
|---|---|---|
| `detect(frame)` | `tuple` | Run inference; see return table below |
| `__enter__` / `__exit__` | — | Clears heatmap accumulator on exit |
| `ObjectDetector.from_model(name, config?)` | context manager | Loads model + yields detector |

**`detect()` return values:**

| Field | Type | Description |
|---|---|---|
| `detected_classes` | `set[str]` | All class names above confidence threshold |
| `annotated_frame` | `np.ndarray` | Original-resolution BGR frame with bounding boxes |
| `object_count` | `int` | Total number of detections |
| `class_confidences` | `dict[str, float]` | Highest confidence per class name |

---

#### `DetectionApp(config)`

| Method | Description |
|---|---|
| `mainloop()` | Start the Tk event loop (blocks) |
| `__enter__` / `__exit__` | Calls `_close_app()` on exit |
| `DetectionApp.run_app(config?)` | Factory context manager; sets CTk theme |

---

#### `DetectionConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"yolov8n.pt"` | YOLOv8 weight file name |
| `target_class` | `str` | `"person"` | Class to watch for in headless mode |
| `confidence_threshold` | `float` | `0.5` | Minimum detection confidence (0.0–1.0) |
| `frame_width` | `int` | `640` | Requested camera frame width |
| `frame_height` | `int` | `480` | Requested camera frame height |
| `detection_size` | `tuple[int,int]` | `(416, 416)` | Resize resolution for YOLO inference |
| `camera_index` | `int` | `0` | OpenCV camera index |
| `target_fps` | `int` | `30` | FPS cap for the detection loop |
| `show_heatmap` | `bool` | `False` | Enable detection heat-map overlay |
| `filter_classes` | `list[str]` | `[]` | Restrict boxes to these classes; empty = all |

**Persistence:**

```python
config = DetectionConfig(confidence_threshold=0.7)
config.save()                 # → ~/.yolov8_app.json

config = DetectionConfig.load()   # restore from file
```

---

#### `ModelLoader.load_model(model_name)` → `YOLO | None`

Tries four strategies in order:

1. Package resource (if running inside a package)
2. Script directory and adjacent `eye/` subfolder
3. Current working directory
4. Auto-download from Ultralytics

---

### Architecture Overview

```
EyeSession
├── EyeSession.gui()       → DetectionApp.run_app() → DetectionApp.__enter__/__exit__
└── EyeSession.headless()  → simple_detect()
                                ├── CameraManager.__enter__/__exit__
                                └── ObjectDetector.from_model().__enter__/__exit__

DetectionApp (GUI)
├── Main thread   ──→  CTk event loop + all widget mutations (via self.after())
│                             ↑
│                       polls frame queue at ~60 Hz
│                             ↑
└── Video thread  ──→  camera.read() → detector.detect() → FramePacket → Queue(maxsize=2)

Resource ownership
──────────────────
CameraManager.__exit__    → cap.release()
ObjectDetector.__exit__   → _heatmap = None
DetectionApp.__exit__     → thread.join(2s) → writer.release() → camera.release() → destroy()
EyeSession.__exit__       → app._close_app() (if GUI) / cv2.destroyAllWindows() (if headless)
```

---

### Examples Summary

```python
# ── EyeSession (recommended) ─────────────────────────────────────────────────

from pyaitk.eye import EyeSession, DetectionConfig

# GUI
with EyeSession.gui() as session:
    session.run()
    print(session.last_detections)

# GUI with custom config
cfg = DetectionConfig(model_name="yolov8s.pt", show_heatmap=True)
with EyeSession.gui(config=cfg) as session:
    session.run()

# Headless
with EyeSession.headless("car") as session:
    detected = session.run()

# ── Functional API ───────────────────────────────────────────────────────────

from pyaitk.eye import launch_gui, simple_detect
launch_gui()
detected = simple_detect("person")

# ── Low-level context managers ───────────────────────────────────────────────

from pyaitk.eye import CameraManager, ObjectDetector
import cv2

with CameraManager(0) as cam:
    with ObjectDetector.from_model("yolov8n.pt") as det:
        ret, frame = cam.read()
        classes, annotated, count, confs = det.detect(frame)
cv2.destroyAllWindows()

# ── DetectionApp directly ────────────────────────────────────────────────────

from pyaitk.eye import DetectionApp, DetectionConfig
with DetectionApp.run_app(DetectionConfig()) as app:
    app.mainloop()

# ── Legacy aliases ───────────────────────────────────────────────────────────

from pyaitk.eye import EYE, OpenEYE
EYE()       # headless detect
OpenEYE()   # launch GUI
```

---

## CLSE - Compositional Latent Synthesis Engine

A complete, self-contained Compositional Latent Synthesis Engine pipeline built entirely on NumPy, NLTK, scikit-learn, and PyTorch — no Stable Diffusion or external model weights required. Converts natural-language prompts into images through a multi-stage NLP → neural model → renderer architecture, with full support for procedural art, visual effects, animation, streaming, and a rich CLI.

---

### What Is This?

The CLSE system is a nine-module package covering every layer of the Compositional Latent Synthesis Engine stack:

| Module | Class / Entry Point | Responsibility |
|---|---|---|
| `TTI_config.py` | `TTIConfig`, `get_config()` | Master config — all tuneable parameters |
| `TTI_core.py` | `TTIImage`, `ImageCanvas`, `ColorUtils`, `ImageIO` | Pixel-level image engine (BMP/PNG/JPEG I/O, drawing) |
| `TTI_art.py` | `ProceduralArt`, `VisualEffects`, `StreamingWriter`, `AnimationEngine` | Procedural art, effects, animation, large-image streaming |
| `TTI_ai.py` | `TTIGenerator`, `NLPAnalyser`, `PromptAnalysis` | NLP pipeline + numpy VAE renderer + colour predictor |
| `TTI_model.py` | `TTIModel`, `TTITrainer`, `TTILoss` | 4-layer transformer VAE (3.8M params), training loop |
| `TTI_dataset.py` | `TTIDataset`, `Vocabulary` | Synthetic 50k-sample dataset generator with caching |
| `TTI_pipeline.py` | `TTIPipeline` | Full end-to-end pipeline connecting all modules |
| `TTI_train.py` | CLI training script | Production training with checkpointing and early stopping |
| `TTI_main.py` | `TTIPipeline` façade + `main()` | Unified entry point + full CLI |

#### Note

> These `*.py` files are inner file of CLSE (Compositional Latent Synthesis Engine).

---

### Quick Start

```python
from pyaitk.CLSE import TTIPipeline

pipe = TTIPipeline()

# Generate from a text prompt
img = pipe.generate("a calm blue ocean at sunset", output="ocean.png")

# Procedural art (no model needed)
img = pipe.art("mandelbrot", output="fractal.png")

# Apply an effect to an existing image
img = pipe.effect("sepia", "photo.png", output="photo_sepia.png")

# Analyse a prompt
analysis = pipe.analyse("dark gothic castle at midnight")
print(analysis.scene_type, analysis.colour_matches)
```

---

### Module Guide

---

#### Configuration

```python
from pyaitk.CLSE import TTIConfig, get_config, update_config, reset_config

# Global singleton (auto-discovers tti_config.json next to the module)
cfg = get_config()

# Read settings
print(cfg.image.default_width)       # 512
print(cfg.ai.model_type)             # "vae_numpy"
print(cfg.art.fractal_max_iter)      # 256
print(cfg.paths.output_dir)          # "tti_output"

# Bulk update
update_config(
    image={"default_width": 1024, "default_height": 1024},
    ai={"seed": 42, "num_inference_steps": 100},
)

# Save / load JSON snapshot
cfg.save("tti_config.json")
cfg2 = TTIConfig.load("tti_config.json")

# Create all output directories
cfg.ensure_dirs()

# Reset to factory defaults
reset_config()
```

**Config sections:**

| Section | Dataclass | Key fields |
|---|---|---|
| `cfg.image` | `ImageConfig` | `default_width`, `default_height`, `default_format`, `background_color`, `jpeg_quality` |
| `cfg.ai` | `AIConfig` | `model_type`, `latent_dim`, `vocab_size`, `num_inference_steps`, `guidance_scale`, `seed` |
| `cfg.art` | `ArtConfig` | `fractal_max_iter`, `blur_default_radius`, `noise_default_intensity`, `animation_fps` |
| `cfg.paths` | `PathConfig` | `output_dir`, `model_dir`, `cache_dir`, `log_dir` |
| `cfg.log` | `LogConfig` | `level`, `log_to_file`, `log_filename`, `show_progress` |

---

#### Image Engine

```python
from pyaitk.CLSE import TTIImage, ImageCanvas, ColorUtils, ImageIO, ImageValidator

# Create a blank image
img = TTIImage(width=512, height=512, bpp=24, background=(20, 30, 60))

# Pixel operations
img.set_pixel(100, 100, (255, 0, 0))
color = img.get_pixel(100, 100)   # (255, 0, 0)

# NumPy array interop
arr = img.to_array()              # shape (H, W, 3), dtype=uint8
img.from_array(arr)

# Save and load
img.save("output.png")
img.save("output.jpg", fmt="jpeg")
img.save("output.bmp", fmt="bmp")
img2 = TTIImage.load("output.png")

# Drawing (ImageCanvas)
canvas = ImageCanvas(img)
canvas.line(0, 0, 511, 511, (255, 255, 0))
canvas.rectangle(50, 50, 200, 200, (255, 100, 0), filled=True)
canvas.circle(256, 256, 100, (0, 200, 255), filled=False)
canvas.fill_background((10, 10, 30))

# Colour utilities
rgb = ColorUtils.hsv_to_rgb(0.6, 0.8, 0.9)
blended = ColorUtils.lerp((255, 0, 0), (0, 0, 255), t=0.5)
rgba = ColorUtils.to_rgba((100, 150, 200))         # adds alpha=255
clamped = ColorUtils.clamp(300)                    # → 255

# Multi-format I/O (static)
img = ImageIO.load("photo.png")
ImageIO.save(img, "photo.jpeg", quality=85)

# Integrity check
ImageValidator.validate(img)    # raises TTIImageError if corrupt
```

**Exception hierarchy:**

```
TTIError
├── TTIImageError   — pixel-level or dimension errors
└── TTIIOError      — file read/write failures
```

---

#### Procedural Art & Effects

#### ProceduralArt

All methods are static and return `TTIImage` objects.

```python
from pyaitk.CLSE import ProceduralArt, VisualEffects, StreamingWriter, AnimationEngine

# Fractals
img = ProceduralArt.mandelbrot_set(width=800, height=600)
img = ProceduralArt.julia_set(800, 600, c_real=-0.7, c_imag=0.27015)
img = ProceduralArt.sierpinski_triangle(512, 512, depth=7)

# Patterns
img = ProceduralArt.plasma(512, 512)
img = ProceduralArt.voronoi(512, 512, n_cells=25, seed=42)
img = ProceduralArt.perlin_noise_image(512, 512, octaves=4)

# Gradients
img = VisualEffects.create_linear_gradient(512, 512, (70, 130, 200), (200, 80, 120))
img = VisualEffects.create_radial_gradient(512, 512, (255, 220, 50), (30, 30, 120))
```

#### VisualEffects

Apply filters to existing `TTIImage` objects (all return the modified image):

```python
img = VisualEffects.blur(img, radius=3)
img = VisualEffects.gaussian_blur(img, sigma=2.0)
img = VisualEffects.sharpen(img, factor=1.5)
img = VisualEffects.edge_detect(img)
img = VisualEffects.emboss(img)
img = VisualEffects.grayscale(img)
img = VisualEffects.sepia(img, strength=0.8)
img = VisualEffects.invert(img)
img = VisualEffects.add_noise(img, intensity=0.15)
img = VisualEffects.pixelate(img, block_size=10)
img = VisualEffects.vignette(img, strength=0.6)
img = VisualEffects.adjust_brightness(img, factor=1.2)
img = VisualEffects.adjust_contrast(img, factor=1.3)
img = VisualEffects.blend(img1, img2, alpha=0.5)
```

#### StreamingWriter — large images without full RAM

```python
from pyaitk.CLSE import StreamingWriter

# Write a 4000×4000 image row by row
with StreamingWriter("large.png", width=4000, height=4000, bpp=24) as sw:
    for y in range(4000):
        row = [(y % 255, 100, 200)] * 4000   # list of RGB tuples
        sw.write_row(row)
```

#### AnimationEngine — frame sequences

```python
from pyaitk.CLSE import AnimationEngine

engine = AnimationEngine(fps=24)
frames = engine.generate_frames(base_img, n_frames=48, mode="zoom")
engine.save_frames(frames, output_dir="frames/", fmt="png")
```

#### CustomBitDepth — non-standard pixel formats

```python
from pyaitk.CLSE import CustomBitDepth

# 16-bit per channel, 3 channels
cbd = CustomBitDepth(width=256, height=256, bits_per_channel=16, n_channels=3)
cbd.set_pixel(0, 0, [65535, 0, 32768])
cbd.save("high_depth.custimg")
cbd.save_preview("preview.png")   # downsample to 8-bit PNG for viewing
```

---

#### AI Engine

```python
from pyaitk.CLSE import TTIGenerator, NLPAnalyser

# NLP analysis
nlp = NLPAnalyser()
analysis = nlp.analyse("a mysterious purple galaxy with glowing stars")

print(analysis.scene_type)         # "starfield"
print(analysis.nouns)              # ["galaxy", "stars"]
print(analysis.adjectives)         # ["mysterious", "purple", "glowing"]
print(analysis.colour_matches)     # [("purple", (138,43,226)), ("gold", (255,215,0)), ...]
print(analysis.modifiers)          # {"mysterious": 0.8, "glowing": 0.6}
print(analysis.filtered_tokens)    # ["mysterious", "purple", "galaxy", "glowing", "stars"]
print(analysis.complexity())       # 0.72  (float 0–1)

# Full generation
from pyaitk.CLSE import get_config
gen = TTIGenerator(get_config())
img = gen.generate("stormy sea at dusk", width=512, height=512, seed=99)
img.save("storm.png")

# Variations
imgs = gen.generate_variations("neon city at night", n_variations=4)
for i, img in enumerate(imgs):
    img.save(f"variation_{i}.png")

# Interpolation between two prompts
imgs = gen.interpolate("sunrise", "midnight", steps=6)
for i, img in enumerate(imgs):
    img.save(f"interp_{i:02d}.png")
```

**AI pipeline (internal stages):**

```
NLPAnalyser       → 256-d SemanticVector  (TF-IDF + PCA, NLTK tokeniser)
ColourPredictor   → PaletteSpec           (sklearn KNN on 170+ colour keywords)
SceneComposer     → 128-d LatentCode      (numpy VAE encoder)
ImageDecoder      → TTIImage              (14 scene-type renderers)
```

---

#### Neural Architecture

```python
from pyaitk.CLSE import TTIModel, TTIModelLarge, ModelConfig, TTITrainer, TTILoss

# Default model (4-layer transformer, 3.8M params)
cfg = ModelConfig(vocab_size=8192, embed_dim=256, n_layers=4, n_heads=8, latent_dim=128)
model = TTIModel(cfg)

# Large variant (6-layer, 512-d, ~14M params)
model_large = TTIModelLarge()

# Forward pass
import torch
tokens = torch.randint(0, 8192, (4, 32))   # batch=4, seq_len=32
out = model(tokens)
# out.scene_logits  : (4, 15)   — 15-class scene prediction
# out.colour_pred   : (4, 18)   — 6 colours × RGB
# out.param_pred    : (4, 64)   — scene renderer parameters
# out.mu, out.logvar: (4, 128)  — VAE latent distribution

# Multi-task loss
criterion = TTILoss(scene_weight=1.0, colour_weight=0.5, param_weight=0.3, kl_weight=0.01)
loss = criterion(out, scene_labels, colour_targets, param_targets)

# Training loop
trainer = TTITrainer(model, dataset, val_dataset, output_dir="tti_models/")
history = trainer.train(epochs=20, batch_size=64, lr=3e-4)
print(history["best_val_loss"])
```

**Model architecture:**

```
TokenEmbedding      — learned token + sinusoidal positional embeddings
TransformerEncoder  — 4× MultiHeadSelfAttention + FFN (BERT-style, pre-LN)
    ├── ColourHead      — 3-layer MLP → 18-d colour palette
    ├── SceneClassifier — 2-layer MLP → 15-class scene logit
    └── ParamDecoder    — VAE: μ/σ → z → 64-d parameter vector
```

---

#### Dataset Generator

```python
from pyaitk.CLSE import TTIDataset, Vocabulary, build_dataset

# Build a 50,000-sample dataset (cached to disk with SHA-256 integrity check)
dataset = build_dataset(
    n_samples=50_000,
    cache_dir="tti_models/",
    force_rebuild=False,     # use cache if available
)

# Train / val / test splits
train_ds, val_ds, test_ds = dataset.splits()
print(len(train_ds), len(val_ds), len(test_ds))   # 40000, 5000, 5000

# DataLoader-compatible access
sample = train_ds[0]
# sample["token_ids"]   : torch.LongTensor (32,)
# sample["scene_label"] : int  (0–14)
# sample["colour_vec"]  : torch.FloatTensor (18,)
# sample["param_vec"]   : torch.FloatTensor (64,)
# sample["prompt"]      : str

# Vocabulary
vocab = Vocabulary.load("tti_models/vocab.json")
ids = vocab.encode("a stormy sea at dusk")   # list of int
text = vocab.decode(ids)                     # str
print(vocab.size)                            # 8192
```

**Dataset statistics (default build):**

| Split | Samples |
|---|---|
| Train | 40,000 |
| Validation | 5,000 |
| Test | 5,000 |
| Vocab size | 8,192 |
| Scene classes | 15 |
| Colour dims | 18 (6 colours × RGB) |
| Parameter dims | 64 |

---

#### Unified Pipeline

```python
from pyaitk.CLSE import TTIPipeline

pipe = TTIPipeline()

# Generate
img = pipe.generate("a bright rainbow over a misty waterfall", output="rainbow.png")

# Variations
imgs = pipe.variations("neon city at night", n=4, output_dir="variations/")

# Interpolation
imgs = pipe.interpolate("sunrise", "midnight", steps=6, output_dir="interp/")

# Procedural art
img = pipe.art("julia", output="julia.png", c_real=-0.4, c_imag=0.6)

# Effects
img = pipe.effect("sepia", input_path="photo.png", output="photo_sepia.png")

# NLP analysis only
info = pipe.analyse("dark gothic castle at midnight")

# Animation
frames = pipe.animate("sunset over the ocean", n_frames=24, output_dir="frames/")

# Stream a very large image (memory-safe)
pipe.stream_large("huge.png", width=4000, height=4000, pattern="gradient")

# Model info
print(pipe.model_info())

# Config management
pipe.show_config()
pipe.save_config("snapshot.json")
```

---

### Architecture Overview

```
TTI_main.py  /  TTI_pipeline.py   ←  unified façade
│
├── TTI_config.py    ←  all settings (reads from config.pbcfg / tti_config.json)
│
├── TTI_ai.py        ←  NLP + VAE renderer (no model weights needed)
│   ├── NLPAnalyser      NLTK tokenise → TF-IDF → PCA → SemanticVector
│   ├── ColourPredictor  KNN on 170+ colour keywords → PaletteSpec
│   ├── SceneComposer    numpy VAE encoder → 128-d LatentCode
│   └── ImageDecoder     14 scene-type renderers → TTIImage
│
├── TTI_model.py     ←  PyTorch transformer VAE (optional, boosts quality)
│   ├── TokenEmbedding   learnable + positional
│   ├── TransformerEncoder  4-layer BERT-style
│   ├── ColourHead       → 18-d palette
│   ├── SceneClassifier  → 15-class scene
│   └── ParamDecoder     → 64-d renderer params (VAE)
│
├── TTI_dataset.py   ←  50k-sample synthetic generator + Vocabulary + DataLoader
│
├── TTI_train.py     ←  training script (gradient ckpt, cosine LR, early stop)
│
├── TTI_core.py      ←  pixel engine (TTIImage, ImageCanvas, ColorUtils, I/O)
│
└── TTI_art.py       ←  procedural art, effects, streaming, animation
```

---

### Examples Summary

```python
from pyaitk.CLSE import TTIPipeline
from pyaitk.CLSE import ProceduralArt, VisualEffects
from pyaitk.CLSE import NLPAnalyser
from pyaitk.CLSE import get_config, update_config

pipe = TTIPipeline()

# Generate
img = pipe.generate("stormy sea at midnight", output="storm.png")

# Batch
pipe.generate_batch(["sunrise", "sunset", "noon"], output_dir="batch/")

# Variations + interpolation
pipe.generate_variations("neon city", n=4, output_dir="vars/")
pipe.interpolate("calm lake", "raging ocean", steps=6, output_dir="interp/")

# Procedural art
pipe.art("mandelbrot", output="mandelbrot.png", width=800, height=600)
pipe.art("julia", output="julia.png", c_real=-0.4, c_imag=0.6)

# Effects
pipe.effect("sepia", "input.png", output="sepia.png")
pipe.effect("vignette", "input.png", strength=0.7, output="vig.png")

# Analysis
info = pipe.analyse("mysterious purple galaxy")
print(info.scene_type, info.complexity())

# Config override
update_config(image={"default_width": 1024}, ai={"seed": 7})

# Full demo
pipe.demo(output_dir="tti_demo/")
```

---

## Camera — Camera Module

A Tkinter-based camera viewer with live QR/barcode scanning, image capture, video recording, and full context-manager support. Wraps OpenCV and pyzbar behind a clean class-based API with thread-safe state management.

---

### What Is This?

`Camera` provides a camera module that combines:

- **Live video preview** — Tkinter GUI window displaying frames from any OpenCV-compatible device
- **QR / barcode scanning** — auto-decodes QR codes and barcodes from every frame via `pyzbar`; accumulates all unique payloads thread-safely
- **Image capture** — saves the current frame as a timestamped `.png` with one method call
- **Video recording** — start/stop writing annotated frames to a timestamped `.avi` file using XVID codec
- **Context-manager protocol** — `Camera.open()` creates the Tk root, opens the device, and releases everything on exit
- **Functional entry-point** — `Start()` launches the GUI and returns all scanned data in one call
- **Auto-install of `pyzbar`** — silently attempts `pip install pyzbar` if the library is missing; degrades gracefully if it still can't be imported
- **Thread-safe design** — separate `RLock` guards for scanned data and video writer; frame loop never blocks on recording

---

### Installation

> On Linux, `pyzbar` also needs the native `zbar` library:
> ```bash
> sudo apt install libzbar0
> ```
> On Windows, the `pyzbar` wheel bundles the DLL automatically.

---

### How to Use

#### 1. Simplest — functional one-liner

Launches the GUI; blocks until the window is closed; returns all scanned payloads.

```python
from pyaitk.Camera import Start

data = Start()
for item in data:
    print(item)
```

#### 2. Context manager (recommended)

```python
from pyaitk.Camera import Camera

with Camera.open() as cam:
    cam.run()              # blocks until the window is closed

print(cam.scanned_data)   # set of all scanned QR/barcode strings
```

#### 3. Custom device and output directory

```python
from pyaitk.Camera import Camera

with Camera.open(device=1, output_dir="./recordings") as cam:
    cam.run()
```

#### 4. Programmatic capture and recording

```python
from pyaitk.Camera import Camera
import tkinter as tk

root = tk.Tk()
cam = Camera(root, device=0, output_dir="./output")

# Capture a single frame immediately
path = cam.capture_image()
print(f"Saved: {path}")

# Start and stop recording
record_path = cam.start_recording()
print(f"Recording to: {record_path}")

# … run some frames …
cam.run()       # blocks until window closed

cam.stop_recording()
cam.close()
```

#### 5. Inspect scanned data before closing

```python
from pyaitk.Camera import Camera
import threading

with Camera.open() as cam:
    # Poll scanned_data from another thread
    def monitor():
        import time
        while not cam.is_closed:
            print("Scanned so far:", cam.scanned_data)
            time.sleep(2.0)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    cam.run()

print("Final scanned data:", cam.scanned_data)
```

#### 6. State checks

```python
with Camera.open() as cam:
    print(cam.is_recording)   # False
    cam.start_recording()
    print(cam.is_recording)   # True
    cam.stop_recording()
    print(cam.is_closed)      # False
    cam.run()

print(cam.is_closed)          # True
```

---

### GUI Reference

The camera window opens with three buttons:

| Button | Action |
|---|---|
| 📸 **Capture Image** | Saves current frame as `captured_YYYYMMDD_HHMMSS.png` in `output_dir`; shows a confirmation dialog |
| 🎥 **Start Recording** / ⏹️ **Stop Recording** | Toggles video recording; saves to `output_YYYYMMDD_HHMMSS.avi` in `output_dir` |
| ❌ **Quit** | Stops recording (if active), releases the camera, and destroys the window |

Closing the window via the title bar `×` button also triggers a clean shutdown.

---

### API Reference

#### `Start(device, output_dir)` → `set[str]`

Functional entry-point. Launches the GUI, blocks until the window is closed, and returns all scanned QR/barcode payloads.

| Parameter    | Type             | Default | Description                            |
|--------------|------------------|---------|----------------------------------------|
| `device`     | `int`            | `0`     | OpenCV camera device index             |
| `output_dir` | `Path` or `str`  | `"."`   | Directory for saved images and videos  |

---

#### `Camera.open(device, output_dir)` → context manager

Class-level `@contextmanager` factory. Creates a `tk.Tk` root, instantiates `Camera`, yields it, and calls `close()` on exit.

```python
with Camera.open(device=0, output_dir="./out") as cam:
    cam.run()
```

---

#### `Camera(window, device, output_dir)`

Direct constructor for advanced use when you manage the Tk root yourself.

| Parameter    | Type             | Default | Description                              |
|--------------|------------------|---------|------------------------------------------|
| `window`     | `tk.Tk`          | —       | Root Tkinter window                      |
| `device`     | `int`            | `0`     | OpenCV capture device index              |
| `output_dir` | `Path` or `str`  | `"."`   | Directory for saved images and videos    |

**Raises:** `CameraError` if the device cannot be opened.

---

#### Instance methods

| Method | Returns | Description |
|---|---|---|
| `run()` | `None` | Enter the Tk main-loop (blocks until window closed) |
| `capture_image()` | `Path` or `None` | Capture current frame to `output_dir`; returns saved path |
| `start_recording()` | `Path` or `None` | Begin writing frames to `.avi`; returns output path |
| `stop_recording()` | `None` | Stop recording and flush the video file |
| `close()` | `None` | Release all resources; safe to call multiple times |

#### Properties

| Property | Type | Description |
|---|---|---|
| `scanned_data` | `set[str]` | Thread-safe snapshot of all decoded QR/barcode payloads |
| `is_recording` | `bool` | `True` while a video is being written |
| `is_closed` | `bool` | `True` after `close()` has been called |

---

### Output Files

All files are written to `output_dir` (default: current directory).

| File pattern | Format | Created by |
|---|---|---|
| `captured_YYYYMMDD_HHMMSS.png` | PNG | `capture_image()` / 📸 button |
| `output_YYYYMMDD_HHMMSS.avi` | AVI (XVID, 20 fps) | `start_recording()` / 🎥 button |

---

### Exception Reference

```
CameraError(RuntimeError)
├── Raised when the camera device cannot be opened
└── Raised when run() is called after the camera is already closed
```

---

### Architecture Notes

- **Frame loop** — driven by `window.after(_FRAME_DELAY_MS, _update_frame)` (10 ms ≈ 100 fps cap); never blocks the Tk event loop
- **QR decoding** — runs on every frame inside the frame loop; new unique payloads are added to `_scanned_data` under `_data_lock`
- **Video writer** — guarded by `_writer_lock` so the recording flag and `cv2.VideoWriter` are always consistent across threads
- **`pyzbar` graceful degradation** — if not installed, a `pip install` is attempted once at import time; if still unavailable, QR scanning is silently disabled and the rest of the module works normally
- **Clean shutdown** — `close()` stops recording, releases the OpenCV capture device, and destroys the Tk window; idempotent (safe to call multiple times)

---

### Examples Summary

```python
# Simplest: launch and collect scanned QR data
from pyaitk.Camera import Start
data = Start()

# Context manager
from pyaitk.Camera import Camera
with Camera.open(device=0, output_dir="./out") as cam:
    cam.run()
print(cam.scanned_data)

# Custom device
with Camera.open(device=1) as cam:
    cam.run()

# Programmatic capture before running
import tkinter as tk
from pyaitk.Camera import Camera
root = tk.Tk()
cam = Camera(root, output_dir="./shots")
cam.capture_image()          # save a frame immediately
cam.start_recording()        # start video
cam.run()                    # blocks
cam.stop_recording()
cam.close()

# State checks
with Camera.open() as cam:
    print(cam.is_recording)  # False
    cam.start_recording()
    print(cam.is_recording)  # True
    cam.run()
```

---

## Memory — Memory Module for Pythonaibrain

A two-tier episodic memory system for Brain and AdvanceBrain. `Memory` provides a thread-safe, JSON-backed key/value store. `SmartMemory` extends it transparently with a full ML pipeline — TF-IDF → Autoencoder → Clustering → IntentClassifier — enabling semantic search over conversation history without changing any call sites.

---

### What Is This?

`Memory` provides two public classes and a factory function:

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

### How to Use

#### 1. Plain `Memory` (classic key/value)

```python
from pyaitk.Memory import Memory

mem = Memory(path="memory.json")

mem.remember("user_name", "Divyanshu")
mem.remember("last_topic", "Python programming")

print(mem.recall("user_name"))    # "Divyanshu"
print(mem.recall("missing_key", default="N/A"))  # "N/A"

mem.save_memory()
mem.load_memory()   # reload from disk
```

#### 2. `SmartMemory` — drop-in with ML extras

```python
from pyaitk.Memory import SmartMemory

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

#### 3. `build_memory()` factory (recommended for `Brain`)

```python
from pyaitk.Memory import build_memory

# Returns SmartMemory if SummarizerAI is available, Memory otherwise
mem = build_memory("memory.json", smart=True, fit_interval=50)
mem.remember("greeting", "Hello!")
mem.save_memory()
```

#### 4. Checking and controlling the summarizer

```python
from pyaitk.Memory import SmartMemory

mem = SmartMemory(path="memory.json", auto_fit=False)

# Manually trigger a (blocking) fit
success = mem.fit_summarizer()
print(success)   # True / False

# Check fit state
print(mem.is_summarizer_fitted())   # True after fit

# Get the report object
report = mem.get_report()   # MemorySummaryReport dataclass or None
```

#### 5. Extended `Memory` API

```python
from pyaitk.Memory import Memory

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

#### 6. Inspecting `MemoryEntry` metadata

```python
from pyaitk.Memory import Memory

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

### API Reference

#### `Memory(path, max_entries, auto_load)`

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

#### `SmartMemory(path, max_entries, auto_load, auto_fit, fit_interval, summarizer_config)`

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

#### `build_memory(path, smart, **kwargs)` → `Memory`

Factory function. Returns `SmartMemory` when `smart=True` and `SummarizerAI` is importable; returns plain `Memory` otherwise.

| Parameter   | Type   | Default        | Description                                        |
|-------------|--------|----------------|----------------------------------------------------|
| `path`      | `str`  | `memory.json`  | Path for JSON persistence                          |
| `smart`     | `bool` | `True`         | Attempt to use `SmartMemory`                       |
| `**kwargs`  | —      | —              | Forwarded to `SmartMemory` or `Memory` constructor |

---

#### `MemoryEntry` fields

| Field           | Type    | Description                                  |
|-----------------|---------|----------------------------------------------|
| `key`           | `str`   | Entry key                                    |
| `value`         | `str`   | Stored value                                 |
| `timestamp`     | `float` | Unix timestamp of creation                   |
| `access_count`  | `int`   | Number of times `recall()` accessed this key |
| `last_accessed` | `float` | Unix timestamp of most recent recall         |

---

### Architecture

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

### File Format

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

### Examples Summary

```python
# Plain Memory
from pyaitk.Memory import Memory
mem = Memory("memory.json")
mem.remember("name", "Divyanshu")
print(mem.recall("name"))
mem.save_memory()

# SmartMemory
from pyaitk.Memory import SmartMemory
mem = SmartMemory("memory.json", auto_fit=True, fit_interval=10)
mem.remember("How are you?", "I'm great!")
results = mem.semantic_search("how are things", top_k=3)
intent = mem.predict_intent("What's the weather?")
mem.export_report("report.json")
mem.save_memory()

# Factory (recommended)
from pyaitk.Memory import build_memory
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

---

## Pythonaibrain Config

The master configuration layer for the entire Pythonaibrain / pyaitk framework. All subsystems — Brain, STT, TTS, NER, TTI, Memory Summarizer, LLM — read their settings from a single `.pbcfg` file via typed dataclass section objects on one unified `AppConfig` instance.

---

### What Is This?

`Config` provides:

- **`.pbcfg` file format** — INI-style config file with `; #` comment support and inline comments
- **`AppConfig`** — unified config manager that reads/writes all sections and exposes them as typed dataclasses
- **22 section dataclasses** — one per subsystem, each with typed fields and sensible defaults
- **Auto-discovery** — searches upward from cwd to find `config.pbcfg`, falls back to built-in defaults
- **Typed `get()` / `set()`** — generic read/write with optional type casting and fallback
- **`generate_default_config()`** — writes a fully-populated default `.pbcfg` to disk
- **JSON export** — serialize the entire config to JSON string or file
- **Module-level singleton** — `get_config()` returns the same instance across the entire application

---

### File Format (`.pbcfg`)

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

### How to Use

#### 1. Module-level singleton (recommended)

```python
from pyaitk.config import get_config

cfg = get_config()   # auto-discovers config.pbcfg from cwd upward
print(cfg.brain.smart_memory)    # True
print(cfg.model.epochs)          # 100
print(cfg.stt.pause_threshold)   # 0.8
```

#### 2. Load a specific file

```python
from pyaitk.config import AppConfig

cfg = AppConfig("myproject.pbcfg")
print(cfg.tts.voice)    # "david"
print(cfg.llm.n_ctx)    # 2048
```

#### 3. Auto-discover

```python
cfg = AppConfig.discover()            # search cwd → home
cfg = AppConfig.discover("/my/dir")   # search from a custom start path
```

#### 4. Build from a dict

```python
cfg = AppConfig.from_dict({
    "brain": {"smart_memory": True, "username": "divyanshu"},
    "model": {"epochs": 50, "learning_rate": 0.0005},
})
```

#### 5. Read, mutate, and save

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

#### 6. Reload from disk

```python
cfg.load()                        # reload from current path
cfg.load("other.pbcfg")           # reload from different file
```

#### 7. Generate a default config file

```python
from pyaitk.config import generate_default_config

path = generate_default_config()              # → ./config.pbcfg
path = generate_default_config("myapp.pbcfg")
```

#### 8. Reset to factory defaults

```python
from pyaitk.config import reset_config

cfg = reset_config()   # clears singleton, returns AppConfig with all defaults
```

#### 9. Export to JSON

```python
json_str = cfg.to_json(indent=2)
cfg.save_json("config.json")
```

#### 10. Inspect all settings

```python
print(cfg.dump())           # human-readable dump of all sections
d = cfg.as_dict()           # nested dict {section: {key: value}}
```

---

### Section Reference

#### `[brain]` → `cfg.brain` (`BrainConfig`)

| Key                  | Type   | Default              | Description                                      |
|----------------------|--------|----------------------|--------------------------------------------------|
| `intents_path`       | `str`  | `./intents.json`     | Path to intents JSON file                        |
| `condition`          | `bool` | `True`               | Enable dynamic intent learning from web search   |
| `smart_memory`       | `bool` | `True`               | Use `SmartMemory` (semantic search + clustering) |
| `memory_path`        | `str`  | `memory.json`        | Path for memory persistence                      |
| `memory_fit_interval`| `int`  | `20`                 | Auto-fit SmartMemory every N stored memories     |
| `username`           | `str`  | `user_name`          | Key used for user name storage in memory         |
| `download`           | `bool` | `False`              | Auto-download NLTK data on Brain init            |

#### `[model]` → `cfg.model` (`ModelConfig`)

| Key              | Type    | Default           | Description                        |
|------------------|---------|-------------------|------------------------------------|
| `model_path`     | `str`   | `model.pth`       | Path to save/load model weights    |
| `dimension_path` | `str`   | `dimensions.json` | Path to save/load vocab/intent map |
| `batch_size`     | `int`   | `8`               | Training batch size                |
| `learning_rate`  | `float` | `0.001`           | Adam optimizer learning rate       |
| `epochs`         | `int`   | `100`             | Training epochs                    |

#### `[llm]` → `cfg.llm` (`LLMConfig`)

| Key          | Type   | Default | Description                                 |
|--------------|--------|---------|---------------------------------------------|
| `n_ctx`      | `int`  | `2048`  | LLM context window size                     |
| `n_threads`  | `int`  | `0`     | CPU threads; `0` = `os.cpu_count()`         |
| `max_tokens` | `int`  | `512`   | Max tokens to generate per response         |
| `verbose`    | `bool` | `False` | Enable llama.cpp verbose logging            |

#### `[tts]` → `cfg.tts` (`TTSConfig`)

| Key            | Type            | Default           | Description                                   |
|----------------|-----------------|-------------------|-----------------------------------------------|
| `rate`         | `int`           | `150`             | Words per minute                              |
| `volume`       | `float`         | `1.0`             | Volume 0.0–1.0 (validated in `__post_init__`) |
| `voice`        | `str`           | `david`           | Voice name fragment (fuzzy matched)           |
| `default_text` | `str`           | `Hello from PyAI` | Fallback text for empty `say()` calls         |
| `output_path`  | `str` or `None` | `None`            | Save to WAV file instead of playing           |

#### `[stt]` → `cfg.stt` (`STTConfig`)

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

#### `[logging]` → `cfg.logging` (`LoggingConfig`)

| Key      | Type  | Default                                              | Description         |
|----------|-------|------------------------------------------------------|---------------------|
| `level`  | `str` | `INFO`                                               | Root logging level  |
| `format` | `str` | `%(asctime)s [%(levelname)s] %(name)s - %(message)s` | Log record format   |

#### `[memory]` → `cfg.memory` (`MemoryConfig`)

| Key         | Type   | Default | Description                             |
|-------------|--------|---------|-----------------------------------------|
| `auto_load` | `bool` | `True`  | Load memory from disk on init           |
| `auto_fit`  | `bool` | `True`  | Auto-fit SmartMemory summarizer on load |

#### `[embedding]` → `cfg.embedding` (`EmbeddingConfig`)

| Key                  | Type    | Default  | Description                                     |
|----------------------|---------|----------|-------------------------------------------------|
| `tfidf_max_features` | `int`   | `5000`   | Max vocabulary size for TF-IDF                  |
| `tfidf_ngram_range`  | `tuple` | `(1, 3)` | N-gram range (stored as `1,3` in file)          |
| `tfidf_sublinear_tf` | `bool`  | `True`   | Apply sublinear TF scaling                      |
| `embed_dim`          | `int`   | `128`    | Learned embedding dimension                     |
| `vocab_size`         | `int`   | `10000`  | Vocabulary size for learned embeddings          |

#### `[clustering]` → `cfg.clustering` (`ClusteringConfig`)

| Key                       | Type               | Default | Description                           |
|---------------------------|--------------------|---------|---------------------------------------|
| `n_clusters`              | `int`              | `8`     | KMeans cluster count                  |
| `kmeans_max_iter`         | `int`              | `300`   | KMeans max iterations                 |
| `kmeans_random_state`     | `int`              | `42`    | KMeans random seed                    |
| `dbscan_eps`              | `float`            | `0.5`   | DBSCAN epsilon radius                 |
| `dbscan_min_samples`      | `int`              | `2`     | DBSCAN minimum samples per cluster    |
| `agglo_linkage`           | `str`              | `ward`  | Agglomerative linkage criterion       |
| `agglo_distance_threshold`| `float` or `None`  | `None`  | Agglomerative distance threshold      |

#### `[classifier]` → `cfg.classifier` (`ClassifierConfig`)

| Key                    | Type    | Default  | Description                                    |
|------------------------|---------|----------|------------------------------------------------|
| `lr_max_iter`          | `int`   | `500`    | Logistic Regression max iterations             |
| `lr_c`                 | `float` | `1.0`    | LR regularization strength (lower = stronger)  |
| `lr_solver`            | `str`   | `lbfgs`  | LR solver                                      |
| `lr_multi_class`       | `str`   | `auto`   | LR multi-class strategy                        |
| `similarity_threshold` | `float` | `0.65`   | Min cosine similarity for pattern matching     |

#### `[summarizer]` → `cfg.summarizer` (`SummarizerConfig`)

| Key                       | Type    | Default | Description                                 |
|---------------------------|---------|---------|---------------------------------------------|
| `latent_dim`              | `int`   | `64`    | Autoencoder latent space dimension          |
| `hidden_dim`              | `int`   | `256`   | Autoencoder hidden layer size               |
| `ae_epochs`               | `int`   | `30`    | Autoencoder training epochs                 |
| `ae_lr`                   | `float` | `0.001` | Autoencoder learning rate                   |
| `ae_batch_size`           | `int`   | `16`    | Autoencoder training batch size             |
| `top_patterns_per_cluster`| `int`   | `3`     | Top patterns to extract per cluster         |
| `min_cluster_size`        | `int`   | `2`     | Minimum cluster size for summarization      |

#### `[tti_image]` → `cfg.tti_image` (`TTIImageConfig`)

| Key               | Type    | Default           | Description                                  |
|-------------------|---------|-------------------|----------------------------------------------|
| `default_width`   | `int`   | `512`             | Output image width in pixels                 |
| `default_height`  | `int`   | `512`             | Output image height in pixels                |
| `default_bpp`     | `int`   | `24`              | Bits per pixel                               |
| `default_format`  | `str`   | `png`             | Output format: `png`, `bmp`, `jpeg`          |
| `background_color`| `tuple` | `(255, 255, 255)` | RGB background color (stored as `255,255,255`) |
| `jpeg_quality`    | `int`   | `92`              | JPEG compression quality (1–95)              |

#### `[tti_ai]` → `cfg.tti_ai` (`TTIAIConfig`)

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

#### `[tti_art]` → `cfg.tti_art` (`TTIArtConfig`)

| Key                       | Type    | Default | Description                                |
|---------------------------|---------|---------|--------------------------------------------|
| `fractal_max_iter`        | `int`   | `256`   | Max iterations for fractal generation      |
| `blur_default_radius`     | `int`   | `2`     | Default Gaussian blur radius               |
| `noise_default_intensity` | `float` | `0.15`  | Default noise overlay intensity            |
| `animation_fps`           | `int`   | `24`    | Frames per second for animations           |
| `streaming_chunk_mb`      | `int`   | `32`    | Chunk size for streaming output (MB)       |

#### `[tti_paths]` → `cfg.tti_paths` (`TTIPathConfig`)

| Key          | Type  | Default      | Description                      |
|--------------|-------|--------------|----------------------------------|
| `output_dir` | `str` | `tti_output` | Directory for generated images   |
| `model_dir`  | `str` | `tti_models` | Directory for saved models       |
| `cache_dir`  | `str` | `tti_cache`  | Directory for cached data        |
| `log_dir`    | `str` | `tti_logs`   | Directory for TTI log files      |

Call `cfg.ensure_tti_dirs()` (or `cfg.tti_paths.ensure_dirs()`) to create all four directories.

#### `[postprocessor]` → `cfg.postprocessor` (`PostprocessorConfig`)

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

#### `[preprocessor]` → `cfg.preprocessor` (`PreprocessorConfig`)

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

### `AppConfig` API Reference

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

### Module-Level Helpers

```python
from pyaitk.config import get_config, reset_config, generate_default_config

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

### Examples Summary

```python
from pyaitk.config import AppConfig, get_config, generate_default_config

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

---

# PyAgent — ZENTRAA CLI Reference

**ZENTRAA** *(Zone for Encrypted Networked Talks & Real-time AI Agent)*  
Powered by **Pythonaibrain v1.1.9** · Author: Divyanshu Sinha  
Encryption: RSA-2048-OAEP · AES-256-GCM · RSA-PSS · Curve25519

---

## Installation

```bash
pip install "pythonaibrain[zentraa]"
```

> **Linux only** — install PortAudio before pip if you plan to use the TIGER AI voice features:
> ```bash
> sudo apt install portaudio19-dev python3-pyaudio
> ```

---

## Quick Start

Every command is available through the `pythonaibrain` (or `pyaitk`) dispatcher:

```
pythonaibrain zentraa <command> [options]
```

Or as a standalone alias:

```
zentraa-server / zentraa-client / zentraa-tiger-ai / zentraa-web
```

Typical startup order:

```
1. zentraa server    ← start first
2. zentraa web       ← optional browser bridge
3. zentraa ai        ← optional TIGER AI agent
4. zentraa client    ← one per human user
```

---

## Commands

### `zentraa server` — TCP Chat Server

Starts the encrypted ZENTRAA chat server that all clients connect to.

```bash
pythonaibrain zentraa server
pythonaibrain zentraa server --host 0.0.0.0 --port 9999
pythonaibrain zentraa server --config /path/to/ZENTRAA.pbcfg
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | `ZENTRAA.pbcfg` | Path to config file |
| `--host` | `-H` | from config | Override bind host |
| `--port` | `-p` | from config | Override bind port |
| `--help` | | | Show help and exit |

---

### `zentraa client` — Chat Client (TUI)

Interactive terminal client for human users. Supports direct messages, broadcasts, and talking to TIGER AI.

```bash
pythonaibrain zentraa client
pythonaibrain zentraa client --host 127.0.0.1 --port 9999 --userid Alice
pythonaibrain zentraa client --config /path/to/ZENTRAA.pbcfg
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | `ZENTRAA.pbcfg` | Path to config file |
| `--host` | `-H` | from config | Server host |
| `--port` | `-p` | from config | Server port |
| `--userid` | `-u` | prompted | Your user ID |
| `--help` | | | Show help and exit |

#### In-chat commands

| Command | Description |
|---|---|
| `<message>` | Broadcast to all users |
| `@<userid> <message>` | Direct message a user |
| `@uid1 @uid2 ... <message>` | Multi-user direct message |
| `@ai <message>` | Ask TIGER AI privately |
| `@ai @<uid> <message>` | Ask AI, share reply with `<uid>` |
| `/help` | Show help |
| `/clear` or `/cls` | Clear the screen |
| `/ai` | Show TIGER AI info |
| `/setting` | View current settings |
| `/users` | List online users |
| `/me <action>` | Send an action / emote message |
| `/whois <userid>` | Show info about a user |
| `/ping` | Ping the server manually |
| `/stats` | Show session statistics |
| `/notify <on\|off>` | Toggle bell notifications |
| `/timestamps <on\|off>` | Toggle message timestamps |
| `/quit` or `/exit` | Disconnect |

#### Keyboard shortcuts

| Key | Action |
|---|---|
| `↑` / `↓` | Browse command history |
| `Tab` | Autocomplete `@userid` or `/command` |
| `Ctrl+C` / `Ctrl+D` | Quit |

---

### `zentraa ai` — TIGER AI Agent

Connects an automated AI agent (TIGER AI) to the server. Other users can query it with `@ai <message>`.

```bash
pythonaibrain zentraa ai
pythonaibrain zentraa ai --smart
pythonaibrain zentraa ai --basic --host 127.0.0.1 --port 9999
pythonaibrain zentraa ai --config /path/to/ZENTRAA.pbcfg
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--config` | `-c` | `ZENTRAA.pbcfg` | Path to config file |
| `--host` | `-H` | from config | Server host |
| `--port` | `-p` | from config | Server port |
| `--smart` | | from config | Force **AdvanceBrain** (LLM mode) |
| `--basic` | | from config | Force **Brain** (intent-matching mode) |
| `--help` | | | Show help and exit |

> `--smart` and `--basic` are mutually exclusive. If neither is passed, the value from `ZENTRAA.pbcfg` is used.

---

### `zentraa web` — HTTP / WebSocket Bridge

Starts the HTTP and WebSocket bridge so browser clients can connect to the ZENTRAA TCP server. The bridge auto-selects a free port starting from `7080` unless `--no-auto-port` is set.

```bash
pythonaibrain zentraa web
pythonaibrain zentraa web --http-port 7080 --tcp-port 9999
pythonaibrain zentraa web --no-auto-port --max-upload-mb 128 --history 1000
```

| Option | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | HTTP bind host |
| `--http-port` | `7080` (auto) | HTTP / WebSocket port |
| `--tcp-host` | `127.0.0.1` | ZENTRAA TCP server host |
| `--tcp-port` | `9999` | ZENTRAA TCP server port |
| `--no-auto-port` | off | Fail instead of scanning for a free port |
| `--max-upload-mb` | `64` | Maximum file upload size in MB |
| `--history` | `500` | Messages stored per conversation |
| `--ping-interval` | `20` | WebSocket ping interval in seconds |
| `--help` | | Show help and exit |

Once running, open your browser at:

```
http://localhost:7080
```

#### Bridge features

- Typing indicators
- Read receipts
- Emoji reactions
- Message delivery confirmations
- Rich user list (online/offline status)
- File upload via `POST /api/upload` (chunked, ≤128 KB per chunk)
- REST `GET /api/users` — online users and metadata
- REST `GET /api/history/{conv_id}` — last N messages

---

## Global CLI Flags

```bash
pythonaibrain --version      # Print version
pythonaibrain --info         # Package metadata + module availability
pythonaibrain --modules      # Per-module availability table
pythonaibrain --help         # Full help
```

---

## Configuration

All commands read `ZENTRAA.pbcfg` from the current directory by default. Pass `--config` to use a different path.

A minimal config example:

```ini
[network]
host = 0.0.0.0
port = 9999

[client]
default_host = 127.0.0.1
default_port = 9999

[ai]
smart_ai = true

[ui]
banner_style = full
```

> `.env` files and RSA key files (`.pem`) in `.zentraa_keys/` are generated locally and are **never** included in the package. Keep them out of version control.

---

## Architecture

```
Browser ──WS/JSON──► HTTP Bridge (zentraa web)
                           │
                      TCP/Encrypted
                           │
                     ZENTRAA Server (zentraa server)
                           │
              ┌────────────┴────────────┐
         Chat Clients              TIGER AI Agent
        (zentraa client)          (zentraa ai)
```

---

## License

- **PyAgent / ZENTRAA**: LGPL-3.0-or-later  
- **pyaitk.CLSE**: AGPL-3.0-or-later (see `pyaitk/CLSE/LICENSE.txt`)

---

## Visit [PyPI](https://pypi.org/project/pythonaibrain) for installation and more details.

## Visit [GitHub](https://github.com/DivyanshuSinha136/TIGER-All-Photos/) for more detail about package.

## Visit [Pythonaibrain Issues](https://github.com/DivyanshuSinha136/TIGER-All-Photos/issues) for any issues.

## View on [Hugging Face](https://huggingface.co/spaces/DivyanshuSinha/Pythonaibrain/blob/main/README.md).

---

**Start building your AI assistant today with Pythonaibrain!**
**Try to ask your doubt with AI releated to this package!**
