# STT — Production-grade Speech-to-Text Module

A fully cross-platform Speech-to-Text library supporting both online (Google Speech Recognition) and offline (PocketSphinx) backends, with automatic engine selection based on network availability.

---

## What Is This?

`stt.py` is a production-quality STT wrapper that provides:

- **Dual engine support** — Google Speech Recognition (online) and PocketSphinx (offline), selected automatically
- **Auto engine detection** — checks network connectivity at runtime and picks the best available engine
- **Context manager interface** — opens the microphone once and releases it cleanly on exit, even on exception
- **Ambient noise calibration** — samples the noise floor before each listen to improve accuracy
- **Configurable capture parameters** — energy threshold, pause detection, phrase time limits, and timeouts
- **Retry logic** — automatically retries on transient service errors with configurable delay
- **Structured logging** — all internal events use Python's `logging` module; no bare `print` statements in library code
- **Custom exception hierarchy** — fine-grained errors (`STTAudioError`, `STTRecognitionError`, `STTServiceError`, `STTEngineError`) for clean error handling
- **CLI entry-point** — run directly from the terminal with `--engine`, `--language`, `--timeout`, and more

---

## Installation

```bash
pip install SpeechRecognition pocketsphinx pyaudio
```

> On Linux, you may also need:
> ```bash
> sudo apt install portaudio19-dev python3-pyaudio
> ```

---

## How to Use

### 1. One-shot (simplest)

```python
from stt import STT

stt = STT()
text = stt.listen()
print(text)
```

### 2. Context manager (recommended)

The microphone is opened once and released cleanly on exit — even if an exception is raised.

```python
from stt import STT

with STT() as stt:
    text = stt.listen()
    print(text)
```

### 3. Force a specific engine

```python
from stt import STT, STTConfig, Engine

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

### 4. Check which engine is active

```python
from stt import STT

with STT() as stt:
    print(stt.active_engine)  # Engine.GOOGLE or Engine.POCKETSPHINX
    text = stt.listen()
```

### 5. Custom configuration

```python
from stt import STT, STTConfig

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

### 6. Multi-turn listening loop

```python
from stt import STT, STTAudioError, STTRecognitionError

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

## Configuration (`STTConfig`)

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

## Engine Selection

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

## Exception Hierarchy

```
STTError
├── STTAudioError        — mic could not be opened, or no speech detected in time
├── STTRecognitionError  — audio captured but speech was unintelligible
├── STTServiceError      — online service was unreachable or returned an error
└── STTEngineError       — PocketSphinx failed to init or missing language data
```

Example error handling:

```python
from stt import STT, STTAudioError, STTRecognitionError, STTServiceError, STTEngineError

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

## CLI Usage

Run directly from the terminal:

```bash
python stt.py [options]
```

### Flags

| Flag                   | Description                                                   |
|------------------------|---------------------------------------------------------------|
| `--engine ENGINE`      | `auto` (default), `google`, or `sphinx`                       |
| `--language TAG`       | BCP-47 language tag (default: `en-US`)                        |
| `--timeout SECONDS`    | Seconds to wait for speech to start (default: `5`)            |
| `--retries N`          | Max retry attempts on service errors (default: `3`)           |
| `--verbose` / `-v`     | Enable debug logging                                          |

### CLI Examples

```bash
# Auto-detect engine and listen
python stt.py

# Force Google with British English
python stt.py --engine google --language en-GB

# Use offline PocketSphinx
python stt.py --engine sphinx

# Set a longer timeout and more retries
python stt.py --timeout 10 --retries 5

# Enable verbose debug output
python stt.py --verbose
```

---

## Examples Summary

```python
# Quickest usage
from stt import STT
text = STT().listen()

# Context manager with config
from stt import STT, STTConfig, Engine
cfg = STTConfig(preferred_engine=Engine.GOOGLE, google_language="fr-FR", timeout=6.0)
with STT(config=cfg) as stt:
    text = stt.listen()
    print(text)

# Resilient loop with error recovery
from stt import STT, STTAudioError, STTRecognitionError
with STT() as stt:
    while True:
        try:
            print(stt.listen())
        except STTAudioError:
            pass   # timed out, try again
        except STTRecognitionError:
            print("Please repeat.")
```
