# TTS — Production-grade Text-to-Speech Module

A fully offline, cross-platform Text-to-Speech library built on the `pyttsx3` backend. Designed for clean lifecycle management, fuzzy voice selection, structured logging, and both programmatic and CLI usage.

---

## What Is This?

`tts.py` is a production-quality TTS wrapper that provides:

- **Offline operation** — no network calls; uses the system's native speech engine via `pyttsx3`
- **Cross-platform support** — works on Windows (SAPI5), macOS (NSSpeechSynthesizer), and Linux (eSpeak)
- **Context manager interface** — the engine is initialised once and torn down cleanly, even on exception
- **Fuzzy voice matching** — resolves voice names from short fragments (e.g. `"zira"`, `"alex"`) with ranked fallback
- **File saving** — synthesise speech to a `.wav` file instead of playing it live
- **Structured logging** — all internal events use Python's `logging` module; no bare `print` statements in library code
- **Custom exception hierarchy** — fine-grained errors (`TTSEngineError`, `TTSVoiceError`, `TTSSpeakError`) for clean error handling
- **Module-level convenience function** — `speak()` for quick one-liner usage
- **CLI entry-point** — run directly from the terminal with `--voice`, `--rate`, `--save`, and more

---

## Installation

```bash
pip install pyttsx3
```

> On Linux, you also need `espeak` installed:
> ```bash
> sudo apt install espeak
> ```

---

## How to Use

### 1. One-shot (simplest)

```python
from tts import TTS

TTS().say("Hello, world!")
```

### 2. Context manager (recommended)

The engine is initialised once and shut down cleanly on exit — even if an exception is raised.

```python
from tts import TTS, TTSConfig

with TTS(TTSConfig(voice="zira", rate=160)) as tts:
    tts.say("Hello!")
    tts.say("How are you?")
```

### 3. Module-level convenience function

```python
from tts import speak

speak("Hello world!")
speak("Bonjour", voice="fiona", rate=140)
```

### 4. Save speech to a WAV file

```python
from tts import TTS, TTSConfig

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

### 5. List available voices

```python
from tts import TTS

with TTS() as tts:
    for name in tts.available_voices():
        print(name)
```

### 6. Get platform-appropriate voice hints

```python
from tts import TTS

with TTS() as tts:
    print(tts.platform_voice_hints())
# e.g. ['david', 'zira', 'mark'] on Windows
```

---

## Configuration (`TTSConfig`)

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

## Voice Matching

Voices are resolved from a short fragment string using a ranked strategy:

1. **Direct substring match** — `"zira"` matches `"Microsoft Zira Desktop"`
2. **Platform-hint-assisted match** — uses known fragments for the current OS
3. **First installed voice** — fallback with a warning logged

**Platform voice hints:**

| Platform | Available fragments                                         |
|----------|-------------------------------------------------------------|
| Windows  | `david`, `zira`, `mark`                                     |
| macOS    | `alex`, `samantha`, `victoria`, `fred`, `daniel`, `fiona`   |
| Linux    | `english`, `english-us`, `english-uk`, `mb-en1`, `mb-fr1`  |

---

## Exception Hierarchy

```
TTSError
├── TTSEngineError   — pyttsx3 failed to initialise
├── TTSVoiceError    — no matching voice found
└── TTSSpeakError    — synthesis failed at runtime
```

Example error handling:

```python
from tts import TTS, TTSConfig, TTSEngineError, TTSVoiceError, TTSSpeakError

try:
    with TTS(TTSConfig(voice="zira")) as tts:
        tts.say("Testing error handling.")
except TTSEngineError as e:
    print(f"Engine failed: {e}")
except TTSVoiceError as e:
    print(f"Voice not found: {e}")
except TTSSpeakError as e:
    print(f"Synthesis error: {e}")
```

---

## CLI Usage

Run directly from the terminal:

```bash
python tts.py [text] [options]
```

### Flags

| Flag                  | Description                                        |
|-----------------------|----------------------------------------------------|
| `text`                | Text to speak (positional, optional)               |
| `--voice NAME`        | Voice name fragment (e.g. `zira`, `alex`)          |
| `--rate WPM`          | Speech rate in words per minute (default: `150`)   |
| `--volume FLOAT`      | Volume from `0.0` to `1.0` (default: `1.0`)        |
| `--save FILE`         | Save output to a WAV file instead of playing       |
| `--list-voices`       | List all installed voices and exit                 |
| `--hints`             | Show platform-appropriate voice hints and exit     |
| `--verbose` / `-v`    | Enable debug logging                               |

### CLI Examples

```bash
# Speak with default settings
python tts.py "Hello from the command line"

# Use a specific voice and rate
python tts.py "Good morning" --voice zira --rate 160

# Save to a WAV file
python tts.py "This will be saved" --save output.wav

# List all installed voices
python tts.py --list-voices

# Show voice hints for the current platform
python tts.py --hints

# Enable verbose debug output
python tts.py "Debug mode" --verbose
```

---

## Examples Summary

```python
# Quickest usage
from tts import speak
speak("Hello!")

# Full control with context manager
from tts import TTS, TTSConfig
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
