# ITT — Image-to-Text (OCR) Utility

A minimal, dependency-light OCR utility built on EasyOCR. Extracts text from images in a single function call, with the recognition model loaded once at startup for efficient repeated use.

---

## What Is This?

`mainitt.py` is a lightweight image-to-text module that provides:

- **Single function API** — `ITT(image_path)` returns all recognised text as a plain string
- **EasyOCR backend** — deep learning-based OCR supporting 80+ languages out of the box
- **Model loaded once** — the `reader` is initialised at module level so repeated calls pay no reload cost
- **Multi-language support** — pass any EasyOCR-supported language codes at call time
- **Zero boilerplate** — no configuration classes, no context managers; just import and call

---

## Installation

```bash
pip install easyocr
```

> EasyOCR downloads model weights on first use (~100 MB). An internet connection is required for the initial download; subsequent runs work fully offline.

---

## How to Use

### 1. Basic usage

```python
from mainitt import ITT

text = ITT("screenshot.png")
print(text)
```

### 2. Extract text from any image format

EasyOCR supports JPEG, PNG, BMP, TIFF, and more.

```python
text = ITT("photo.jpg")
text = ITT("scan.bmp")
text = ITT("document.tiff")
```

### 3. Multi-language recognition

Pass a list of BCP-47 / EasyOCR language codes as the second argument.

```python
# English + French
text = ITT("menu.jpg", languages=["en", "fr"])

# English + Hindi
text = ITT("sign.png", languages=["en", "hi"])
```

> Note: The module-level `reader` is initialised with `['en']` only. To use other languages reliably, re-initialise `easyocr.Reader` with the desired codes before calling `readtext`.

### 4. Batch processing multiple images

```python
from mainitt import ITT
from pathlib import Path

results = {}
for img in Path("./images").glob("*.png"):
    results[img.name] = ITT(str(img))

for name, text in results.items():
    print(f"{name}: {text[:80]}")
```

### 5. Using the result downstream

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

## API Reference

### `ITT(image_path, languages)` → `str`

Runs OCR on the given image and returns all detected text joined into a single space-separated string.

| Parameter    | Type        | Default    | Description                                      |
|--------------|-------------|------------|--------------------------------------------------|
| `image_path` | `str`       | *(required)* | Path to the image file                         |
| `languages`  | `list[str]` | `['en']`   | EasyOCR language codes to use for recognition   |

**Returns:** `str` — all detected text regions joined by spaces, in detection order.

---

## Module-level `reader`

```python
reader = easyocr.Reader(['en'])
```

The reader is created once when the module is imported. This means:

- Model files are downloaded on the **first import** only
- All subsequent `ITT()` calls reuse the same loaded model — no per-call overhead
- If you need languages beyond English, create your own `easyocr.Reader` instance with the required codes

---

## Supported Languages (selected)

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

## Notes

- **Detection order** — text regions are returned in the order EasyOCR detects them, which follows a rough top-to-bottom, left-to-right reading order but may vary for complex layouts.
- **Low-quality images** — blurry, low-contrast, or heavily compressed images will reduce accuracy. Pre-processing with a library like Pillow (resize, sharpen, greyscale) can improve results.
- **GPU acceleration** — EasyOCR uses the GPU automatically if a CUDA-capable device is available. On CPU, recognition is slower but fully functional.

---

## Examples Summary

```python
from mainitt import ITT

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
