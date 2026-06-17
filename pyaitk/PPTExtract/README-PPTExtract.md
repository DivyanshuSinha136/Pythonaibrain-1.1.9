# PPTXExtractor — PowerPoint Content Extraction Utility

A straightforward utility for extracting text, images, and tables from `.pptx` PowerPoint files using `python-pptx`. Processes all slides and organises extracted content by slide number.

---

## What Is This?

`PPTXExtractor` is a class-based PPTX extraction tool that provides:

- **Text extraction** — pulls all non-empty text from every shape across all slides
- **Image extraction** — saves embedded images to disk, preserving their original format (PNG, JPEG, etc.)
- **Table extraction** — reads all table shapes and returns row/cell data as nested lists
- **All-in-one extraction** — a single `extract_all()` call returns text, images, and tables together
- **Auto output directory** — creates the image output folder automatically if it doesn't exist
- **Slide-indexed results** — all output is keyed by slide number (1-based) for easy lookup

---

## Installation

```bash
pip install python-pptx
```

---

## How to Use

### 1. Extract everything at once

```python
from pptx_extractor import PPTXExtractor

extractor = PPTXExtractor("presentation.pptx")
data = extractor.extract_all()

# data["texts"]  → {slide_num: [str, ...]}
# data["images"] → {slide_num: [image_path, ...]}
# data["tables"] → {slide_num: [[[cell, ...], ...], ...]}
```

### 2. Extract text only

```python
extractor = PPTXExtractor("presentation.pptx")
texts = extractor.extract_text()

for slide_num, lines in texts.items():
    print(f"Slide {slide_num}:")
    for line in lines:
        print(f"  {line}")
```

### 3. Extract and save images

```python
extractor = PPTXExtractor("presentation.pptx", image_output_dir="my_images")
images = extractor.extract_images()

for slide_num, paths in images.items():
    for path in paths:
        print(f"Slide {slide_num}: saved → {path}")
```

Images are saved as `slide{N}_image{M}.{ext}` inside the output directory.

### 4. Extract tables

```python
extractor = PPTXExtractor("presentation.pptx")
tables = extractor.extract_tables()

for slide_num, slide_tables in tables.items():
    for table in slide_tables:
        for row in table:
            print("\t".join(row))
```

### 5. Iterate all content by slide

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

## API Reference

### `PPTXExtractor(pptx_path, image_output_dir)`

| Parameter          | Type  | Default               | Description                                      |
|--------------------|-------|-----------------------|--------------------------------------------------|
| `pptx_path`        | `str` | *(required)*          | Path to the `.pptx` file                         |
| `image_output_dir` | `str` | `"extracted_images"`  | Directory where extracted images will be saved   |

The image output directory is created automatically if it does not exist.

---

### `extract_text()` → `dict[int, list[str]]`

Returns all non-empty text from every shape across all slides.

```python
{
    1: ["Title of Slide One", "Bullet point A", "Bullet point B"],
    2: ["Slide Two heading", "Some body text"],
}
```

---

### `extract_images()` → `dict[int, list[str]]`

Saves all embedded images to `image_output_dir` and returns their file paths.

```python
{
    1: ["extracted_images/slide1_image1.png"],
    3: ["extracted_images/slide3_image1.jpeg", "extracted_images/slide3_image2.png"],
}
```

Filenames follow the pattern: `slide{slide_num}_image{shape_num}.{ext}`

---

### `extract_tables()` → `dict[int, list[list[list[str]]]]`

Returns table data as nested lists: slide → list of tables → list of rows → list of cell strings.

```python
{
    2: [
        [["Header A", "Header B"], ["Row 1A", "Row 1B"], ["Row 2A", "Row 2B"]],
    ]
}
```

---

### `extract_all()` → `dict`

Runs all three extractors and returns a combined dictionary:

```python
{
    "texts":  { 1: [...], 2: [...] },
    "images": { 1: [...], 3: [...] },
    "tables": { 2: [...] },
}
```

---

## CLI Usage

Run directly from the terminal (edit the filename inside the script):

```bash
python __init__.py
```

By default it opens `your_presentation.pptx` in the current directory and prints all extracted content to stdout.

**Output format:**
```
Slide 1 Texts:
- Company Overview
- Q3 2024 Results

Slide 1 Images:
- extracted_images/slide1_image1.png

Slide 2 Tables:
Region      Revenue
North       $1.2M
South       $0.9M
---
```

---

## Examples Summary

```python
from pptx_extractor import PPTXExtractor

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