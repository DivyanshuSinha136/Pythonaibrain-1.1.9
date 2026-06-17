# PDF Text Extractor — PDF Extraction Utility

A lightweight, production-ready utility for extracting text content from PDF files using PyMuPDF (`fitz`). Handles multi-page documents, corrupted files, and partial extraction failures gracefully.

---

## What Is This?

`__init__.py` is a focused PDF text extraction module that provides:

- **Full document extraction** — reads all pages and joins them into a single string
- **Per-page fault tolerance** — if a single page fails, extraction continues for the rest
- **Configurable page separator** — control how pages are joined in the output string
- **Structured logging** — all warnings and errors go through Python's `logging` module
- **Custom exception** — `PDFExtractionError` wraps PyMuPDF errors for clean upstream handling
- **Input validation** — checks for empty paths, missing files, and non-file paths before opening
- **CLI entry-point** — run directly from the terminal to preview extracted text

---

## Installation

```bash
pip install PyMuPDF
```

> PyMuPDF installs as `fitz` — no separate import needed beyond `import fitz`.

---

## How to Use

### 1. Basic extraction

```python
from PTT import extract_text_from_pdf

text = extract_text_from_pdf("document.pdf")
print(text)
```

### 2. Custom page separator

```python
text = extract_text_from_pdf("report.pdf", page_separator="\n---\n")
print(text)
```

### 3. Custom encoding

```python
text = extract_text_from_pdf("document.pdf", encoding="latin-1")
```

### 4. With error handling

```python
from PTT import extract_text_from_pdf, PDFExtractionError

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

### 5. Processing multiple files

```python
from pathlib import Path
from PTT import extract_text_from_pdf, PDFExtractionError

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

## API Reference

### `extract_text_from_pdf(pdf_path, encoding, page_separator)`

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

## Exception Reference

```
PDFExtractionError
└── Wraps fitz.FileDataError and other PyMuPDF runtime errors
```

`PDFExtractionError` is the single catch-all for PDF-level failures. `FileNotFoundError` and `ValueError` are raised directly for path/input issues and do not need to be caught as `PDFExtractionError`.

---

## Behaviour Notes

- **Empty PDF** — if the document has zero pages, an empty string `""` is returned and a warning is logged.
- **Page-level errors** — if one page fails to extract, that page contributes an empty string and extraction continues for remaining pages. The error is logged.
- **No partial results lost** — all successfully extracted pages are still returned even if some pages failed.

---

## CLI Usage

Run directly from the terminal to extract and preview text from a PDF:

```bash
python __init__.py [pdf_file]
```

If no file is specified, it defaults to `sample.pdf` in the current directory.

### CLI Examples

```bash
# Extract from a specific file
python __init__.py report.pdf

# Use default sample.pdf
python __init__.py
```

**Output:**
```
Successfully extracted 14302 characters from report.pdf

First 500 characters:
--------------------------------------------------------------------------------
Introduction

This report covers the key findings from Q3 2024...
```

---

## Examples Summary

```python
# Minimal usage
from PTT import extract_text_from_pdf
text = extract_text_from_pdf("file.pdf")

# Custom separator between pages
text = extract_text_from_pdf("file.pdf", page_separator="\n--- PAGE BREAK ---\n")

# Full error handling
from pdf import extract_text_from_pdf, PDFExtractionError
try:
    text = extract_text_from_pdf("file.pdf")
except FileNotFoundError:
    print("File not found.")
except PDFExtractionError as e:
    print(f"Extraction failed: {e}")

# Batch processing a folder
from pathlib import Path
from pdf import extract_text_from_pdf, PDFExtractionError
for f in Path("./docs").glob("*.pdf"):
    try:
        print(f"{f.name}: {len(extract_text_from_pdf(str(f)))} chars")
    except PDFExtractionError:
        print(f"{f.name}: failed")
```
