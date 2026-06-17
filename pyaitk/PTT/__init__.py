"""PDF text extraction utility.

This module provides functionality to extract text content from PDF files
using PyMuPDF (fitz).
"""

import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when PDF text extraction fails."""
    pass


def extract_text_from_pdf(
    pdf_path: str,
    encoding: str = "utf-8",
    page_separator: str = "\n\n"
) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file to extract text from.
        encoding: Text encoding to use (default: utf-8).
        page_separator: String to insert between pages (default: double newline).
    
    Returns:
        Extracted text content from all pages of the PDF.
    
    Raises:
        PDFExtractionError: If the PDF cannot be opened or read.
        FileNotFoundError: If the specified PDF file does not exist.
        ValueError: If pdf_path is None or empty.
    
    Example:
        >>> text = extract_text_from_pdf("document.pdf")
        >>> print(text[:100])
    """
    if not pdf_path:
        raise ValueError("pdf_path cannot be None or empty")
    
    path = Path(pdf_path)
    
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")
    
    try:
        text_pages = []
        
        with fitz.open(pdf_path) as doc:
            if doc.page_count == 0:
                logger.warning(f"PDF has no pages: {pdf_path}")
                return ""
            
            for page_num, page in enumerate(doc, start=1):
                try:
                    page_text = page.get_text(encoding=encoding)
                    text_pages.append(page_text)
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num}: {e}")
                    # Continue processing other pages
                    text_pages.append("")
        
        return page_separator.join(text_pages)
    
    except fitz.FileDataError as e:
        raise PDFExtractionError(f"Invalid or corrupted PDF file: {pdf_path}") from e
    except Exception as e:
        raise PDFExtractionError(f"Failed to extract text from PDF: {str(e)}") from e


if __name__ == "__main__":
    # Example usage with error handling
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = "sample.pdf"
    
    try:
        extracted_text = extract_text_from_pdf(pdf_file)
        print(f"Successfully extracted {len(extracted_text)} characters from {pdf_file}")
        print("\nFirst 500 characters:")
        print("-" * 80)
        print(extracted_text[:500])
    except (PDFExtractionError, FileNotFoundError, ValueError) as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
