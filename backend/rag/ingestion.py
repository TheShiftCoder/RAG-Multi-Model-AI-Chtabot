"""Document Ingestion and Text Extraction Module."""

import re
import logging
from pathlib import Path
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def clean_text(raw_text: str) -> str:
    """Normalizes whitespace and removes null bytes and unprintable control characters."""
    if not raw_text:
        return ""
    # Strip null bytes and non-printable characters except standard whitespace
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", raw_text)
    # Normalize multiple newlines and spaces
    cleaned = re.sub(r"\r\n|\r", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def extract_text_from_file(file_path: Path, file_type: str) -> str:
    """Extracts and normalizes text from PDF or TXT files."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_type.lower().strip(".")
    if ext == "pdf":
        try:
            reader = PdfReader(str(file_path))
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages_text.append(text)
            extracted = "\n\n".join(pages_text)
            return clean_text(extracted)
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise ValueError(f"Could not read PDF contents: {e}")

    elif ext == "txt":
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return clean_text(f.read())
        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {e}")
            raise ValueError(f"Could not read text file: {e}")

    else:
        raise ValueError(f"Unsupported file format for text extraction: {ext}")

