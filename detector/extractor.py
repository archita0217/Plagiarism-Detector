"""
extractor.py
------------
Turns an uploaded file into plain text.

Supported: .txt .md .rtf .pdf .docx
PDF uses pdfplumber (better layout handling) and falls back to pypdf.
"""

from __future__ import annotations

import re
from pathlib import Path

import config


class ExtractionError(Exception):
    """Raised when a file cannot be read as text."""


def extract_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise ExtractionError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        raise ExtractionError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
        )

    if suffix == ".pdf":
        text = _from_pdf(path)
    elif suffix == ".docx":
        text = _from_docx(path)
    else:
        text = _from_plain(path)

    text = _normalise(text)
    if len(text.split()) < config.CHUNK_MIN_WORDS:
        raise ExtractionError(
            "The document contains too little readable text. "
            "If it is a scanned PDF, run OCR on it first."
        )
    return text


# ------------------------------------------------------------------ readers
def _from_plain(path: Path) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ExtractionError("Could not decode the text file.")


def _from_pdf(path: Path) -> str:
    pages: list[str] = []

    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        if any(p.strip() for p in pages):
            return "\n".join(pages)
    except ImportError:
        pass
    except Exception:
        pages = []

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except ImportError as exc:
        raise ExtractionError(
            "Install a PDF reader first:  pip install pdfplumber pypdf"
        ) from exc
    except Exception as exc:
        raise ExtractionError(f"Could not read the PDF: {exc}") from exc


def _from_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as exc:
        raise ExtractionError(
            "Install the Word reader first:  pip install python-docx"
        ) from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"Could not read the Word file: {exc}") from exc

    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


# ------------------------------------------------------------------ cleanup
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_MULTI_SPACE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _normalise(text: str) -> str:
    """Repair line-wrapping artefacts that PDFs are famous for."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # "informa-\ntion" -> "information"
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
