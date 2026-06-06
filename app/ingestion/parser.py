# app/ingestion/parser.py

from pathlib import Path
from typing import List, Dict
import uuid

from bs4 import BeautifulSoup
from unstructured.partition.auto import partition


class DocumentParser:
    """
    Wrapper around Unstructured IO.

    Converts a document into a list of page-level records.
    
    Simple hybrid parser:
    - TXT → direct read
    - HTML → BeautifulSoup
    - Everything else → unstructured (PDF, DOCX, PPTX, etc.)
    
    """
    @staticmethod
    def parse_document(file_path: Path, doc_id: str) -> List[Dict]:
        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            return DocumentParser._parse_txt(file_path, doc_id)

        if suffix == ".html":
            return DocumentParser._parse_html(file_path, doc_id)

        # PDF, DOCX, PPTX, etc.
        return DocumentParser._parse_with_unstructured(file_path, doc_id)

    # ─────────────────────────────
    # TXT (lightweight)
    # ─────────────────────────────

    @staticmethod
    def _parse_txt(file_path: Path, doc_id: str) -> List[Dict]:
        text = file_path.read_text(encoding="utf-8", errors="ignore")

        return [{
            "text": text,
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": ".txt",
            "page_number": 1,
            "doc_id": doc_id,
        }]

    # ─────────────────────────────
    # HTML (lightweight)
    # ─────────────────────────────

    @staticmethod
    def _parse_html(file_path: Path, doc_id: str) -> List[Dict]:
        html = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        # remove noise
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n")

        # clean empty lines
        text = "\n".join(
            line.strip()
            for line in text.splitlines()
            if line.strip()
        )

        return [{
            "text": text,
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": ".html",
            "page_number": 1,
            "doc_id": doc_id,
        }]

    # ─────────────────────────────
    # FALLBACK (PDF / DOCX / PPTX / etc.)
    # ─────────────────────────────

    @staticmethod
    def _parse_with_unstructured(file_path: Path, doc_id: str) -> List[Dict]:
        elements = partition(str(file_path))

        page_texts: dict[int, list[str]] = {}

        for el in elements:
            page_number = getattr(el.metadata, "page_number", None)
            page_number = int(page_number) if page_number else 1

            page_texts.setdefault(page_number, []).append(str(el))

        pages = []

        for page_number, texts in page_texts.items():
            pages.append({
                "text": " ".join(texts).strip(),
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "page_number": page_number,
                "doc_id": doc_id,
            })

        return pages