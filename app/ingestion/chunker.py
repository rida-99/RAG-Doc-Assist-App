# app/ingestion/chunker.py

from typing import List, Dict
import uuid

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

from app.core.config import settings


class ChunkingStrategy:
    """
    Split parsed documents into smaller chunks while preserving metadata.
    """

    def __init__(
        self,
        method: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        encoding_name: str = "cl100k_base",
    ):
        self.method = method or settings.chunk_method
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding_name = encoding_name

    def chunk_text(self, text: str) -> List[str]:
        """
        Split a single text into chunks.
        """

        if self.method == "fixed":
            splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.encoding_name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        elif self.method == "recursive":
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.encoding_name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ".",
                    " ",
                    "",
                ],
            )

        else:
            raise ValueError(
                f"Unknown chunking method '{self.method}'. "
                "Choose 'fixed' or 'recursive'."
            )

        return splitter.split_text(text)

    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Convert parsed pages into chunk records.

        Input:
            [
                {
                    "text": "...",
                    "source": "...",
                    "file_name": "...",
                    ...
                }
            ]

        Output:
            [
                {
                    "id": "...",
                    "text": "...",
                    "metadata": {...}
                }
            ]
        """

        chunks = []

        for page in pages:
            text_chunks = self.chunk_text(page["text"])

            for chunk_text in text_chunks:
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "metadata": {
                            "source": page["source"],
                            "doc_id": page["doc_id"],
                            "file_name": page["file_name"],
                            "file_type": page["file_type"],
                            "page_number": page["page_number"],
                            "chunk_method": self.method,
                        },
                    }
                )

        return chunks