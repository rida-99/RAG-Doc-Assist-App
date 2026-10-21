
# app/ingestion/pipeline.py

import logging
import threading
import uuid

from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.ingestion.validator import validate_and_save
from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import ChunkingStrategy
from app.ingestion.embedder import Embedder


logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    Orchestrates the complete document ingestion process:

    1. Validate and save the uploaded document.
    2. Parse the document into page-level records.
    3. Split the extracted text into chunks.
    4. Generate embeddings for each chunk.
    5. Return the processed document and its metadata.

    Vector storage is handled separately by the retrieval layer.
    """

    def __init__(
        self,
        chunker: ChunkingStrategy | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.chunker = chunker or ChunkingStrategy()

        # Load the embedding model only when it is first needed.
        self._embedder = embedder
        self._embedder_lock = threading.Lock()

    def _get_embedder(self) -> Embedder:
        """Initialize the embedding model once per pipeline instance."""
        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = Embedder()

        return self._embedder

    async def ingest(self, file: UploadFile) -> dict[str, Any]:
        """
        Ingest an uploaded file asynchronously.

        Returns:
            A dictionary containing document metadata and embedded chunks.
        """

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="A filename is required.",
            )

        original_filename = Path(file.filename).name

        # Step 1: Validate the upload and save the original file.
        saved_path, sha256 = await validate_and_save(file)

        # Generate a unique identifier for this ingestion.
        doc_id = str(uuid.uuid4())

        logger.info(
            "Starting ingestion: doc_id=%s, filename=%s",
            doc_id,
            original_filename,
        )

        # Steps 2-4: Run synchronous document processing outside
        # the asynchronous event loop.
        result = await run_in_threadpool(
            self._process_document,
            saved_path,
            original_filename,
            doc_id,
            sha256,
        )

        logger.info(
            "Ingestion completed: doc_id=%s, chunks=%d",
            doc_id,
            result["chunk_count"],
        )

        return result

    def _process_document(
        self,
        file_path: Path,
        original_filename: str,
        doc_id: str,
        sha256: str,
    ) -> dict[str, Any]:
        """Parse, chunk and embed a previously validated document."""

        # Step 2: Parse the document.
        try:
            pages = DocumentParser.parse_document(
                file_path=file_path,
                doc_id=doc_id,
            )
        except Exception as exc:
            logger.warning(
                "Document parsing failed: doc_id=%s, error=%s",
                doc_id,
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=422,
                detail="Unable to extract text from this document.",
            ) from exc

        # Reject documents that contain no extractable text.
        if not pages or not any(
            page.get("text", "").strip() for page in pages
        ):
            raise HTTPException(
                status_code=422,
                detail="The document contains no extractable text.",
            )

        # The validator saves files under a hash-based filename.
        # Preserve the original filename in the document metadata.
        for page in pages:
            page["file_name"] = original_filename

        # Step 3: Split the extracted text into chunks.
        chunks = self.chunker.chunk_pages(pages)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text chunks could be generated.",
            )

        # Step 4: Generate embeddings.
        embedder = self._get_embedder()
        embedded_chunks = embedder.embed_chunks(chunks)

        # Step 5: Return the ingestion result.
        return {
            "doc_id": doc_id,
            "sha256": sha256,
            "file_name": original_filename,
            "file_type": file_path.suffix.lower(),
            "saved_path": str(file_path),
            "page_count": len(pages),
            "chunk_count": len(embedded_chunks),
            "chunks": embedded_chunks,
        }

# test pipeline
if __name__ == "__main__":
    import asyncio
    import sys
    from pprint import pprint
    from fastapi import UploadFile

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    upload_file = UploadFile(filename=file_path.name, file=open(file_path, "rb"))

    pipeline = DocumentIngestionPipeline()
    result = asyncio.run(pipeline.ingest(upload_file))
    pprint(result)
    
# run with
# python -m app.ingestion.pipeline ".\app\ingestion\RAG text.pdf"