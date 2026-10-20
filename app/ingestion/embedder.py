# app/ingestion/embedder.py

from pathlib import Path
from typing import List, Dict
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.ingestion.chunker import ChunkingStrategy
from app.ingestion.parser import DocumentParser


class Embedder:
    """
    Handles embedding generation for RAG chunks using SentenceTransformer.
    """

    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(settings.embed_model)
        print("Embedding model loaded.")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert list of texts into embeddings.
        """
        texts = [f"passage: {t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=settings.embed_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        return np.array(embeddings)

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Add embeddings to chunk objects.

        Input:
            [
                {
                    "id": "...",
                    "text": "...",
                    "metadata": {...}
                }
            ]

        Output:
            same list + "embedding"
        """

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()

        return chunks

# test embedder
if __name__ == "__main__":
    import sys
    from pprint import pprint

    if len(sys.argv) < 2:
        print("Usage: python embedder.py <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    doc_id = str(uuid.uuid4())

    # Parse document into pages
    pages = DocumentParser.parse_document(file_path, doc_id)

    # Chunk pages into smaller pieces
    chunker = ChunkingStrategy()
    chunks = chunker.chunk_pages(pages)

    # Embed chunks
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(chunks)

    pprint(embedded_chunks)
    

# run with
# python -m app.ingestion.embedder ".\app\ingestion\RAG text.pdf"