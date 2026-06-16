# app/ingestion/embedder.py

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


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