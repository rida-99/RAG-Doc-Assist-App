
# app/retrieval/retriever.py

import logging
import math
import threading

from typing import Any

from app.core.config import settings
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore


logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant document chunks for a user query.

    Responsibilities:
    - Encode the user query using the document embedding model.
    - Search the ChromaDB vector store.
    - Filter irrelevant results using cosine distance.
    - Remove duplicate passages.
    - Prepare a source-aware context for the LLM.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: Embedder | None = None,
        top_k: int | None = None,
        fetch_k: int | None = None,
        max_distance: float = 0.35,
    ) -> None:
        self.vector_store = vector_store or VectorStore()

        # Reuse the existing embedding model when provided.
        self._embedder = embedder
        self._embedder_lock = threading.Lock()

        self.top_k = (
            top_k
            if top_k is not None
            else getattr(settings, "retrieval_top_k", 5)
        )

        self.fetch_k = (
            fetch_k
            if fetch_k is not None
            else max(self.top_k * 3, self.top_k)
        )

        self.max_distance = max_distance

        if self.top_k < 1:
            raise ValueError("top_k must be positive.")

        if self.fetch_k < self.top_k:
            raise ValueError("fetch_k must be >= top_k.")

        if not math.isfinite(max_distance) or not 0 <= max_distance <= 2:
            raise ValueError(
                "max_distance must be between 0 and 2."
            )

    def _get_embedder(self) -> Embedder:
        """Load the embedding model lazily and only once."""

        if self._embedder is None:
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = Embedder()

        return self._embedder

    @staticmethod
    def _validate_query(query: str) -> str:
        """Validate and normalize the user's question."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        return query.strip()

    def encode_query(self, query: str) -> list[float]:
        """
        Encode a query using the multilingual E5 Instruct format.

        Document embeddings are generated separately by Embedder,
        which uses the 'passage:' prefix.
        """

        query = self._validate_query(query)

        formatted_query = (
            "Instruct: Given a question, retrieve passages "
            "that answer the question\n"
            f"Query: {query}"
        )

        embedder = self._get_embedder()

        # Do not call embedder.embed_texts() here: that method
        # adds the 'passage:' prefix intended for documents.
        embedding = embedder.model.encode(
            [formatted_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        vector = [float(value) for value in embedding]

        if not vector or not all(
            math.isfinite(value) for value in vector
        ):
            raise ValueError("The query embedding is invalid.")

        return vector

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        fetch_k: int | None = None,
        max_distance: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve, filter and rank relevant chunks.

        Args:
            query: User question.
            top_k: Maximum number of final results.
            fetch_k: Number of candidates to retrieve initially.
            max_distance: Maximum accepted cosine distance.
            where: Optional ChromaDB metadata filter.

        Returns:
            A ranked list of chunks containing text, metadata,
            distance and similarity.
        """

        query = self._validate_query(query)

        final_k = self.top_k if top_k is None else top_k
        candidate_k = (
            self.fetch_k if fetch_k is None else fetch_k
        )
        distance_limit = (
            self.max_distance
            if max_distance is None
            else max_distance
        )

        if final_k < 1:
            raise ValueError("top_k must be positive.")

        if candidate_k < final_k:
            raise ValueError("fetch_k must be >= top_k.")

        if (
            not math.isfinite(distance_limit)
            or not 0 <= distance_limit <= 2
        ):
            raise ValueError(
                "max_distance must be between 0 and 2."
            )

        # Step 1: Encode the question.
        query_embedding = self.encode_query(query)

        # Step 2: Retrieve more candidates than ultimately needed.
        candidates = self.vector_store.search_by_embedding(
            query_embedding=query_embedding,
            top_k=candidate_k,
            where=where,
        )

        # Step 3: Filter and deduplicate candidates.
        results: list[dict[str, Any]] = []
        seen_texts: set[str] = set()

        for candidate in candidates:
            distance = candidate.get("distance")

            if (
                not isinstance(distance, (int, float))
                or not math.isfinite(distance)
            ):
                continue

            distance = float(distance)

            # Lower cosine distance means greater similarity.
            if distance > distance_limit:
                continue

            text = candidate.get("text", "")

            if not isinstance(text, str) or not text.strip():
                continue

            # Normalize whitespace and case for deduplication.
            normalized_text = " ".join(text.casefold().split())

            if normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)

            results.append(
                {
                    "id": candidate["id"],
                    "text": text,
                    "metadata": candidate.get("metadata") or {},
                    "distance": distance,
                    "similarity": 1.0 - distance,
                }
            )

        # Step 4: Keep the best results.
        results.sort(key=lambda item: item["distance"])
        results = results[:final_k]

        logger.info(
            "Retrieval completed: candidates=%d, accepted=%d",
            len(candidates),
            len(results),
        )

        return results

    @staticmethod
    def build_context(
        results: list[dict[str, Any]],
    ) -> str:
        """
        Build a source-aware text context for the LLM.

        The source labels help the LLM identify where information
        came from. They do not guarantee factual citations.
        """

        if not results:
            return (
                "No sufficiently relevant document passages "
                "were found."
            )

        sections: list[str] = []

        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}

            file_name = metadata.get("file_name", "Unknown source")
            page_number = metadata.get("page_number", "Unknown")
            doc_id = metadata.get("doc_id", "Unknown")

            sections.append(
                f"[Source {index}: {file_name}, "
                f"page {page_number}, document {doc_id}]\n"
                f"{result['text']}"
            )

        return "\n\n---\n\n".join(sections)


if __name__ == "__main__":
    vector_store = VectorStore(
        persist_directory="storage/test_vector_store",
        collection_name="test_documents",
    )
    embedder = Embedder()
    retriever = Retriever(vector_store=vector_store, embedder=embedder, top_k=5, fetch_k=15, max_distance=0.35)
    try:
        question = input("Enter a question: ")
        results = retriever.retrieve(question)

        print(f"Number of relevant chunks: {len(results)}")
        print("\n===== CONTEXT FOR LLM =====")
        print(retriever.build_context(results))
    finally:
        retriever.vector_store.close()


# run with
# python -m app.retrieval.retriever