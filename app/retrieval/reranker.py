
"""
Cross-encoder reranking for the RAG document assistant.

The reranker receives candidate chunks from the vector retriever,
scores each query-chunk pair, and returns the most relevant chunks first.
"""

import logging
import os
from typing import Any, Optional, Sequence, Union

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Multilingual reranking model suitable for French and English.
DEFAULT_RERANKER_MODEL = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

Document = Union[str, dict[str, Any]]


class Reranker:
    """Rerank retrieved documents using a cross-encoder."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        text_key: str = "text",
        device: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = (
            model_name
            or os.getenv(
                "RERANKER_MODEL",
                DEFAULT_RERANKER_MODEL,
            )
        )
        self.text_key = text_key
        self.device = device
        self.batch_size = batch_size
        self._model: Optional[CrossEncoder] = None

    def _load_model(self) -> CrossEncoder:
        """Load the model only when reranking is first requested."""
        if self._model is None:
            logger.info(
                "Loading cross-encoder model: %s",
                self.model_name,
            )

            kwargs = {}
            if self.device:
                kwargs["device"] = self.device

            self._model = CrossEncoder(
                self.model_name,
                **kwargs,
            )

            logger.info("Cross-encoder model loaded.")

        return self._model

    def _get_text(self, document: Document) -> str:
        """Extract text from a string or a document dictionary."""
        if isinstance(document, str):
            return document

        if not isinstance(document, dict):
            raise TypeError(
                "Each document must be a string or a dictionary."
            )

        # Support common chunk formats used in RAG applications.
        for key in (
            self.text_key,
            "text",
            "content",
            "page_content",
            "document",
        ):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value

        raise ValueError(
            f"No text field found in document. "
            f"Expected one of: {self.text_key}, text, content, "
            "page_content, document."
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Rank documents by cross-encoder relevance.

        Args:
            query: User's question.
            documents: Candidate chunks from the retriever.
            top_k: Maximum number of results. None returns all.

        Returns:
            Documents ordered by descending reranker score.

            Dictionary documents retain their original fields and
            receive an additional 'rerank_score' field.
            String documents remain strings.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        if not documents:
            return []

        texts = [self._get_text(doc) for doc in documents]

        model = self._load_model()
        pairs = [(query, text) for text in texts]

        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        if top_k is not None:
            ranked = ranked[:top_k]

        results: list[Document] = []

        for document, score in ranked:
            score = float(score)

            if isinstance(document, dict):
                result = document.copy()
                result["rerank_score"] = score
                results.append(result)
            else:
                results.append(document)

        logger.info(
            "Reranked %d documents; returning %d.",
            len(documents),
            len(results),
        )

        return results


# Reuse one instance so the model is not loaded for every request.
reranker = Reranker()


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_k: Optional[int] = None,
) -> list[Document]:
    """Convenience function for integration with the RAG pipeline."""
    return reranker.rerank(
        query=query,
        documents=documents,
        top_k=top_k,
    )
