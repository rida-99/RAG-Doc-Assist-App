
"""
LLM Router
----------
Selects and invokes the configured LLM provider.

Supported providers:
- Ollama
- OpenAI
- Anthropic (Claude)

Expected provider interface:
    provider.generate(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

The provider can return either a string or a dictionary containing
the generated answer.
"""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMRouterError(Exception):
    """Base exception for LLM router errors."""


class UnknownProviderError(LLMRouterError):
    """Raised when the requested LLM provider is unknown."""


class LLMGenerationError(LLMRouterError):
    """Raised when an LLM provider fails to generate an answer."""


class LLMRouter:
    """Routes generation requests to the selected LLM provider."""

    PROVIDER_ALIASES = {
        "ollama": "ollama",
        "openai": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
    }

    DEFAULT_MODELS = {
        "ollama": "llama3.2:3b",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-latest",
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        provider_instance: Optional[Any] = None,
    ):
        """
        Args:
            provider: Provider name. Defaults to LLM_PROVIDER.
            model: Model name. Defaults to the selected provider's model.
            temperature: Generation temperature.
            max_tokens: Maximum number of generated tokens.
            provider_instance: Optional provider instance for dependency
                injection and testing.
        """
        provider_name = (
            provider or os.getenv("LLM_PROVIDER", "ollama")
        ).strip().lower()

        if provider_name not in self.PROVIDER_ALIASES:
            supported = ", ".join(self.PROVIDER_ALIASES)
            raise UnknownProviderError(
                f"Unknown LLM provider '{provider_name}'. "
                f"Supported providers: {supported}"
            )

        self.provider_name = self.PROVIDER_ALIASES[provider_name]
        self.model = (
            model
            or os.getenv("LLM_MODEL")
            or self.DEFAULT_MODELS[self.provider_name]
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")

        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0.")

        self.provider = (
            provider_instance
            if provider_instance is not None
            else self._create_provider()
        )

    def _create_provider(self) -> Any:
        """Create the selected provider using lazy imports."""

        try:
            if self.provider_name == "ollama":
                from app.generation.providers.ollama import (
                    OllamaProvider,
                )
                return OllamaProvider()

            if self.provider_name == "openai":
                from app.generation.providers.openai import (
                    OpenAIProvider,
                )
                return OpenAIProvider()

            if self.provider_name == "anthropic":
                from app.generation.providers.anthropic import (
                    AnthropicProvider,
                )
                return AnthropicProvider()

        except ImportError as exc:
            raise LLMRouterError(
                f"Could not import the {self.provider_name} provider. "
                "Check the provider module and its dependencies."
            ) from exc

        raise UnknownProviderError(
            f"Unsupported provider: {self.provider_name}"
        )

    @staticmethod
    def _extract_messages(prompt_payload: Any) -> list[dict]:
        """Accept the structured prompt payload or a messages list."""

        if isinstance(prompt_payload, dict):
            messages = prompt_payload.get("messages")

            if messages is None:
                raise ValueError(
                    "Prompt payload must contain a 'messages' field."
                )

        elif isinstance(prompt_payload, list):
            messages = prompt_payload

        else:
            raise TypeError(
                "Expected a prompt payload dictionary or a messages list."
            )

        if not messages:
            raise ValueError("The messages list cannot be empty.")

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary.")

            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Unsupported message role: {message.get('role')}"
                )

            if not isinstance(message.get("content"), str):
                raise ValueError(
                    "Each message must contain string content."
                )

        return messages

    @staticmethod
    def _extract_answer(result: Any) -> str:
        """Normalize provider output into a plain answer string."""

        if isinstance(result, str):
            answer = result

        elif isinstance(result, dict):
            answer = (
                result.get("answer")
                or result.get("text")
                or result.get("content")
            )

            if not isinstance(answer, str):
                raise LLMGenerationError(
                    "Provider returned a dictionary without a "
                    "valid answer, text, or content field."
                )

        else:
            raise LLMGenerationError(
                f"Unsupported provider response type: "
                f"{type(result).__name__}"
            )

        answer = answer.strip()

        if not answer:
            raise LLMGenerationError(
                "The LLM provider returned an empty answer."
            )

        return answer

    def generate(
        self,
        prompt_payload: Any,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Generate an answer from a prompt_builder payload.

        Returns:
            {
                "answer": "...",
                "provider": "ollama",
                "model": "llama3.2:3b",
                "sources": [...],
                "context_stats": {...}
            }
        """
        messages = self._extract_messages(prompt_payload)

        effective_temperature = (
            self.temperature
            if temperature is None
            else temperature
        )
        effective_max_tokens = (
            self.max_tokens
            if max_tokens is None
            else max_tokens
        )

        if not 0 <= effective_temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")

        if effective_max_tokens < 1:
            raise ValueError("max_tokens must be greater than 0.")

        try:
            result = self.provider.generate(
                messages=messages,
                model=self.model,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )

            answer = self._extract_answer(result)

        except LLMRouterError:
            raise

        except Exception as exc:
            logger.exception(
                "Generation failed with provider '%s' and model '%s'.",
                self.provider_name,
                self.model,
            )
            raise LLMGenerationError(
                f"Generation failed using {self.provider_name}: {exc}"
            ) from exc

        payload = (
            prompt_payload
            if isinstance(prompt_payload, dict)
            else {}
        )

        return {
            "answer": answer,
            "provider": self.provider_name,
            "model": self.model,
            "sources": payload.get("sources", []),
            "context_stats": payload.get("context_stats", {}),
        }


def generate_answer(
    prompt_payload: Any,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> dict:
    """Convenience function for one generation request."""

    router = LLMRouter(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return router.generate(prompt_payload)



# test prompt builder 
if __name__ == "__main__":
    from app.generation.prompt_builder import build_prompt_payload
    from app.retrieval.vector_store import VectorStore
    from app.ingestion.embedder import Embedder
    from app.retrieval.retriever import Retriever

    query = "What is the difference between retrieval and generation in LLMs?"

    retriever = Retriever(
        vector_store=VectorStore(
            persist_directory="storage/test_vector_store",
            collection_name="test_documents",
        ),
        embedder=Embedder(),
        top_k=5,
        fetch_k=15,
        max_distance=0.35,
    )

    try:
        documents = retriever.retrieve(query)

        payload = build_prompt_payload(
            query=query,
            documents=documents,
            max_context_chars=12000,
            system_instruction="You are a helpful assistant.",
        )

        result = generate_answer(
            prompt_payload=payload,   # contient messages + sources + context_stats
            provider="ollama",
            model="llama3.2:3b",
            temperature=0.5,
            max_tokens=150,
        )

        print("Answer:", result["answer"])
        print("Provider:", result["provider"])
        print("Model:", result["model"])
        print("Sources:", result["sources"])
        print("Context stats:", result["context_stats"])
    finally:
        retriever.vector_store.close()


# run with
# python -m app.generation.llm_router