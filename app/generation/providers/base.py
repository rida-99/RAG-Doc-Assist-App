
"""
Base LLM Provider
-----------------
Defines the common interface for all LLM providers.

Every provider (Ollama, OpenAI, Anthropic) must implement
the same generate() method.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """

    def __init__(self):
        """Initialize the provider."""
        pass

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a response using the configured LLM.

        Args:
            messages:
                Conversation messages.

                Example:
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "Explain RAG."
                    }
                ]

            model:
                Name of the LLM model.

            temperature:
                Controls response randomness.

            max_tokens:
                Maximum number of generated tokens.

        Returns:
            Generated response as a string.

        Raises:
            NotImplementedError:
                If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Return a readable provider representation."""
        return f"{self.__class__.__name__}()"


