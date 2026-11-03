
"""
Ollama Provider
---------------
Integrates locally hosted Ollama models into the RAG application.

Uses Ollama's /api/chat endpoint.
"""

import logging
from typing import Optional

import requests

from app.generation.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Provider for locally hosted Ollama models."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        """
        Initialize the Ollama provider.

        Args:
            base_url: Ollama server URL.
            timeout: HTTP request timeout in seconds.
        """
        super().__init__()

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.chat_endpoint = f"{self.base_url}/api/chat"

        self.session = requests.Session()

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3.2:3b",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate an answer using Ollama.

        Args:
            messages: Conversation messages.
            model: Ollama model name.
            temperature: Controls response randomness.
            max_tokens: Maximum generated tokens.

        Returns:
            Generated response as a string.

        Raises:
            ConnectionError: If Ollama is unavailable.
            RuntimeError: If Ollama returns an invalid response.
        """

        if not messages:
            raise ValueError("Messages cannot be empty.")

        if not model:
            raise ValueError("Model name cannot be empty.")

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = self.session.post(
                self.chat_endpoint,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()

            answer = data.get("message", {}).get("content")

            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(
                    "Ollama returned an empty or invalid response."
                )

            logger.info(
                "Successfully generated response using Ollama model: %s",
                model,
            )

            return answer.strip()

        except requests.exceptions.ConnectionError as exc:
            logger.error("Could not connect to Ollama: %s", exc)

            raise ConnectionError(
                "Ollama server is unavailable. "
                "Make sure Ollama is running at "
                f"{self.base_url}."
            ) from exc

        except requests.exceptions.Timeout as exc:
            logger.error("Ollama request timed out: %s", exc)

            raise TimeoutError(
                f"Ollama generation timed out after {self.timeout} seconds."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            logger.error("Ollama HTTP error: %s", exc)

            raise RuntimeError(
                f"Ollama returned an HTTP error: {exc}"
            ) from exc

        except ValueError as exc:
            logger.error("Invalid Ollama response: %s", exc)

            raise RuntimeError(
                "Ollama returned invalid JSON."
            ) from exc

    def check_connection(self) -> bool:
        """
        Check whether the Ollama server is reachable.

        Returns:
            True if Ollama is reachable, otherwise False.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )

            response.raise_for_status()
            return True

        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """
        Return the locally available Ollama model names.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )

            response.raise_for_status()

            data = response.json()

            return [
                model["name"]
                for model in data.get("models", [])
                if "name" in model
            ]

        except requests.RequestException as exc:
            logger.error("Could not retrieve Ollama models: %s", exc)
            raise ConnectionError(
                "Could not retrieve models from Ollama."
            ) from exc

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

