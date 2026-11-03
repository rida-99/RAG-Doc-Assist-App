"""
OpenAI Provider
---------------
Integrates OpenAI models into the RAG application.

Uses the official OpenAI Python SDK and Chat Completions API.
"""

import logging
import os
from typing import Optional

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from app.generation.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI hosted language models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY.
            base_url: Optional custom API endpoint.
            timeout: HTTP request timeout in seconds.
        """
        super().__init__()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is missing. "
                "Set the OPENAI_API_KEY environment variable."
            )

        self.timeout = timeout

        client_options = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }

        if base_url:
            client_options["base_url"] = base_url

        self.client = OpenAI(**client_options)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate an answer using an OpenAI model.

        Args:
            messages: Conversation messages.
            model: OpenAI model name.
            temperature: Controls response randomness.
            max_tokens: Maximum number of generated tokens.

        Returns:
            Generated response as a string.

        Raises:
            ValueError: If the input is invalid.
            ConnectionError: If the API cannot be reached.
            TimeoutError: If the request times out.
            RuntimeError: If generation fails.
        """

        if not messages:
            raise ValueError("Messages cannot be empty.")

        if not model:
            raise ValueError("Model name cannot be empty.")

        if not 0 <= temperature <= 2:
            raise ValueError(
                "Temperature must be between 0 and 2."
            )

        if max_tokens < 1:
            raise ValueError(
                "max_tokens must be greater than 0."
            )

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary.")

            if message.get("role") not in {
                "system", "user", "assistant"
            }:
                raise ValueError(
                    f"Unsupported message role: {message.get('role')}"
                )

            if not isinstance(message.get("content"), str):
                raise ValueError(
                    "Each message must contain string content."
                )

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )

            answer = response.choices[0].message.content

            if not answer or not answer.strip():
                raise RuntimeError(
                    "OpenAI returned an empty response."
                )

            logger.info(
                "Successfully generated response using OpenAI model: %s",
                model,
            )

            return answer.strip()

        except AuthenticationError as exc:
            logger.error("OpenAI authentication failed: %s", exc)

            raise RuntimeError(
                "OpenAI authentication failed. "
                "Check your OPENAI_API_KEY."
            ) from exc

        except RateLimitError as exc:
            logger.error("OpenAI rate limit reached: %s", exc)

            raise RuntimeError(
                "OpenAI rate limit reached. "
                "Check your API quota and billing."
            ) from exc

        except APITimeoutError as exc:
            logger.error("OpenAI request timed out: %s", exc)

            raise TimeoutError(
                f"OpenAI request timed out after {self.timeout} seconds."
            ) from exc

        except APIConnectionError as exc:
            logger.error("Could not connect to OpenAI: %s", exc)

            raise ConnectionError(
                "Could not connect to the OpenAI API."
            ) from exc

        except APIStatusError as exc:
            logger.error(
                "OpenAI API returned status %s: %s",
                exc.status_code,
                exc,
            )

            raise RuntimeError(
                f"OpenAI API error (HTTP {exc.status_code})."
            ) from exc

    def close(self) -> None:
        """Close the OpenAI HTTP client."""
        self.client.close()