
"""
Anthropic Provider
------------------
Integrates Anthropic Claude models into the RAG application.

Uses the official Anthropic Python SDK and Messages API.
"""

import logging
import os
from typing import Optional

from anthropic import (
    Anthropic,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from app.generation.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY.
            timeout: HTTP request timeout in seconds.
        """
        super().__init__()

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key is missing. "
                "Set the ANTHROPIC_API_KEY environment variable."
            )

        self.timeout = timeout

        self.client = Anthropic(
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate an answer using an Anthropic Claude model.

        Args:
            messages: Conversation messages.
            model: Anthropic model name.
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

        if not 0 <= temperature <= 1:
            raise ValueError(
                "Anthropic temperature must be between 0 and 1."
            )

        if max_tokens < 1:
            raise ValueError(
                "max_tokens must be greater than 0."
            )

        system_messages = []
        conversation_messages = []

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(
                    "Each message must be a dictionary."
                )

            role = message.get("role")
            content = message.get("content")

            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"Unsupported message role: {role}"
                )

            if not isinstance(content, str):
                raise ValueError(
                    "Each message must contain string content."
                )

            if role == "system":
                system_messages.append(content)

            else:
                conversation_messages.append({
                    "role": role,
                    "content": content,
                })

        if not conversation_messages:
            raise ValueError(
                "At least one user or assistant message is required."
            )

        # Anthropic uses a separate system parameter.
        system_prompt = "\n\n".join(system_messages)

        request_params = {
            "model": model,
            "messages": conversation_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        try:
            response = self.client.messages.create(
                **request_params
            )

            # Extract generated text blocks.
            answer = "\n".join(
                block.text
                for block in response.content
                if block.type == "text"
            )

            if not answer.strip():
                raise RuntimeError(
                    "Anthropic returned an empty response."
                )

            logger.info(
                "Successfully generated response using Claude model: %s",
                model,
            )

            return answer.strip()

        except AuthenticationError as exc:
            logger.error(
                "Anthropic authentication failed: %s",
                exc,
            )

            raise RuntimeError(
                "Anthropic authentication failed. "
                "Check your ANTHROPIC_API_KEY."
            ) from exc

        except RateLimitError as exc:
            logger.error(
                "Anthropic rate limit reached: %s",
                exc,
            )

            raise RuntimeError(
                "Anthropic rate limit reached. "
                "Check your API usage and billing."
            ) from exc

        except APITimeoutError as exc:
            logger.error(
                "Anthropic request timed out: %s",
                exc,
            )

            raise TimeoutError(
                f"Anthropic request timed out after "
                f"{self.timeout} seconds."
            ) from exc

        except APIConnectionError as exc:
            logger.error(
                "Could not connect to Anthropic: %s",
                exc,
            )

            raise ConnectionError(
                "Could not connect to the Anthropic API."
            ) from exc

        except APIStatusError as exc:
            logger.error(
                "Anthropic API returned status %s: %s",
                exc.status_code,
                exc,
            )

            raise RuntimeError(
                f"Anthropic API error (HTTP {exc.status_code})."
            ) from exc

    def close(self) -> None:
        """Close the Anthropic HTTP client."""
        self.client.close()
