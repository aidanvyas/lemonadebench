"""Factory for creating appropriate AI players based on model names."""

import logging
from typing import Any

from .base_player import BasePlayer

logger = logging.getLogger(__name__)


class PlayerFactory:
    """Factory for creating AI players based on model provider."""

    # Map of model prefixes to their providers
    MODEL_PROVIDERS = {
        # OpenAI models
        "gpt-4.1": "openai",
        "gpt-5": "openai",
        "o1": "openai",
        "o3": "openai",
        "o4": "openai",
    }

    @classmethod
    def create_player(
        cls, model_name: str, api_key: str | None = None, **kwargs: Any
    ) -> BasePlayer:
        """Create an appropriate AI player based on the model name.

        Args:
            model_name: Name of the model (e.g., 'gpt-4.1-nano', 'gemini-2.0-flash-exp')
            api_key: Optional API key (uses environment variables if not provided)
            **kwargs: Additional arguments to pass to the player constructor

        Returns:
            An appropriate AI player instance

        Raises:
            ValueError: If the model provider cannot be determined
        """
        provider = cls._get_provider(model_name)

        logger.info(f"Creating {provider} player for model {model_name}")

        if provider == "openai":
            from .openai_player import OpenAIPlayer  # lazy import to avoid test deps

            return OpenAIPlayer(model_name=model_name, api_key=api_key, **kwargs)

        raise ValueError(
            f"Unknown model provider for '{model_name}'. "
            f"Supported prefixes: {list(cls.MODEL_PROVIDERS.keys())}"
        )

    @classmethod
    def _get_provider(cls, model_name: str) -> str:
        """Determine the provider based on the model name.

        Args:
            model_name: Name of the model

        Returns:
            Provider name ('openai' or 'gemini')

        Raises:
            ValueError: If provider cannot be determined
        """
        model_lower = model_name.lower()

        # Check each prefix
        for prefix, provider in cls.MODEL_PROVIDERS.items():
            if model_lower.startswith(prefix):
                return provider

        # If no match found, raise an error
        raise ValueError(
            f"Cannot determine provider for model '{model_name}'. "
            f"Model should start with one of: {list(cls.MODEL_PROVIDERS.keys())}"
        )

    @classmethod
    def is_supported_model(cls, model_name: str) -> bool:
        """Check if a model is supported.

        Args:
            model_name: Name of the model

        Returns:
            True if the model is supported, False otherwise
        """
        try:
            cls._get_provider(model_name)
            return True
        except ValueError:
            return False
