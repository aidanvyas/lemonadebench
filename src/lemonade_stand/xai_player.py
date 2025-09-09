"""xAI (Grok) player scaffold for Lemonade Stand.

This scaffolds support for xAI's Grok models with tool/function calling. Imports
are lazy to keep tests independent of the SDK.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base_player import BasePlayer
from .business_game import BusinessGame
from .game_recorder import GameRecorder

logger = logging.getLogger(__name__)


class XAIPlayer(BasePlayer):
    """AI player using xAI Grok models (e.g., grok-2)."""

    def __init__(
        self,
        model_name: str = "grok-2",
        api_key: str | None = None,
        *,
        api_max_retries: int = 3,
        api_backoff: float = 1.0,
    ) -> None:
        super().__init__(model_name, api_key)

        self.api_max_retries = api_max_retries
        self.api_backoff = api_backoff

        # Pricing placeholders (per 1M tokens); adjust when official.
        self.model_pricing = {
            "grok-2": {"input": 1.00, "cached_input": 0.25, "output": 4.00},
        }

        # Lazy client init; the python SDK may not be installed.
        self._client = None

        key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        if not key:
            logger.warning("XAI/GROK API key not set; player will raise on use")
        else:
            try:
                # No official xAI SDK at the time of writing; future-proof placeholder
                # Implement when SDK or OpenAI-compatible endpoint is available.
                self._client = object()  # sentinel
            except ModuleNotFoundError:  # pragma: no cover
                logger.warning("xAI SDK not installed; cannot use XAIPlayer")
                self._client = None

    def close(self) -> None:  # pragma: no cover - no-op
        pass

    def play_turn(
        self, game: BusinessGame, recorder: GameRecorder | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "XAIPlayer play_turn not implemented yet. This is a scaffold."
        )

    def calculate_cost(self) -> dict[str, float]:
        pricing = self.model_pricing.get(
            self.model_name, {"input": 1.0, "cached_input": 0.5, "output": 2.0}
        )
        non_cached_input = (
            self.total_token_usage["input_tokens"]
            - self.total_token_usage["cached_input_tokens"]
        )
        input_cost = (non_cached_input / 1_000_000) * pricing["input"]
        cached_cost = (
            self.total_token_usage["cached_input_tokens"] / 1_000_000
        ) * pricing["cached_input"]
        output_cost = (self.total_token_usage["output_tokens"] / 1_000_000) * pricing[
            "output"
        ]
        return {
            "input_cost": input_cost,
            "cached_cost": cached_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + cached_cost + output_cost,
            "total_tokens": self.total_token_usage["total_tokens"],
        }

    def reset(self) -> None:
        self.errors = []
        self.total_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }
