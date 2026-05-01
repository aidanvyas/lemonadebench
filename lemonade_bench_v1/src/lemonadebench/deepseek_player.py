"""DeepSeek-based player for the Lemonade Stand business game.

Uses DeepSeek's Chat Completions API (OpenAI-compatible) with thinking mode
enabled so reasoning_content is captured turn-by-turn. Conversation history
is maintained client-side (DeepSeek is a stateless API).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from .business_game import BusinessGame
from .game_recorder import GameRecorder
from .tool_schemas import (
    CheckInventoryParams,
    CheckMorningPricesParams,
    GetHistoricalSupplyCostsParams,
    OpenForBusinessParams,
    OrderSuppliesParams,
    PurchaseAutomationParams,
    SetOperatingHoursParams,
    SetPriceParams,
)

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

SUPPORTED_MODELS = {"deepseek-v4-flash", "deepseek-v4-pro"}

# Per 1M tokens. "input" = cache miss; "cached_input" = cache hit.
# v4-pro reflects the 75%-off promo price (active until 2026/05/31 15:59 UTC).
# Update to $0.0145 / $1.74 / $3.48 once the promo ends.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-flash": {
        "input": 0.14,
        "cached_input": 0.0028,
        "output": 0.28,
    },
    "deepseek-v4-pro": {
        "input": 0.435,
        "cached_input": 0.003625,
        "output": 0.87,
    },
}


class DeepSeekPlayer:
    """AI player that uses DeepSeek's Chat Completions API."""

    def __init__(
        self,
        model_name: str = "deepseek-v4-flash",
        api_key: str | None = None,
        *,
        reasoning_effort: str = "high",
        thinking_enabled: bool = True,
        api_max_retries: int = 3,
        api_backoff: float = 1.0,
    ) -> None:
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported DeepSeek model: {model_name}. "
                f"Supported: {sorted(SUPPORTED_MODELS)}",
            )

        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.thinking_enabled = thinking_enabled
        self.api_max_retries = api_max_retries
        self.api_backoff = api_backoff

        # Client-side conversation history (DeepSeek is stateless).
        # Holds dicts and ChatCompletionMessage objects interchangeably — the
        # OpenAI SDK accepts both shapes.
        self.messages: list[Any] = []

        self.reasoning_summaries: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []

        self.total_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }

        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key not found (set DEEPSEEK_API_KEY)")
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def close(self) -> None:
        try:
            close_method = getattr(self.client, "close", None)
            if callable(close_method):
                close_method()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to close DeepSeek client: {exc}")

    def __enter__(self) -> DeepSeekPlayer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.close()

    def get_tools(self) -> list[dict[str, Any]]:
        """Define available tools in Chat Completions format."""
        return [
            self._build_tool(
                "check_morning_prices",
                "Check today's supply costs for all items",
                CheckMorningPricesParams,
            ),
            self._build_tool(
                "check_inventory",
                "View current inventory levels and expiration dates",
                CheckInventoryParams,
            ),
            self._build_tool(
                "order_supplies",
                "Purchase supplies (delivered instantly)",
                OrderSuppliesParams,
            ),
            self._build_tool(
                "set_operating_hours",
                "Set today's operating hours",
                SetOperatingHoursParams,
            ),
            self._build_tool(
                "set_price",
                "Set the price for a lemonade",
                SetPriceParams,
            ),
            self._build_tool(
                "get_historical_supply_costs",
                "Analyze supply price trends",
                GetHistoricalSupplyCostsParams,
            ),
            self._build_tool(
                "purchase_automation",
                (
                    "One-time purchase that eliminates the labor portion of "
                    "the hourly operating cost for the rest of the game"
                ),
                PurchaseAutomationParams,
            ),
            self._build_tool(
                "open_for_business",
                "Open the stand for business (must set price and hours first)",
                OpenForBusinessParams,
            ),
        ]

    def _build_tool(
        self,
        name: str,
        description: str,
        params_model: type[BaseModel],
    ) -> dict[str, Any]:
        schema = params_model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }

    def execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        game: BusinessGame,
    ) -> str:
        try:
            result: Any
            if tool_name == "check_morning_prices":
                result = game.check_morning_prices()
            elif tool_name == "check_inventory":
                result = game.check_inventory()
            elif tool_name == "order_supplies":
                result = game.order_supplies(**args)
            elif tool_name == "set_operating_hours":
                result = game.set_operating_hours(**args)
            elif tool_name == "set_price":
                result = game.set_price(**args)
            elif tool_name == "get_historical_supply_costs":
                result = game.get_historical_supply_costs()
            elif tool_name == "purchase_automation":
                result = game.purchase_automation()
            elif tool_name == "open_for_business":
                result = game.open_for_business()
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)}, default=str)

    def play_turn(
        self,
        game: BusinessGame,
        recorder: GameRecorder | None = None,
    ) -> dict[str, Any]:
        """Play one turn (one in-game day) of the game."""
        # Seed conversation with system instructions on first day.
        if not self.messages:
            self.messages.append(
                {"role": "system", "content": game.get_system_instructions()},
            )

        # Add today's user message.
        self.messages.append(
            {"role": "user", "content": game.get_day_summary(is_first_attempt=True)},
        )

        max_attempts = 10
        attempts = 0
        all_tool_calls_this_turn: list[str] = []

        while attempts < max_attempts:
            attempts += 1
            try:
                logger.info(f"Day {game.current_day}, sub-turn {attempts}")

                kwargs = self._build_request_kwargs()
                start_time = time.time()
                response = self._create_with_retry(**kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                self._update_token_usage(response)
                message = response.choices[0].message

                # Capture reasoning trace before anything else can mutate state.
                self._extract_reasoning_summary(message, game, attempts)

                # Append assistant message verbatim (carries content,
                # reasoning_content, tool_calls — required for tool-call turns).
                self.messages.append(message)

                tool_calls = message.tool_calls or []
                tool_executions: list[dict[str, Any]] = []
                opened_for_business = False

                for tool_call in tool_calls:
                    name = tool_call.function.name
                    args = (
                        json.loads(tool_call.function.arguments)
                        if tool_call.function.arguments
                        else {}
                    )
                    result_str = self.execute_tool(name, args, game)
                    all_tool_calls_this_turn.append(name)

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        },
                    )
                    tool_executions.append(
                        {
                            "tool": name,
                            "arguments": args,
                            "result": json.loads(result_str),
                        },
                    )

                    if name == "open_for_business":
                        result_dict = json.loads(result_str)
                        if result_dict.get("success", False):
                            opened_for_business = True

                if recorder:
                    recorder.record_interaction(
                        attempt=attempts,
                        request=kwargs,
                        response=response,
                        tool_executions=tool_executions,
                        duration_ms=duration_ms,
                        conversation_id=None,
                    )

                if opened_for_business:
                    return {
                        "success": True,
                        "attempts": attempts,
                        "tool_calls": all_tool_calls_this_turn,
                        "opened_for_business": True,
                    }

                if not tool_calls:
                    logger.info(f"Sub-turn {attempts}: no tool calls; ending day")
                    break

            except Exception as e:
                logger.error(f"Error in turn: {e}")
                self.errors.append({"day": game.current_day, "error": str(e)})
                if recorder and hasattr(recorder, "record_error"):
                    recorder.record_error(str(e))

        return {
            "success": False,
            "error": (
                "Max attempts reached. "
                "Did not call open_for_business() to start the day."
            ),
            "attempts": attempts,
            "tool_calls": all_tool_calls_this_turn,
        }

    def _build_request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": self.messages,
            "tools": self.get_tools(),
            "reasoning_effort": self.reasoning_effort,
        }
        if self.thinking_enabled:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        return kwargs

    def _create_with_retry(self, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                attempt += 1
                if attempt > self.api_max_retries:
                    logger.error(
                        f"DeepSeek call failed after {attempt - 1} retries: {e}"
                    )
                    raise
                delay = self.api_backoff * (2 ** (attempt - 1))
                msg = (
                    f"DeepSeek call error: {e}. "
                    f"Retry {attempt}/{self.api_max_retries} "
                    f"in {delay:.1f}s"
                )
                logger.warning(msg)
                time.sleep(delay)

    def _extract_reasoning_summary(
        self,
        message: Any,
        game: BusinessGame,
        attempts: int,
    ) -> None:
        reasoning = getattr(message, "reasoning_content", None)
        if not reasoning:
            return
        self.reasoning_summaries.append(
            {
                "day": game.current_day,
                "attempt": attempts,
                "summary": reasoning,
                "summary_type": "reasoning_content",
                "effort": self.reasoning_effort,
            },
        )
        if game.current_day == 1 and attempts == 1:
            logger.info(f"Captured reasoning summary: {reasoning[:200]}...")

    def _update_token_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        self.total_token_usage["input_tokens"] += prompt_tokens
        self.total_token_usage["output_tokens"] += completion_tokens
        self.total_token_usage["total_tokens"] += total_tokens

        details = getattr(usage, "completion_tokens_details", None)
        if details:
            reasoning = getattr(details, "reasoning_tokens", 0) or 0
            self.total_token_usage["reasoning_tokens"] += reasoning

        # DeepSeek-specific cache fields.
        cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
        if cache_hit is None:
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                cache_hit = getattr(prompt_details, "cached_tokens", 0) or 0
        self.total_token_usage["cached_input_tokens"] += cache_hit or 0

    def calculate_cost(self) -> dict[str, float]:
        pricing = MODEL_PRICING.get(
            self.model_name,
            {"input": 1.0, "cached_input": 0.5, "output": 2.0},
        )

        cached = self.total_token_usage["cached_input_tokens"]
        non_cached_input = self.total_token_usage["input_tokens"] - cached

        input_cost = (non_cached_input / 1_000_000) * pricing["input"]
        cached_cost = (cached / 1_000_000) * pricing["cached_input"]
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
        self.messages = []
        self.reasoning_summaries = []
        self.errors = []
        self.total_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }
