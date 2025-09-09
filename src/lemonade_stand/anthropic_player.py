"""Anthropic-based player (Claude) for the Lemonade Stand business game.

This is a scaffold to add full support for Anthropic's Messages API with
tool/function calling. It mirrors the structure of other players while
keeping imports lazy so tests don't require the SDK.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any

from .base_player import BasePlayer
from .business_game import BusinessGame
from .game_recorder import GameRecorder

logger = logging.getLogger(__name__)


class AnthropicPlayer(BasePlayer):
    """AI player using Anthropic's Claude models (e.g., claude-3.5-sonnet)."""

    def __init__(
        self,
        model_name: str = "claude-3.5-sonnet",
        api_key: str | None = None,
        *,
        api_max_retries: int = 3,
        api_backoff: float = 1.0,
    ) -> None:
        super().__init__(model_name, api_key)

        self.api_max_retries = api_max_retries
        self.api_backoff = api_backoff

        # For compatibility with OpenAI player
        self.reasoning_summaries: list[dict[str, Any]] = []

        # Rate limit tracking
        self.rate_limit_tier = 1  # Default to Tier 1
        self.token_usage_history = deque(maxlen=60)  # Track last 60 seconds
        self.request_history = deque(maxlen=60)  # Track request timestamps

        # Rate limits by tier (for Claude Haiku 3)
        self.rate_limits = {
            1: {"rpm": 50, "itpm": 50_000, "otpm": 10_000},
            2: {"rpm": 1_000, "itpm": 100_000, "otpm": 20_000},
            3: {"rpm": 2_000, "itpm": 200_000, "otpm": 40_000},
            4: {"rpm": 4_000, "itpm": 400_000, "otpm": 80_000},
        }

        # Official Anthropic pricing (per 1M tokens)
        self.model_pricing = {
            # Claude Opus 4.1
            "claude-opus-4.1": {"input": 15.00, "cached_input": 1.50, "output": 75.00},
            # Claude Sonnet 4 (using ≤200K pricing as default since game prompts are small)
            "claude-4-sonnet": {"input": 3.00, "cached_input": 0.30, "output": 15.00},
            "claude-sonnet-4-20250514": {
                "input": 3.00,
                "cached_input": 0.30,
                "output": 15.00,
            },
            "claude-3.5-sonnet": {"input": 3.00, "cached_input": 0.30, "output": 15.00},
            "claude-3.5-sonnet-20241022": {
                "input": 3.00,
                "cached_input": 0.30,
                "output": 15.00,
            },
            # Claude Haiku 3.5
            "claude-3.5-haiku": {"input": 0.80, "cached_input": 0.08, "output": 4.00},
            "claude-3.5-haiku-20241022": {
                "input": 0.80,
                "cached_input": 0.08,
                "output": 4.00,
            },
            # Legacy Haiku 3.0 (older pricing)
            "claude-3-haiku-20240307": {
                "input": 0.25,
                "cached_input": 0.03,
                "output": 1.25,
            },
        }

        # Model max OUTPUT token limits (based on official documentation)
        self.model_max_tokens = {
            # Opus models - up to 32,000 tokens
            "claude-opus-4.1": 32000,
            "claude-3-opus": 32000,
            "claude-3-opus-20240229": 32000,
            # Sonnet models - up to 64,000 tokens
            "claude-4-sonnet": 64000,
            "claude-sonnet-4-20250514": 64000,
            "claude-3.5-sonnet": 64000,
            "claude-3.5-sonnet-20241022": 64000,
            "claude-3-sonnet-20240229": 64000,
            # Haiku models - newer versions support 8,192, legacy is 4,096
            "claude-3.5-haiku": 8192,
            "claude-3.5-haiku-20241022": 8192,
            "claude-3-haiku-20240307": 4096,  # Legacy model has lower limit
            # Legacy models
            "claude-2.1": 8192,
            "claude-2.0": 8192,
            # For Bedrock, we might want to cap at 4096 due to burndown throttling
            # but for direct API usage, we can use the full limits
        }

        # Lazy import to avoid test-time dependency
        self._client = None
        self._Anthropic = None

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            # Defer raising until used; allows constructing in environments without keys
            logger.warning("ANTHROPIC_API_KEY not set; player will raise on use")
        else:
            try:
                from anthropic import Anthropic  # type: ignore

                self._Anthropic = Anthropic
                self._client = Anthropic(api_key=key)
            except ModuleNotFoundError:  # pragma: no cover
                logger.warning(
                    "anthropic package not installed; install `anthropic` to use AnthropicPlayer"
                )
                self._client = None

    def close(self) -> None:
        # Anthropic client doesn't require explicit close
        pass

    def _check_rate_limits(self, estimated_tokens: int) -> float:
        """Check if we're approaching rate limits and return delay if needed.

        Args:
            estimated_tokens: Estimated tokens for the next request

        Returns:
            Seconds to wait before making the request
        """
        current_time = time.time()
        limits = self.rate_limits[self.rate_limit_tier]

        # Clean up old history (older than 60 seconds)
        while (
            self.token_usage_history
            and self.token_usage_history[0][0] < current_time - 60
        ):
            self.token_usage_history.popleft()
        while self.request_history and self.request_history[0] < current_time - 60:
            self.request_history.popleft()

        # Calculate current usage
        recent_tokens = sum(tokens for _, tokens in self.token_usage_history)
        recent_requests = len(self.request_history)

        # Check if adding this request would exceed limits
        if (
            recent_tokens + estimated_tokens > limits["itpm"] * 0.9
        ):  # Stay at 90% of limit
            # Calculate how long to wait
            wait_time = 60 - (current_time - self.token_usage_history[0][0]) + 1
            logger.info(
                f"Approaching token limit ({recent_tokens + estimated_tokens}/{limits['itpm']}), waiting {wait_time:.1f}s"
            )
            return wait_time

        if recent_requests >= limits["rpm"] * 0.9:  # Stay at 90% of limit
            wait_time = 60 - (current_time - self.request_history[0]) + 1
            logger.info(
                f"Approaching request limit ({recent_requests}/{limits['rpm']}), waiting {wait_time:.1f}s"
            )
            return wait_time

        # Add small delay to smooth out bursts (prevent acceleration limits)
        if recent_tokens > limits["itpm"] * 0.5:  # If over 50% capacity, slow down
            return 0.5  # Half second delay

        return 0  # No delay needed

    def _record_usage(self, input_tokens: int, _output_tokens: int) -> None:
        """Record token and request usage for rate limiting.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
        """
        current_time = time.time()
        self.token_usage_history.append((current_time, input_tokens))
        self.request_history.append(current_time)

    def _tools_for_anthropic(self) -> list[dict[str, Any]]:
        """Convert BasePlayer tools to Anthropic's tool schema."""
        tools = []
        for tool in self.get_tools():
            tools.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool["parameters"],
                }
            )
        return tools

    def play_turn(
        self, game: BusinessGame, recorder: GameRecorder | None = None
    ) -> dict[str, Any]:
        """Play one turn using Anthropic Messages API with tool use."""
        # Ensure client available
        if self._client is None:
            raise RuntimeError(
                "Anthropic client not initialized. Install `anthropic` and set ANTHROPIC_API_KEY."
            )

        # Build prompts and tools
        system_prompt = game._get_system_prompt()
        user_prompt = game.get_turn_prompt()
        tools = self._tools_for_anthropic()

        # Conversation per Anthropic: list of messages with content blocks
        conversation: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]

        max_attempts = 10
        attempts = 0
        all_tool_calls_this_turn: list[str] = []

        while attempts < max_attempts:
            attempts += 1
            try:
                # Estimate tokens for rate limiting (rough estimate)
                estimated_tokens = len(str(conversation)) // 4 + len(system_prompt) // 4

                # Check rate limits and wait if necessary
                wait_time = self._check_rate_limits(estimated_tokens)
                if wait_time > 0:
                    time.sleep(wait_time)

                # Create message with model-specific max_tokens
                # For the lemonade game, we could optimize by using ~4096 tokens since
                # responses are typically 500-1000 tokens, but using the model's full
                # capacity ensures we never truncate and costs nothing extra
                max_tokens = self.model_max_tokens.get(self.model_name, 8192)

                start_time = time.time()

                # Claude 4 models require streaming for long operations
                if (
                    "claude-4" in self.model_name
                    or "claude-sonnet-4" in self.model_name
                ):
                    # Use streaming for Claude 4 models
                    with self._client.messages.stream(
                        model=self.model_name,
                        system=system_prompt,
                        messages=conversation,
                        tools=tools,
                        max_tokens=max_tokens,
                    ) as stream:
                        response = stream.get_final_message()
                else:
                    # Regular non-streaming for other models
                    response = self._client.messages.create(
                        model=self.model_name,
                        system=system_prompt,
                        messages=conversation,
                        tools=tools,
                        max_tokens=max_tokens,
                    )
                (time.time() - start_time) * 1000  # ms

                # Update token usage if present
                usage = getattr(response, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
                    self.total_token_usage["input_tokens"] += input_tokens
                    self.total_token_usage["output_tokens"] += output_tokens
                    total = input_tokens + output_tokens
                    self.total_token_usage["total_tokens"] += total

                    # Record for rate limiting
                    self._record_usage(input_tokens, output_tokens)

                # Process response content blocks
                tool_results_blocks: list[dict[str, Any]] = []
                tool_calls_made: list[str] = []
                tool_results_for_record: list[dict[str, Any]] = []
                opened_successfully = False

                for block in getattr(response, "content", []) or []:
                    btype = getattr(block, "type", None)
                    if btype == "tool_use":
                        tool_name = getattr(block, "name", "")
                        tool_input = getattr(block, "input", {}) or {}
                        tool_id = getattr(block, "id", None)

                        result_json_str = self.execute_tool(
                            tool_name, dict(tool_input), game
                        )
                        tool_calls_made.append(tool_name)
                        all_tool_calls_this_turn.append(tool_name)

                        # Record execution for recorder
                        try:
                            import json as _json

                            parsed_res = _json.loads(result_json_str)
                        except Exception:
                            parsed_res = {"raw": result_json_str}
                        tool_results_for_record.append(
                            {
                                "tool": tool_name,
                                "arguments": dict(tool_input),
                                "result": parsed_res,
                            }
                        )

                        # Detect success
                        if (
                            tool_name == "open_for_business"
                            and isinstance(parsed_res, dict)
                            and parsed_res.get("success")
                        ):
                            opened_successfully = True

                        # Build tool_result block to feed back
                        # Anthropic expects content as text or array; send JSON string
                        tool_results_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result_json_str,
                            }
                        )

                # Recorder integration
                if recorder:
                    # Convert request/response to dict-ish for recording
                    request_dict = {
                        "model": self.model_name,
                        "system": system_prompt,
                        "messages": conversation,
                        "tools": tools,
                    }
                    resp_dict = {
                        "model": getattr(response, "model", self.model_name),
                        "content": [
                            {
                                "type": getattr(b, "type", None),
                                "text": getattr(b, "text", None),
                                "name": getattr(b, "name", None),
                                "input": getattr(b, "input", None),
                                "id": getattr(b, "id", None),
                            }
                            for b in (getattr(response, "content", []) or [])
                        ],
                        "usage": {
                            "input_tokens": getattr(usage, "input_tokens", 0)
                            if usage
                            else 0,
                            "output_tokens": getattr(usage, "output_tokens", 0)
                            if usage
                            else 0,
                            "total_tokens": (
                                (getattr(usage, "input_tokens", 0) or 0)
                                + (getattr(usage, "output_tokens", 0) or 0)
                                if usage
                                else 0
                            ),
                        },
                    }
                    recorder.record_interaction(
                        attempt=attempts,
                        request=request_dict,
                        response=resp_dict,
                        tool_executions=tool_results_for_record,
                        duration_ms=0,
                    )

                if opened_successfully:
                    return {
                        "success": True,
                        "attempts": attempts,
                        "tool_calls": all_tool_calls_this_turn,
                        "opened_for_business": True,
                    }

                # If we executed tools, we need to append the assistant's message first
                # then send tool results as a user message
                if tool_results_blocks:
                    # Add the assistant's message with tool use blocks
                    assistant_content = []
                    for block in getattr(response, "content", []) or []:
                        if getattr(block, "type", None) == "tool_use":
                            assistant_content.append(
                                {
                                    "type": "tool_use",
                                    "id": getattr(block, "id", None),
                                    "name": getattr(block, "name", ""),
                                    "input": getattr(block, "input", {}),
                                }
                            )
                        elif getattr(block, "type", None) == "text":
                            assistant_content.append(
                                {"type": "text", "text": getattr(block, "text", "")}
                            )

                    if assistant_content:
                        conversation.append(
                            {"role": "assistant", "content": assistant_content}
                        )

                    # Now add tool results as user message
                    conversation.append(
                        {"role": "user", "content": tool_results_blocks}
                    )
                    continue

                # If no tool calls were made, gently nudge
                if not tool_calls_made:
                    conversation.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please proceed with the remaining steps and call open_for_business() when ready.",
                                }
                            ],
                        }
                    )

            except Exception as e:
                error_str = str(e)
                logger.error(f"Anthropic play_turn error: {error_str}")

                # Handle rate limit errors specially
                if "rate_limit_error" in error_str or "429" in error_str:
                    # Extract retry-after if available
                    if "retry_after" in error_str:
                        try:
                            # Try to extract the retry-after value
                            import re

                            match = re.search(
                                r'retry[_-]after["\s:]+(\d+)', error_str.lower()
                            )
                            if match:
                                retry_after = int(match.group(1))
                                logger.info(
                                    f"Rate limit hit, waiting {retry_after} seconds..."
                                )
                                time.sleep(retry_after + 1)  # Add 1 second buffer
                            else:
                                # Default exponential backoff for rate limits
                                wait_time = min(
                                    60, self.api_backoff * (2 ** (attempts - 1))
                                )
                                logger.info(
                                    f"Rate limit hit, waiting {wait_time} seconds..."
                                )
                                time.sleep(wait_time)
                        except Exception:
                            # Fallback to exponential backoff
                            wait_time = min(
                                60, self.api_backoff * (2 ** (attempts - 1))
                            )
                            logger.info(
                                f"Rate limit hit, waiting {wait_time} seconds..."
                            )
                            time.sleep(wait_time)
                    else:
                        # Default exponential backoff for rate limits
                        wait_time = min(60, self.api_backoff * (2 ** (attempts - 1)))
                        logger.info(f"Rate limit hit, waiting {wait_time} seconds...")
                        time.sleep(wait_time)

                    # Don't count rate limit errors as real errors
                    if attempts < max_attempts:
                        continue

                self.errors.append({"day": game.current_day, "error": error_str})
                # Continue to next attempt

        return {
            "success": False,
            "error": "Max attempts reached. Did not call open_for_business() to start the day.",
            "attempts": attempts,
            "tool_calls": all_tool_calls_this_turn,
        }

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
        self.reasoning_summaries = []
        self.total_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }
        # Clear rate limit tracking
        self.token_usage_history.clear()
        self.request_history.clear()
