"""OpenAI-based player for the Lemonade Stand business game."""

import json
import logging
import os
import time
from typing import Any

from openai import OpenAI

from .business_game import BusinessGame
from .game_recorder import GameRecorder

logger = logging.getLogger(__name__)

class OpenAIPlayer:
    """AI player that uses OpenAI's API to play the lemonade stand business game."""

    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        api_key: str | None = None,
        include_reasoning_summary: bool = True,
    ) -> None:
        """Initialize the AI player.

        Args:
            model_name: OpenAI model to use
            api_key: OpenAI API key (uses env var if not provided)
            include_reasoning_summary: Whether to request reasoning summaries for o* models
        """
        self.model_name = model_name
        self.include_reasoning_summary = include_reasoning_summary

        # For stateless approach - minimal tracking
        self.reasoning_summaries: list[dict[str, Any]] = []

        # Token tracking
        self.total_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }

        # Cost tracking
        self.model_pricing = {
            "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
            "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
            "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
            "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
            "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
        }

        # Check if this is a reasoning model
        self.is_reasoning_model = model_name.startswith(("o1", "o3", "o4"))

        # Initialize OpenAI client (synchronous)
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")
        self.client = OpenAI(api_key=api_key)

    def close(self) -> None:
        """Close the underlying OpenAI client."""
        try:
            close_method = getattr(self.client, "close", None)
            if callable(close_method):
                close_method()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to close OpenAI client: {exc}")

    # Support use as a context manager
    def __enter__(self) -> "OpenAIPlayer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.close()

        # Track errors
        self.errors: list[dict[str, Any]] = []

    def get_tools(self) -> list[dict[str, Any]]:
        """Define available tools for the AI."""
        return [
            self._tool_check_morning_prices(),
            self._tool_check_inventory(),
            self._tool_order_supplies(),
            self._tool_set_operating_hours(),
            self._tool_set_price(),
            self._tool_get_historical_supply_costs(),
            self._tool_open_for_business(),
        ]

    def _tool_check_morning_prices(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "check_morning_prices",
            "description": "Check today's supply costs for all items",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def _tool_check_inventory(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "check_inventory",
            "description": "View current inventory levels and expiration dates",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def _tool_order_supplies(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "order_supplies",
            "description": "Purchase supplies (delivered instantly)",
            "parameters": {
                "type": "object",
                "properties": {
                    "cups": {
                        "type": "integer",
                        "description": "Number of cups to order",
                        "minimum": 0,
                    },
                    "lemons": {
                        "type": "integer",
                        "description": "Number of lemons to order",
                        "minimum": 0,
                    },
                    "sugar": {
                        "type": "integer",
                        "description": "Amount of sugar to order",
                        "minimum": 0,
                    },
                    "water": {
                        "type": "integer",
                        "description": "Amount of water to order",
                        "minimum": 0,
                    },
                },
                "required": ["cups", "lemons", "sugar", "water"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def _tool_set_operating_hours(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "set_operating_hours",
            "description": "Set today's operating hours",
            "parameters": {
                "type": "object",
                "properties": {
                    "open_hour": {
                        "type": "integer",
                        "description": "Opening hour (0-23)",
                        "minimum": 0,
                        "maximum": 23,
                    },
                    "close_hour": {
                        "type": "integer",
                        "description": "Closing hour (1-24, must be > open_hour)",
                        "minimum": 1,
                        "maximum": 24,
                    },
                },
                "required": ["open_hour", "close_hour"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def _tool_set_price(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "set_price",
            "description": "Set the price for a lemonade",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "Price per lemonade",
                        "minimum": 0,
                    }
                },
                "required": ["price"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def _tool_get_historical_supply_costs(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "get_historical_supply_costs",
            "description": "Analyze supply price trends",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def _tool_open_for_business(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": "open_for_business",
            "description": "Open the stand for business (must set price and hours first)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def execute_tool(
        self, tool_name: str, args: dict[str, Any], game: BusinessGame
    ) -> str:
        """Execute a tool with given arguments.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool
            game: The game instance

        Returns:
            JSON string with the result
        """
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
            elif tool_name == "open_for_business":
                result = game.open_for_business()
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def play_turn(
        self, game: BusinessGame, recorder: GameRecorder | None = None
    ) -> dict[str, Any]:
        """Play one turn of the game using OpenAI Responses API (stateless).

        Args:
            game: The BusinessGame instance
            recorder: Optional GameRecorder to record all interactions

        Returns:
            Dictionary with success status and attempt information
        """
        prompt = game.get_turn_prompt()
        max_attempts = 10
        attempts = 0
        all_tool_calls_this_turn: list[str] = []
        conversation = [{"role": "user", "content": prompt}]

        while attempts < max_attempts:
            attempts += 1
            try:
                if attempts <= 2:
                    logger.info(f"Day {game.current_day}, Attempt {attempts}")
                    if attempts > 1:
                        logger.info(
                            f"  Progress: {list(set(all_tool_calls_this_turn))}"
                        )

                # Build request
                kwargs = self._build_request_kwargs(conversation, game)

                # Time the API call
                start_time = time.time()
                response = self.client.responses.create(**kwargs)
                duration_ms = int((time.time() - start_time) * 1000)

                # Extract data from response
                self._extract_reasoning_summary(response, game, attempts)
                self._update_token_usage(response)

                # Process the response output
                (
                    tool_calls_made,
                    tool_results,
                    assistant_message,
                    success,
                ) = self._process_output(
                    response, game, attempts, all_tool_calls_this_turn
                )

                # Record the interaction if recorder is provided
                if recorder:
                    # Build list of tool executions
                    tool_executions = []
                    for tool_result in tool_results:
                        tool_executions.append({
                            "tool": tool_result["name"],
                            "arguments": self._get_tool_args_from_response(response, tool_result["name"]),
                            "result": json.loads(tool_result["result"]),
                        })

                    recorder.record_interaction(
                        attempt=attempts,
                        request=kwargs,
                        response=response,
                        tool_executions=tool_executions,
                        duration_ms=duration_ms,
                    )

                if success is not None:
                    return success

                if tool_results:
                    self._append_tool_results_to_conversation(
                        tool_results, conversation
                    )

                if not tool_calls_made:
                    logger.info(f"Attempt {attempts}: No tool calls made")
            except Exception as e:
                logger.error(f"Error in turn: {e}")
                self.errors.append({"day": game.current_day, "error": str(e)})
                if attempts < max_attempts:
                    logger.warning(f"Error on attempt {attempts}, will retry")

                # Record the error if recorder is provided
                if recorder and hasattr(recorder, "record_error"):
                    recorder.record_error(str(e))

        return self._max_attempts_response(attempts, all_tool_calls_this_turn)

    def _get_tool_args_from_response(self, response: Any, tool_name: str) -> dict[str, Any]:
        """Extract tool arguments from response for a specific tool call."""
        for item in response.output:
            if item.type == "function_call" and item.name == tool_name:
                return json.loads(item.arguments) if item.arguments else {}
        return {}

    def _max_attempts_response(
        self, attempts: int, all_tool_calls: list[str]
    ) -> dict[str, Any]:
        return {
            "success": False,
            "error": "Max attempts reached. Did not call open_for_business() to start the day.",
            "attempts": attempts,
            "tool_calls": all_tool_calls,
        }

    def _build_request_kwargs(
        self, conversation: list[dict[str, Any]], game: BusinessGame
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": conversation,
            "tools": self.get_tools(),
            "instructions": game._get_system_prompt(),
        }
        if self.is_reasoning_model:
            kwargs["reasoning"] = {"effort": "medium"}
            kwargs["max_output_tokens"] = 25000
            if self.include_reasoning_summary:
                kwargs["reasoning"]["summary"] = "auto"
        return kwargs

    def _extract_reasoning_summary(
        self, response: Any, game: BusinessGame, attempts: int
    ) -> None:
        if not (self.is_reasoning_model and hasattr(response, "output")):
            return
        for item in response.output:
            if (
                hasattr(item, "type")
                and item.type == "reasoning"
                and hasattr(item, "summary")
                and item.summary
            ):
                reasoning_text = None
                if (
                    isinstance(item.summary, list)
                    and len(item.summary) > 0
                    and hasattr(item.summary[0], "text")
                ):
                    reasoning_text = item.summary[0].text
                self.reasoning_summaries.append(
                    {
                        "day": game.current_day,
                        "attempt": attempts,
                        "summary": reasoning_text,
                        "summary_type": response.reasoning.summary
                        if hasattr(response, "reasoning")
                        else None,
                        "effort": response.reasoning.effort
                        if hasattr(response, "reasoning")
                        else None,
                    }
                )
                if game.current_day == 1 and attempts == 1:
                    logger.info(
                        f"Captured reasoning summary: {reasoning_text[:200]}..."
                        if reasoning_text
                        else "No reasoning text"
                    )

    def _update_token_usage(self, response: Any) -> None:
        if not hasattr(response, "usage"):
            return
        usage = response.usage

        # The Responses API uses different field names than Chat Completions API
        # Try both field names to support both APIs
        input_tokens = getattr(usage, "input_tokens", 0) or getattr(
            usage, "prompt_tokens", 0
        )
        output_tokens = getattr(usage, "output_tokens", 0) or getattr(
            usage, "completion_tokens", 0
        )
        total_tokens = getattr(usage, "total_tokens", 0)

        self.total_token_usage["input_tokens"] += input_tokens
        self.total_token_usage["output_tokens"] += output_tokens
        self.total_token_usage["total_tokens"] += total_tokens

        # Handle token details (different field names in different APIs)
        if hasattr(usage, "input_tokens_details") or hasattr(
            usage, "prompt_tokens_details"
        ):
            details: Any = getattr(usage, "input_tokens_details", None) or getattr(
                usage, "prompt_tokens_details", None
            )
            if details:
                cached = getattr(details, "cached_tokens", 0)
                self.total_token_usage["cached_input_tokens"] += cached

        if hasattr(usage, "output_tokens_details") or hasattr(usage, "completion_tokens_details"):
            details = getattr(usage, "output_tokens_details", None) or getattr(usage, "completion_tokens_details", None)
            if details:
                reasoning = getattr(details, "reasoning_tokens", 0)
                self.total_token_usage["reasoning_tokens"] += reasoning

    def _process_output(
        self,
        response: Any,
        game: BusinessGame,
        attempts: int,
        all_tool_calls_this_turn: list[str],
    ) -> tuple[
        list[str], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None
    ]:
        tool_calls_made: list[str] = []
        tool_results: list[dict[str, Any]] = []
        assistant_message: dict[str, Any] | None = None

        if attempts <= 4:
            logger.info(f"Response has {len(response.output)} output items")

        for item in response.output:
            if item.type == "function_call":
                args = json.loads(item.arguments) if item.arguments else {}
                result = self.execute_tool(item.name, args, game)
                tool_calls_made.append(item.name)
                all_tool_calls_this_turn.append(item.name)
                tool_results.append(
                    {"name": item.name, "result": result, "id": item.id}
                )
                if item.name == "open_for_business":
                    result_dict = json.loads(result)
                    if result_dict.get("success", False):
                        logger.info("open_for_business succeeded - day complete")
                        return (
                            tool_calls_made,
                            tool_results,
                            assistant_message,
                            {
                                "success": True,
                                "attempts": attempts,
                                "tool_calls": all_tool_calls_this_turn,
                                "opened_for_business": True,
                            },
                        )
                if attempts <= 2:
                    logger.info(f"Executed {item.name}, result: {result[:100]}...")
            elif item.type == "text":
                if not assistant_message:
                    assistant_message = {"role": "assistant", "content": item.text}
                elif isinstance(assistant_message["content"], str):
                    assistant_message["content"] += "\n" + item.text
                else:
                    assistant_message["content"].append(
                        {"type": "text", "text": item.text}
                    )

        return tool_calls_made, tool_results, assistant_message, None

    def _append_tool_results_to_conversation(
        self, tool_results: list[dict[str, Any]], conversation: list[dict[str, Any]]
    ) -> None:
        results_message = "Here are the results of the tool calls:\n\n"
        for tool_result in tool_results:
            results_message += (
                f"{tool_result['name']} result:\n{tool_result['result']}\n\n"
            )
        results_message += "Please continue with the next steps."
        conversation.append({"role": "user", "content": results_message})

    def calculate_cost(self) -> dict[str, float]:
        """Calculate the total cost of API usage.

        Returns:
            Cost breakdown and total
        """
        pricing = self.model_pricing.get(
            self.model_name, {"input": 1.0, "cached_input": 0.5, "output": 2.0}
        )

        # Calculate costs (pricing is per 1M tokens)
        non_cached_input = (
            self.total_token_usage["input_tokens"]
            - self.total_token_usage["cached_input_tokens"]
        )

        input_cost = (non_cached_input / 1_000_000) * pricing["input"]
        cached_cost = (
            self.total_token_usage["cached_input_tokens"] / 1_000_000
        ) * pricing["cached_input"]

        # For reasoning models, output tokens include reasoning tokens
        # The API charges for all output tokens at the output rate
        output_cost = (self.total_token_usage["output_tokens"] / 1_000_000) * pricing[
            "output"
        ]

        # Note: reasoning_tokens are already included in output_tokens for billing
        # So we don't need to add them separately

        return {
            "input_cost": input_cost,
            "cached_cost": cached_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + cached_cost + output_cost,
            "total_tokens": self.total_token_usage["total_tokens"],
        }

    def reset(self) -> None:
        """Reset the player for a new game."""
        self.reasoning_summaries = []
        self.errors = []
        self.total_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }
