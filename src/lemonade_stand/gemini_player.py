"""Gemini-based player for the Lemonade Stand business game."""

import json
import logging
import os
import time
from typing import Any

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool

from .base_player import BasePlayer
from .business_game import BusinessGame
from .game_recorder import GameRecorder

logger = logging.getLogger(__name__)


class GeminiPlayer(BasePlayer):
    """AI player that uses Google's Gemini API to play the lemonade stand business game."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        api_key: str | None = None,
        *,
        api_max_retries: int = 3,
        api_backoff: float = 1.0,
    ) -> None:
        """Initialize the Gemini AI player.

        Args:
            model_name: Gemini model to use (e.g., 'gemini-2.0-flash-exp', 'gemini-1.5-pro')
            api_key: Gemini API key (uses env var if not provided)
            api_max_retries: Number of times to retry failed API calls
            api_backoff: Initial backoff delay (seconds) for retries
        """
        super().__init__(model_name, api_key)

        self.api_max_retries = api_max_retries
        self.api_backoff = api_backoff

        # For compatibility with OpenAI player
        self.reasoning_summaries: list[dict[str, Any]] = []

        # Model pricing (per 1M tokens)
        self.model_pricing = {
            "gemini-2.0-flash-exp": {
                "input": 0.00,
                "cached_input": 0.00,
                "output": 0.00,
            },  # Free during experimental phase
            "gemini-2.5-flash-lite": {
                "input": 0.10,
                "cached_input": 0.025,
                "output": 0.40,
            },
            "gemini-2.5-flash": {"input": 0.30, "cached_input": 0.075, "output": 2.50},
            "gemini-2.5-pro": {"input": 1.25, "cached_input": 0.3125, "output": 10.00},
            "gemini-1.5-flash": {
                "input": 0.075,
                "cached_input": 0.01875,
                "output": 0.30,
            },
            "gemini-1.5-flash-8b": {
                "input": 0.0375,
                "cached_input": 0.009375,
                "output": 0.15,
            },
            "gemini-1.5-pro": {"input": 1.25, "cached_input": 0.3125, "output": 5.00},
            "gemini-1.0-pro": {"input": 0.50, "cached_input": 0.125, "output": 1.50},
        }

        # Initialize Gemini client
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable"
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def close(self) -> None:
        """Close the underlying Gemini client."""
        # Gemini SDK doesn't require explicit cleanup
        pass

    def _convert_tools_to_gemini(self) -> list[Tool]:
        """Convert tool definitions to Gemini format."""
        function_declarations = []

        for tool in self.get_tools():
            # Convert OpenAI tool format to Gemini FunctionDeclaration
            func_decl = FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"],
            )
            function_declarations.append(func_decl)

        return [Tool(function_declarations=function_declarations)]

    def play_turn(
        self, game: BusinessGame, recorder: GameRecorder | None = None
    ) -> dict[str, Any]:
        """Play one turn of the game using Gemini API.

        Args:
            game: The BusinessGame instance
            recorder: Optional GameRecorder to record all interactions

        Returns:
            Dictionary with success status and attempt information
        """
        prompt = game.get_turn_prompt()
        system_prompt = game._get_system_prompt()

        max_attempts = 10
        attempts = 0
        all_tool_calls_this_turn: list[str] = []

        # Initialize chat with system prompt
        chat = self.model.start_chat(
            history=[],
            enable_automatic_function_calling=False,  # We'll handle function calls manually
        )

        # Combine system prompt with user prompt for first message
        full_prompt = f"{system_prompt}\n\n{prompt}"

        while attempts < max_attempts:
            attempts += 1
            try:
                if attempts <= 2:
                    logger.info(f"Day {game.current_day}, Attempt {attempts}")
                    if attempts > 1:
                        logger.info(
                            f"  Progress: {list(set(all_tool_calls_this_turn))}"
                        )

                # Prepare the request
                tools = self._convert_tools_to_gemini()

                # Time the API call
                start_time = time.time()

                # Send message with tools
                response = self._send_with_retry(
                    chat=chat,
                    message=full_prompt
                    if attempts == 1
                    else "Please continue with the next steps.",
                    tools=tools,
                )

                duration_ms = int((time.time() - start_time) * 1000)

                # Update token usage if available
                if hasattr(response, "usage_metadata"):
                    self._update_token_usage(response.usage_metadata)

                # Process function calls
                tool_calls_made = []
                tool_results = []

                for part in response.parts:
                    if hasattr(part, "function_call"):
                        fc = part.function_call
                        tool_name = fc.name
                        args = dict(fc.args) if fc.args else {}

                        # Execute the tool
                        result = self.execute_tool(tool_name, args, game)
                        tool_calls_made.append(tool_name)
                        all_tool_calls_this_turn.append(tool_name)
                        tool_results.append(
                            {"name": tool_name, "result": result, "args": args}
                        )

                        # Check if we successfully opened for business
                        if tool_name == "open_for_business":
                            result_dict = json.loads(result)
                            if result_dict.get("success", False):
                                logger.info(
                                    "open_for_business succeeded - day complete"
                                )

                                # Record the interaction if recorder is provided
                                if recorder:
                                    self._record_interaction(
                                        recorder,
                                        attempts,
                                        response,
                                        tool_results,
                                        duration_ms,
                                        tools,
                                    )

                                return {
                                    "success": True,
                                    "attempts": attempts,
                                    "tool_calls": all_tool_calls_this_turn,
                                    "opened_for_business": True,
                                }

                        if attempts <= 2:
                            logger.info(
                                f"Executed {tool_name}, result: {result[:100]}..."
                            )

                # Record the interaction if recorder is provided
                if recorder:
                    self._record_interaction(
                        recorder, attempts, response, tool_results, duration_ms, tools
                    )

                # If we made tool calls, send the results back to continue the conversation
                if tool_results:
                    # Format tool results for the next message
                    results_message = "Here are the results of the tool calls:\n\n"
                    for tool_result in tool_results:
                        results_message += f"{tool_result['name']} result:\n{tool_result['result']}\n\n"

                    # Continue the conversation with tool results
                    full_prompt = results_message

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

        return {
            "success": False,
            "error": "Max attempts reached. Did not call open_for_business() to start the day.",
            "attempts": attempts,
            "tool_calls": all_tool_calls_this_turn,
        }

    def _send_with_retry(self, chat, message: str, tools: list[Tool]) -> Any:
        """Send message to Gemini with exponential backoff."""
        # Add delay to respect free tier rate limits
        # Free tier: Flash=10 RPM (6s between), Pro=5 RPM (12s between)
        if "pro" in self.model_name.lower():
            time.sleep(12)  # Pro: 5 RPM on free tier
        else:
            time.sleep(6)  # Flash: 10 RPM on free tier

        attempt = 0
        while True:
            try:
                return chat.send_message(
                    message,
                    tools=tools,
                    tool_config={"function_calling_config": {"mode": "ANY"}},
                )
            except Exception as e:
                error_str = str(e)
                attempt += 1

                # Check if it's a rate limit error and extract retry delay
                if "429" in error_str and "retry_delay" in error_str:
                    # Try to extract the retry delay from the error message
                    import re

                    match = re.search(r"seconds:\s*(\d+)", error_str)
                    if match:
                        delay = int(match.group(1)) + 2  # Add 2 seconds buffer
                        logger.warning(
                            f"Rate limited. Waiting {delay} seconds before retry..."
                        )
                        time.sleep(delay)
                        continue

                if attempt > self.api_max_retries:
                    logger.error(f"Gemini call failed after {attempt - 1} retries: {e}")
                    raise

                delay = self.api_backoff * (2 ** (attempt - 1))
                logger.warning(
                    f"Gemini call error: {e}. Retry {attempt}/{self.api_max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)

    def _update_token_usage(self, usage_metadata: Any) -> None:
        """Update token usage from Gemini response."""
        if not usage_metadata:
            return

        # Gemini uses different field names
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0)
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0)
        total_tokens = getattr(usage_metadata, "total_token_count", 0)
        cached_tokens = getattr(usage_metadata, "cached_content_token_count", 0)

        self.total_token_usage["input_tokens"] += input_tokens
        self.total_token_usage["output_tokens"] += output_tokens
        self.total_token_usage["total_tokens"] += total_tokens
        self.total_token_usage["cached_input_tokens"] += cached_tokens

    def _record_interaction(
        self,
        recorder: GameRecorder,
        attempts: int,
        response: Any,
        tool_results: list[dict[str, Any]],
        duration_ms: int,
        tools: list[Tool],
    ) -> None:
        """Record the interaction with the recorder."""
        # Build tool executions list
        tool_executions = []
        for tool_result in tool_results:
            tool_executions.append(
                {
                    "tool": tool_result["name"],
                    "arguments": tool_result["args"],
                    "result": json.loads(tool_result["result"]),
                }
            )

        # Convert response to a format the recorder can handle
        response_text = ""
        import contextlib

        with contextlib.suppress(Exception):
            response_text = response.text if hasattr(response, "text") else ""

        response_dict = {
            "model": self.model_name,
            "text": response_text,
            "parts": [str(part) for part in response.parts]
            if hasattr(response, "parts")
            else [],
            "usage_metadata": {
                "prompt_token_count": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ),
                "candidates_token_count": getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ),
                "total_token_count": getattr(
                    response.usage_metadata, "total_token_count", 0
                ),
                "cached_content_token_count": getattr(
                    response.usage_metadata, "cached_content_token_count", 0
                ),
            }
            if hasattr(response, "usage_metadata")
            else {},
        }

        # Convert tools to dict format for recording
        tools_dict = []
        if tools:
            for tool in tools:
                if hasattr(tool, "function_declarations"):
                    for func in tool.function_declarations:
                        tools_dict.append(
                            {
                                "name": func.name,
                                "description": func.description,
                                "parameters": func.parameters,
                            }
                        )

        request_dict = {
            "model": self.model_name,
            "tools": tools_dict,
        }

        recorder.record_interaction(
            attempt=attempts,
            request=request_dict,
            response=response_dict,
            tool_executions=tool_executions,
            duration_ms=duration_ms,
        )

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
        """Reset the player for a new game."""
        self.errors = []
        self.total_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }
