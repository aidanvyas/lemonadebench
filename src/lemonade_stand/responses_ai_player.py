"""AI player using the new Responses API for reasoning models."""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .comprehensive_recorder import ComprehensiveRecorder
from .simple_game import SimpleLemonadeGame

load_dotenv()
logger = logging.getLogger(__name__)


class ResponsesAIPlayer:
    """AI player that uses the Responses API for reasoning models."""

    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        include_summary: bool = False,
        include_calculator: bool = False,
    ) -> None:
        """Initialize AI player with Responses API.

        Args:
            model_name: Model to use (gpt-4.1-nano, gpt-4.1, o4-mini, o3, etc.)
            api_key: OpenAI API key
            reasoning_effort: "low", "medium", or "high" (only for reasoning models)
            include_summary: Whether to request reasoning summaries (only for reasoning models)
            include_calculator: Whether to include calculator tool
        """
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.include_summary = include_summary
        self.tool_call_count = 0
        self.tool_call_history = []
        self.calculator_history = []  # Track all calculator expressions
        self.previous_response_id = None  # For conversation continuity
        self.reasoning_summaries = []  # Store reasoning summaries
        self._include_calculator = include_calculator
        self.recorder = None  # Will be set when recording is enabled

        # Store function outputs for next turn
        self.pending_function_outputs = []

        # Track total token usage
        self.total_token_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0,
            'total_tokens': 0,
            'cached_input_tokens': 0  # Track cached tokens separately
        }

        # Check if this is a reasoning model
        self.is_reasoning_model = model_name.startswith(('o1', 'o3', 'o4'))

        # Model pricing (per 1M tokens)
        self.model_pricing = {
            'gpt-4.1-nano': {'input': 0.10, 'cached_input': 0.025, 'output': 0.40},
            'gpt-4.1-mini': {'input': 0.40, 'cached_input': 0.10, 'output': 1.60},
            'gpt-4.1': {'input': 2.00, 'cached_input': 0.50, 'output': 8.00},
            'o3': {'input': 2.00, 'cached_input': 0.50, 'output': 8.00},
            'o4-mini': {'input': 1.10, 'cached_input': 0.275, 'output': 4.40},
        }

        # Get API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")

        self.client = OpenAI(api_key=api_key)

    def enable_recording(self, output_dir: str = "analysis/raw_data"):
        """Enable comprehensive recording of all interactions."""
        self.recorder = ComprehensiveRecorder(output_dir)

    def get_tools(self) -> list[dict[str, Any]]:
        """Define available tools for the AI."""
        tools = [
            {
                "type": "function",
                "name": "get_historical_data",
                "description": "Get historical pricing and profit data for past days",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Number of past days to retrieve (0 for all history)",
                        }
                    },
                    "required": ["days"],
                    "additionalProperties": False
                },
                "strict": True
            },
            {
                "type": "function",
                "name": "set_price",
                "description": "Set the lemonade price for today",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "price": {
                            "type": "number",
                            "description": "The price to set (non-negative, up to 2 decimal places)",
                            "minimum": 0,
                        }
                    },
                    "required": ["price"],
                    "additionalProperties": False
                },
                "strict": True
            },
        ]

        # Optionally add calculator
        if self._include_calculator:
            tools.append({
                "type": "function",
                "name": "calculate",
                "description": "Perform arithmetic calculations to analyze profitability",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate (e.g., '2 * 50', '100 - 25 * 1.5')",
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": False
                },
                "strict": True
            })

        return tools

    def execute_tool(
        self, tool_name: str, arguments: dict, game: SimpleLemonadeGame
    ) -> str:
        """Execute a tool call and return the result."""
        if tool_name == "get_historical_data":
            days = arguments.get("days", 0)
            history = game.history

            if days > 0 and len(history) > days:
                history = history[-days:]

            if not history:
                return "No historical data available yet."

            result = "Historical data:\n"
            for record in history:
                result += f"Day {record['day']}: Price=${record['price']:.2f}, Customers={record['customers']}, Profit=${record['profit']:.2f}\n"
            return result

        elif tool_name == "set_price":
            price = arguments.get("price", 1.00)
            return f"Price will be set to ${price:.2f}"

        elif tool_name == "calculate":
            try:
                expression = arguments.get("expression", "")
                # Track calculator usage
                self.calculator_history.append({
                    'day': game.current_day,
                    'expression': expression
                })
                # Only allow basic math operations
                allowed_chars = "0123456789+-*/()., "
                if all(c in allowed_chars for c in expression):
                    result = eval(expression)
                    return f"Result: {expression} = {result}"
                else:
                    return "Error: Invalid characters in expression"
            except Exception as e:
                return f"Error: {str(e)}"

        return "Unknown tool"

    def get_system_prompt(self, game: SimpleLemonadeGame) -> str:
        """Get system prompt for the game."""
        tools_list = [
            "- get_historical_data(days): Review past pricing and profit data",
            "- set_price(price): Set today's lemonade price",
            "- calculate(expression): Perform arithmetic calculations"
        ]

        tools_text = "\n".join(tools_list)

        base_prompt = (
            "You run a lemonade stand. Your goal is to maximize total profit over 100 days.\n\n"
            f"Available tools:\n{tools_text}\n\n"
            "Each morning, you should set the price for lemonade using the set_price tool.\n"
            "If you don't set a price, it will default to yesterday's price.\n\n"
        )

        # Add prompt variations
        if hasattr(game, '_use_suggested_price') and game._use_suggested_price:
            base_prompt += f"Suggested starting price: ${game.suggested_starting_price:.2f}\n\n"
        elif hasattr(game, '_use_exploration_hint') and game._use_exploration_hint:
            base_prompt += "Try different prices to discover which price maximizes your daily profit.\n\n"

        base_prompt += "Remember: Your goal is to maximize total profit. Think carefully about the relationship between price, demand, and profit."

        return base_prompt

    def get_turn_prompt(self, game: SimpleLemonadeGame) -> str:
        """Get prompt for current turn."""
        state = game.get_state()

        if state["last_result"]:
            last = state["last_result"]
            return (
                f"Day {state['day']}. "
                f"Yesterday's profit: ${last['profit']:.2f}. "
                "Analyze the results and set today's price."
            )
        else:
            return "Day 1. Set your price for the first day."

    def calculate_cost(self) -> dict:
        """Calculate the total cost based on token usage and model pricing."""
        pricing = self.model_pricing.get(self.model_name)
        if not pricing:
            return {"error": f"Pricing not available for model {self.model_name}"}

        # Calculate costs (pricing is per 1M tokens)
        # Non-cached input tokens = total input - cached input
        non_cached_input_tokens = self.total_token_usage['input_tokens'] - self.total_token_usage['cached_input_tokens']
        non_cached_input_cost = (non_cached_input_tokens / 1_000_000) * pricing['input']
        cached_input_cost = (self.total_token_usage['cached_input_tokens'] / 1_000_000) * pricing['cached_input']
        output_cost = (self.total_token_usage['output_tokens'] / 1_000_000) * pricing['output']

        total_cost = non_cached_input_cost + cached_input_cost + output_cost

        return {
            'non_cached_input_cost': non_cached_input_cost,
            'cached_input_cost': cached_input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost,
            'model': self.model_name,
            'non_cached_input_tokens': non_cached_input_tokens,
            'cached_input_tokens': self.total_token_usage['cached_input_tokens'],
            'output_tokens': self.total_token_usage['output_tokens']
        }

    def make_decision(self, game: SimpleLemonadeGame) -> float:
        """Make a pricing decision using the Responses API."""
        # Build input messages
        input_messages = []

        # If we have pending function outputs from previous turn, add them first
        if self.pending_function_outputs:
            input_messages.extend(self.pending_function_outputs)
            self.pending_function_outputs = []  # Clear after use

        # Add current turn's prompt
        input_messages.append({
            "role": "user",
            "content": self.get_turn_prompt(game)
        })

        # Track tools used this turn
        day_tools = []

        try:
            # Build request parameters
            kwargs = {
                "model": self.model_name,
                "input": input_messages,
                "tools": self.get_tools(),
                "instructions": self.get_system_prompt(game),  # Use instructions parameter
            }

            # Add reasoning parameters only for reasoning models
            if self.is_reasoning_model:
                kwargs["reasoning"] = {"effort": self.reasoning_effort or "medium"}
                kwargs["max_output_tokens"] = 25000  # Reserve space for reasoning

                # Add reasoning summary if requested
                if self.include_summary:
                    kwargs["reasoning"]["summary"] = "auto"

            # Use previous response ID for conversation continuity
            if self.previous_response_id:
                kwargs["previous_response_id"] = self.previous_response_id

            # Record request if enabled
            if self.recorder:
                self.recorder.record_request(game.current_day, kwargs)

            response = self.client.responses.create(**kwargs)

            # Record response if enabled
            if self.recorder:
                self.recorder.record_response(game.current_day, response)

            # Store response ID for next turn
            self.previous_response_id = response.id

            # Extract reasoning summary if available
            if hasattr(response, 'reasoning') and hasattr(response.reasoning, 'summary'):
                self.reasoning_summaries.append({
                    "day": game.current_day,
                    "summary": response.reasoning.summary
                })
                logger.info(f"Reasoning summary: {response.reasoning.summary}")

            # Process the output items
            price = None  # Must be set explicitly
            price_was_set = False
            function_calls = []

            for item in response.output:
                if item.type == "function_call":
                    self.tool_call_count += 1
                    function_name = item.name
                    function_args = json.loads(item.arguments)
                    day_tools.append(function_name)

                    # Execute the tool
                    result = self.execute_tool(function_name, function_args, game)
                    logger.debug(f"Tool {function_name} result: {result}")

                    # Record tool execution if enabled
                    if self.recorder:
                        self.recorder.record_tool_execution(
                            game.current_day, function_name, function_args, result
                        )

                    # Store function call for response
                    function_calls.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": str(result)
                    })

                    # Capture price if set_price was called
                    if function_name == "set_price":
                        price = float(function_args.get("price", 1.00))
                        price_was_set = True

            # Record tool usage for this day
            self.tool_call_history.append({
                "day": game.current_day,
                "tools": day_tools
            })

            # Store function outputs for next turn
            self.pending_function_outputs = function_calls

            # Log usage details and accumulate token counts
            if hasattr(response, 'usage'):
                usage = response.usage
                # Extract token counts
                input_tokens = getattr(usage, 'input_tokens', 0)
                output_tokens = getattr(usage, 'output_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)

                # Check for cached tokens in prompt_tokens_details
                cached_input_tokens = 0
                if hasattr(usage, 'prompt_tokens_details'):
                    cached_input_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
                    # Debug logging
                    if game.current_day <= 3:
                        logger.info(f"Day {game.current_day} usage details: {usage}")
                        if hasattr(usage, 'prompt_tokens'):
                            logger.info(f"  prompt_tokens: {getattr(usage, 'prompt_tokens', 'N/A')}")
                        if hasattr(usage, 'prompt_tokens_details'):
                            details = usage.prompt_tokens_details
                            logger.info(f"  prompt_tokens_details: {details}")

                reasoning_tokens = 0
                if hasattr(usage, 'output_tokens_details'):
                    reasoning_tokens = getattr(usage.output_tokens_details, 'reasoning_tokens', 0)

                # Update totals
                self.total_token_usage['input_tokens'] += input_tokens
                self.total_token_usage['output_tokens'] += output_tokens
                self.total_token_usage['reasoning_tokens'] += reasoning_tokens
                self.total_token_usage['total_tokens'] += total_tokens
                self.total_token_usage['cached_input_tokens'] += cached_input_tokens

                price_str = f"${price:.2f}" if price is not None else "NOT SET"
                logger.info(
                    f"Day {game.current_day}: Price={price_str}, "
                    f"Tokens (in/out/reason): {input_tokens}/{output_tokens}/{reasoning_tokens}, "
                    f"Tools: {day_tools}"
                )

            # If price wasn't set, use suggested price as fallback
            if not price_was_set:
                logger.warning(f"Day {game.current_day}: set_price not called, using suggested price")
                price = game.suggested_starting_price

            return round(price, 2)

        except Exception as e:
            logger.error(f"Error in Responses API call: {e}")

            # Record error if enabled
            if self.recorder:
                self.recorder.record_error(game.current_day, e)

            # Record the day even on error
            self.tool_call_history.append({
                "day": game.current_day,
                "tools": []
            })
            # Clear pending outputs on error
            self.pending_function_outputs = []
            return game.suggested_starting_price

    def play_game(self, game: SimpleLemonadeGame) -> list[dict]:
        """Play a full game using the Responses API."""
        results = []

        while not game.game_over:
            price = self.make_decision(game)
            result = game.play_turn(price)
            results.append(result)

            # Record game state if enabled
            if self.recorder:
                self.recorder.record_game_state(
                    result['day'],
                    result['price'],
                    result['customers'],
                    result['profit'],
                    result['cash']
                )

            # Get tools used for this day
            day_entry = self.tool_call_history[-1] if self.tool_call_history else None
            tools_used = day_entry["tools"] if day_entry else []

            logger.info(
                f"Day {result['day']}: Price=${price:.2f}, "
                f"Profit=${result['profit']:.2f}, "
                f"Tools: {tools_used}"
            )

        # Save recording if enabled
        if self.recorder:
            test_name = "game"  # Default test name
            if hasattr(game, '_test_name'):
                test_name = game._test_name
            self.recorder.save(test_name, self.model_name)

        return results
