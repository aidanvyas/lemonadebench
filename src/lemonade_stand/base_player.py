"""Base player class for AI players in the Lemonade Stand business game."""

from abc import ABC, abstractmethod
from typing import Any

from .business_game import BusinessGame
from .game_recorder import GameRecorder


class BasePlayer(ABC):
    """Abstract base class for AI players."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        """Initialize the base player.

        Args:
            model_name: Model identifier
            api_key: API key for the model provider
        """
        self.model_name = model_name
        self.api_key = api_key

        # Token tracking
        self.total_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
        }

        # Error tracking
        self.errors: list[dict[str, Any]] = []

        # Reasoning summaries (for o1/o3 models that support reasoning)
        self.reasoning_summaries: list[dict[str, Any]] = []

        # Model pricing (to be overridden by subclasses)
        self.model_pricing: dict[str, dict[str, float]] = {}

    @abstractmethod
    def play_turn(
        self, game: BusinessGame, recorder: GameRecorder | None = None
    ) -> dict[str, Any]:
        """Play one turn of the game.

        Args:
            game: The BusinessGame instance
            recorder: Optional GameRecorder to record all interactions

        Returns:
            Dictionary with success status and attempt information
        """
        pass

    @abstractmethod
    def calculate_cost(self) -> dict[str, float]:
        """Calculate the total cost of API usage.

        Returns:
            Cost breakdown and total
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the player for a new game."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close any underlying connections."""
        pass

    def __enter__(self) -> "BasePlayer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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
                        "description": "Number of cups to order (minimum 0)",
                    },
                    "lemons": {
                        "type": "integer",
                        "description": "Number of lemons to order (minimum 0)",
                    },
                    "sugar": {
                        "type": "integer",
                        "description": "Amount of sugar to order (minimum 0)",
                    },
                    "water": {
                        "type": "integer",
                        "description": "Amount of water to order (minimum 0)",
                    },
                },
                "required": ["cups", "lemons", "sugar", "water"],
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
                    },
                    "close_hour": {
                        "type": "integer",
                        "description": "Closing hour (1-24, must be > open_hour)",
                    },
                },
                "required": ["open_hour", "close_hour"],
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
                        "description": "Price per lemonade (minimum 0)",
                    }
                },
                "required": ["price"],
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
        import json

        try:
            result: Any

            # Convert float arguments to int for certain tools
            if tool_name == "set_operating_hours":
                if "open_hour" in args:
                    args["open_hour"] = int(args["open_hour"])
                if "close_hour" in args:
                    args["close_hour"] = int(args["close_hour"])
            elif tool_name == "order_supplies":
                for key in ["cups", "lemons", "sugar", "water"]:
                    if key in args:
                        args[key] = int(args[key])

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

            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)}, default=str)
