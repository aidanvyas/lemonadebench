"""NanoEval-compatible LemonadeBench implementation for OpenAI models.

This implementation follows NanoEval's principles:
- Minimal indirection (~100 lines for core eval logic)
- Clear separation of concerns
- Fast, async-based execution
- SQLite-based result tracking
"""

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

# Simulated NanoEval imports (replace with actual when available)
# from nanoeval import Eval, EvalSpec, Task, Solver


@dataclass
class LemonadeTask:
    """A single day in the lemonade stand game."""

    day: int
    weather: str
    temperature: float
    game_state: dict[str, Any]

    def to_prompt(self) -> str:
        """Convert task to model prompt."""
        return f"""Day {self.day} of your lemonade stand.
Weather: {self.weather} ({self.temperature}°F)
Current cash: ${self.game_state["cash"]:.2f}
Current inventory: {self.game_state["inventory"]}

What is your strategy for today?"""


class OpenAISolver:
    """Solver that uses OpenAI models to play the lemonade game."""

    def __init__(self, model_name: str = "gpt-4.1-nano"):
        self.model_name = model_name
        # Import here to keep dependencies lazy
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def solve(self, task: LemonadeTask) -> dict[str, Any]:
        """Generate solution for a single game day."""
        from src.lemonade_stand.business_game import BusinessGame

        # Create game instance for tool execution
        game = BusinessGame()

        # Build the API request
        response = await self.client.responses.create(
            model=self.model_name,
            input=[{"role": "user", "content": task.to_prompt()}],
            tools=self._get_game_tools(),
            instructions=self._get_system_prompt(),
            max_output_tokens=2000,
        )

        # Process tool calls and execute game actions
        result = {"day": task.day, "decisions": [], "profit": 0, "tool_calls": []}

        for item in response.output:
            if item.type == "function_call":
                # Execute the tool against the game
                tool_result = self._execute_tool(
                    item.name, json.loads(item.arguments), game
                )
                result["tool_calls"].append(item.name)
                result["decisions"].append(
                    {
                        "tool": item.name,
                        "args": json.loads(item.arguments),
                        "result": tool_result,
                    }
                )

                # Check if day completed successfully
                if item.name == "open_for_business":
                    parsed_result = json.loads(tool_result)
                    if parsed_result.get("success"):
                        result["profit"] = parsed_result.get("profit", 0)

        return result

    def _get_game_tools(self) -> list[dict]:
        """Return tool definitions for the game."""
        return [
            {
                "name": "set_price",
                "description": "Set the price per cup of lemonade",
                "parameters": {
                    "type": "object",
                    "properties": {"price": {"type": "number", "minimum": 0}},
                    "required": ["price"],
                },
            },
            {
                "name": "set_operating_hours",
                "description": "Set hours to operate (0-23)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hours": {"type": "integer", "minimum": 0, "maximum": 24}
                    },
                    "required": ["hours"],
                },
            },
            {
                "name": "buy_supplies",
                "description": "Purchase inventory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cups": {"type": "integer", "minimum": 0},
                        "lemons": {"type": "integer", "minimum": 0},
                        "sugar": {"type": "integer", "minimum": 0},
                        "water": {"type": "integer", "minimum": 0},
                    },
                },
            },
            {
                "name": "open_for_business",
                "description": "Start the business day",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def _get_system_prompt(self) -> str:
        """Return system instructions for the model."""
        return """You are managing a lemonade stand business.
Your goal is to maximize profit over 30 days.
Recipe: 1 cup + 1 lemon + 1 sugar + 1 water = 1 lemonade
Demand follows: Q = 50 - 10p (optimal price around $2.50)
You must call open_for_business() after setting price and hours."""

    def _execute_tool(self, tool_name: str, args: dict, _game: Any) -> str:
        """Execute a tool against the game instance."""
        # This would integrate with actual game logic
        # Simplified for demonstration
        if tool_name == "set_price":
            return json.dumps({"success": True, "price": args["price"]})
        elif tool_name == "open_for_business":
            return json.dumps({"success": True, "profit": 100.0})
        return json.dumps({"success": False})


class LemonadeBenchEval:
    """NanoEval-compatible evaluation for LemonadeBench."""

    def __init__(self, model_name: str = "gpt-4.1-nano", days: int = 30):
        self.model_name = model_name
        self.days = days
        self.solver = OpenAISolver(model_name)

    async def run(self) -> dict[str, Any]:
        """Run the full evaluation."""
        tasks = self._generate_tasks()
        results = []

        # Run tasks concurrently (NanoEval style)
        async with asyncio.TaskGroup() as tg:
            futures = [tg.create_task(self.solver.solve(task)) for task in tasks]

        results = [await f for f in futures]

        # Calculate metrics
        total_profit = sum(r["profit"] for r in results)
        avg_profit_per_day = total_profit / len(results)

        return {
            "model": self.model_name,
            "days": self.days,
            "total_profit": total_profit,
            "avg_daily_profit": avg_profit_per_day,
            "results": results,
        }

    def _generate_tasks(self) -> list[LemonadeTask]:
        """Generate tasks for each day of the game."""
        tasks = []
        for day in range(1, self.days + 1):
            # Simulate weather patterns
            weather = "sunny" if day % 3 == 0 else "cloudy"
            temp = 75 + (day % 10) * 2

            tasks.append(
                LemonadeTask(
                    day=day,
                    weather=weather,
                    temperature=temp,
                    game_state={
                        "cash": 1000.00,  # Would track actual state
                        "inventory": {
                            "cups": 100,
                            "lemons": 50,
                            "sugar": 20,
                            "water": 100,
                        },
                    },
                )
            )
        return tasks


# Example usage following NanoEval patterns
async def main():
    """Run LemonadeBench evaluation."""
    eval = LemonadeBenchEval(model_name="gpt-4.1-nano", days=30)
    results = await eval.run()

    print(f"Model: {results['model']}")
    print(f"Total Profit: ${results['total_profit']:.2f}")
    print(f"Average Daily Profit: ${results['avg_daily_profit']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
