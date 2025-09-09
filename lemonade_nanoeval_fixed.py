#!/usr/bin/env python3
"""
NanoEval-compatible LemonadeBench implementation (CORRECTED).

This properly handles the sequential nature of the game where each day
depends on the previous day's state.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

# Import the actual game components
from src.lemonade_stand.business_game import BusinessGame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameTask:
    """Represents a complete game evaluation task."""

    task_id: str
    model: str
    days: int
    game_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DayResult:
    """Result from a single day."""

    day: int
    success: bool
    profit: float
    revenue: float
    costs: float
    cash_remaining: float
    inventory: dict[str, int]
    decisions: dict[str, Any]
    error: str | None = None


@dataclass
class GameResult:
    """Result from a complete game."""

    task_id: str
    model: str
    total_days: int
    total_profit: float
    final_cash: float
    days_succeeded: int
    day_results: list[DayResult]
    token_usage: dict[str, int]
    total_duration_ms: int


class LemonadeSolver:
    """Async solver for lemonade game using OpenAI models."""

    def __init__(self, model: str = "gpt-4.1-nano", api_key: str | None = None):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.total_tokens = {"input": 0, "output": 0, "cached": 0}

    async def solve_game(self, task: GameTask) -> GameResult:
        """Solve a complete game sequentially."""
        start_time = time.time()

        # Initialize game
        game = BusinessGame(total_days=task.days, **task.game_params)

        day_results = []
        total_profit = 0
        days_succeeded = 0

        # Play each day sequentially (MUST be sequential!)
        for day in range(1, task.days + 1):
            logger.info(f"Playing day {day}/{task.days} for {task.model}")

            # Play one day
            day_result = await self._play_day(game, day)
            day_results.append(day_result)

            if day_result.success:
                days_succeeded += 1
                total_profit += day_result.profit
            else:
                logger.warning(f"Day {day} failed: {day_result.error}")

        # Get final game state
        final_metrics = game.get_current_metrics()

        return GameResult(
            task_id=task.task_id,
            model=task.model,
            total_days=task.days,
            total_profit=float(final_metrics.get("total_profit", 0)),
            final_cash=float(game.cash),
            days_succeeded=days_succeeded,
            day_results=day_results,
            token_usage=self.total_tokens.copy(),
            total_duration_ms=int((time.time() - start_time) * 1000),
        )

    async def _play_day(self, game: BusinessGame, day: int) -> DayResult:
        """Play a single day of the game."""
        try:
            # Get initial state
            initial_cash = float(game.cash)
            {
                item: game.inventory.get_available(item)
                for item in ["cups", "lemons", "sugar", "water"]
            }

            # Execute the turn
            result = await self._execute_turn(game)

            # Calculate day metrics
            final_cash = float(game.cash)
            profit = final_cash - initial_cash

            # Get decisions from result
            decisions = result.get("decisions", {})

            return DayResult(
                day=day,
                success=result["success"],
                profit=profit,
                revenue=result.get("revenue", 0),
                costs=result.get("costs", 0),
                cash_remaining=final_cash,
                inventory={
                    item: game.inventory.get_available(item)
                    for item in ["cups", "lemons", "sugar", "water"]
                },
                decisions=decisions,
                error=result.get("error"),
            )

        except Exception as e:
            logger.error(f"Day {day} failed with exception: {e}")
            return DayResult(
                day=day,
                success=False,
                profit=0,
                revenue=0,
                costs=0,
                cash_remaining=float(game.cash),
                inventory={},
                decisions={},
                error=str(e),
            )

    async def _execute_turn(self, game: BusinessGame) -> dict:
        """Execute a single turn with the AI."""
        from src.lemonade_stand.openai_player import OpenAIPlayer

        # Create a temporary player to get tools
        player = OpenAIPlayer(self.model)
        tools = player.get_tools()

        # Get prompts
        system_prompt = game._get_system_prompt()
        user_prompt = game.get_turn_prompt()

        conversation = [{"role": "user", "content": user_prompt}]
        max_attempts = 5
        decisions = {}

        for _attempt in range(max_attempts):
            # Call OpenAI API
            response = await self.client.responses.create(
                model=self.model,
                input=conversation,
                tools=tools,
                instructions=system_prompt,
                max_output_tokens=2000,
            )

            # Track token usage
            if hasattr(response, "usage"):
                self.total_tokens["input"] += getattr(response.usage, "input_tokens", 0)
                self.total_tokens["output"] += getattr(
                    response.usage, "output_tokens", 0
                )
                if hasattr(response.usage, "input_tokens_details"):
                    details = response.usage.input_tokens_details
                    if hasattr(details, "cached_tokens"):
                        self.total_tokens["cached"] += details.cached_tokens

            # Process tool calls
            tool_results = []
            for item in response.output:
                if item.type == "function_call":
                    args = json.loads(item.arguments) if item.arguments else {}
                    result = player.execute_tool(item.name, args, game)

                    # Track decisions
                    if item.name in [
                        "set_price",
                        "set_operating_hours",
                        "buy_supplies",
                    ]:
                        decisions[item.name] = args

                    tool_results.append({"name": item.name, "result": result})

                    # Check if day completed
                    if item.name == "open_for_business":
                        result_dict = json.loads(result)
                        if result_dict.get("success"):
                            return {
                                "success": True,
                                "decisions": decisions,
                                "revenue": result_dict.get("revenue", 0),
                                "costs": result_dict.get("costs", 0),
                            }

            # Continue conversation if needed
            if tool_results:
                feedback = "Tool results:\n"
                for tr in tool_results:
                    feedback += f"{tr['name']}: {tr['result']}\n"
                conversation.append({"role": "user", "content": feedback})

        return {
            "success": False,
            "error": "Max attempts reached without opening for business",
            "decisions": decisions,
        }


class LemonadeBenchEval:
    """NanoEval-style evaluation harness for LemonadeBench."""

    def __init__(
        self,
        models: list[str],
        days: int = 30,
        games_per_model: int = 1,
        db_path: Path | None = None,
    ):
        self.models = models if isinstance(models, list) else [models]
        self.days = days
        self.games_per_model = games_per_model
        self.db_path = db_path or Path(
            f"lemonade_eval_{datetime.now():%Y%m%d_%H%M%S}.db"
        )
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for tracking results."""
        conn = sqlite3.connect(self.db_path)

        # Game-level results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS games (
                task_id TEXT PRIMARY KEY,
                model TEXT,
                total_days INTEGER,
                total_profit REAL,
                final_cash REAL,
                days_succeeded INTEGER,
                token_input INTEGER,
                token_output INTEGER,
                token_cached INTEGER,
                duration_ms INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Day-level results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS days (
                task_id TEXT,
                day INTEGER,
                success BOOLEAN,
                profit REAL,
                revenue REAL,
                costs REAL,
                cash_remaining REAL,
                inventory TEXT,
                decisions TEXT,
                error TEXT,
                PRIMARY KEY (task_id, day),
                FOREIGN KEY (task_id) REFERENCES games(task_id)
            )
        """)

        conn.commit()
        conn.close()

    async def run(self) -> dict[str, Any]:
        """Run evaluation for all models."""
        all_results = {}

        # Run games for each model
        for model in self.models:
            logger.info(f"\nEvaluating {model}")
            model_results = []

            # Run multiple games per model for statistical robustness
            for game_num in range(1, self.games_per_model + 1):
                task = GameTask(
                    task_id=f"{model}_game{game_num}_{int(time.time() * 1000)}",
                    model=model,
                    days=self.days,
                )

                # Create solver for this model
                solver = LemonadeSolver(model)

                # Run the game (days are sequential within each game)
                logger.info(
                    f"Starting game {game_num}/{self.games_per_model} for {model}"
                )
                result = await solver.solve_game(task)
                model_results.append(result)

                # Save to database
                self._save_game_result(result)

                # Log summary
                logger.info(
                    f"Game {game_num} complete: "
                    f"Profit=${result.total_profit:.2f}, "
                    f"Success={result.days_succeeded}/{result.total_days} days"
                )

            all_results[model] = model_results

        # Generate summary statistics
        return self._generate_summary(all_results)

    def _save_game_result(self, result: GameResult):
        """Save game and day results to database."""
        conn = sqlite3.connect(self.db_path)

        # Save game-level result
        conn.execute(
            """
            INSERT INTO games
            (task_id, model, total_days, total_profit, final_cash, days_succeeded,
             token_input, token_output, token_cached, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                result.task_id,
                result.model,
                result.total_days,
                result.total_profit,
                result.final_cash,
                result.days_succeeded,
                result.token_usage.get("input", 0),
                result.token_usage.get("output", 0),
                result.token_usage.get("cached", 0),
                result.total_duration_ms,
            ),
        )

        # Save day-level results
        for day_result in result.day_results:
            conn.execute(
                """
                INSERT INTO days
                (task_id, day, success, profit, revenue, costs, cash_remaining,
                 inventory, decisions, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.task_id,
                    day_result.day,
                    day_result.success,
                    day_result.profit,
                    day_result.revenue,
                    day_result.costs,
                    day_result.cash_remaining,
                    json.dumps(day_result.inventory),
                    json.dumps(day_result.decisions),
                    day_result.error,
                ),
            )

        conn.commit()
        conn.close()

    def _generate_summary(self, all_results: dict[str, list[GameResult]]) -> dict:
        """Generate summary statistics."""
        summary = {
            "evaluation": {
                "days_per_game": self.days,
                "games_per_model": self.games_per_model,
                "database": str(self.db_path),
            },
            "models": {},
        }

        for model, results in all_results.items():
            profits = [r.total_profit for r in results]
            tokens = [
                r.token_usage.get("input", 0) + r.token_usage.get("output", 0)
                for r in results
            ]

            summary["models"][model] = {
                "games_played": len(results),
                "avg_profit": sum(profits) / len(profits) if profits else 0,
                "max_profit": max(profits) if profits else 0,
                "min_profit": min(profits) if profits else 0,
                "total_tokens": sum(tokens),
                "avg_duration_seconds": sum(r.total_duration_ms for r in results)
                / len(results)
                / 1000,
            }

        return summary


async def main():
    """Example usage with multiple models."""
    # Evaluate multiple models
    eval = LemonadeBenchEval(
        models=["gpt-4.1-nano"],  # Add more models as needed
        days=30,
        games_per_model=2,  # Run 2 games per model for variance
    )

    print("Starting LemonadeBench evaluation")
    print(f"Database: {eval.db_path}")

    results = await eval.run()

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for model, stats in results["models"].items():
        print(f"\n{model}:")
        print(f"  Average Profit: ${stats['avg_profit']:.2f}")
        print(
            f"  Min/Max Profit: ${stats['min_profit']:.2f} / ${stats['max_profit']:.2f}"
        )
        print(f"  Total Tokens: {stats['total_tokens']:,}")
        print(f"  Avg Duration: {stats['avg_duration_seconds']:.1f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
