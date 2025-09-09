#!/usr/bin/env python3
"""
NanoEval-compatible LemonadeBench implementation.

This provides a clean, minimal interface to run LemonadeBench evaluations
using NanoEval's design principles. The implementation is under 200 lines
and provides full game functionality.
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
    """Represents a single turn in the lemonade game."""

    task_id: str
    day: int
    game: BusinessGame
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result from executing a game task."""

    task_id: str
    day: int
    success: bool
    profit: float
    revenue: float
    costs: float
    decisions: dict[str, Any]
    token_usage: dict[str, int]
    duration_ms: int
    error: str | None = None


class LemonadeSolver:
    """Async solver for lemonade game tasks using OpenAI models."""

    def __init__(self, model: str = "gpt-4.1-nano", api_key: str | None = None):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.token_usage = {"input": 0, "output": 0, "cached": 0}

    async def solve(self, task: GameTask) -> TaskResult:
        """Solve a single game day task."""
        start_time = time.time()
        game = task.game

        try:
            # Get prompts from the game
            system_prompt = game._get_system_prompt()
            user_prompt = game.get_turn_prompt()

            # Execute turn with OpenAI
            result = await self._execute_turn(game, system_prompt, user_prompt)

            # Extract metrics from game state
            metrics = game.get_current_metrics()

            duration_ms = int((time.time() - start_time) * 1000)

            return TaskResult(
                task_id=task.task_id,
                day=task.day,
                success=result["success"],
                profit=float(metrics.get("profit_today", 0)),
                revenue=float(metrics.get("revenue_today", 0)),
                costs=float(metrics.get("costs_today", 0)),
                decisions=result.get("decisions", {}),
                token_usage=result.get("token_usage", {}),
                duration_ms=duration_ms,
                error=result.get("error"),
            )

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            return TaskResult(
                task_id=task.task_id,
                day=task.day,
                success=False,
                profit=0,
                revenue=0,
                costs=0,
                decisions={},
                token_usage={},
                duration_ms=int((time.time() - start_time) * 1000),
                error=str(e),
            )

    async def _execute_turn(self, game: BusinessGame, system: str, prompt: str) -> dict:
        """Execute a single turn with the AI."""
        from src.lemonade_stand.openai_player import OpenAIPlayer

        # Create a temporary player to get tools
        player = OpenAIPlayer(self.model)
        tools = player.get_tools()

        conversation = [{"role": "user", "content": prompt}]
        max_attempts = 5
        decisions = {}

        for _attempt in range(max_attempts):
            # Call OpenAI API
            response = await self.client.responses.create(
                model=self.model,
                input=conversation,
                tools=tools,
                instructions=system,
                max_output_tokens=2000,
            )

            # Track token usage
            if hasattr(response, "usage"):
                self.token_usage["input"] += response.usage.input_tokens
                self.token_usage["output"] += response.usage.output_tokens

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
                                "token_usage": self.token_usage.copy(),
                            }

            # Continue conversation if needed
            if tool_results:
                feedback = "Tool results:\n"
                for tr in tool_results:
                    feedback += f"{tr['name']}: {tr['result']}\n"
                conversation.append({"role": "user", "content": feedback})

        return {
            "success": False,
            "error": "Max attempts reached",
            "decisions": decisions,
            "token_usage": self.token_usage.copy(),
        }


class LemonadeBenchEval:
    """NanoEval-style evaluation harness for LemonadeBench."""

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        days: int = 30,
        parallel: int = 5,
        db_path: Path | None = None,
    ):
        self.model = model
        self.days = days
        self.parallel = parallel
        self.db_path = db_path or Path(
            f"lemonade_{model}_{datetime.now():%Y%m%d_%H%M%S}.db"
        )
        self.solver = LemonadeSolver(model)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for tracking results."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                task_id TEXT PRIMARY KEY,
                day INTEGER,
                success BOOLEAN,
                profit REAL,
                revenue REAL,
                costs REAL,
                decisions TEXT,
                token_usage TEXT,
                duration_ms INTEGER,
                error TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    async def run(self) -> dict[str, Any]:
        """Run the full evaluation."""
        game = BusinessGame(total_days=self.days)
        tasks = []

        # Generate tasks for each day
        for day in range(1, self.days + 1):
            task = GameTask(
                task_id=f"{self.model}_day{day}_{time.time()}", day=day, game=game
            )
            tasks.append(task)

        # Run tasks with controlled parallelism
        results = []
        for i in range(0, len(tasks), self.parallel):
            batch = tasks[i : i + self.parallel]
            batch_results = await asyncio.gather(
                *[self.solver.solve(task) for task in batch]
            )
            results.extend(batch_results)

            # Save to database
            self._save_results(batch_results)

            # Progress update
            logger.info(
                f"Completed {min(i + self.parallel, len(tasks))}/{len(tasks)} days"
            )

        # Calculate final metrics
        total_profit = sum(r.profit for r in results if r.success)
        success_rate = sum(1 for r in results if r.success) / len(results)
        total_tokens = sum(
            r.token_usage.get("input", 0) + r.token_usage.get("output", 0)
            for r in results
        )

        return {
            "model": self.model,
            "days": self.days,
            "total_profit": total_profit,
            "success_rate": success_rate,
            "total_tokens": total_tokens,
            "avg_duration_ms": sum(r.duration_ms for r in results) / len(results),
            "results": [self._result_to_dict(r) for r in results],
        }

    def _save_results(self, results: list[TaskResult]):
        """Save results to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        for result in results:
            conn.execute(
                """
                INSERT OR REPLACE INTO results
                (task_id, day, success, profit, revenue, costs, decisions, token_usage, duration_ms, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.task_id,
                    result.day,
                    result.success,
                    result.profit,
                    result.revenue,
                    result.costs,
                    json.dumps(result.decisions),
                    json.dumps(result.token_usage),
                    result.duration_ms,
                    result.error,
                ),
            )
        conn.commit()
        conn.close()

    def _result_to_dict(self, result: TaskResult) -> dict:
        """Convert TaskResult to dictionary."""
        return {
            "day": result.day,
            "success": result.success,
            "profit": result.profit,
            "revenue": result.revenue,
            "costs": result.costs,
            "decisions": result.decisions,
            "duration_ms": result.duration_ms,
        }


async def main():
    """Example usage."""
    # Run evaluation
    eval = LemonadeBenchEval(
        model="gpt-4.1-nano",
        days=30,
        parallel=5,  # Run 5 days concurrently
    )

    print(f"Starting LemonadeBench evaluation for {eval.model}")
    print(f"Database: {eval.db_path}")

    results = await eval.run()

    # Print summary
    print("\n" + "=" * 50)
    print(f"Model: {results['model']}")
    print(f"Days: {results['days']}")
    print(f"Total Profit: ${results['total_profit']:.2f}")
    print(f"Success Rate: {results['success_rate'] * 100:.1f}%")
    print(f"Total Tokens: {results['total_tokens']:,}")
    print(f"Avg Duration: {results['avg_duration_ms']:.0f}ms per day")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
