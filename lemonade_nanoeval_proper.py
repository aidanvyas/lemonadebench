"""
LemonadeBench NanoEval implementation following the official API pattern.

This implementation properly handles:
- Sequential days within each game (required by game mechanics)
- Parallel execution across multiple independent games
- Proper integration with NanoEval's task/solver architecture
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import nanoeval
from nanoeval.evaluation import EvalSpec, RunnerArgs
from nanoeval.json_recorder import json_recorder
from nanoeval.setup import nanoeval_entrypoint
from nanoeval.solvers.base import Solver
from nanoeval.task import Task

from src.lemonade_stand.business_game import BusinessGame
from src.lemonade_stand.openai_player import OpenAIPlayer

logger = logging.getLogger(__name__)


@dataclass
class LemonadeGameTask(Task):
    """
    Represents a complete 30-day lemonade game as a single task.

    Each task is one independent game that runs sequentially for 30 days.
    Multiple tasks can run in parallel (different game instances).
    """

    game_id: str
    days: int = 30
    starting_cash: float = 1000.0

    def get_prompt(self) -> str:
        """Return task description."""
        return f"Run a {self.days}-day lemonade stand game (Game ID: {self.game_id})"


@dataclass
class LemonadeGameResult:
    """Result from running a complete game."""

    game_id: str
    total_profit: float
    final_cash: float
    days_succeeded: int
    total_days: int
    daily_profits: list[float]
    token_usage: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "total_profit": self.total_profit,
            "final_cash": self.final_cash,
            "success_rate": self.days_succeeded / self.total_days,
            "avg_daily_profit": self.total_profit / self.total_days,
            "daily_profits": self.daily_profits,
            "token_usage": self.token_usage,
        }


class OpenAILemonadeSolver(Solver):
    """
    Solver that plays complete lemonade games using OpenAI models.

    Important: Each game's 30 days must run sequentially (game state dependency).
    Different games can run in parallel.
    """

    def __init__(self, model: str = "gpt-4.1-nano"):
        self.model = model

    async def solve(self, task: LemonadeGameTask) -> LemonadeGameResult:
        """
        Solve a complete game task by playing all days sequentially.

        This method will be called in parallel for different game instances,
        but within each game, days must be played in order.
        """
        # Initialize game and player
        game = BusinessGame(total_days=task.days, starting_cash=task.starting_cash)
        player = OpenAIPlayer(model_name=self.model)

        daily_profits = []
        days_succeeded = 0

        # Play each day SEQUENTIALLY (required by game mechanics)
        for day in range(1, task.days + 1):
            initial_cash = float(game.cash)

            try:
                # Play one turn
                result = player.play_turn(game)

                if result.get("success"):
                    days_succeeded += 1

                # Calculate daily profit
                final_cash = float(game.cash)
                daily_profit = final_cash - initial_cash
                daily_profits.append(daily_profit)

                # Advance to next day (handles expiration, etc.)
                if day < task.days:
                    game.current_day = day + 1

            except Exception as e:
                logger.error(f"Day {day} failed for game {task.game_id}: {e}")
                daily_profits.append(0.0)

        # Get final metrics
        final_metrics = game.get_current_metrics()

        return LemonadeGameResult(
            game_id=task.game_id,
            total_profit=float(final_metrics.get("total_profit", 0)),
            final_cash=float(game.cash),
            days_succeeded=days_succeeded,
            total_days=task.days,
            daily_profits=daily_profits,
            token_usage=player.total_token_usage,
        )


class LemonadeBenchEval:
    """
    NanoEval-compatible evaluation for LemonadeBench.

    Creates multiple independent game tasks that can run in parallel.
    """

    def __init__(
        self, solver: OpenAILemonadeSolver, num_games: int = 5, days_per_game: int = 30
    ):
        self.solver = solver
        self.num_games = num_games
        self.days_per_game = days_per_game
        self.tasks = self._generate_tasks()

    def _generate_tasks(self) -> list[LemonadeGameTask]:
        """Generate independent game tasks."""
        tasks = []
        for i in range(self.num_games):
            tasks.append(
                LemonadeGameTask(game_id=f"game_{i + 1}", days=self.days_per_game)
            )
        return tasks

    async def evaluate(self) -> dict[str, Any]:
        """Run evaluation and compute metrics."""
        results = []

        # Each task is a complete game that will run its days sequentially
        # But different games can run in parallel via NanoEval's concurrency
        for task in self.tasks:
            result = await self.solver.solve(task)
            results.append(result)

        # Compute aggregate metrics
        total_profits = [r.total_profit for r in results]
        success_rates = [r.days_succeeded / r.total_days for r in results]

        return {
            "num_games": self.num_games,
            "days_per_game": self.days_per_game,
            "model": self.solver.model,
            "metrics": {
                "avg_total_profit": sum(total_profits) / len(total_profits),
                "max_profit": max(total_profits),
                "min_profit": min(total_profits),
                "avg_success_rate": sum(success_rates) / len(success_rates),
            },
            "games": [r.to_dict() for r in results],
        }


async def main() -> None:
    """Main entry point following NanoEval pattern."""

    # Create evaluation spec
    report = await nanoeval.run(
        EvalSpec(
            eval=LemonadeBenchEval(
                solver=OpenAILemonadeSolver(model="gpt-4.1-nano"),
                num_games=10,  # Run 10 independent games
                days_per_game=30,
            ),
            runner=RunnerArgs(
                concurrency=5,  # Run up to 5 games in parallel
                experimental_use_multiprocessing=False,  # Games are I/O bound
                enable_slackbot=False,
                recorder=json_recorder(),
            ),
        )
    )

    # Verify key metrics are present
    assert "avg_total_profit" in report["metrics"]
    assert "avg_success_rate" in report["metrics"]

    # Print summary
    print("\n" + "=" * 60)
    print("LEMONADEBENCH EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {report['model']}")
    print(f"Games Run: {report['num_games']} x {report['days_per_game']} days")
    print(f"Average Total Profit: ${report['metrics']['avg_total_profit']:.2f}")
    print(
        f"Profit Range: ${report['metrics']['min_profit']:.2f} - ${report['metrics']['max_profit']:.2f}"
    )
    print(f"Average Success Rate: {report['metrics']['avg_success_rate'] * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    # Use NanoEval's entry point for proper setup
    nanoeval_entrypoint(main())
