#!/usr/bin/env python3
"""Benchmark runner for LemonadeBench v0.5 - Business simulation with inventory management."""

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime

from tqdm import tqdm
from pathlib import Path
from typing import Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv

from src.lemonade_stand import (
    OpenAIPlayer,
    BusinessGame,
    GameRecorder,
    BenchmarkRecorder,
)

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_single_game(
    model_name: str,
    game_number: int,
    days: int = 30,
    starting_cash: float = 1000,
    seed: int = None,
) -> dict[str, Any]:
    """Run a single lemonade business game.

    Args:
        model_name: AI model to use
        game_number: Game number for logging
        days: Number of days to simulate
        starting_cash: Starting cash amount
        seed: Random seed for reproducibility

    Returns:
        Game results dictionary
    """
    logger.info(f"Starting game {game_number} with {model_name}")
    start_time = time.time()

    # Initialize game and player
    game = BusinessGame(days=days, starting_cash=starting_cash, seed=seed)
    player = OpenAIPlayer(model_name=model_name)

    # Initialize GameRecorder
    recorder = GameRecorder(
        model=model_name,
        game_number=game_number,
        parameters={
            "days": days,
            "starting_cash": starting_cash,
            "seed": seed,
        },
    )

    # Track additional metrics
    days_with_stockouts = 0
    total_expired_items = {"cups": 0, "lemons": 0, "sugar": 0, "water": 0}
    total_customers_lost = 0
    daily_cash_history = []
    turn_attempts = []

    try:
        # Play the game day by day with progress bar
        day_bar = tqdm(
            total=days,
            desc=f"Game {game_number} Days",
            leave=False,
            position=2,
        )
        while not game.is_game_over():
            # Start new day
            day_info = game.start_new_day()
            daily_cash_history.append(game.cash)

            # Update progress bar
            day_bar.update(1)

            # Record game state at start of day
            supply_costs = game.check_morning_prices()["prices"]
            recorder.start_day(
                day_number=game.current_day,
                game_state={
                    "cash": game.cash,
                    "inventory": game.check_inventory(),
                    "expired_items": day_info["expired_items"],
                    "supply_costs": supply_costs,
                },
            )

            # Track expired items
            if day_info["expired_items"]:
                for item, quantity in day_info["expired_items"].items():
                    total_expired_items[item] += quantity

            # Log progress periodically
            if game.current_day % 10 == 0 or game.current_day == 1:
                logger.info(f"  Day {game.current_day}/{days} - Cash: ${game.cash:.2f}")

            # AI plays the turn - pass the recorder
            turn_result = player.play_turn(game, recorder=recorder)
            turn_attempts.append(turn_result.get("attempts", 1))

            if not turn_result["success"]:
                logger.error(
                    f"  Failed to complete day {game.current_day}: {turn_result.get('error')}"
                )
                break

            # Simulate the day
            day_result = game.simulate_day()

            # Track metrics
            if day_result["customers_lost"] > 0:
                days_with_stockouts += 1
                total_customers_lost += day_result["customers_lost"]

            # Record game state at end of day
            recorder.end_day(
                game_state_after={
                    "cash": game.cash,
                    "inventory": game.check_inventory(),
                    "day_result": day_result,
                    "price": game.price,
                    "open_hour": game.open_hour,
                    "close_hour": game.close_hour,
                },
                total_attempts=turn_result.get("attempts", 1),
            )

        day_bar.close()

        # Get final results
        final_results = game.get_final_results()
        cost_info = player.calculate_cost()

        # Calculate additional metrics
        total_expired_value = sum(
            total_expired_items[item] * game.inventory.base_costs[item]
            for item in total_expired_items
        )

        duration = time.time() - start_time

        # Record final results
        recorder.record_final_results(
            results=final_results, total_cost=cost_info["total_cost"]
        )

        logger.info(
            f"Completed game {game_number}: "
            f"Profit=${final_results['total_profit']:.2f}, "
            f"Customers={final_results['total_customers']}, "
            f"Duration={duration:.1f}s"
        )

        return {
            "game_number": game_number,
            "model": model_name,
            "success": True,
            "starting_cash": starting_cash,
            "days_played": final_results["days_played"],
            "final_cash": final_results["final_cash"],
            "total_profit": final_results["total_profit"],
            "total_revenue": final_results["total_revenue"],
            "total_operating_cost": final_results["total_operating_cost"],
            "total_customers": final_results["total_customers"],
            "total_customers_lost": total_customers_lost,
            "days_with_stockouts": days_with_stockouts,
            "stockout_rate": days_with_stockouts / final_results["days_played"],
            "average_daily_profit": final_results["average_daily_profit"],
            "final_inventory_value": final_results["inventory_value"],
            "total_expired_items": total_expired_items,
            "total_expired_value": total_expired_value,
            "daily_cash_history": daily_cash_history,
            "average_turn_attempts": statistics.mean(turn_attempts)
            if turn_attempts
            else 0,
            "token_usage": player.total_token_usage,
            "cost_info": cost_info,
            "reasoning_summaries": player.reasoning_summaries,
            "ai_errors": player.errors,
            "duration_seconds": duration,
            "game_history": game.history,  # Full game history for detailed analysis
            "supply_cost_history": game.supply_cost_history,
            "recorder": recorder,  # Include recorder for saving later
        }

    except Exception as e:
        logger.error(f"Fatal error in game {game_number}: {e}")
        import traceback

        traceback.print_exc()

        return {
            "game_number": game_number,
            "model": model_name,
            "success": False,
            "starting_cash": starting_cash,
            "error": str(e),
            "days_played": game.current_day,
            "duration_seconds": time.time() - start_time,
        }


def aggregate_results(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate results from multiple games.

    Args:
        games: List of individual game results

    Returns:
        Aggregated statistics
    """
    successful_games = [g for g in games if g.get("success", False)]

    if not successful_games:
        return {
            "num_games": len(games),
            "successful_games": 0,
            "failed_games": len(games),
            "errors": [
                g.get("error", "Unknown error")
                for g in games
                if not g.get("success", False)
            ],
        }

    # Extract metrics from successful games
    total_profits = [g["total_profit"] for g in successful_games]
    total_customers = [g["total_customers"] for g in successful_games]
    customers_lost = [g["total_customers_lost"] for g in successful_games]
    stockout_rates = [g["stockout_rate"] for g in successful_games]
    expired_values = [g["total_expired_value"] for g in successful_games]
    durations = [g["duration_seconds"] for g in successful_games]

    # Aggregate token usage
    total_tokens = sum(g["token_usage"]["total_tokens"] for g in successful_games)
    total_cost = sum(g["cost_info"]["total_cost"] for g in successful_games)

    # Aggregate expired items
    total_expired = {"cups": 0, "lemons": 0, "sugar": 0, "water": 0}
    for g in successful_games:
        for item, count in g["total_expired_items"].items():
            total_expired[item] += count

    return {
        "num_games": len(games),
        "successful_games": len(successful_games),
        "failed_games": len(games) - len(successful_games),
        "total_profit": {
            "mean": statistics.mean(total_profits),
            "std": statistics.stdev(total_profits) if len(total_profits) > 1 else 0,
            "min": min(total_profits),
            "max": max(total_profits),
            "values": total_profits,
        },
        "total_customers": {
            "mean": statistics.mean(total_customers),
            "std": statistics.stdev(total_customers) if len(total_customers) > 1 else 0,
            "total": sum(total_customers),
        },
        "customers_lost": {
            "mean": statistics.mean(customers_lost),
            "total": sum(customers_lost),
        },
        "stockout_rate": {
            "mean": statistics.mean(stockout_rates),
            "std": statistics.stdev(stockout_rates) if len(stockout_rates) > 1 else 0,
        },
        "expired_value": {
            "mean": statistics.mean(expired_values),
            "total": sum(expired_values),
        },
        "total_expired_items": total_expired,
        "duration": {"mean": statistics.mean(durations), "total": sum(durations)},
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "cost_per_game": total_cost / len(successful_games) if successful_games else 0,
        "individual_games": games,
    }


def main():
    """Run the v0.5 benchmark."""
    parser = argparse.ArgumentParser(
        description="Run LemonadeBench v0.5 - Business Simulation"
    )
    parser.add_argument(
        "--games", type=int, default=1, help="Number of games to run per model"
    )
    parser.add_argument("--days", type=int, default=30, help="Number of days per game")
    parser.add_argument(
        "--models", nargs="+", default=["gpt-4.1-nano"], help="Models to test"
    )
    parser.add_argument(
        "--starting-cash", type=float, default=1000, help="Starting cash for each game"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-analysis", action="store_true", help="Skip automatic analysis generation"
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("LEMONADEBENCH v0.5 - Business Simulation with Inventory Management")
    logger.info("=" * 70)
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info(f"Games per model: {args.games}")
    logger.info(f"Days per game: {args.days}")
    logger.info(f"Starting cash: ${args.starting_cash}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Initialize benchmark recorder
    benchmark_recorder = BenchmarkRecorder(
        parameters={
            "models": args.models,
            "games_per_model": args.games,
            "days_per_game": args.days,
            "starting_cash": args.starting_cash,
            "seed": args.seed,
        }
    )

    # Run benchmark for each model
    all_results = {}
    overall_start = time.time()

    for model in tqdm(args.models, desc="Models", position=0, leave=False):
        logger.info(f"\nTesting model: {model}")
        logger.info("-" * 50)

        model_start = time.time()
        games = []

        # Run multiple games
        games_bar = tqdm(
            range(1, args.games + 1),
            desc=f"{model} Games",
            position=1,
            leave=False,
        )
        for game_num in games_bar:
            # Use different seed for each game if base seed provided
            game_seed = (args.seed + game_num) if args.seed else None

            result = run_single_game(
                model_name=model,
                game_number=game_num,
                days=args.days,
                starting_cash=args.starting_cash,
                seed=game_seed,
            )

            # Extract recorder and add to benchmark recorder
            if result.get("success") and "recorder" in result:
                benchmark_recorder.add_game_recording(result["recorder"])

            # Remove recorder from result before appending (to avoid duplication)
            result_copy = result.copy()
            result_copy.pop("recorder", None)
            games.append(result_copy)

            # Log progress
            if result["success"]:
                games_bar.set_postfix(status="done")
            else:
                games_bar.set_postfix(status="failed")

        games_bar.close()

        # Aggregate results for this model
        model_results = aggregate_results(games)
        model_results["model"] = model
        model_results["duration"] = time.time() - model_start

        all_results[model] = model_results

        # Summary for this model
        logger.info(f"\n{model} Summary:")
        logger.info(
            f"  Successful games: {model_results['successful_games']}/{args.games}"
        )
        if model_results["successful_games"] > 0:
            logger.info(
                f"  Average profit: ${model_results['total_profit']['mean']:.2f}"
            )
            logger.info(
                f"  Average customers: {model_results['total_customers']['mean']:.0f}"
            )
            logger.info(
                f"  Average stockout rate: {model_results['stockout_rate']['mean']:.1%}"
            )
            logger.info(f"  Total cost: ${model_results['total_cost']:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_str = "-".join(args.models) if len(args.models) > 1 else args.models[0]

    # Save comprehensive recording
    recording_filename = f"results/json/{models_str}_{args.games}games_{args.days}days_v05_{timestamp}_full.json"
    filename = f"results/json/{models_str}_{args.games}games_{args.days}days_v05_{timestamp}.json"

    Path("results/json").mkdir(parents=True, exist_ok=True)

    # Save the comprehensive recording
    benchmark_recorder.save_to_file(Path(recording_filename))
    logger.info(f"Full recording saved to: {recording_filename}")

    with open(filename, "w") as f:
        json.dump(
            {
                "version": "0.5",
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "models": args.models,
                    "games_per_model": args.games,
                    "days_per_game": args.days,
                    "starting_cash": args.starting_cash,
                    "seed": args.seed,
                },
                "results": all_results,
                "total_duration_seconds": time.time() - overall_start,
            },
            f,
            indent=2,
        )

    # Final summary
    logger.info(f"\n{'=' * 70}")
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total duration: {(time.time() - overall_start) / 60:.1f} minutes")
    logger.info(f"Results saved to: {filename}")
    logger.info(f"Full recording saved to: {recording_filename}")

    # Print comparison if multiple models
    if len(args.models) > 1:
        logger.info("\nModel Comparison:")
        logger.info(
            f"{'Model':<15} {'Avg Profit':<12} {'Customers':<10} {'Stockouts':<10} {'Cost/Game':<10}"
        )
        logger.info("-" * 60)

        for model, results in all_results.items():
            if results["successful_games"] > 0:
                logger.info(
                    f"{model:<15} "
                    f"${results['total_profit']['mean']:<11.2f} "
                    f"{results['total_customers']['mean']:<9.0f} "
                    f"{results['stockout_rate']['mean']:<9.1%} "
                    f"${results['cost_per_game']:<9.4f}"
                )

    # Run analysis if not skipped
    if not args.no_analysis:
        logger.info("\nGenerating analysis report...")
        from analysis.analyze_results import analyze_results

        try:
            # Pass the full recording filename to analyze_results
            analyze_results(recording_filename)
            logger.info("Analysis complete!")
        except Exception as e:
            logger.error(f"Failed to generate analysis: {e}")
            logger.info(
                "You can manually run: python analysis/analyze_results.py --file "
                + recording_filename
            )


if __name__ == "__main__":
    main()
