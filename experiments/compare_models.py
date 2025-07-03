"""Compare different models on the simple lemonade pricing game."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
import logging
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from src.lemonade_stand.simple_game import SimpleLemonadeGame
from src.lemonade_stand.responses_ai_player import ResponsesAIPlayer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_game(model_name: str, run_number: int, days: int = 100, 
                   condition: str = "suggested") -> dict:
    """Run a single game with specified model.
    
    Args:
        model_name: Model to use
        run_number: Run number for this trial
        days: Number of days to simulate
        condition: One of "suggested", "no_guidance", "exploration"
    """
    game = SimpleLemonadeGame(days=days)
    
    # Set game condition
    game._use_suggested_price = (condition == "suggested")
    game._use_exploration_hint = (condition == "exploration")

    # Use Responses API - it's the future!
    player = ResponsesAIPlayer(
        model_name=model_name,
        include_calculator=(condition == "exploration"),  # Give calculator for exploration
    )

    # Track total API time
    start_time = time.time()
    results = player.play_game(game)
    api_time = time.time() - start_time

    # Calculate metrics
    total_profit = sum(r["profit"] for r in results)
    final_cash = results[-1]["cash"] if results else 100.0
    avg_price = sum(r["price"] for r in results) / len(results) if results else 0
    avg_customers = (
        sum(r["customers"] for r in results) / len(results) if results else 0
    )
    
    # Get token usage if available
    token_usage = getattr(player, 'total_token_usage', {
        'input_tokens': 0,
        'output_tokens': 0,
        'reasoning_tokens': 0,
        'total_tokens': 0
    })

    return {
        "model": model_name,
        "condition": condition,
        "run": run_number,
        "total_profit": total_profit,
        "final_cash": final_cash,
        "days_survived": len(results),
        "avg_price": avg_price,
        "avg_customers": avg_customers,
        "api_time_seconds": api_time,
        "tool_calls": player.tool_call_count,
        "tool_call_history": player.tool_call_history,
        "token_usage": token_usage,
        "daily_results": results,
    }


def compare_models(
    models: list[str],
    conditions: list[str],
    runs_per_model: int = 3,
    days: int = 100,
) -> dict:
    """Compare multiple models across different conditions.
    
    Args:
        models: List of model names
        conditions: List of conditions ("suggested", "no_guidance", "exploration")
        runs_per_model: Runs per model-condition combination
        days: Days per game
    """
    all_results = []

    for model in models:
        for condition in conditions:
            logger.info(f"\nTesting {model} with condition: {condition}")

            for run in range(runs_per_model):
                logger.info(f"Run {run + 1}/{runs_per_model}")
                result = run_single_game(model, run + 1, days, condition)
                all_results.append(result)

                logger.info(
                    f"Total profit: ${result['total_profit']:.2f}, "
                    f"Avg price: ${result['avg_price']:.2f}, "
                    f"Time: {result['api_time_seconds']:.1f}s, "
                    f"Tool calls: {result['tool_calls']}"
                )

    return {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "conditions": conditions,
        "runs_per_model": runs_per_model,
        "days": days,
        "results": all_results,
    }


def analyze_results(comparison_data: dict, show_plots: bool = False) -> None:
    """Analyze and visualize comparison results.
    
    Args:
        comparison_data: Results data
        show_plots: Whether to generate and show plots
    """
    results = comparison_data["results"]

    # Group by model and condition
    grouped = {}
    for r in results:
        key = f"{r['model']}_{r.get('condition', 'suggested')}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    # Print summary table
    print("\n" + "=" * 120)
    print("COMPARISON RESULTS")
    print("=" * 120)
    print(
        f"{'Model+Condition':<30} {'Avg Profit':<12} {'Avg Price':<12} {'Avg Customers':<15} {'Avg Time (s)':<12} {'Tool Calls':<12}"
    )
    print("-" * 120)

    for key, runs in sorted(grouped.items()):
        avg_profit = sum(r["total_profit"] for r in runs) / len(runs)
        avg_price = sum(r["avg_price"] for r in runs) / len(runs)
        avg_customers = sum(r["avg_customers"] for r in runs) / len(runs)
        avg_time = sum(r["api_time_seconds"] for r in runs) / len(runs)
        avg_tool_calls = sum(r["tool_calls"] for r in runs) / len(runs)

        print(
            f"{key:<30} ${avg_profit:<11.2f} ${avg_price:<11.2f} {avg_customers:<14.1f} {avg_time:<11.1f} {avg_tool_calls:<11.1f}"
        )

    print("=" * 120)
    
    # Print token usage summary if available
    print("\nTOKEN USAGE SUMMARY")
    print("=" * 120)
    print(f"{'Model+Condition':<30} {'Input Tokens':<15} {'Output Tokens':<15} {'Reasoning':<15} {'Total':<15}")
    print("-" * 120)
    
    for key, runs in sorted(grouped.items()):
        # Average token usage
        if runs and runs[0].get('token_usage'):
            avg_input = sum(r.get('token_usage', {}).get('input_tokens', 0) for r in runs) / len(runs)
            avg_output = sum(r.get('token_usage', {}).get('output_tokens', 0) for r in runs) / len(runs)
            avg_reasoning = sum(r.get('token_usage', {}).get('reasoning_tokens', 0) for r in runs) / len(runs)
            avg_total = sum(r.get('token_usage', {}).get('total_tokens', 0) for r in runs) / len(runs)
            
            print(f"{key:<30} {avg_input:<14.0f} {avg_output:<14.0f} {avg_reasoning:<14.0f} {avg_total:<14.0f}")
    
    print("=" * 120)

    # Create visualizations only if requested
    if not show_plots:
        return
        
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Model Comparison Results", fontsize=16)

    # Plot 1: Profit by condition for each model
    ax1 = axes[0, 0]
    
    # Reorganize data by model
    model_conditions = {}
    for key, runs in grouped.items():
        model = runs[0]["model"]
        condition = runs[0].get("condition", "suggested")
        if model not in model_conditions:
            model_conditions[model] = {}
        avg_profit = sum(r["total_profit"] for r in runs) / len(runs)
        model_conditions[model][condition] = avg_profit
    
    # Plot grouped bar chart
    conditions = comparison_data.get("conditions", ["suggested", "no_guidance", "exploration"])
    x = range(len(model_conditions))
    width = 0.25
    
    for i, condition in enumerate(conditions):
        profits = [model_conditions[model].get(condition, 0) for model in model_conditions]
        ax1.bar([xi + i*width for xi in x], profits, width, label=condition)
    
    ax1.set_title("Average Profit by Model and Condition")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Average Profit ($)")
    ax1.set_xticks([xi + width for xi in x])
    ax1.set_xticklabels(list(model_conditions.keys()))
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Price evolution for suggested condition
    ax2 = axes[0, 1]
    for key, runs in grouped.items():
        if "suggested" in key and runs:
            model = runs[0]["model"]
            prices = [d["price"] for d in runs[0]["daily_results"]]
            days = [d["day"] for d in runs[0]["daily_results"]]
            ax2.plot(days, prices, label=model, alpha=0.7)
    ax2.set_title("Price Evolution - Suggested Condition (First Run)")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average price by condition
    ax3 = axes[1, 0]
    
    # Similar grouped bar for average prices
    model_avg_prices = {}
    for key, runs in grouped.items():
        model = runs[0]["model"]
        condition = runs[0].get("condition", "suggested")
        if model not in model_avg_prices:
            model_avg_prices[model] = {}
        avg_price = sum(r["avg_price"] for r in runs) / len(runs)
        model_avg_prices[model][condition] = avg_price
    
    for i, condition in enumerate(conditions):
        prices = [model_avg_prices[model].get(condition, 0) for model in model_avg_prices]
        ax3.bar([xi + i*width for xi in x], prices, width, label=condition)
    
    ax3.set_title("Average Price by Model and Condition")
    ax3.set_xlabel("Model")
    ax3.set_ylabel("Average Price ($)")
    ax3.set_xticks([xi + width for xi in x])
    ax3.set_xticklabels(list(model_avg_prices.keys()))
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Tool usage by condition
    ax4 = axes[1, 1]
    
    model_tool_usage = {}
    for key, runs in grouped.items():
        model = runs[0]["model"]
        condition = runs[0].get("condition", "suggested")
        if model not in model_tool_usage:
            model_tool_usage[model] = {}
        avg_tools = sum(r["tool_calls"] for r in runs) / len(runs)
        model_tool_usage[model][condition] = avg_tools
    
    for i, condition in enumerate(conditions):
        tools = [model_tool_usage[model].get(condition, 0) for model in model_tool_usage]
        ax4.bar([xi + i*width for xi in x], tools, width, label=condition)
    
    ax4.set_title("Average Tool Calls by Model and Condition")
    ax4.set_xlabel("Model")
    ax4.set_ylabel("Average Tool Calls")
    ax4.set_xticks([xi + width for xi in x])
    ax4.set_xticklabels(list(model_tool_usage.keys()))
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
    logger.info(f"Plots saved to {plots_dir / filename}")

    if show_plots:
        plt.show()


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare AI models on lemonade pricing"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4.1-nano", "gpt-4.1-mini", "o4-mini"],
        help="Models to compare (allowed: gpt-4.1-nano, gpt-4.1-mini, gpt-4.1, o4-mini, o3)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["suggested", "no_guidance", "exploration"],
        help="Conditions to test (suggested, no_guidance, exploration)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per model-condition combination",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=100,
        help="Number of days per game",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate and show plots (default: save results only)",
    )

    args = parser.parse_args()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
        )
        return

    # Run comparison
    logger.info(f"Comparing models: {args.models}")
    logger.info(f"Testing conditions: {args.conditions}")
    logger.info(f"Runs per model-condition: {args.runs}")
    logger.info(f"Days per game: {args.days}")

    comparison_data = compare_models(
        models=args.models,
        conditions=args.conditions,
        runs_per_model=args.runs,
        days=args.days,
    )

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = results_dir / filename

    with open(filepath, "w") as f:
        json.dump(comparison_data, f, indent=2)

    logger.info(f"Results saved to {filepath}")

    # Always analyze and print summary
    analyze_results(comparison_data, show_plots=args.plots)


if __name__ == "__main__":
    main()
