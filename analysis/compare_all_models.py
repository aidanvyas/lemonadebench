#!/usr/bin/env python3
"""Compare all 8 models from separate benchmark runs."""

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Model colors - consistent palette for all 8 models
MODEL_COLORS = {
    "gpt-4.1-nano": "#FF6B6B",  # Red
    "gpt-4.1-mini": "#4ECDC4",  # Teal
    "gpt-4.1": "#45B7D1",  # Blue
    "o4-mini": "#DDA0DD",  # Plum
    "o3": "#96CEB4",  # Green
    "gpt-5-nano": "#FFB347",  # Orange
    "gpt-5-mini": "#B19CD9",  # Purple
    "gpt-5": "#77DD77",  # Pastel Green
}


def load_benchmark_data(filename: str) -> list[dict[str, Any]]:
    """Load benchmark data from a full recording file."""
    with open(filename) as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, dict) and "games" in data:
        return data["games"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown format in {filename}")


def extract_profit_trajectory(game_data: dict[str, Any]) -> list[float]:
    """Extract daily profit history from a game."""
    cash_history = []
    starting_cash = game_data["parameters"]["starting_cash"]

    for day_data in game_data.get("days", []):
        if "game_state_after" in day_data:
            cash = day_data["game_state_after"]["cash"]
            cash_history.append(cash - starting_cash)

    return cash_history


def main():
    """Generate comprehensive comparison of all 8 models."""

    # Load data from both benchmark runs
    original_5_file = "results/json/gpt-4.1-nano-gpt-4.1-mini-gpt-4.1-o4-mini-o3_1games_30days_v05_20250713_220015_full.json"
    gpt5_family_file = "results/json/gpt-5-gpt-5-mini-gpt-5-nano_1games_30days_v05_20250810_010501_full.json"

    print("Loading benchmark data...")
    original_5_data = load_benchmark_data(original_5_file)
    gpt5_family_data = load_benchmark_data(gpt5_family_file)

    # Combine all data
    all_games = original_5_data + gpt5_family_data

    # Group by model and extract trajectories
    model_trajectories = {}
    model_final_profits = {}
    model_costs = {}

    for game_data in all_games:
        model = game_data["model"]
        trajectory = extract_profit_trajectory(game_data)

        if trajectory:
            if model not in model_trajectories:
                model_trajectories[model] = []
                model_final_profits[model] = []
                model_costs[model] = []

            model_trajectories[model].append(trajectory)

            # Get final profit and cost
            final_results = game_data.get("final_results", {})
            model_final_profits[model].append(
                final_results.get("total_profit", trajectory[-1] if trajectory else 0)
            )
            model_costs[model].append(game_data.get("total_cost", 0))

    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Profit Trajectories
    ax1 = plt.subplot(2, 2, (1, 2))

    # Sort models by final profit for consistent ordering
    avg_profits = {}
    for model in model_trajectories:
        if model_final_profits[model]:
            avg_profits[model] = statistics.mean(model_final_profits[model])

    sorted_models = sorted(
        avg_profits.keys(), key=lambda x: avg_profits[x], reverse=True
    )

    # Plot trajectories
    for model in sorted_models:
        trajectories = model_trajectories[model]
        if not trajectories:
            continue

        # Calculate average trajectory
        max_days = max(len(traj) for traj in trajectories)
        avg_trajectory = []

        for day in range(max_days):
            day_profits = [traj[day] for traj in trajectories if day < len(traj)]
            if day_profits:
                avg_trajectory.append(statistics.mean(day_profits))

        if avg_trajectory:
            days = list(range(1, len(avg_trajectory) + 1))

            label = f"{model} (${avg_profits[model]:.0f})"
            ax1.plot(
                days,
                avg_trajectory,
                color=MODEL_COLORS.get(model, "#808080"),
                label=label,
                linewidth=2.5,
                marker="o" if len(avg_trajectory) <= 10 else None,
                markersize=4,
                alpha=0.9,
            )

    ax1.set_xlabel("Day", fontsize=12)
    ax1.set_ylabel("Cumulative Profit ($)", fontsize=12)
    ax1.set_title("Profit Trajectories - All 8 Models", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Subplot 2: Final Profit Comparison (Bar Chart)
    ax2 = plt.subplot(2, 2, 3)

    models = list(sorted_models)
    profits = [avg_profits[m] for m in models]
    colors = [MODEL_COLORS.get(m, "#808080") for m in models]

    bars = ax2.bar(range(len(models)), profits, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.set_ylabel("Final Profit ($)", fontsize=12)
    ax2.set_title("Final Profit Comparison", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, profit in zip(bars, profits, strict=False):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"${profit:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Subplot 3: Cost Efficiency (Profit per Dollar)
    ax3 = plt.subplot(2, 2, 4)

    avg_costs = {
        model: statistics.mean(model_costs[model])
        for model in models
        if model_costs[model]
    }
    profit_per_dollar = [
        avg_profits[m] / avg_costs[m] if avg_costs.get(m, 0) > 0 else 0 for m in models
    ]

    bars = ax3.bar(range(len(models)), profit_per_dollar, color=colors, alpha=0.8)
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha="right")
    ax3.set_ylabel("Profit per API Dollar", fontsize=12)
    ax3.set_title("Cost Efficiency", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, efficiency in zip(bars, profit_per_dollar, strict=False):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{efficiency:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Overall title
    fig.suptitle(
        "LemonadeBench v0.5: Comprehensive 8-Model Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    output_dir = Path("results/plots/all_models_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "comprehensive_8_model_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nPlot saved to: {output_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':<15} {'Avg Profit':<12} {'API Cost':<10} {'Profit/Dollar':<15} {'Rank'}"
    )
    print("-" * 80)

    for i, model in enumerate(sorted_models, 1):
        profit = avg_profits[model]
        cost = avg_costs.get(model, 0)
        efficiency = profit / cost if cost > 0 else 0
        print(f"{model:<15} ${profit:<11.2f} ${cost:<9.4f} {efficiency:<14.0f} #{i}")

    print("\nKey Findings:")
    print(
        f"1. Best performer: {sorted_models[0]} with ${avg_profits[sorted_models[0]]:.2f} profit"
    )
    print(
        f"2. Most cost-efficient: {models[profit_per_dollar.index(max(profit_per_dollar))]}"
    )
    print(
        f"3. GPT-5 family average: ${statistics.mean([avg_profits[m] for m in ['gpt-5', 'gpt-5-mini', 'gpt-5-nano'] if m in avg_profits]):.2f}"
    )
    print(
        f"4. GPT-4.1 family average: ${statistics.mean([avg_profits[m] for m in ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano'] if m in avg_profits]):.2f}"
    )
    print(
        f"5. Reasoning models (o3, o4-mini) average: ${statistics.mean([avg_profits[m] for m in ['o3', 'o4-mini'] if m in avg_profits]):.2f}"
    )


if __name__ == "__main__":
    main()
