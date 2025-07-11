#!/usr/bin/env python3
"""Analyze v0.5 benchmark results with comprehensive metrics, LaTeX tables, and plots."""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).parent.parent))

# Import plotting libraries
try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: uv add matplotlib")


def generate_latex_table(report, output_file):
    """Generate a LaTeX table with key metrics from v0.5 results.

    Args:
        report: Metrics report from generate_metrics_report
        output_file: Path to save the .tex file
    """
    latex = r"""\begin{table}[h]
\centering
\caption{LemonadeBench v0.5 - Business Simulation Results}
\label{tab:lemonadebench_v05}
\begin{tabular}{|l|r|r|r|r|r|r|r|}
\hline
\textbf{Model} & \textbf{Days} & \textbf{Profit (\$)} & \textbf{Profit SD} & \textbf{Customers} & \textbf{Service} & \textbf{Stockouts} & \textbf{Cost (\$)} \\
\hline
"""

    # Add row for each model
    for model, stats in report["model_comparison"].items():
        # Extract model-specific data
        model_games = [
            g for g in report.get("individual_metrics", []) if g.model == model
        ]
        profits = [g.total_profit for g in model_games] if model_games else []
        profit_std = statistics.stdev(profits) if len(profits) > 1 else 0

        # Calculate average customers from game metrics
        avg_customers = (
            sum(g.total_customers_served for g in model_games) / len(model_games)
            if model_games
            else 0
        )
        avg_stockout_rate = (
            sum(g.stockout_rate for g in model_games) / len(model_games)
            if model_games
            else 0
        )

        latex += f"{model:<15} & "
        latex += f"{stats['avg_days_survived']:>4.1f} & "
        latex += f"{stats['avg_profit']:>8.2f} & "
        latex += f"{profit_std:>7.2f} & "
        latex += f"{avg_customers:>9.0f} & "
        latex += f"{stats['avg_service_rate'] * 100:>7.1f}\\% & "
        latex += f"{avg_stockout_rate * 100:>9.1f}\\% & "
        latex += f"{stats['avg_cost_per_day']:>6.4f} \\\\\n"
        latex += r"\hline" + "\n"

    latex += r"""\end{tabular}
\end{table}"""

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)

    print(f"LaTeX table saved to: {output_path}")


def generate_profit_plots(data, output_dir):
    """Generate profit-over-time plots for each model.

    Args:
        data: Raw benchmark data with game histories
        output_dir: Directory to save plot files
    """
    if not PLOTTING_AVAILABLE:
        print("Skipping plots - matplotlib not available")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up the plot style
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors for different models
    colors = {
        "gpt-4.1-nano": "#FF6B6B",
        "gpt-4.1-mini": "#4ECDC4",
        "gpt-4.1": "#45B7D1",
        "o3": "#96CEB4",
        "o4-mini": "#DDA0DD",
        "claude-3-haiku": "#F7DC6F",
        "claude-3.5-sonnet": "#BB8FCE",
    }

    # Plot each model's profit trajectory
    for model, model_results in data["results"].items():
        for game in model_results["individual_games"]:
            if game["success"] and "daily_cash_history" in game:
                days = list(range(len(game["daily_cash_history"])))
                cash_history = game["daily_cash_history"]

                # Calculate profit history (cash - starting cash)
                starting_cash = data["parameters"]["starting_cash"]
                profit_history = [cash - starting_cash for cash in cash_history]

                # Plot with transparency for multiple games
                alpha = 0.7 if model_results["num_games"] > 1 else 1.0
                ax.plot(
                    days,
                    profit_history,
                    color=colors.get(model, "#808080"),
                    label=f"{model} (Game {game['game_number']})",
                    alpha=alpha,
                    linewidth=2,
                )

    # Add theoretical optimal line (if we know it)
    # Assuming optimal daily profit of ~$625 from the metrics
    if data["parameters"]["days_per_game"] > 0:
        days_range = range(data["parameters"]["days_per_game"])
        optimal_profit = [625.54 * day for day in days_range]
        ax.plot(
            days_range,
            optimal_profit,
            "k--",
            label="Theoretical Optimal",
            alpha=0.5,
            linewidth=2,
        )

    # Formatting
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Cumulative Profit ($)", fontsize=12)
    ax.set_title(
        "LemonadeBench v0.5: Profit Over Time by Model", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels, strict=False):
        model_name = label.split(" (Game")[0]
        if model_name not in [lbl.split(" (Game")[0] for lbl in unique_labels]:
            unique_labels.append(label)
            unique_handles.append(handle)

    ax.legend(
        unique_handles,
        [lbl.split(" (Game")[0] for lbl in unique_labels],
        loc="upper left",
        framealpha=0.9,
    )

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_path / f"profit_over_time_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Profit plot saved to: {plot_file}")


def analyze_results(filename: str = None) -> None:
    """Analyze results from the given filename or the latest results.
    
    Args:
        filename: Path to the JSON results file
    """
    if filename:
        with open(filename) as f:
            data = json.load(f)
    else:
        # Find the latest file
        results_dir = Path("results/json")
        if results_dir.exists():
            json_files = sorted(
                results_dir.glob("*_full.json"),  # Look for comprehensive recordings first
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not json_files:  # Fall back to regular files
                json_files = sorted(
                    results_dir.glob("*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            if json_files:
                filename = str(json_files[0])
                print(f"Analyzing: {filename}")
                with open(filename) as f:
                    data = json.load(f)
            else:
                print("No result files found")
                return
        else:
            print("No results directory found")
            return
    
    # Check if this is a comprehensive recording
    if "benchmark_metadata" in data and "games" in data:
        # New comprehensive format
        analyze_comprehensive_format(data, filename)
    else:
        print("Error: This file is not in the comprehensive recording format.")
        print("Please run a new benchmark to generate data in the correct format.")


def analyze_comprehensive_format(data: Dict[str, Any], filename: str) -> None:
    """Analyze the new comprehensive recording format."""
    print("\n" + "=" * 80)
    print("LEMONADEBENCH v0.5 - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    metadata = data["benchmark_metadata"]
    params = metadata["parameters"]
    
    print(f"\nBenchmark Configuration:")
    print(f"  Models: {', '.join(params['models'])}")
    print(f"  Games per model: {params['games_per_model']}")
    print(f"  Days per game: {params['days_per_game']}")
    print(f"  Starting cash: ${params['starting_cash']}")
    print(f"  Duration: {metadata['total_duration_seconds']:.1f} seconds")
    
    # Analyze each game
    model_stats = {}
    
    for game_data in data["games"]:
        model = game_data["model"]
        if model not in model_stats:
            model_stats[model] = {
                "games": [],
                "total_tokens": 0,
                "total_cost": 0,
                "total_interactions": 0,
                "tool_usage": {},
            }
        
        # Extract key metrics from final results
        if game_data.get("final_results"):
            model_stats[model]["games"].append(game_data["final_results"])
            model_stats[model]["total_tokens"] += game_data.get("total_tokens", 0)
            model_stats[model]["total_cost"] += game_data.get("total_cost", 0)
        
        # Analyze interactions
        for day_data in game_data.get("days", []):
            for interaction in day_data.get("interactions", []):
                model_stats[model]["total_interactions"] += 1
                
                # Count tool usage
                for tool_exec in interaction.get("tool_executions", []):
                    tool_name = tool_exec["tool"]
                    if tool_name not in model_stats[model]["tool_usage"]:
                        model_stats[model]["tool_usage"][tool_name] = 0
                    model_stats[model]["tool_usage"][tool_name] += 1
    
    # Print model comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    for model, stats in model_stats.items():
        print(f"\n--- {model} ---")
        if stats["games"]:
            profits = [g["total_profit"] for g in stats["games"]]
            print(f"Games completed: {len(stats['games'])}")
            print(f"Average profit: ${statistics.mean(profits):.2f}")
            if len(profits) > 1:
                print(f"Profit std dev: ${statistics.stdev(profits):.2f}")
            print(f"Total tokens: {stats['total_tokens']:,}")
            print(f"Total cost: ${stats['total_cost']:.4f}")
            print(f"Total interactions: {stats['total_interactions']}")
            
            print("\nTool usage:")
            for tool, count in sorted(stats["tool_usage"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool}: {count}")
    
    # Save analysis outputs
    base_name = Path(filename).stem.replace("_full", "")
    
    # Generate LaTeX table
    latex_file = f"results/latex/{base_name}_analysis.tex"
    generate_comprehensive_latex_table(model_stats, latex_file, params)
    
    # Generate plots
    plot_dir = f"results/plots/{base_name}"
    generate_comprehensive_plots(data, plot_dir)


def generate_comprehensive_latex_table(model_stats: Dict[str, Any], output_file: str, params: Dict[str, Any]) -> None:
    """Generate LaTeX table from comprehensive data."""
    latex = r"""\begin{table}[h]
\centering
\caption{LemonadeBench v0.5 - Comprehensive Results}
\label{tab:lemonadebench_v05_comprehensive}
\begin{tabular}{|l|r|r|r|r|r|r|}
\hline
\textbf{Model} & \textbf{Games} & \textbf{Avg Profit (\$)} & \textbf{Std Dev (\$)} & \textbf{Tokens} & \textbf{Cost (\$)} & \textbf{Interactions} \\
\hline
"""
    
    for model, stats in sorted(model_stats.items()):
        if stats["games"]:
            profits = [g["total_profit"] for g in stats["games"]]
            avg_profit = statistics.mean(profits)
            std_profit = statistics.stdev(profits) if len(profits) > 1 else 0
            
            latex += f"{model} & "
            latex += f"{len(stats['games'])} & "
            latex += f"{avg_profit:.2f} & "
            latex += f"{std_profit:.2f} & "
            latex += f"{stats['total_tokens']:,} & "
            latex += f"{stats['total_cost']:.4f} & "
            latex += f"{stats['total_interactions']} \\\\"
            latex += "\n\\hline\n"
    
    latex += r"""\end{tabular}
\end{table}"""
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to: {output_path}")


def generate_comprehensive_plots(data: Dict[str, Any], output_dir: str) -> None:
    """Generate plots from comprehensive data."""
    if not PLOTTING_AVAILABLE:
        print("\nSkipping plots - matplotlib not available")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Profit trajectories
    plt.figure(figsize=(12, 8))
    
    colors = {
        "gpt-4.1-nano": "#FF6B6B",
        "gpt-4.1-mini": "#4ECDC4",
        "gpt-4.1": "#45B7D1",
        "o3": "#96CEB4",
        "o4-mini": "#DDA0DD",
        "claude-3-haiku": "#F7DC6F",
        "claude-3.5-sonnet": "#BB8FCE",
    }
    
    for game_data in data["games"]:
        model = game_data["model"]
        game_id = game_data["game_id"]
        
        # Extract daily cash history from game states
        cash_history = []
        for day_data in game_data.get("days", []):
            if "game_state_after" in day_data:
                cash_history.append(day_data["game_state_after"]["cash"])
        
        if cash_history:
            days = list(range(len(cash_history)))
            starting_cash = game_data["parameters"]["starting_cash"]
            profit_history = [cash - starting_cash for cash in cash_history]
            
            plt.plot(
                days,
                profit_history,
                color=colors.get(model, "#808080"),
                label=f"{model} (Game {game_id})",
                alpha=0.7,
                linewidth=2,
            )
    
    plt.xlabel("Day", fontsize=12)
    plt.ylabel("Cumulative Profit ($)", fontsize=12)
    plt.title("LemonadeBench v0.5: Profit Trajectories", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    
    # Custom legend with unique models only
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    unique_models = {}
    for handle, label in zip(handles, labels):
        model = label.split(" (Game")[0]
        if model not in unique_models:
            unique_models[model] = handle
    
    plt.legend(
        unique_models.values(),
        unique_models.keys(),
        loc="upper left",
        framealpha=0.9
    )
    
    plot_file = output_path / "profit_trajectories.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Profit plot saved to: {plot_file}")
    
    # Plot 2: Tool usage heatmap
    # ... additional plots can be added here




def main():
    parser = argparse.ArgumentParser(description="Analyze LemonadeBench v0.5 results")
    parser.add_argument(
        "--file", help="JSON results file from v0.5 benchmark"
    )
    parser.add_argument(
        "--latest", action="store_true", help="Analyze the most recent result file"
    )

    args = parser.parse_args()

    if args.latest or not args.file:
        analyze_results()
    else:
        analyze_results(args.file)


if __name__ == "__main__":
    main()
