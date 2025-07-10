#!/usr/bin/env python3
"""Analyze v0.5 benchmark results with comprehensive metrics, LaTeX tables, and plots."""

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime
import statistics

sys.path.append(str(Path(__file__).parent.parent))

from src.lemonade_stand.comprehensive_recorder import (
    MetricsAnalyzer,
    generate_metrics_report,
    print_metrics_summary,
    save_metrics_report,
)

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: uv pip install matplotlib")


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
        model_games = [g for g in report.get("individual_metrics", []) if g.model == model]
        profits = [g.total_profit for g in model_games] if model_games else []
        profit_std = statistics.stdev(profits) if len(profits) > 1 else 0
        
        # Calculate average customers from game metrics
        avg_customers = sum(g.total_customers_served for g in model_games) / len(model_games) if model_games else 0
        avg_stockout_rate = sum(g.stockout_rate for g in model_games) / len(model_games) if model_games else 0
        
        latex += f"{model:<15} & "
        latex += f"{stats['avg_days_survived']:>4.1f} & "
        latex += f"{stats['avg_profit']:>8.2f} & "
        latex += f"{profit_std:>7.2f} & "
        latex += f"{avg_customers:>9.0f} & "
        latex += f"{stats['avg_service_rate']*100:>7.1f}\\% & "
        latex += f"{avg_stockout_rate*100:>9.1f}\\% & "
        latex += f"{stats['avg_cost_per_day']:>6.4f} \\\\\n"
        latex += r"\hline" + "\n"
    
    latex += r"""\end{tabular}
\end{table}"""
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
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
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors for different models
    colors = {
        'gpt-4.1-nano': '#FF6B6B',
        'gpt-4.1-mini': '#4ECDC4',
        'gpt-4.1': '#45B7D1',
        'o3': '#96CEB4',
        'o4-mini': '#DDA0DD',
        'claude-3-haiku': '#F7DC6F',
        'claude-3.5-sonnet': '#BB8FCE'
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
                ax.plot(days, profit_history, 
                       color=colors.get(model, '#808080'),
                       label=f"{model} (Game {game['game_number']})",
                       alpha=alpha, linewidth=2)
    
    # Add theoretical optimal line (if we know it)
    # Assuming optimal daily profit of ~$625 from the metrics
    if data["parameters"]["days_per_game"] > 0:
        days_range = range(data["parameters"]["days_per_game"])
        optimal_profit = [625.54 * day for day in days_range]
        ax.plot(days_range, optimal_profit, 'k--', 
               label='Theoretical Optimal', alpha=0.5, linewidth=2)
    
    # Formatting
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Cumulative Profit ($)', fontsize=12)
    ax.set_title('LemonadeBench v0.5: Profit Over Time by Model', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        model_name = label.split(' (Game')[0]
        if model_name not in [l.split(' (Game')[0] for l in unique_labels]:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    ax.legend(unique_handles, [l.split(' (Game')[0] for l in unique_labels], 
             loc='upper left', framealpha=0.9)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = output_path / f"profit_over_time_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Profit plot saved to: {plot_file}")
    


def main():
    parser = argparse.ArgumentParser(description="Analyze LemonadeBench v0.5 results")
    parser.add_argument("input_file", nargs='?', help="JSON results file from v0.5 benchmark")
    parser.add_argument("--save-report", help="Save detailed metrics report to file")
    parser.add_argument(
        "--compare-models", action="store_true", help="Show detailed model comparison"
    )
    parser.add_argument(
        "--latex", help="Generate LaTeX table and save to specified file"
    )
    parser.add_argument(
        "--plots", help="Generate plots and save to specified directory"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available result files"
    )
    parser.add_argument(
        "--latest", action="store_true", help="Analyze the most recent result file"
    )

    args = parser.parse_args()

    # Handle --list option
    if args.list:
        results_dir = Path("results/json")
        if results_dir.exists():
            json_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            print("Available result files:")
            for i, file in enumerate(json_files[:20]):  # Show latest 20
                print(f"{i+1:3d}. {file.name}")
        else:
            print("No results directory found")
        return

    # Handle --latest option
    input_file = args.input_file
    if args.latest:
        results_dir = Path("results/json")
        if results_dir.exists():
            json_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if json_files:
                input_file = str(json_files[0])
                print(f"Analyzing latest file: {input_file}")
            else:
                print("No result files found")
                return
        else:
            print("No results directory found")
            return

    # Load results
    with open(input_file) as f:
        data = json.load(f)

    if data.get("version") != "0.5":
        print(
            f"Warning: This file appears to be from version {data.get('version', 'unknown')}, not v0.5"
        )

    # Extract all games
    analyzer = MetricsAnalyzer()
    all_game_metrics = []

    for model, model_results in data["results"].items():
        for game in model_results["individual_games"]:
            if game["success"]:
                # Add model info if not present
                if "model" not in game:
                    game["model"] = model

                # Add days target if not present
                if "days_target" not in game:
                    game["days_target"] = data["parameters"]["days_per_game"]

                game_metrics = analyzer.analyze_game(game)
                all_game_metrics.append(game_metrics)

    if not all_game_metrics:
        print("No successful games found to analyze!")
        return

    # Generate report
    report = generate_metrics_report(all_game_metrics)
    print_metrics_summary(report)

    # Save if requested
    if args.save_report:
        save_metrics_report(report, args.save_report)
        print(f"\nDetailed report saved to: {args.save_report}")

    # Show detailed model comparison if requested
    if args.compare_models and "model_comparison" in report:
        print("\n" + "=" * 80)
        print("DETAILED MODEL COMPARISON")
        print("=" * 80)

        for model, stats in report["model_comparison"].items():
            print(f"\n--- {model} ---")
            print(f"Games Analyzed: {stats['games']}")
            print(
                f"Average Days Survived: {stats['avg_days_survived']:.1f} / {data['parameters']['days_per_game']}"
            )
            print(f"Average Total Profit: ${stats['avg_profit']:.2f}")
            print(f"Average Service Rate: {stats['avg_service_rate']:.1%}")
            print(f"Optimal Price Discovery: {stats['price_discovery_rate']:.1%}")
            print(f"Average Cost per Day: ${stats['avg_cost_per_day']:.4f}")

            # Find best and worst games for this model
            model_games = [g for g in all_game_metrics if g.model == model]
            if model_games:
                best_game = max(model_games, key=lambda g: g.total_profit)
                worst_game = min(model_games, key=lambda g: g.total_profit)

                print(f"\nBest Game:")
                print(f"  - Survived: {best_game.days_survived} days")
                print(f"  - Profit: ${best_game.total_profit:.2f}")
                print(f"  - Service Rate: {best_game.overall_service_rate:.1%}")

                print(f"\nWorst Game:")
                print(f"  - Survived: {worst_game.days_survived} days")
                print(f"  - Profit: ${worst_game.total_profit:.2f}")
                print(f"  - Service Rate: {worst_game.overall_service_rate:.1%}")

    # Generate LaTeX table if requested
    if args.latex:
        # Add the game metrics to the report for LaTeX generation
        report["individual_metrics"] = all_game_metrics
        generate_latex_table(report, args.latex)

    # Generate plots if requested
    if args.plots:
        generate_profit_plots(data, args.plots)


if __name__ == "__main__":
    main()
