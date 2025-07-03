#!/usr/bin/env python3
"""Generate plots from saved comparison results."""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


def load_results(filepath: Path) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_plots(comparison_data: dict, output_dir: Path = None) -> None:
    """Generate plots from comparison data."""
    results = comparison_data["results"]
    
    # Group by model and condition
    grouped = {}
    for r in results:
        key = f"{r['model']}_{r.get('condition', 'suggested')}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
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
            # Average across all runs
            all_prices = []
            for run in runs:
                prices = [d["price"] for d in run["daily_results"]]
                all_prices.append(prices)
            
            # Calculate average price per day
            max_days = max(len(prices) for prices in all_prices)
            avg_prices = []
            for day in range(max_days):
                day_prices = [prices[day] for prices in all_prices if day < len(prices)]
                avg_prices.append(sum(day_prices) / len(day_prices))
            
            days = list(range(1, len(avg_prices) + 1))
            ax2.plot(days, avg_prices, label=model, marker='o', markersize=4, alpha=0.7)
    
    ax2.set_title("Average Price Evolution - Suggested Condition")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Token usage by model and condition
    ax3 = axes[1, 0]
    
    model_token_usage = {}
    for key, runs in grouped.items():
        model = runs[0]["model"]
        condition = runs[0].get("condition", "suggested")
        if model not in model_token_usage:
            model_token_usage[model] = {}
        
        # Average token usage
        avg_tokens = 0
        count = 0
        for run in runs:
            if 'token_usage' in run:
                avg_tokens += run['token_usage'].get('total_tokens', 0)
                count += 1
        
        if count > 0:
            model_token_usage[model][condition] = avg_tokens / count
    
    for i, condition in enumerate(conditions):
        tokens = [model_token_usage[model].get(condition, 0) for model in model_token_usage]
        ax3.bar([xi + i*width for xi in x], tokens, width, label=condition)
    
    ax3.set_title("Average Total Token Usage by Model and Condition")
    ax3.set_xlabel("Model")
    ax3.set_ylabel("Average Total Tokens")
    ax3.set_xticks([xi + width for xi in x])
    ax3.set_xticklabels(list(model_token_usage.keys()))
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Profit distribution boxplot
    ax4 = axes[1, 1]
    
    # Prepare data for boxplot
    plot_data = []
    plot_labels = []
    
    for key in sorted(grouped.keys()):
        runs = grouped[key]
        profits = [r["total_profit"] for r in runs]
        if profits:  # Only add if there's data
            plot_data.append(profits)
            plot_labels.append(key.replace('_', '\n'))
    
    if plot_data:
        ax4.boxplot(plot_data, labels=plot_labels)
        ax4.set_title("Profit Distribution by Model and Condition")
        ax4.set_ylabel("Total Profit ($)")
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    # Use timestamp from results or current time
    timestamp = comparison_data.get("timestamp", datetime.now().isoformat())
    timestamp_str = timestamp.replace(":", "").replace("-", "").replace(".", "_")[:15]
    filename = f"comparison_{timestamp_str}.png"
    filepath = output_dir / filename
    
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Plots saved to {filepath}")
    
    return filepath


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate plots from saved results")
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: plots/)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots after generating"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        return 1
    
    # Load and process results
    print(f"Loading results from {args.results_file}")
    comparison_data = load_results(args.results_file)
    
    # Generate plots
    filepath = generate_plots(comparison_data, args.output_dir)
    
    if args.show:
        plt.show()
    
    return 0


if __name__ == "__main__":
    exit(main())