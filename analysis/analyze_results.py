#!/usr/bin/env python3
"""Analyze LemonadeBench results - simplified version with list, latex, and plots."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(filepath: Path) -> dict:
    """Load results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def list_results(recent: int = 10) -> None:
    """List all saved experiment results."""
    results_dir = Path("results/json")
    if not results_dir.exists():
        print("No results found. Run experiments first.")
        return

    # Find all JSON files
    result_files = sorted(
        results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True
    )

    if not result_files:
        print("No result files found.")
        return

    print(
        f"\nFound {len(result_files)} experiment files. Showing {min(recent, len(result_files))} most recent:\n"
    )
    print("=" * 100)

    for i, filepath in enumerate(result_files[:recent]):
        try:
            data = load_results(filepath)

            # Extract metadata
            models = data.get("models", [data.get("model", "unknown")])
            if not isinstance(models, list):
                models = [models]
            days = data.get("days", 30)
            runs_per_test = data.get("runs_per_test", 1)
            timestamp = data.get("timestamp", "unknown")

            # Get test conditions
            if "aggregated_results" in data:
                conditions = sorted(
                    {r["test_name"] for r in data["aggregated_results"]}
                )
                total_tokens = sum(
                    r.get("total_tokens", 0) for r in data["aggregated_results"]
                )
            else:
                conditions = data.get("tests_run", [])
                total_tokens = data.get("total_tokens", 0)

            print(f"\n{i + 1}. {filepath.name}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Models: {', '.join(models)}")
            print(f"   Conditions: {', '.join(conditions) if conditions else 'N/A'}")
            print(f"   Configuration: {runs_per_test} runs × {days} days")
            print(f"   Total tokens: {total_tokens:,}")

        except Exception as e:
            print(f"   Error reading file: {e}")

    print("\n" + "=" * 100)
    print("\nTo analyze: python analyze_results.py --latest --latex --plots")


def generate_latex_table(
    test_name: str, test_results: list[dict], output_path: Path
) -> None:
    """Generate LaTeX table for a single test condition."""

    # Group by model
    model_data = {}
    for result in test_results:
        model = result.get("model", "unknown")
        if model not in model_data:
            model_data[model] = []
        model_data[model].append(result)

    # Start LaTeX table
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{test_name}}}")
    lines.append("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|c|c|}")
    lines.append("\\hline")

    # Header
    lines.append(
        "\\multirow{2}{*}{Model} & \\multirow{2}{*}{Default} & \\multicolumn{4}{c|}{Profit (\\$)} & \\multicolumn{3}{c|}{Tool Calls} & \\multicolumn{3}{c|}{Tokens} & \\multirow{2}{*}{Cost (\\$)} \\\\"
    )
    lines.append("\\cline{3-12}")
    lines.append(
        "& & Min & Mean & Std & Max & set\\_price & calc & history & Input & Reason & Output & \\\\"
    )
    lines.append("\\hline")

    # Data rows
    for model in sorted(model_data.keys()):
        runs = model_data[model]

        # Extract data from individual runs
        profits = []
        tool_calls = {"set_price": [], "calculate": [], "get_historical_data": []}
        tokens = {"input": [], "reasoning": [], "output": []}
        costs = []

        for run in runs:
            # Get individual run data
            if "individual_runs" in run:
                for individual_run in run["individual_runs"]:
                    profits.append(individual_run["total_profit"])

                    # Tool calls
                    breakdown = individual_run.get("tool_call_breakdown", {})
                    tool_calls["set_price"].append(breakdown.get("set_price", 0))
                    tool_calls["calculate"].append(breakdown.get("calculate", 0))
                    tool_calls["get_historical_data"].append(
                        breakdown.get("get_historical_data", 0)
                    )

                    # Tokens
                    usage = individual_run.get("token_usage", {})
                    tokens["input"].append(usage.get("input_tokens", 0))
                    tokens["reasoning"].append(usage.get("reasoning_tokens", 0))
                    tokens["output"].append(usage.get("output_tokens", 0))

                    # Cost
                    cost_info = individual_run.get("cost_info", {})
                    costs.append(cost_info.get("total_cost", 0))

        # Calculate statistics
        if profits:
            # Determine default profit based on test type
            if "Suggested" in test_name:
                default_profit = 2250  # $1.00 * 75 customers * 30 days
            elif "Inverse" in test_name:
                default_profit = 1500  # $2.00 * 25 customers * 30 days
            else:
                default_profit = 0  # No default for exploration/no guidance

            # Format row
            row = [
                model,
                f"{default_profit:,}" if default_profit > 0 else "N/A",
                f"{min(profits):.0f}",
                f"{statistics.mean(profits):.0f}",
                f"{statistics.stdev(profits) if len(profits) > 1 else 0:.0f}",
                f"{max(profits):.0f}",
                f"{statistics.mean(tool_calls['set_price']):.1f}",
                f"{statistics.mean(tool_calls['calculate']):.1f}",
                f"{statistics.mean(tool_calls['get_historical_data']):.1f}",
                f"{statistics.mean(tokens['input']) / 1000:.1f}k",
                f"{statistics.mean(tokens['reasoning']):.0f}",
                f"{statistics.mean(tokens['output']):.0f}",
                f"{statistics.mean(costs):.4f}",
            ]

            lines.append(" & ".join(row) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Created: {output_path}")


def generate_latex_tables(data: dict) -> None:
    """Generate LaTeX tables for all test conditions."""
    print("\nGenerating LaTeX tables...")

    # Get aggregated results
    if "aggregated_results" not in data:
        print("Error: No aggregated results found in data")
        return

    results = data["aggregated_results"]

    # Group by test condition
    test_groups = {}
    for result in results:
        test_name = result["test_name"]
        if test_name not in test_groups:
            test_groups[test_name] = []
        test_groups[test_name].append(result)

    # Generate table for each test
    for test_name, test_results in test_groups.items():
        # Create safe filename
        safe_name = (
            test_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        )
        output_path = Path(f"results/tex/{safe_name}.tex")
        generate_latex_table(test_name, test_results, output_path)


def generate_price_discovery_plot(data: dict) -> None:
    """Generate beautiful price discovery visualization."""
    print("\nGenerating price discovery plot...")

    # Get aggregated results
    if "aggregated_results" not in data:
        print("Error: No aggregated results found in data")
        return

    results = data["aggregated_results"]
    days = data.get("days", 30)

    # Group by test condition and model
    test_groups = {}
    for result in results:
        test_name = result["test_name"]
        model = result.get("model", "unknown")

        if test_name not in test_groups:
            test_groups[test_name] = {}

        # Extract price data from individual runs
        all_prices = []
        if "individual_runs" in result:
            for run in result["individual_runs"]:
                if "prices" in run:
                    all_prices.append(run["prices"])

        if all_prices:
            test_groups[test_name][model] = all_prices

    # Set up the plot with beautiful styling
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Price Discovery Across Test Conditions", fontsize=18, fontweight="bold"
    )

    # Color palette - professional and colorblind-friendly
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Test order for consistent layout
    test_order = [
        "Suggested Price",
        "No Guidance",
        "Exploration Hint",
        "Inverse Demand",
    ]

    for idx, (ax, test_name) in enumerate(zip(axes.flat, test_order, strict=False)):
        if test_name not in test_groups:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(test_name)
            continue

        # Plot each model
        for model_idx, (model, price_runs) in enumerate(
            sorted(test_groups[test_name].items())
        ):
            color = colors[model_idx % len(colors)]

            # Calculate mean and std for each day
            day_prices = []
            day_stds = []

            for day in range(days):
                day_values = [run[day] for run in price_runs if day < len(run)]
                if day_values:
                    day_prices.append(statistics.mean(day_values))
                    day_stds.append(
                        statistics.stdev(day_values) if len(day_values) > 1 else 0
                    )

            days_array = list(range(1, len(day_prices) + 1))

            # Plot mean line
            ax.plot(
                days_array,
                day_prices,
                color=color,
                linewidth=2.5,
                label=model,
                alpha=0.9,
            )

            # Add confidence interval
            if day_stds:
                lower = [p - s for p, s in zip(day_prices, day_stds, strict=False)]
                upper = [p + s for p, s in zip(day_prices, day_stds, strict=False)]
                ax.fill_between(days_array, lower, upper, color=color, alpha=0.2)

        # Add optimal price line
        optimal_price = 2.0 if "Inverse" not in test_name else 1.0
        ax.axhline(
            y=optimal_price,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Optimal",
        )

        # Styling
        ax.set_title(test_name, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Day", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=12)
        ax.set_xlim(0, days + 1)
        ax.set_ylim(0, 4.5)
        ax.grid(True, alpha=0.3)

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

    # Adjust layout
    plt.tight_layout()

    # Save with high quality
    output_path = Path("results/plots/price_discovery.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  Created: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze LemonadeBench experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    # List all experiments
  %(prog)s --latest --latex          # Generate LaTeX tables for latest
  %(prog)s --latest --plots          # Generate price discovery plot
  %(prog)s experiment.json --latex   # Analyze specific file
        """,
    )

    parser.add_argument("input_file", nargs="?", help="JSON results file to analyze")
    parser.add_argument(
        "--latest", action="store_true", help="Use the most recent results file"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all saved experiments"
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=10,
        help="With --list, show N most recent (default: 10)",
    )
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    parser.add_argument(
        "--plots", action="store_true", help="Generate price discovery plot"
    )

    args = parser.parse_args()

    # Handle --list mode
    if args.list:
        list_results(args.recent)
        return

    # Determine input file
    if args.latest or not args.input_file:
        # Find the most recent results file
        results_dir = Path("results/json")
        if not results_dir.exists():
            print("Error: No results/json directory found")
            return

        json_files = sorted(
            results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )

        if not json_files:
            print("Error: No results files found")
            return

        input_file = json_files[0]
        print(f"Using latest results: {input_file.name}")
    else:
        input_file = Path(args.input_file)
        if not input_file.exists():
            print(f"Error: File not found: {input_file}")
            return

    # Load data
    data = load_results(input_file)

    # Generate requested outputs
    if args.latex:
        generate_latex_tables(data)

    if args.plots:
        generate_price_discovery_plot(data)

    if not args.latex and not args.plots:
        print("\nNo output requested. Use --latex and/or --plots to generate outputs.")
        print("Use --help for more information.")


if __name__ == "__main__":
    main()
