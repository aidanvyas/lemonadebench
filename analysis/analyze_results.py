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

import matplotlib.pyplot as plt


def calculate_business_metrics(model: str, stats: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate business efficiency metrics for a model."""
    
    # Find all games for this model in the comprehensive data
    model_games = [game for game in data["games"] if game["model"] == model]
    
    # Initialize aggregators
    all_prices = set()
    total_weighted_purchase_cost = 0
    total_quantity_purchased = 0
    total_expired_value = 0
    total_lost_customer_profit = 0
    total_excess_stock_value = 0
    
    for game_data in model_games:
        # 1. Extract distinct prices tried
        for day_data in game_data.get("days", []):
            for interaction in day_data.get("interactions", []):
                for tool_exec in interaction.get("tool_executions", []):
                    if tool_exec["tool"] == "set_price":
                        price = tool_exec["arguments"].get("price")
                        if price:
                            all_prices.add(price)
        
        # 2. Calculate weighted average purchase price
        for day_data in game_data.get("days", []):
            supply_costs = day_data.get("game_state_before", {}).get("supply_costs", {})
            for interaction in day_data.get("interactions", []):
                for tool_exec in interaction.get("tool_executions", []):
                    if tool_exec["tool"] == "order_supplies":
                        ordered = tool_exec.get("arguments", {})
                        for item, quantity in ordered.items():
                            if quantity > 0 and item in supply_costs:
                                total_weighted_purchase_cost += supply_costs[item] * quantity
                                total_quantity_purchased += quantity
        
        # 3. Calculate expired goods value
        for day_data in game_data.get("days", []):
            expired_items = day_data.get("game_state_before", {}).get("expired_items", {})
            # Use base costs for expired items (approximate)
            base_costs = {"cups": 0.05, "lemons": 0.20, "sugar": 0.10, "water": 0.02}
            for item, quantity in expired_items.items():
                if item in base_costs:
                    total_expired_value += base_costs[item] * quantity
        
        # 4. Calculate lost customer profit (from final results)
        final_results = game_data.get("final_results", {})
        customers_lost = final_results.get("total_customers_lost", 0)
        customers_served = final_results.get("total_customers", 0)
        revenue = final_results.get("total_revenue", 0)
        if customers_served > 0:
            revenue_per_customer = revenue / customers_served
            total_lost_customer_profit += customers_lost * revenue_per_customer
        
        # 5. Calculate excess stock value (final inventory)
        final_inventory_value = final_results.get("final_inventory_value", 0)
        total_excess_stock_value += final_inventory_value
    
    # Calculate averages
    num_games = len(model_games)
    avg_purchase_price = (total_weighted_purchase_cost / total_quantity_purchased 
                         if total_quantity_purchased > 0 else 0)
    
    return {
        "distinct_prices": len(all_prices),
        "avg_purchase_price": avg_purchase_price,
        "expired_value": total_expired_value / num_games if num_games > 0 else 0,
        "lost_customer_profit": total_lost_customer_profit / num_games if num_games > 0 else 0,
        "excess_stock_value": total_excess_stock_value / num_games if num_games > 0 else 0,
    }



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
    generate_comprehensive_latex_table(model_stats, latex_file, params, data)
    
    # Generate plots
    plot_dir = f"results/plots/{base_name}"
    generate_comprehensive_plots(data, plot_dir)


def generate_comprehensive_latex_table(model_stats: Dict[str, Any], output_file: str, params: Dict[str, Any], data: Dict[str, Any]) -> None:
    """Generate LaTeX table with business efficiency metrics."""
    
    # Check if any model has multiple games
    multiple_games = any(len(stats["games"]) > 1 for stats in model_stats.values())
    
    # Build dynamic header based on whether we have multiple games
    if multiple_games:
        header = r"""\begin{table}[h]
\centering
\caption{LemonadeBench v0.5 - Business Efficiency Analysis}
\label{tab:lemonadebench_v05_efficiency}
\begin{tabular}{|l|r|r|r|r|r|r|r|r|r|r|r|}
\hline
\textbf{Model} & \textbf{Games} & \textbf{Avg Profit (\$)} & \textbf{Std Dev} & \textbf{Avg Tools} & \textbf{Avg Cost} & \textbf{Avg Time (s)} & \textbf{Prices} & \textbf{Avg Buy Price} & \textbf{Expired (\$)} & \textbf{Lost Profit (\$)} & \textbf{Excess (\$)} \\
\hline
"""
    else:
        header = r"""\begin{table}[h]
\centering
\caption{LemonadeBench v0.5 - Business Efficiency Analysis}
\label{tab:lemonadebench_v05_efficiency}
\begin{tabular}{|l|r|r|r|r|r|r|r|r|r|}
\hline
\textbf{Model} & \textbf{Profit (\$)} & \textbf{Tools} & \textbf{Cost (\$)} & \textbf{Time (s)} & \textbf{Prices} & \textbf{Avg Buy Price} & \textbf{Expired (\$)} & \textbf{Lost Profit (\$)} & \textbf{Excess (\$)} \\
\hline
"""
    
    latex = header
    
    for model, stats in sorted(model_stats.items()):
        if stats["games"]:
            # Calculate business efficiency metrics
            efficiency_metrics = calculate_business_metrics(model, stats, data)
            
            if multiple_games:
                profits = [g["total_profit"] for g in stats["games"]]
                avg_profit = statistics.mean(profits)
                std_profit = statistics.stdev(profits) if len(profits) > 1 else 0
                avg_tools = stats["total_interactions"] / len(stats["games"])
                avg_cost = stats["total_cost"] / len(stats["games"])
                avg_time = sum(g["duration_seconds"] for g in stats["games"]) / len(stats["games"])
                
                latex += f"{model} & "
                latex += f"{len(stats['games'])} & "
                latex += f"{avg_profit:.2f} & "
                latex += f"{std_profit:.2f} & "
                latex += f"{avg_tools:.1f} & "
                latex += f"{avg_cost:.4f} & "
                latex += f"{avg_time:.1f} & "
            else:
                profit = stats["games"][0]["total_profit"]
                tools = stats["total_interactions"]
                cost = stats["total_cost"]
                time_taken = stats["games"][0]["duration_seconds"]
                
                latex += f"{model} & "
                latex += f"{profit:.2f} & "
                latex += f"{tools} & "
                latex += f"{cost:.4f} & "
                latex += f"{time_taken:.1f} & "
            
            # Add business efficiency metrics (same for both formats)
            latex += f"{efficiency_metrics['distinct_prices']} & "
            latex += f"{efficiency_metrics['avg_purchase_price']:.3f} & "
            latex += f"{efficiency_metrics['expired_value']:.2f} & "
            latex += f"{efficiency_metrics['lost_customer_profit']:.2f} & "
            latex += f"{efficiency_metrics['excess_stock_value']:.2f} \\\\"
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
    """Generate average profit trajectory plot."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    colors = {
        "gpt-4.1-nano": "#FF6B6B",
        "gpt-4.1-mini": "#4ECDC4",
        "gpt-4.1": "#45B7D1",
        "o3": "#96CEB4",
        "o4-mini": "#DDA0DD",
    }
    
    # Group games by model
    model_trajectories = {}
    max_days = 0
    
    for game_data in data["games"]:
        model = game_data["model"]
        if model not in model_trajectories:
            model_trajectories[model] = []
        
        # Extract daily cash history from game states
        cash_history = []
        for day_data in game_data.get("days", []):
            if "game_state_after" in day_data:
                cash_history.append(day_data["game_state_after"]["cash"])
        
        if cash_history:
            starting_cash = game_data["parameters"]["starting_cash"]
            profit_history = [cash - starting_cash for cash in cash_history]
            model_trajectories[model].append(profit_history)
            max_days = max(max_days, len(profit_history))
    
    # Calculate and plot average trajectory for each model
    for model, trajectories in model_trajectories.items():
        if not trajectories:
            continue
            
        # Calculate average trajectory
        avg_trajectory = []
        for day in range(max_days):
            day_profits = [traj[day] for traj in trajectories if day < len(traj)]
            if day_profits:
                avg_trajectory.append(statistics.mean(day_profits))
            else:
                break  # No more data for this day
        
        if avg_trajectory:
            days = list(range(len(avg_trajectory)))
            
            # Plot average trajectory
            plt.plot(
                days,
                avg_trajectory,
                color=colors.get(model, "#808080"),
                label=f"{model} (avg of {len(trajectories)} games)" if len(trajectories) > 1 else model,
                linewidth=3,
                marker='o',
                markersize=4,
                alpha=0.9
            )
            
            # Add confidence interval if multiple games
            if len(trajectories) > 1:
                std_trajectory = []
                for day in range(len(avg_trajectory)):
                    day_profits = [traj[day] for traj in trajectories if day < len(traj)]
                    if len(day_profits) > 1:
                        std_trajectory.append(statistics.stdev(day_profits))
                    else:
                        std_trajectory.append(0)
                
                upper_bound = [avg + std for avg, std in zip(avg_trajectory, std_trajectory)]
                lower_bound = [avg - std for avg, std in zip(avg_trajectory, std_trajectory)]
                
                plt.fill_between(
                    days,
                    lower_bound,
                    upper_bound,
                    color=colors.get(model, "#808080"),
                    alpha=0.2
                )
    
    plt.xlabel("Day", fontsize=14)
    plt.ylabel("Cumulative Profit ($)", fontsize=14)
    plt.title("LemonadeBench v0.5: Average Profit Trajectories by Model", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    
    plt.legend(loc="upper left", framealpha=0.9, fontsize=12)
    
    plot_file = output_path / "average_profit_trajectories.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Average profit trajectory plot saved to: {plot_file}")




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
