#!/usr/bin/env python3
"""Analyze v0.5 benchmark results with FINAL efficiency metrics per user specifications."""
import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt

# Game constants
REFERENCE_PRICE = 2.69  # User-specified reference price for pricing loss calculations
BASE_DEMAND = 50
PRICE_SENSITIVITY = 10
INGREDIENT_COST = 0.37  # Cost per lemonade (sum of base costs)
OPERATING_COST_PER_HOUR = 5

# Base supply costs (from business_game.py)
BASE_COSTS = {
    "cups": 0.05,
    "lemons": 0.20,
    "sugar": 0.10,
    "water": 0.02
}

# Hour multipliers (ACTUAL from business_game.py)
HOUR_MULTIPLIERS = {
    0: 0.0,   # 12-1am: Closed
    1: 0.0,   # 1-2am: Closed
    2: 0.0,   # 2-3am: Closed
    3: 0.0,   # 3-4am: Closed
    4: 0.0,   # 4-5am: Closed
    5: 0.0,   # 5-6am: Closed
    6: 0.3,   # 6-7am: Early morning (30% of base)
    7: 0.5,   # 7-8am: Morning commute
    8: 0.7,   # 8-9am: Morning
    9: 0.8,   # 9-10am: Mid-morning
    10: 1.0,  # 10-11am: Late morning (100% base)
    11: 1.2,  # 11am-12pm: Pre-lunch
    12: 1.5,  # 12-1pm: Lunch peak (150% of base)
    13: 1.3,  # 1-2pm: Post-lunch
    14: 0.9,  # 2-3pm: Afternoon
    15: 0.8,  # 3-4pm: Mid-afternoon
    16: 0.9,  # 4-5pm: Late afternoon
    17: 1.1,  # 5-6pm: Evening commute
    18: 1.0,  # 6-7pm: Early evening
    19: 0.7,  # 7-8pm: Evening
    20: 0.4,  # 8-9pm: Late evening (40% of base)
    21: 0.0,  # 9-10pm: Closed
    22: 0.0,  # 10-11pm: Closed
    23: 0.0,  # 11pm-12am: Closed
}


def calculate_business_metrics_final(model: str, stats: dict[str, Any], full_data: list[dict[str, Any]]) -> dict[str, float]:
    """Calculate final business efficiency metrics with user-specified sign conventions."""
    # Find the game data for this model
    game_data = None
    for game in full_data:
        if game.get("model") == model:
            game_data = game
            break
    
    if not game_data:
        return {}
    
    # Initialize metrics
    total_purchasing_efficiency = 0
    total_expired_loss = 0
    total_excess_loss = 0
    total_stockout_loss = 0
    total_pricing_loss = 0
    total_scheduling_loss = 0
    
    num_games = 1  # For now, assuming 1 game per model
    
    # PURCHASING EFFICIENCY: Compare weighted avg purchase price vs base cost
    # Positive = saved money (bought below base price)
    # Negative = overpaid (bought above base price)
    
    # Get final results first
    final_result = game_data.get("final_results", {})
    
    # Track actual costs paid and quantities
    actual_costs_by_item = {"cups": 0, "lemons": 0, "sugar": 0, "water": 0}
    quantities_by_item = {"cups": 0, "lemons": 0, "sugar": 0, "water": 0}
    
    # Look through all days for purchases and their prices
    for day_data in game_data.get("days", []):
        # Get the morning supply costs for this day
        morning_costs = day_data.get("game_state_before", {}).get("supply_costs", {})
        
        # Find any purchases made this day
        for interaction in day_data.get("interactions", []):
            for tool_exec in interaction.get("tool_executions", []):
                if tool_exec.get("tool") == "order_supplies":
                    result = tool_exec.get("result", {})
                    if result.get("success"):
                        ordered = result.get("ordered", {})
                        # Calculate cost for each item using morning prices
                        for item in ["cups", "lemons", "sugar", "water"]:
                            qty = ordered.get(item, 0)
                            if qty > 0:
                                cost_per_unit = morning_costs.get(item, BASE_COSTS[item])
                                actual_costs_by_item[item] += qty * cost_per_unit
                                quantities_by_item[item] += qty
    
    # Calculate purchasing efficiency
    total_purchasing_efficiency = 0
    for item in ["cups", "lemons", "sugar", "water"]:
        if quantities_by_item[item] > 0:
            base_cost = quantities_by_item[item] * BASE_COSTS[item]
            actual_cost = actual_costs_by_item[item]
            # Positive if saved money, negative if overpaid
            total_purchasing_efficiency += base_cost - actual_cost
    
    # EXPIRED LOSSES: Use game engine's expired_items tracking (like original)
    total_expired_loss = 0
    for day_data in game_data.get("days", []):
        expired_items = day_data.get("game_state_before", {}).get("expired_items", {})
        # Use base costs for expired items
        for item, quantity in expired_items.items():
            if item in BASE_COSTS:
                total_expired_loss += BASE_COSTS[item] * quantity
    
    # EXCESS LOSSES (negative = loss) 
    ending_value = final_result.get("inventory_value", 0)
    total_excess_loss = ending_value
    
    # STOCKOUT LOSSES: Lost profit from turning customers away
    # Use actual prices for each day's lost sales
    total_stockout_loss = 0
    
    for day_data in game_data.get("days", []):
        day_result = day_data.get("game_state_after", {}).get("day_result", {})
        if day_result:
            customers_lost = day_result.get("customers_lost", 0)
            if customers_lost > 0:
                # Use the actual price for this day
                actual_price = day_data.get("game_state_after", {}).get("price", 0)
                if actual_price > 0:
                    # Profit margin at actual price
                    profit_margin = actual_price - INGREDIENT_COST
                    total_stockout_loss += customers_lost * profit_margin
    
    # PRICING LOSSES: Counterfactual with demand elasticity
    # "If inventory and hours were the same, what would profit be if you changed prices to $2.69?"
    for day_data in game_data.get("days", []):
        day_result = day_data.get("game_state_after", {}).get("day_result", {})
        if day_result:
            actual_price = day_data.get("game_state_after", {}).get("price", 0)
            start_hour = day_result.get("open_hour", 0)
            end_hour = day_result.get("close_hour", 0)
            
            # Get actual inventory available (can_make at start of day)
            can_make = day_data.get("game_state_before", {}).get("inventory", {}).get("can_make", 0)
            
            if actual_price > 0 and start_hour < end_hour:
                # ACTUAL SCENARIO (what really happened)
                actual_total_demand = 0
                for hour in range(start_hour, end_hour):
                    if hour in HOUR_MULTIPLIERS:
                        hour_demand = (BASE_DEMAND - PRICE_SENSITIVITY * actual_price) * HOUR_MULTIPLIERS[hour]
                        actual_total_demand += max(0, hour_demand)
                
                # Use actual customers served 
                actual_customers_served = day_result.get("customers_served", 0)
                actual_sales = actual_customers_served
                actual_revenue = actual_sales * actual_price
                actual_ingredient_costs = actual_sales * INGREDIENT_COST
                actual_operating_costs = (end_hour - start_hour) * OPERATING_COST_PER_HOUR
                actual_profit = actual_revenue - actual_ingredient_costs - actual_operating_costs
                
                # OPTIMAL PRICING SCENARIO (same supply capacity, same hours, different price)
                optimal_total_demand = 0
                for hour in range(start_hour, end_hour):  # SAME HOURS
                    if hour in HOUR_MULTIPLIERS:
                        hour_demand = (BASE_DEMAND - PRICE_SENSITIVITY * REFERENCE_PRICE) * HOUR_MULTIPLIERS[hour]
                        optimal_total_demand += max(0, hour_demand)
                
                # Use same supply constraint logic
                if actual_total_demand >= actual_customers_served:
                    # Hours were the constraint - pricing change affects demand
                    optimal_sales = min(optimal_total_demand, actual_customers_served * (optimal_total_demand / actual_total_demand))
                else:
                    # Supply was the constraint - pricing change limited by supply
                    optimal_sales = min(optimal_total_demand, actual_customers_served)
                optimal_revenue = optimal_sales * REFERENCE_PRICE
                optimal_ingredient_costs = optimal_sales * INGREDIENT_COST
                optimal_operating_costs = (end_hour - start_hour) * OPERATING_COST_PER_HOUR  # SAME OPERATING HOURS
                optimal_profit = optimal_revenue - optimal_ingredient_costs - optimal_operating_costs
                
                # Pricing loss: actual vs optimal (negative = loss from suboptimal pricing)
                day_pricing_loss = actual_profit - optimal_profit
                total_pricing_loss += day_pricing_loss
    
    # SCHEDULING LOSSES: Counterfactual with global optimal hours
    # "Given my price and inventory, how much am I losing by not using globally optimal hours (6am-8pm)?"
    OPTIMAL_START_HOUR = 6  # 6am
    OPTIMAL_END_HOUR = 21   # 8pm (hour 20 is last operable hour)
    
    for day_data in game_data.get("days", []):
        day_result = day_data.get("game_state_after", {}).get("day_result", {})
        if day_result:
            actual_price = day_data.get("game_state_after", {}).get("price", 0)
            start_hour = day_result.get("open_hour", 0)
            end_hour = day_result.get("close_hour", 0)
            
            # Get actual inventory available
            can_make = day_data.get("game_state_before", {}).get("inventory", {}).get("can_make", 0)
            
            if actual_price > 0 and start_hour < end_hour:
                # ACTUAL SCENARIO (what really happened)
                actual_total_demand = 0
                for hour in range(start_hour, end_hour):
                    if hour in HOUR_MULTIPLIERS:
                        hour_demand = (BASE_DEMAND - PRICE_SENSITIVITY * actual_price) * HOUR_MULTIPLIERS[hour]
                        actual_total_demand += max(0, hour_demand)
                
                # Use actual customers served as the achieved constraint
                actual_customers_served = day_result.get("customers_served", 0)
                actual_sales = actual_customers_served  # What they actually achieved
                actual_revenue = actual_sales * actual_price
                actual_ingredient_costs = actual_sales * INGREDIENT_COST
                actual_operating_costs = (end_hour - start_hour) * OPERATING_COST_PER_HOUR
                actual_profit = actual_revenue - actual_ingredient_costs - actual_operating_costs
                
                # OPTIMAL SCHEDULING SCENARIO (same supply capacity, same price, global optimal hours)
                optimal_total_demand = 0
                for hour in range(OPTIMAL_START_HOUR, OPTIMAL_END_HOUR):  # 6am-8pm
                    if hour in HOUR_MULTIPLIERS:
                        hour_demand = (BASE_DEMAND - PRICE_SENSITIVITY * actual_price) * HOUR_MULTIPLIERS[hour]  # SAME PRICE
                        optimal_total_demand += max(0, hour_demand)
                
                # Key insight: If they were constrained by hours (demand > served), extending helps
                # If they were constrained by supply (demand <= served), extending doesn't help
                if actual_total_demand >= actual_customers_served:
                    # Hours were the constraint - extending hours allows more sales
                    optimal_sales = min(optimal_total_demand, actual_customers_served * (optimal_total_demand / actual_total_demand))
                else:
                    # Supply was the constraint - extending hours doesn't help
                    optimal_sales = actual_customers_served
                optimal_revenue = optimal_sales * actual_price  # SAME PRICE
                optimal_ingredient_costs = optimal_sales * INGREDIENT_COST
                optimal_operating_costs = (OPTIMAL_END_HOUR - OPTIMAL_START_HOUR) * OPERATING_COST_PER_HOUR  # 15 hours
                optimal_profit = optimal_revenue - optimal_ingredient_costs - optimal_operating_costs
                
                # Scheduling loss: actual vs optimal 
                # Positive = doing better than optimal (e.g., selling out efficiently with fewer hours)
                # Negative = doing worse than optimal (e.g., inefficient hours, 24/7 operation)
                day_scheduling_loss = actual_profit - optimal_profit
                total_scheduling_loss += day_scheduling_loss
    
    # Return metrics with user-specified sign conventions
    return {
        "purchasing": total_purchasing_efficiency / num_games if num_games > 0 else 0,  # + = saved
        "expired": -total_expired_loss / num_games if num_games > 0 else 0,            # - = loss
        "excess": -total_excess_loss / num_games if num_games > 0 else 0,              # - = loss
        "scheduling": total_scheduling_loss / num_games if num_games > 0 else 0,       # +/- 
        "pricing": total_pricing_loss / num_games if num_games > 0 else 0,             # - = loss (always)
        "stockout": -total_stockout_loss / num_games if num_games > 0 else 0,          # - = loss
    }

def generate_comprehensive_plots(data: dict[str, Any], output_dir: str) -> None:
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

def generate_computational_requirements_table(model_stats: dict[str, Any], output_file: str, data: list[dict[str, Any]]) -> None:
    """Generate LaTeX table with computational requirements and detailed tool usage."""
    
    header = r"""\begin{table}[h]
\centering
\caption{LemonadeBench v0.5 - Computational Requirements and Tool Usage}
\label{tab:computational_requirements}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|r|r|r|r|r|r|r|r|r|r|r|}
\hline
\textbf{Model} & \textbf{Profit (\$)} & \textbf{Total} & \textbf{Check} & \textbf{Check} & \textbf{Order} & \textbf{Set} & \textbf{Set} & \textbf{Historical} & \textbf{Open} & \textbf{Time (s)} & \textbf{API Cost (\$)} \\
 & & \textbf{Calls} & \textbf{Inv.} & \textbf{Prices} & \textbf{Supplies} & \textbf{Price} & \textbf{Hours} & \textbf{Data} & \textbf{Business} & & \\
\hline
"""
    
    latex = header
    
    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        if stats["games"]:
            profits = [g["total_profit"] for g in stats["games"]]
            avg_profit = statistics.mean(profits)
            
            # Get detailed tool usage
            tool_counts = {
                "check_inventory": stats["tool_usage"].get("check_inventory", 0),
                "check_morning_prices": stats["tool_usage"].get("check_morning_prices", 0),
                "order_supplies": stats["tool_usage"].get("order_supplies", 0),
                "set_price": stats["tool_usage"].get("set_price", 0),
                "set_operating_hours": stats["tool_usage"].get("set_operating_hours", 0),
                "get_historical_supply_costs": stats["tool_usage"].get("get_historical_supply_costs", 0),
                "open_for_business": stats["tool_usage"].get("open_for_business", 0),
            }
            
            total_tools = sum(tool_counts.values())
            
            # Get duration and cost
            model_games = [game for game in data if game.get("model") == model]
            avg_time = sum(g.get("duration_seconds", 0) for g in model_games) / len(model_games) if model_games else 0
            avg_cost = stats["total_cost"] / len(stats["games"])
            
            # Format profit with commas
            profit_str = f"{int(avg_profit):,}"
            
            latex += f"{model} & {profit_str} & {total_tools} & "
            latex += f"{tool_counts['check_inventory']} & "
            latex += f"{tool_counts['check_morning_prices']} & "
            latex += f"{tool_counts['order_supplies']} & "
            latex += f"{tool_counts['set_price']} & "
            latex += f"{tool_counts['set_operating_hours']} & "
            latex += f"{tool_counts['get_historical_supply_costs']} & "
            latex += f"{tool_counts['open_for_business']} & "
            latex += f"{avg_time:.1f} & "
            latex += f"{avg_cost:.4f} \\\\\n"
            latex += "\\hline\n"
    
    latex += r"""\end{tabular}
}%
\end{table}
"""
    
    # Write to file
    with open(output_file, "w") as f:
        f.write(latex)
    
    print(f"Computational requirements table saved to: {output_file}")


def generate_comprehensive_latex_table(model_stats: dict[str, Any], output_file: str, data: list[dict[str, Any]]) -> None:
    """Generate LaTeX table with business efficiency metrics."""
    
    header = r"""\begin{table}[h]
\centering
\caption{LemonadeBench v0.5 - Business Efficiency Analysis}
\label{tab:lemonadebench_v05_efficiency}
\begin{tabular}{|l|r|r|r|r|r|r|r|r|r|r|}
\hline
\textbf{Model} & \textbf{Profit (\$)} & \textbf{Purchasing (\$)} & \textbf{Expired (\$)} & \textbf{Excess (\$)} & \textbf{Scheduling (\$)} & \textbf{Pricing (\$)} & \textbf{Stockout (\$)} & \textbf{Tools} & \textbf{Time (s)} & \textbf{Cost (\$)} \\
\hline
"""
    
    latex = header
    
    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        if stats["games"]:
            # Calculate business efficiency metrics
            efficiency = calculate_business_metrics_final(model, stats, data)
            
            profits = [g["total_profit"] for g in stats["games"]]
            avg_profit = statistics.mean(profits)
            avg_tools = stats["total_interactions"] / len(stats["games"])
            avg_cost = stats["total_cost"] / len(stats["games"])
            
            # Get duration from actual game data
            model_games = [game for game in data if game.get("model") == model]
            avg_time = sum(g.get("duration_seconds", 0) for g in model_games) / len(model_games) if model_games else 0
            
            latex += f"{model} & "
            latex += f"{avg_profit:.2f} & "
            latex += f"{efficiency.get('purchasing', 0):.2f} & "
            latex += f"{efficiency.get('expired', 0):.2f} & "
            latex += f"{efficiency.get('excess', 0):.2f} & "
            latex += f"{efficiency.get('scheduling', 0):.2f} & "
            latex += f"{efficiency.get('pricing', 0):.2f} & "
            latex += f"{efficiency.get('stockout', 0):.2f} & "
            latex += f"{avg_tools:.0f} & "
            latex += f"{avg_time:.1f} & "
            latex += f"{avg_cost:.4f} \\\\\n"
            latex += "\\hline\n"
    
    latex += r"""\end{tabular}
\end{table}
"""
    
    # Write to file
    with open(output_file, "w") as f:
        f.write(latex)
    
    print(f"LaTeX table saved to: {output_file}")

def analyze_comprehensive_format(data: list[dict[str, Any]], filename: str):
    """Analyze comprehensive v0.5 format results."""
    print(f"\nAnalyzing: {filename}")
    print("=" * 80)
    
    # Group by model and collect comprehensive stats
    model_stats = {}
    
    for game_data in data:
        model = game_data.get("model", "unknown")
        
        if model not in model_stats:
            model_stats[model] = {
                "games": [],
                "total_tokens": 0,
                "total_cost": 0,
                "total_interactions": 0,
                "tool_usage": {},
                "avg_supply_cost_per_lemonade": INGREDIENT_COST,
            }
        
        stats = model_stats[model]
        
        # Add game data
        final_results = game_data.get("final_results", {})
        game_info = {
            "total_profit": final_results.get("total_profit", 0),
            "total_revenue": final_results.get("total_revenue", 0),
            "days_played": final_results.get("days_played", 0)
        }
        stats["games"].append(game_info)
        
        # Token and cost tracking
        stats["total_tokens"] += game_data.get("total_tokens", 0)
        stats["total_cost"] += game_data.get("total_cost", 0)
        
        # Count tool usage
        total_tools = 0
        for day_data in game_data.get("days", []):
            for interaction in day_data.get("interactions", []):
                for tool_exec in interaction.get("tool_executions", []):
                    tool_name = tool_exec.get("tool", "unknown")
                    total_tools += 1
                    if tool_name not in stats["tool_usage"]:
                        stats["tool_usage"][tool_name] = 0
                    stats["tool_usage"][tool_name] += 1
        
        stats["total_interactions"] += total_tools
    
    # Print model comparison (like original analyze_results.py)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
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
    
    # Calculate and print efficiency metrics
    print("\n\nBusiness Efficiency Breakdown (FINAL):")
    print("=" * 120)
    print("| Model | Profit ($) | Purchasing | Expired | Excess | Scheduling | Pricing | Stockout |")
    print("|-------|------------|------------|---------|--------|------------|---------|----------|")
    
    # Store for LaTeX generation
    efficiency_data = []
    
    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        efficiency = calculate_business_metrics_final(model, stats, data)
        
        if efficiency and stats["games"]:
            profits = [g["total_profit"] for g in stats["games"]]
            avg_profit = statistics.mean(profits)
            print(f"| {model:<12} | {avg_profit:>10.2f} | "
                  f"{efficiency['purchasing']:>+10.2f} | {efficiency['expired']:>7.2f} | "
                  f"{efficiency['excess']:>6.2f} | {efficiency['scheduling']:>+10.2f} | "
                  f"{efficiency['pricing']:>7.2f} | {efficiency['stockout']:>8.2f} |")
            
            efficiency_data.append({
                "model": model,
                "profit": avg_profit,
                **efficiency
            })
    
    # Generate outputs like original analyze_results.py
    base_name = Path(filename).stem.replace("_full", "")
    
    # Generate LaTeX table
    latex_file = f"results/latex/{base_name}_final_efficiency.tex"
    generate_efficiency_latex(efficiency_data, filename)
    
    # Generate comprehensive LaTeX table (efficiency metrics)
    comprehensive_latex_file = f"results/latex/{base_name}_final_analysis.tex"
    generate_comprehensive_latex_table(model_stats, comprehensive_latex_file, data)
    
    # Generate computational requirements table
    computational_file = f"results/latex/{base_name}_computational_requirements.tex"
    generate_computational_requirements_table(model_stats, computational_file, data)
    
    # Generate plots
    plot_dir = f"results/plots/{base_name}"
    generate_comprehensive_plots({"games": data}, plot_dir)

def generate_efficiency_latex(data: list[dict[str, Any]], filename: str):
    """Generate LaTeX table for efficiency metrics."""
    # Create results/latex directory if it doesn't exist
    latex_dir = Path("results/latex")
    latex_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    base_name = Path(filename).stem
    latex_file = latex_dir / f"{base_name}_final_efficiency.tex"
    
    with open(latex_file, "w") as f:
        f.write("% Business Efficiency Breakdown (Final Metrics)\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Business Efficiency Breakdown}\n")
        f.write("\\label{tab:efficiency_breakdown}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lccccccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Profit (\\$)} & \\textbf{Purchasing} & \\textbf{Expired} & ")
        f.write("\\textbf{Excess} & \\textbf{Scheduling} & \\textbf{Pricing} & \\textbf{Stockout} \\\\\n")
        f.write("\\hline\n")
        
        for row in data:
            model = row['model']
            # Format numbers with appropriate signs
            purchasing = f"{row['purchasing']:+.2f}" if abs(row['purchasing']) > 0.01 else "0.00"
            expired = f"{row['expired']:.2f}"
            excess = f"{row['excess']:.2f}"
            scheduling = f"{row['scheduling']:+.2f}" if abs(row['scheduling']) > 0.01 else "0.00"
            pricing = f"{row['pricing']:.2f}"
            stockout = f"{row['stockout']:.2f}"
            
            f.write(f"{model} & {row['profit']:.2f} & {purchasing} & {expired} & ")
            f.write(f"{excess} & {scheduling} & {pricing} & {stockout} \\\\\n")
            f.write("\\hline\n")
        
        f.write("\\end{tabular}\n")
        f.write("}%\n")
        f.write("\\end{table}\n")
    
    print(f"\nLaTeX table saved to: {latex_file}")

def analyze_results():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze v0.5 benchmark results")
    parser.add_argument("--file", help="Specific results file to analyze")
    parser.add_argument("--latest", action="store_true", help="Analyze the latest results file")
    args = parser.parse_args()
    
    results_dir = Path("results/json")
    
    if args.latest:
        # Find the latest full results file
        json_files = list(results_dir.glob("*_full.json"))
        if not json_files:
            print("No results files found!")
            sys.exit(1)
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        filename = str(latest_file)
    elif args.file:
        filename = args.file
    else:
        print("Please specify --file or --latest")
        sys.exit(1)
    
    # Load the results
    with open(filename) as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict) and "games" in data:
        # v0.5 format with metadata
        analyze_comprehensive_format(data["games"], filename)
    elif isinstance(data, list) and len(data) > 0 and "model" in data[0]:
        # Direct list of games
        analyze_comprehensive_format(data, filename)
    else:
        print("Unknown format or empty results")

def main():
    """Entry point."""
    analyze_results()

if __name__ == "__main__":
    main()