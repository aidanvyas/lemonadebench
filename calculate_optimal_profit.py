#!/usr/bin/env python3
"""Calculate theoretical optimal profit for LemonadeBench."""

import json
from typing import Dict, Any

# Hourly demand multipliers from business_game.py
HOURLY_MULTIPLIERS = {
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

def calculate_average_supply_cost_per_lemonade(data: Dict[str, Any]) -> float:
    """Calculate average supply cost per lemonade from comprehensive data."""
    total_cost = 0
    total_days = 0
    
    for game in data["games"]:
        for day_data in game.get("days", []):
            supply_costs = day_data.get("game_state_before", {}).get("supply_costs", {})
            if supply_costs:
                # 1 lemonade = 1 cup + 1 lemon + 1 sugar + 1 water
                day_cost = supply_costs.get("cups", 0) + supply_costs.get("lemons", 0) + \
                          supply_costs.get("sugar", 0) + supply_costs.get("water", 0)
                total_cost += day_cost
                total_days += 1
    
    return total_cost / total_days if total_days > 0 else 0.38  # fallback

def calculate_optimal_price(avg_supply_cost: float) -> float:
    """Calculate profit-maximizing price.
    
    Profit = (50 - 10p) * (p - cost) - 5
    Derivative: 50 - 10p - 10p + 10*cost = 0
    Optimal p = 2.5 + 0.5 * cost
    """
    return 2.5 + 0.5 * avg_supply_cost

def calculate_hourly_optimal_profit(price: float, supply_cost: float, hour: int) -> float:
    """Calculate optimal profit for one hour."""
    if HOURLY_MULTIPLIERS[hour] == 0.0:
        return 0.0  # Closed hours
    
    # Base demand: Q = 50 - 10p
    base_demand = 50 - 10 * price
    
    # Actual demand with time multiplier
    actual_demand = base_demand * HOURLY_MULTIPLIERS[hour]
    
    # Profit = customers * margin - operating_cost
    margin_per_customer = price - supply_cost
    revenue = actual_demand * price
    supply_costs = actual_demand * supply_cost
    operating_cost = 5.0  # $5/hour
    
    profit = revenue - supply_costs - operating_cost
    
    return max(0, profit)  # Only open if profitable

def calculate_daily_optimal_profit(avg_supply_cost: float) -> Dict[str, Any]:
    """Calculate optimal profit for one day."""
    optimal_price = calculate_optimal_price(avg_supply_cost)
    
    hourly_breakdown = {}
    total_daily_profit = 0
    total_customers = 0
    total_revenue = 0
    total_supply_costs = 0
    total_operating_costs = 0
    operating_hours = 0
    
    for hour in range(24):
        if HOURLY_MULTIPLIERS[hour] > 0:
            profit = calculate_hourly_optimal_profit(optimal_price, avg_supply_cost, hour)
            
            if profit > 0:  # Only include profitable hours
                base_demand = 50 - 10 * optimal_price
                customers = base_demand * HOURLY_MULTIPLIERS[hour]
                revenue = customers * optimal_price
                supply_cost = customers * avg_supply_cost
                
                hourly_breakdown[hour] = {
                    "customers": customers,
                    "revenue": revenue,
                    "supply_cost": supply_cost,
                    "operating_cost": 5.0,
                    "profit": profit
                }
                
                total_customers += customers
                total_revenue += revenue
                total_supply_costs += supply_cost
                total_operating_costs += 5.0
                total_daily_profit += profit
                operating_hours += 1
    
    return {
        "optimal_price": optimal_price,
        "avg_supply_cost": avg_supply_cost,
        "total_daily_profit": total_daily_profit,
        "total_customers": total_customers,
        "total_revenue": total_revenue,
        "total_supply_costs": total_supply_costs,
        "total_operating_costs": total_operating_costs,
        "operating_hours": operating_hours,
        "hourly_breakdown": hourly_breakdown
    }

def main():
    """Calculate optimal profit for the test run."""
    
    # Load the comprehensive data
    filename = "results/json/gpt-4.1-nano_1games_5days_v05_20250711_011015_full.json"
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Calculate average supply cost
    avg_supply_cost = calculate_average_supply_cost_per_lemonade(data)
    print(f"Average supply cost per lemonade: ${avg_supply_cost:.4f}")
    
    # Calculate optimal daily profit
    optimal_day = calculate_daily_optimal_profit(avg_supply_cost)
    
    print(f"\n{'='*60}")
    print("THEORETICAL OPTIMAL DAILY PERFORMANCE")
    print(f"{'='*60}")
    print(f"Optimal price: ${optimal_day['optimal_price']:.2f}")
    print(f"Operating hours: {optimal_day['operating_hours']}")
    print(f"Total customers: {optimal_day['total_customers']:.1f}")
    print(f"Total revenue: ${optimal_day['total_revenue']:.2f}")
    print(f"Total supply costs: ${optimal_day['total_supply_costs']:.2f}")
    print(f"Total operating costs: ${optimal_day['total_operating_costs']:.2f}")
    print(f"DAILY OPTIMAL PROFIT: ${optimal_day['total_daily_profit']:.2f}")
    
    print(f"\n{'='*60}")
    print("HOURLY BREAKDOWN")
    print(f"{'='*60}")
    print(f"{'Hour':<4} {'Customers':<9} {'Revenue':<8} {'Profit':<8}")
    print("-" * 35)
    
    for hour in sorted(optimal_day['hourly_breakdown'].keys()):
        hb = optimal_day['hourly_breakdown'][hour]
        print(f"{hour:2d}   {hb['customers']:7.1f}   ${hb['revenue']:6.2f}   ${hb['profit']:6.2f}")
    
    # Calculate 5-day comparison
    total_5_day_optimal = optimal_day['total_daily_profit'] * 5
    actual_profit = 115.48  # From our test
    
    print(f"\n{'='*60}")
    print("5-DAY COMPARISON")
    print(f"{'='*60}")
    print(f"Theoretical optimal (5 days): ${total_5_day_optimal:.2f}")
    print(f"Actual profit (5 days): ${actual_profit:.2f}")
    print(f"Efficiency: {(actual_profit / total_5_day_optimal) * 100:.1f}%")
    print(f"Total profit gap: ${total_5_day_optimal - actual_profit:.2f}")

if __name__ == "__main__":
    main()