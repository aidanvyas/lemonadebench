#!/usr/bin/env python3
"""Run four lemonade stand tests with multiple runs and detailed timing."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.lemonade_stand.responses_ai_player import ResponsesAIPlayer
from src.lemonade_stand.simple_game import SimpleLemonadeGame


class InverseDemandGame2(SimpleLemonadeGame):
    """Lemonade game with Q = 100 - 50p (optimal at $1.00)."""

    def __init__(self, days: int = 100):
        super().__init__(days)
        self.suggested_starting_price = 1.50  # Start above optimal

    def calculate_demand(self, price: float) -> int:
        """Calculate demand based on price.

        Uses demand function: Q = 100 - 50p
        This gives optimal price at $1.00 (50 customers, $50 revenue)
        At $1.50: 25 customers, $37.50 revenue
        """
        if price < 0:
            return 0
        customers = 100 - 50 * price
        customers = max(0, int(customers))
        return customers


def run_test(test_name: str, game: SimpleLemonadeGame, use_suggested: bool,
             use_exploration: bool, days: int = 30, model: str = "gpt-4.1-nano") -> dict:
    """Run a single test condition."""
    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"{'='*70}")

    # Configure game
    game._use_suggested_price = use_suggested
    game._use_exploration_hint = use_exploration

    # Create player (always with memory via previous_response_id)
    player = ResponsesAIPlayer(
        model_name=model,
        include_calculator=True
    )

    # Show starting conditions
    if hasattr(game, 'suggested_starting_price'):
        print(f"Starting price: ${game.suggested_starting_price:.2f}")

    # Run game with timing
    prices = []
    profits = []
    start_time = time.time()

    for day in range(1, days + 1):
        price = player.make_decision(game)
        result = game.play_turn(price)
        prices.append(price)
        profits.append(result['profit'])

        # Show first 3 and last 2 days
        if day <= 3 or day > days - 2:
            print(f"Day {day}: Price=${price:.2f}, Customers={result['customers']}, "
                  f"Profit=${result['profit']:.2f}")
        elif day == 4:
            print("...")

    end_time = time.time()
    duration = end_time - start_time

    # Calculate results
    total_profit = sum(profits)
    unique_prices = sorted(set(prices))
    avg_price = sum(prices) / len(prices)

    # Find optimal for this game
    if isinstance(game, InverseDemandGame2):
        optimal_price = 1.00
        optimal_profit = 50.00
    else:
        optimal_price = 2.00
        optimal_profit = 100.00

    days_at_optimal = sum(1 for p in prices if abs(p - optimal_price) < 0.01)

    # Calculate tool call breakdown
    tool_call_breakdown = {}
    for day_record in player.tool_call_history:
        for tool in day_record['tools']:
            tool_call_breakdown[tool] = tool_call_breakdown.get(tool, 0) + 1

    # Calculate cost
    cost_info = player.calculate_cost()

    # Results summary
    print("\nResults:")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"Average price: ${avg_price:.2f}")
    print(f"Unique prices tried: {unique_prices}")
    print(f"Days at optimal (${optimal_price:.2f}): {days_at_optimal}/{days}")
    print(f"Total tokens: {player.total_token_usage['total_tokens']:,}")
    print(f"Tool calls: {tool_call_breakdown}")
    print(f"Estimated cost: ${cost_info.get('total_cost', 0):.4f}")
    print(f"Duration: {duration:.1f}s")

    return {
        'test_name': test_name,
        'total_profit': total_profit,
        'average_price': avg_price,
        'unique_prices': unique_prices,
        'days_at_optimal': days_at_optimal,
        'optimal_price': optimal_price,
        'optimal_daily_profit': optimal_profit,
        'prices': prices,
        'profits': profits,
        'token_usage': player.total_token_usage,
        'tool_calls': player.tool_call_count,
        'tool_call_breakdown': tool_call_breakdown,
        'tool_call_history': player.tool_call_history,
        'cost_info': cost_info,
        'duration_seconds': duration
    }


def main():
    """Run all four test conditions with multiple runs."""
    parser = argparse.ArgumentParser(description="Run four lemonade stand tests")
    parser.add_argument("--days", type=int, default=30, help="Days per game")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--runs", type=int, default=1, help="Runs per test")
    args = parser.parse_args()
    
    print("LEMONADEBENCH - Four Test Conditions")
    print(f"Model: {args.model}")
    print(f"Days: {args.days}")
    print(f"Runs per test: {args.runs}")
    print("Memory: Enabled (previous_response_id)")
    print()

    all_results = []
    experiment_start = time.time()

    # Test configurations
    test_configs = [
        ("Suggested Price", True, False, SimpleLemonadeGame),
        ("No Guidance", False, False, SimpleLemonadeGame),
        ("Exploration Hint", False, True, SimpleLemonadeGame),
        ("Inverse Demand (100-50p)", True, True, InverseDemandGame2),
    ]
    
    # Run each test with multiple runs
    for test_name, use_suggested, use_exploration, game_class in test_configs:
        test_start = time.time()
        test_results = []
        
        print(f"\n{'='*70}")
        print(f"STARTING TEST: {test_name}")
        print(f"{'='*70}")
        
        for run_num in range(1, args.runs + 1):
            print(f"\n--- Run {run_num}/{args.runs} ---")
            game = game_class(days=args.days)
            
            result = run_test(
                test_name,
                game,
                use_suggested=use_suggested,
                use_exploration=use_exploration,
                days=args.days,
                model=args.model
            )
            result['run_number'] = run_num
            test_results.append(result)
        
        test_duration = time.time() - test_start
        
        # Aggregate results for this test
        test_summary = {
            'test_name': test_name,
            'runs': test_results,
            'test_duration_seconds': test_duration,
            'average_profit': sum(r['total_profit'] for r in test_results) / len(test_results),
            'average_efficiency': sum(r['total_profit'] / (r['optimal_daily_profit'] * args.days) for r in test_results) / len(test_results) * 100,
        }
        all_results.append(test_summary)

    experiment_duration = time.time() - experiment_start
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*70}\n")

    total_tokens = 0
    total_cost = 0
    
    for test in all_results:
        print(f"{test['test_name']}:")
        print(f"  Runs: {len(test['runs'])}")
        print(f"  Average profit: ${test['average_profit']:.2f}")
        print(f"  Average efficiency: {test['average_efficiency']:.1f}%")
        print(f"  Test duration: {test['test_duration_seconds']:.1f}s")
        
        # Show per-run details with timing
        for run in test['runs']:
            tokens = run['token_usage']['total_tokens']
            cost = run['cost_info'].get('total_cost', 0)
            total_tokens += tokens
            total_cost += cost
            print(f"    Run {run['run_number']}: ${run['total_profit']:.2f} profit, "
                  f"{len(run['unique_prices'])} unique prices, "
                  f"{run['duration_seconds']:.1f}s, {tokens:,} tokens")
        print()

    print(f"\nTotal experiment duration: {experiment_duration:.1f}s")
    print(f"Total tokens across all runs: {total_tokens:,}")
    print(f"Total estimated cost: ${total_cost:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/four_tests_multi_{timestamp}.json"
    Path("results").mkdir(exist_ok=True)

    with open(filename, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'days': args.days,
            'runs_per_test': args.runs,
            'experiment_duration_seconds': experiment_duration,
            'tests': all_results
        }, f, indent=2)

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()