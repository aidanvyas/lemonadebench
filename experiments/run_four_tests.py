#!/usr/bin/env python3
"""Run the four main test conditions for the paper."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import json
from datetime import datetime
from src.lemonade_stand.simple_game import SimpleLemonadeGame
from src.lemonade_stand.responses_ai_player import ResponsesAIPlayer

# Set logging to WARNING to reduce noise
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


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
             use_exploration: bool, days: int = 30) -> dict:
    """Run a single test condition."""
    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"{'='*70}")
    
    # Configure game
    game._use_suggested_price = use_suggested
    game._use_exploration_hint = use_exploration
    
    # Create player (always with memory via previous_response_id)
    player = ResponsesAIPlayer(
        model_name="gpt-4.1-nano",
        include_calculator=True
    )
    
    # Show starting conditions
    if hasattr(game, 'suggested_starting_price'):
        print(f"Starting price: ${game.suggested_starting_price:.2f}")
    
    # Run game
    prices = []
    profits = []
    start_time = datetime.now()
    
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
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
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
    
    # Results summary
    print(f"\nResults:")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"Average price: ${avg_price:.2f}")
    print(f"Unique prices tried: {unique_prices}")
    print(f"Days at optimal (${optimal_price:.2f}): {days_at_optimal}/{days}")
    print(f"Total tokens: {player.total_token_usage['total_tokens']:,}")
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
        'duration_seconds': duration
    }


def main():
    """Run all four test conditions."""
    print("LEMONADEBENCH - Four Test Conditions")
    print("Model: gpt-4.1-nano")
    print("Days: 30")
    print("Memory: Enabled (previous_response_id)")
    
    results = []
    
    # Test 1: Suggested price ($1.00)
    game1 = SimpleLemonadeGame(days=30)
    results.append(run_test(
        "Suggested Price",
        game1,
        use_suggested=True,
        use_exploration=False
    ))
    
    # Test 2: No guidance
    game2 = SimpleLemonadeGame(days=30)
    results.append(run_test(
        "No Guidance",
        game2,
        use_suggested=False,
        use_exploration=False
    ))
    
    # Test 3: Exploration hint
    game3 = SimpleLemonadeGame(days=30)
    results.append(run_test(
        "Exploration Hint",
        game3,
        use_suggested=False,
        use_exploration=True
    ))
    
    # Test 4: Inverse demand (Q = 100 - 50p, start at $1.50)
    game4 = InverseDemandGame2(days=30)
    results.append(run_test(
        "Inverse Demand (100-50p)",
        game4,
        use_suggested=True,
        use_exploration=True
    ))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*70}\n")
    
    total_tokens = 0
    for r in results:
        print(f"{r['test_name']}:")
        print(f"  Total profit: ${r['total_profit']:.2f} "
              f"(optimal: ${r['optimal_daily_profit'] * 30:.2f})")
        print(f"  Efficiency: {r['total_profit'] / (r['optimal_daily_profit'] * 30) * 100:.1f}%")
        print(f"  Unique prices: {len(r['unique_prices'])}")
        print(f"  Days at optimal: {r['days_at_optimal']}/30")
        print(f"  Tokens used: {r['token_usage']['total_tokens']:,}")
        print()
        total_tokens += r['token_usage']['total_tokens']
    
    print(f"Total tokens across all tests: {total_tokens:,}")
    print(f"Estimated cost (4.1-nano): ${total_tokens * 0.15 / 1_000_000:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/four_tests_{timestamp}.json"
    Path("results").mkdir(exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model': 'gpt-4.1-nano',
            'days': 30,
            'tests': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()