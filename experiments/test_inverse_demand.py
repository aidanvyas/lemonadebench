#!/usr/bin/env python3
"""Test inverse demand scenario where optimal is $1 but we start at $2."""

import logging
from ..src.lemonade_stand.simple_game import SimpleLemonadeGame
from ..src.lemonade_stand.responses_ai_player import ResponsesAIPlayer

logging.basicConfig(level=logging.INFO)


class InverseDemandGame(SimpleLemonadeGame):
    """Lemonade game with demand function where $1 is optimal."""
    
    def __init__(self, days: int = 100):
        super().__init__(days)
        self.suggested_starting_price = 2.00  # Start above optimal
        
    def calculate_demand(self, price: float) -> int:
        """Calculate demand based on price.
        
        Uses demand function: Q = 50 - 25p
        This gives optimal price at $1.00 (25 customers, $25 revenue)
        """
        if price < 0:
            return 0
        customers = 50 - 25 * price
        customers = max(0, int(customers))
        return customers
    
    def get_initial_prompt(self) -> str:
        """Get initial system prompt with modified suggestion."""
        base_prompt = (
            "You run a lemonade stand. Your goal is to maximize total profit over 100 days.\n\n"
            "Each day:\n"
            "- You must set a price for lemonade (per cup)\n"
            "- The number of customers depends on your price\n"
            "- Lower prices attract more customers, higher prices mean fewer customers\n"
            "- Your profit = price × number of customers\n\n"
        )
        
        if self._use_suggested_price:
            base_prompt += f"Suggested starting price: ${self.suggested_starting_price:.2f}\n\n"
        
        if self._use_exploration_hint:
            base_prompt += "Try different prices to discover which price maximizes your daily profit.\n\n"
            
        base_prompt += "Set your price for Day 1."
        
        return base_prompt


def test_inverse_demand():
    """Test with inverse demand where we start high and optimal is low."""
    print("="*60)
    print("INVERSE DEMAND TEST")
    print("Demand: Q = 50 - 25p (optimal at $1.00)")
    print("Starting at $2.00 (above optimal)")
    print("="*60)
    
    # Test configuration
    game = InverseDemandGame(days=30)
    game._use_suggested_price = True
    game._use_exploration_hint = True
    
    player = ResponsesAIPlayer(
        model_name="gpt-4.1-nano",
        include_calculator=True
    )
    
    print(f"\nSystem Prompt:\n{player.get_system_prompt(game)}\n")
    
    # Run game
    total_profit = 0
    prices = []
    
    for day in range(1, 31):
        price = player.make_decision(game)
        result = game.play_turn(price)
        prices.append(price)
        total_profit += result['profit']
        
        if day <= 5 or day > 25:  # Show first 5 and last 5 days
            print(f"Day {day}: Price=${price:.2f}, Customers={result['customers']}, Profit=${result['profit']:.2f}")
        elif day == 6:
            print("...")
    
    # Analysis
    print(f"\nTotal profit: ${total_profit:.2f}")
    print(f"Optimal daily profit at $1.00: $25.00")
    print(f"Actual average daily profit: ${total_profit/30:.2f}")
    
    # Check if model discovered optimal
    optimal_days = sum(1 for p in prices if abs(p - 1.00) < 0.01)
    print(f"\nDays at optimal price ($1.00): {optimal_days}/30")
    
    unique_prices = sorted(set(prices))
    print(f"Unique prices tried: {unique_prices}")
    
    # Token usage
    print(f"\nTotal tokens used: {player.total_token_usage['total_tokens']:,}")
    print(f"Tool calls: {player.tool_call_count}")
    print(f"Has conversation memory: {player.previous_response_id is not None}")


if __name__ == "__main__":
    test_inverse_demand()