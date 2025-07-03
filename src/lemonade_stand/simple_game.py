"""Simplified price-only version of the lemonade stand game."""

import logging

logger = logging.getLogger(__name__)


class SimpleLemonadeGame:
    """Simplified game focusing only on price optimization."""

    def __init__(self, days: int = 100) -> None:
        """Initialize simple game with fixed parameters."""
        self.days = days
        self.current_day = 1

        # Game parameters
        self.suggested_starting_price = 1.00

        # Cost structure (kept for future extensibility)
        self.daily_fixed_cost = 0.0
        self.cost_per_lemonade = 0.0

        # Game state
        self.cash = 100.0  # Starting cash
        self.history: list[dict] = []
        self.game_over = False

        # Prompt control flags (for experiments)
        self._use_suggested_price = False
        self._use_exploration_hint = False

    def calculate_demand(self, price: float) -> int:
        """Calculate demand based on price.

        Uses a simple linear demand function: Q = 100 - 25p

        Demand structure:
        - Starting price: $1.00 → 75 customers ($75 revenue)
        - Optimal price: $2.00 → 50 customers ($100 revenue)
        - Zero demand at: $4.00

        Args:
            price: Price per lemonade in dollars.

        Returns:
            Number of customers at the given price.
        """
        if price < 0:
            # Negative prices not allowed
            return 0

        # Linear demand: Q = 100 - 25p
        customers = 100 - 25 * price

        # Ensure non-negative demand
        customers = max(0, int(customers))

        return customers

    def play_turn(self, price: float) -> dict:
        """Play one turn with the given price.

        Args:
            price: Price to charge per lemonade.

        Returns:
            Dictionary containing turn results including profit and customers.

        Raises:
            ValueError: If game is already over.
        """
        if self.game_over:
            raise ValueError("Game is already over")

        # Calculate demand
        customers = self.calculate_demand(price)

        # Calculate financials
        revenue = customers * price
        variable_costs = customers * self.cost_per_lemonade
        total_costs = self.daily_fixed_cost + variable_costs
        base_profit = revenue - total_costs

        # Add random variation (-10% to +10%)
        # random_factor = random.uniform(-10, 10)
        # profit = base_profit * (1 + random_factor / 100)
        profit = base_profit  # Randomness disabled for testing

        # Update cash
        self.cash += profit

        # Record results (round financial values for display)
        result = {
            "day": self.current_day,
            "price": round(price, 2),
            "customers": customers,
            "revenue": round(revenue, 2),
            "costs": round(total_costs, 2),
            "profit": round(profit, 2),
            "cash": round(self.cash, 2),
        }

        self.history.append(result)

        # Check game over conditions
        self.current_day += 1
        if self.current_day > self.days or self.cash < 0:
            self.game_over = True

        return result

    def get_state(self) -> dict:
        """Get current game state.

        Returns:
            Dictionary containing current day, cash, days remaining,
            game status, and last turn's results.
        """
        return {
            "day": self.current_day,
            "cash": self.cash,
            "days_remaining": self.days - self.current_day + 1,
            "game_over": self.game_over,
            "last_result": self.history[-1] if self.history else None,
        }
