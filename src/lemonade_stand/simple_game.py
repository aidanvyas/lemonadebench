"""Simplified price-only version of the lemonade stand game."""

import logging

logger = logging.getLogger(__name__)


class SimpleLemonadeGame:
    """Simplified game focusing only on price optimization."""

    def __init__(
        self, days: int = 100, demand_intercept: float = 100, demand_slope: float = 25
    ) -> None:
        """Initialize simple game with configurable demand function.

        Args:
            days: Number of days to simulate
            demand_intercept: 'a' in demand function Q = a - b*p
            demand_slope: 'b' in demand function Q = a - b*p
        """
        self.days = days
        self.current_day = 1

        # Demand function parameters
        self.demand_intercept = demand_intercept
        self.demand_slope = demand_slope

        # Calculate optimal price: p* = a / (2b)
        self.optimal_price = demand_intercept / (2 * demand_slope)

        # Game parameters
        self.suggested_starting_price = None  # Must be explicitly set

        # Game state
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

        # Linear demand: Q = a - b*p
        customers = self.demand_intercept - self.demand_slope * price

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

        # Calculate financials (no costs, no randomness)
        revenue = customers * price
        profit = revenue  # No costs or randomness

        # Record results (round financial values for display)
        result = {
            "day": self.current_day,
            "price": round(price, 2),
            "customers": customers,
            "revenue": round(revenue, 2),
            "costs": 0.0,
            "profit": round(profit, 2),
        }

        self.history.append(result)

        # Check game over conditions
        self.current_day += 1
        if self.current_day > self.days:
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
            "days_remaining": self.days - self.current_day + 1,
            "game_over": self.game_over,
            "last_result": self.history[-1] if self.history else None,
        }
