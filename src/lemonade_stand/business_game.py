"""Main game engine for the lemonade stand business simulation."""

import random
from collections import deque
from typing import Any

# Game configuration constants
DEFAULT_STARTING_CASH = 1000.0
DEFAULT_HOURLY_OPERATING_COST = 5.0
DEFAULT_TOTAL_DAYS = 30
LEMONADE_RECIPE = {"cups": 1, "lemons": 1, "sugar": 1, "water": 1}


class Inventory:
    """Manages perishable inventory with FIFO expiration tracking."""

    def __init__(self):
        """Initialize empty inventory with shelf life definitions."""
        # Store items as deques of (quantity, expiry_day) tuples
        self.items: dict[str, deque[tuple[int, int]]] = {
            "cups": deque(),
            "lemons": deque(),
            "sugar": deque(),
            "water": deque(),
        }

        # Shelf life in days for each item type
        self.shelf_life: dict[str, float] = {
            "cups": 30,
            "lemons": 7,
            "sugar": 60,
            "water": float("inf"),  # Water never expires
        }

        # Base costs for reference (actual costs vary daily)
        self.base_costs: dict[str, float] = {
            "cups": 0.05,
            "lemons": 0.20,
            "sugar": 0.10,
            "water": 0.02,
        }

    def add_items(self, item_type: str, quantity: int, current_day: int) -> None:
        """Add items to inventory with expiration date.

        Args:
            item_type: Type of item ('cups', 'lemons', 'sugar', 'water')
            quantity: Number of items to add
            current_day: Current day number for calculating expiry
        """
        if item_type not in self.items:
            raise ValueError(f"Unknown item type: {item_type}")

        if quantity <= 0:
            return

        # Calculate expiry day (infinite for water)
        if self.shelf_life[item_type] == float("inf"):
            expiry_day = float("inf")
        else:
            expiry_day = current_day + self.shelf_life[item_type]

        # Add to inventory queue
        self.items[item_type].append((quantity, expiry_day))

    def get_available(self, item_type: str) -> int:
        """Get total available quantity of an item type.

        Args:
            item_type: Type of item to check

        Returns:
            Total quantity available
        """
        if item_type not in self.items:
            return 0

        return sum(quantity for quantity, _ in self.items[item_type])

    def get_inventory_details(self) -> dict[str, list[dict[str, Any]]]:
        """Get detailed inventory information including expiration dates.

        Returns:
            Dictionary with item types as keys and list of batches as values
        """
        details = {}
        for item_type, batches in self.items.items():
            details[item_type] = []
            for quantity, expiry in batches:
                batch_info = {
                    "quantity": quantity,
                    "expires_day": expiry if expiry != float("inf") else "never",
                }
                details[item_type].append(batch_info)
        return details

    def use_items(self, recipe: dict[str, int]) -> bool:
        """Use items according to recipe, FIFO style.

        Args:
            recipe: Dictionary of item_type -> quantity needed

        Returns:
            True if all items were available and used, False otherwise
        """
        # First check if we have enough of everything
        for item_type, needed in recipe.items():
            if self.get_available(item_type) < needed:
                return False

        # Use items FIFO
        for item_type, needed in recipe.items():
            remaining_needed = needed

            while remaining_needed > 0 and self.items[item_type]:
                quantity, expiry = self.items[item_type][0]

                if quantity <= remaining_needed:
                    # Use entire batch
                    self.items[item_type].popleft()
                    remaining_needed -= quantity
                else:
                    # Use part of batch
                    self.items[item_type][0] = (quantity - remaining_needed, expiry)
                    remaining_needed = 0

        return True

    def remove_expired(self, current_day: int) -> dict[str, int]:
        """Remove expired items from inventory.

        Args:
            current_day: Current day number

        Returns:
            Dictionary of item_type -> quantity expired
        """
        expired = {}

        for item_type, batches in self.items.items():
            expired_quantity = 0

            # Remove expired batches from front of queue
            while batches and batches[0][1] <= current_day:
                quantity, _ = batches.popleft()
                expired_quantity += quantity

            if expired_quantity > 0:
                expired[item_type] = expired_quantity

        return expired

    def get_total_value(self) -> float:
        """Calculate total value of inventory at base costs.

        Returns:
            Total value in dollars
        """
        total = 0.0
        for item_type in self.items:
            quantity = self.get_available(item_type)
            total += quantity * self.base_costs[item_type]
        return total

    def can_make_lemonade(self) -> int:
        """Calculate how many lemonades can be made with current inventory.

        Returns:
            Maximum number of lemonades possible (limited by scarcest ingredient)
        """
        # Recipe: 1 of each item per lemonade
        return min(
            self.get_available("cups"),
            self.get_available("lemons"),
            self.get_available("sugar"),
            self.get_available("water"),
        )


class DemandModel:
    """Calculates customer demand based on price, time of day, and random variation."""

    # Hourly demand multipliers for all 24 hours
    HOURLY_MULTIPLIERS: dict[int, float] = {
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

    def __init__(
        self, base_demand_intercept: float = 50, price_sensitivity: float = 10
    ):
        """Initialize demand model.

        Args:
            base_demand_intercept: Maximum customers per hour at price=0
            price_sensitivity: How much demand decreases per dollar of price
        """
        self.base_demand_intercept = base_demand_intercept
        self.price_sensitivity = price_sensitivity
        self._rng: random.Random | None = None

    def set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducible simulations.

        Args:
            seed: Random seed value
        """
        self._rng = random.Random(seed)

    def calculate_base_demand(self, price: float) -> float:
        """Calculate base hourly demand at given price.

        Uses linear demand curve: demand = intercept - sensitivity * price

        Args:
            price: Price per lemonade

        Returns:
            Base demand (before time-of-day and random adjustments)
        """
        demand = self.base_demand_intercept - self.price_sensitivity * price
        return max(0, demand)  # Demand can't be negative

    def get_hour_multiplier(self, hour: int) -> float:
        """Get demand multiplier for given hour.

        Args:
            hour: Hour of day (0-23)

        Returns:
            Multiplier value (0.0 means closed)
        """
        return self.HOURLY_MULTIPLIERS[hour]

    def calculate_customers(
        self, price: float, hour: int, random_variation: bool = True
    ) -> int:
        """Calculate actual number of customers for a given hour.

        Args:
            price: Price per lemonade
            hour: Hour of day (0-23)
            random_variation: Whether to apply ±10% random variation

        Returns:
            Number of customers (rounded to nearest integer)
        """
        # Get base demand from price
        base_demand = self.calculate_base_demand(price)

        # Apply time-of-day multiplier
        hour_multiplier = self.get_hour_multiplier(hour)
        demand_with_time = base_demand * hour_multiplier

        # Apply random variation if enabled
        if random_variation:
            if self._rng:
                variation = self._rng.uniform(0.9, 1.1)
            else:
                variation = random.uniform(0.9, 1.1)
            final_demand = demand_with_time * variation
        else:
            final_demand = demand_with_time

        # Round to nearest integer
        return max(0, round(final_demand))

    def calculate_daily_customers(
        self,
        price: float,
        open_hour: int,
        close_hour: int,
        random_variation: bool = True,
    ) -> dict[int, int]:
        """Calculate customers for each hour of operation.

        Args:
            price: Price per lemonade
            open_hour: Opening hour (inclusive)
            close_hour: Closing hour (exclusive)
            random_variation: Whether to apply random variation

        Returns:
            Dictionary mapping hour -> number of customers
        """
        customers_by_hour = {}

        for hour in range(open_hour, close_hour):
            if hour in self.HOURLY_MULTIPLIERS:
                customers = self.calculate_customers(price, hour, random_variation)
                customers_by_hour[hour] = customers

        return customers_by_hour


class BusinessGame:
    """Lemonade stand business simulation with inventory management."""

    def __init__(
        self,
        days: int = DEFAULT_TOTAL_DAYS,
        starting_cash: float = DEFAULT_STARTING_CASH,
        hourly_operating_cost: float = DEFAULT_HOURLY_OPERATING_COST,
        seed: int | None = None,
    ):
        """Initialize the business game.

        Args:
            days: Total number of days to play
            starting_cash: Initial cash balance
            hourly_operating_cost: Cost per hour of operation
            seed: Random seed for reproducibility
        """
        self.total_days = days
        self.current_day = 0
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.hourly_operating_cost = hourly_operating_cost

        # Initialize components
        self.inventory = Inventory()
        self.demand_model = DemandModel()

        # Set random seed if provided
        if seed is not None:
            self.rng = random.Random(seed)
            self.demand_model.set_random_seed(seed)
        else:
            self.rng = random.Random()

        # Daily state tracking
        self.today_supply_costs: dict[str, float] = {}
        self.price_set = False
        self.hours_set = False
        self.open_hour: int | None = None
        self.close_hour: int | None = None
        self.price: float | None = None

        # History tracking
        self.history: list[dict[str, Any]] = []
        self.supply_cost_history: list[dict[str, float]] = []

        # Yesterday's profit for display
        self.yesterday_profit: float | None = None

        # Recipe for making lemonade
        self.recipe = LEMONADE_RECIPE.copy()

    def start_new_day(self) -> dict[str, Any]:
        """Start a new day: handle expiration, generate costs, reset state.

        Returns:
            Dictionary with day start information
        """
        self.current_day += 1

        # Remove expired inventory
        expired = self.inventory.remove_expired(self.current_day)

        # Generate today's supply costs (±10% variation)
        self.today_supply_costs = {}
        for item, base_cost in self.inventory.base_costs.items():
            variation = self.rng.uniform(0.9, 1.1)
            self.today_supply_costs[item] = round(base_cost * variation, 4)

        # Store in history
        self.supply_cost_history.append(
            {"day": self.current_day, **self.today_supply_costs}
        )

        # Reset daily state
        self.price_set = False
        self.hours_set = False
        self.open_hour = None
        self.close_hour = None
        self.price = None

        return {"day": self.current_day, "expired_items": expired, "cash": self.cash}

    def check_morning_prices(self) -> dict[str, Any]:
        """Check today's supply costs.

        Returns:
            Dictionary with supply costs
        """
        return {
            "success": True,
            "prices": self.today_supply_costs.copy()
        }

    def check_inventory(self) -> dict[str, Any]:
        """Check current inventory levels and expiration dates.

        Returns:
            Inventory details with quantities and expiration
        """
        return {
            "summary": {
                item: self.inventory.get_available(item)
                for item in ["cups", "lemons", "sugar", "water"]
            },
            "details": self.inventory.get_inventory_details(),
            "can_make": self.inventory.can_make_lemonade(),
        }

    def order_supplies(
        self, cups: int = 0, lemons: int = 0, sugar: int = 0, water: int = 0
    ) -> dict[str, Any]:
        """Order supplies for immediate delivery.

        Args:
            cups: Number of cups to order
            lemons: Number of lemons to order
            sugar: Amount of sugar to order
            water: Amount of water to order

        Returns:
            Order confirmation or error
        """
        # Validate quantities
        if any(q < 0 for q in [cups, lemons, sugar, water]):
            return {"success": False, "error": "Cannot order negative quantities"}

        # Calculate total cost
        total_cost = (
            cups * self.today_supply_costs["cups"]
            + lemons * self.today_supply_costs["lemons"]
            + sugar * self.today_supply_costs["sugar"]
            + water * self.today_supply_costs["water"]
        )

        # Check if enough cash
        if total_cost > self.cash:
            return {
                "success": False,
                "error": f"Insufficient funds. Cost: ${total_cost:.2f}, Available: ${self.cash:.2f}",
            }

        # Process order
        self.cash -= total_cost

        # Add to inventory
        self.inventory.add_items("cups", cups, self.current_day)
        self.inventory.add_items("lemons", lemons, self.current_day)
        self.inventory.add_items("sugar", sugar, self.current_day)
        self.inventory.add_items("water", water, self.current_day)

        return {
            "success": True,
            "ordered": {"cups": cups, "lemons": lemons, "sugar": sugar, "water": water},
            "total_cost": total_cost,
            "remaining_cash": self.cash,
        }

    def set_operating_hours(self, open_hour: int, close_hour: int) -> dict[str, Any]:
        """Set today's operating hours.

        Args:
            open_hour: Opening hour (0-23)
            close_hour: Closing hour (1-24, must be > open_hour)

        Returns:
            Confirmation or error
        """
        # Validate hours
        if open_hour < 0 or open_hour > 23:
            return {
                "success": False,
                "error": f"Invalid open hour: {open_hour}. Must be between 0-23.",
            }

        if close_hour < 1 or close_hour > 24:
            return {
                "success": False,
                "error": f"Invalid close hour: {close_hour}. Must be between 1-24.",
            }

        if close_hour <= open_hour:
            return {
                "success": False,
                "error": f"Close hour ({close_hour}) must be after open hour ({open_hour}).",
            }

        self.open_hour = open_hour
        self.close_hour = close_hour
        self.hours_set = True

        return {
            "success": True,
            "open_hour": open_hour,
            "close_hour": close_hour,
            "hours_open": close_hour - open_hour,
        }

    def set_price(self, price: float) -> dict[str, Any]:
        """Set today's lemonade price.

        Args:
            price: Price per lemonade (must be >= 0)

        Returns:
            Confirmation or error
        """
        if price < 0:
            return {"success": False, "error": "Price cannot be negative."}

        self.price = round(price, 2)
        self.price_set = True

        return {"success": True, "price": self.price}

    def open_for_business(self) -> dict[str, Any]:
        """Attempt to open the stand for business today.

        This must be called after setting price and operating hours.

        Returns:
            Dict with success status and error details if not ready
        """
        ready, missing = self.check_ready_for_next_day()

        if not ready:
            return {
                "success": False,
                "error": "Cannot open for business - required actions not completed",
                "missing_actions": missing,
                "hint": "You must set both price and operating hours before opening",
            }

        return {
            "success": True,
            "message": f"Ready to open! Hours: {self.open_hour}-{self.close_hour}, Price: ${self.price:.2f}. The stand is now open for business and the day will play out automatically.",
        }

    def simulate_day(self) -> dict[str, Any]:
        """Simulate the day's business after all decisions are made.

        Returns:
            Day's results
        """
        # Check required actions
        if not self.price_set:
            return {
                "success": False,
                "error": "Cannot simulate day: price not set. Call set_price() first."
            }

        if not self.hours_set:
            return {
                "success": False,
                "error": "Cannot simulate day: hours not set. Call set_operating_hours() first."
            }

        # Calculate customers for each hour
        hourly_customers = self.demand_model.calculate_daily_customers(
            self.price, self.open_hour, self.close_hour
        )

        # Simulate sales hour by hour
        hourly_sales = {}
        total_customers_served = 0
        total_customers_lost = 0

        for hour, potential_customers in hourly_customers.items():
            # Check inventory
            can_make = self.inventory.can_make_lemonade()

            if can_make >= potential_customers:
                # Serve all customers
                served = potential_customers
                lost = 0
            else:
                # Can only serve what we can make
                served = can_make
                lost = potential_customers - can_make

            # Use inventory for served customers
            if served > 0:
                for _ in range(served):
                    self.inventory.use_items(self.recipe)

            hourly_sales[hour] = {
                "customers_wanted": potential_customers,
                "customers_served": served,
                "customers_lost": lost,
            }

            total_customers_served += served
            total_customers_lost += lost

        # Calculate financials
        revenue = total_customers_served * self.price
        operating_hours = self.close_hour - self.open_hour
        operating_cost = operating_hours * self.hourly_operating_cost
        profit = revenue - operating_cost

        # Update cash
        self.cash += profit
        self.yesterday_profit = profit

        # Create day result
        day_result = {
            "day": self.current_day,
            "price": self.price,
            "open_hour": self.open_hour,
            "close_hour": self.close_hour,
            "hours_open": operating_hours,
            "customers_served": total_customers_served,
            "customers_lost": total_customers_lost,
            "revenue": revenue,
            "operating_cost": operating_cost,
            "profit": profit,
            "cash": self.cash,
            "hourly_sales": hourly_sales,
        }

        # Store in history
        self.history.append(day_result)

        # Return with success indicator
        return {"success": True, **day_result}

    def get_historical_supply_costs(self) -> list[dict[str, float]]:
        """Get historical supply cost data.

        Returns:
            List of daily supply costs
        """
        return self.supply_cost_history.copy()

    def check_ready_for_next_day(self) -> tuple[bool, list[str]]:
        """Check if all required actions have been taken.

        Returns:
            Tuple of (ready, missing_actions)
        """
        missing = []

        if not self.price_set:
            missing.append("set_price() - not yet called")

        if not self.hours_set:
            missing.append("set_operating_hours() - not yet called")

        return len(missing) == 0, missing

    def get_turn_prompt(self) -> str:
        """Generate the prompt for the current turn.

        Returns:
            Prompt string for the AI
        """
        # For stateless approach, always return the same format with historical table
        profit_msg = (
            f" You made ${self.yesterday_profit:.2f} yesterday."
            if self.yesterday_profit is not None
            else ""
        )
        base_prompt = f"""Day {self.current_day} of {self.total_days}.{profit_msg}
Current cash: ${self.cash:.2f}
{self._get_historical_table()}
Remember to:
1. Check inventory and morning prices
2. Order supplies if needed
3. Set price and operating hours
4. Call open_for_business() to start the day

What would you like to do?"""
        if self.current_day == 0:
            return f"{self._get_system_prompt()}\n{base_prompt}"
        return base_prompt

    def _get_system_prompt(self) -> str:
        """Get the full system prompt for day 1.

        Returns:
            System prompt string
        """
        return f"""You run a lemonade stand for {self.total_days} days. Your goal is to maximize total profit (cash in bank after {self.total_days} days).

BUSINESS MECHANICS:
- Starting capital: $1000
- Operating cost: $5 per hour the stand is open
- Recipe: 1 lemonade = 1 cup + 1 lemon + 1 sugar + 1 water (all required)
- You can only sell lemonade if you have ALL ingredients in stock

INVENTORY MANAGEMENT:
- Items have different shelf lives:
  * Cups: 30 days
  * Sugar: 60 days
  * Water: Never expires
  * Lemons: 7 days
- Expired items are automatically discarded each morning
- Supplies are delivered instantly when ordered

DAILY WORKFLOW:
1. Morning: Check inventory and supply prices
2. Decisions: Order supplies, set price and operating hours
3. IMPORTANT: Call open_for_business() after setting price and hours
4. Evening: Review profit/loss and customer data

AVAILABLE TOOLS:
- check_inventory(): View current stock and expiration dates
- check_morning_prices(): See today's supply costs
- get_historical_supply_costs(): Analyze supply price trends
- order_supplies(cups, lemons, sugar, water): Purchase supplies
- set_price(price): Set today's lemonade price
- set_operating_hours(open_hour, close_hour): Set today's operating hours
- open_for_business(): REQUIRED - Open the stand after setting price and hours

IMPORTANT: You MUST call open_for_business() after setting your price and operating hours. The stand will not operate until you do this.
{self._get_historical_table()}
Today is Day {self.current_day}. You have ${self.cash:.2f} in cash. What would you like to do?"""

    def _get_historical_table(self) -> str:
        """Generate a table of complete performance history.

        Returns:
            Formatted table string showing all days
        """
        if not self.history:
            return ""

        table = "\nHISTORICAL PERFORMANCE:\n"
        table += "Day | Profit     | Customers | Hours Open | Ran Out\n"
        table += "----|------------|-----------|------------|--------\n"

        # Show ALL days
        for day in self.history:
            ran_out = "Yes" if day["customers_lost"] > 0 else "No"
            hours = f"{day['open_hour']}-{day['close_hour']}"
            table += f"{day['day']:3} | ${day['profit']:9.2f} | {day['customers_served']:9} | {hours:^10} | {ran_out:^7}\n"

        return table

    def is_game_over(self) -> bool:
        """Check if the game has ended.

        Returns:
            True if game is over
        """
        # Game ends after all days or if bankrupt
        return self.current_day >= self.total_days or self.cash < 0

    def get_final_results(self) -> dict[str, Any]:
        """Get final game results.

        Returns:
            Summary of game performance
        """
        total_revenue = sum(day["revenue"] for day in self.history)
        total_operating_cost = sum(day["operating_cost"] for day in self.history)
        total_customers = sum(day["customers_served"] for day in self.history)
        total_lost_sales = sum(day["customers_lost"] for day in self.history)

        return {
            "days_played": self.current_day,
            "final_cash": self.cash,
            "total_profit": self.cash - self.starting_cash,  # Profit over starting capital
            "total_revenue": total_revenue,
            "total_operating_cost": total_operating_cost,
            "total_customers": total_customers,
            "total_lost_sales": total_lost_sales,
            "average_daily_profit": (self.cash - self.starting_cash) / self.current_day
            if self.current_day > 0
            else 0,
            "inventory_value": self.inventory.get_total_value(),
        }
