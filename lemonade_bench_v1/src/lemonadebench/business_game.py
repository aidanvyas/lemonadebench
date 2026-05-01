"""Main game engine for the lemonade stand business simulation."""

import math
import random
from collections import deque
from decimal import ROUND_HALF_UP, Decimal, getcontext
from typing import Any, ClassVar

from .utils import to_decimal

# Configure decimal rounding for accurate currency calculations
getcontext().rounding = ROUND_HALF_UP
TWOPLACES = to_decimal("0.01")

# Game configuration constants
DEFAULT_STARTING_CASH = to_decimal("1000.00")
DEFAULT_LABOR_COST_PER_HOUR = to_decimal("3.00")
DEFAULT_UTILITIES_COST_PER_HOUR = to_decimal("2.00")
DEFAULT_AUTOMATION_COST = to_decimal("1000.00")
DEFAULT_TOTAL_DAYS = 100
LEMONADE_RECIPE = {"cups": 1, "lemons": 1, "sugar": 1, "water": 1}

# Advertising parameters (hidden from the model — must be inferred from sales).
DEFAULT_AD_SQRT_SCALE = 10.0
DEFAULT_AD_MULT_CAP = 0.20
DEFAULT_AD_DECAY = 0.80
DEFAULT_AD_VAR_LO = 0.90
DEFAULT_AD_VAR_HI = 1.10
DEFAULT_AD_LIFETIME_DAYS = 7  # campaign contributes 0 once age >= this

# Loan parameters (rate is daily; range exposed to the model).
DEFAULT_LOAN_CAP = to_decimal("10000.00")
DEFAULT_LOAN_RATE_LO = to_decimal("0.009")  # 0.9%/day
DEFAULT_LOAN_RATE_HI = to_decimal("0.011")  # 1.1%/day


class Inventory:
    """Manages perishable inventory with FIFO expiration tracking."""

    def __init__(self) -> None:
        """Initialize empty inventory with shelf life definitions."""
        # Store items as deques of (quantity, expiry_day) tuples
        self.items: dict[str, deque[tuple[int, float]]] = {
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
        self.base_costs: dict[str, Decimal] = {
            "cups": to_decimal("0.05"),
            "lemons": to_decimal("0.20"),
            "sugar": to_decimal("0.10"),
            "water": to_decimal("0.02"),
        }

    def add_items(self, item_type: str, quantity: int, current_day: int) -> None:
        """Add items to inventory with expiration date.

        Args:
            item_type: Type of item ('cups', 'lemons', 'sugar', 'water')
            quantity: Number of items to add
            current_day: Current day number for calculating expiry

        """
        expiry_day = current_day + self.shelf_life[item_type]
        self.items[item_type].append((quantity, expiry_day))

    def get_available(self, item_type: str) -> int:
        """Get total available quantity of an item type.

        Args:
            item_type: Type of item to check

        Returns:
            Total quantity available

        """
        return sum(quantity for quantity, _ in self.items[item_type])

    def get_inventory_details(self) -> dict[str, list[dict[str, Any]]]:
        """Get detailed inventory information including expiration dates.

        Returns:
            Dictionary with item types as keys and list of batches as values

        """
        details: dict[str, list[dict[str, Any]]] = {}
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
                    self.items[item_type][0] = (
                        quantity - remaining_needed,
                        expiry,
                    )
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

    def get_total_value(self) -> Decimal:
        """Calculate total value of inventory at base costs.

        Returns:
            Total value in dollars

        """
        total = to_decimal("0")
        for item_type in self.items:
            quantity = self.get_available(item_type)
            total += to_decimal(quantity) * self.base_costs[item_type]
        return total.quantize(TWOPLACES)

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

    # Default random variation percentage (±10%)
    DEFAULT_VARIATION_PCT = 0.10

    # Hourly demand multipliers for all 24 hours
    HOURLY_MULTIPLIERS: ClassVar[dict[int, float]] = {
        0: 0.0,  # 12-1am: No demand
        1: 0.0,  # 1-2am: No demand
        2: 0.0,  # 2-3am: No demand
        3: 0.0,  # 3-4am: No demand
        4: 0.0,  # 4-5am: No demand
        5: 0.0,  # 5-6am: No demand
        6: 0.3,  # 6-7am: Early morning (30% of base)
        7: 0.5,  # 7-8am: Morning commute
        8: 0.7,  # 8-9am: Morning
        9: 0.8,  # 9-10am: Mid-morning
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
        21: 0.0,  # 9-10pm: No demand
        22: 0.0,  # 10-11pm: No demand
        23: 0.0,  # 11pm-12am: No demand
    }

    def __init__(
        self,
        base_demand_intercept: float = 50,
        price_sensitivity: float = 10,
        variation_pct: float = DEFAULT_VARIATION_PCT,
    ) -> None:
        """Initialize demand model.

        Args:
            base_demand_intercept: Maximum customers per hour at price=0
            price_sensitivity: How much demand decreases per dollar of price
            variation_pct: Random variation percentage (default 0.10 for ±10%)

        """
        self.base_demand_intercept = base_demand_intercept
        self.price_sensitivity = price_sensitivity
        self.variation_pct = variation_pct

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
        self,
        price: float,
        hour: int,
    ) -> int:
        """Calculate actual number of customers for a given hour.

        Args:
            price: Price per lemonade
            hour: Hour of day (0-23)

        Returns:
            Number of customers (rounded to nearest integer)

        """
        # Get base demand from price
        base_demand = self.calculate_base_demand(price)

        # Apply time-of-day multiplier
        hour_multiplier = self.get_hour_multiplier(hour)
        demand_with_time = base_demand * hour_multiplier

        # Apply random variation (e.g., ±10% by default)
        variation_min = 1.0 - self.variation_pct
        variation_max = 1.0 + self.variation_pct
        variation = random.uniform(variation_min, variation_max)
        final_demand = demand_with_time * variation

        # Round to nearest integer
        return max(0, round(final_demand))

    def calculate_daily_customers(
        self,
        price: float,
        open_hour: int,
        close_hour: int,
    ) -> dict[int, int]:
        """Calculate customers for each hour of operation.

        Args:
            price: Price per lemonade
            open_hour: Opening hour (inclusive)
            close_hour: Closing hour (exclusive)

        Returns:
            Dictionary mapping hour -> number of customers

        """
        customers_by_hour = {}

        for hour in range(open_hour, close_hour):
            if hour in self.HOURLY_MULTIPLIERS:
                customers = self.calculate_customers(price, hour)
                customers_by_hour[hour] = customers

        return customers_by_hour


class BusinessGame:
    """Lemonade stand business simulation with inventory management."""

    def __init__(
        self,
        days: int = DEFAULT_TOTAL_DAYS,
        starting_cash: Decimal | float = DEFAULT_STARTING_CASH,
        labor_cost_per_hour: Decimal | float = DEFAULT_LABOR_COST_PER_HOUR,
        utilities_cost_per_hour: Decimal | float = DEFAULT_UTILITIES_COST_PER_HOUR,
        automation_cost: Decimal | float = DEFAULT_AUTOMATION_COST,
        ad_sqrt_scale: float = DEFAULT_AD_SQRT_SCALE,
        ad_mult_cap: float = DEFAULT_AD_MULT_CAP,
        ad_decay: float = DEFAULT_AD_DECAY,
        ad_var_lo: float = DEFAULT_AD_VAR_LO,
        ad_var_hi: float = DEFAULT_AD_VAR_HI,
        ad_lifetime_days: int = DEFAULT_AD_LIFETIME_DAYS,
        loan_cap: Decimal | float = DEFAULT_LOAN_CAP,
        loan_rate_lo: Decimal | float = DEFAULT_LOAN_RATE_LO,
        loan_rate_hi: Decimal | float = DEFAULT_LOAN_RATE_HI,
    ) -> None:
        """Initialize the business game.

        Args:
            days: Total number of days to play
            starting_cash: Initial cash balance
            labor_cost_per_hour: Wage paid per hour the stand is open
                (eliminated by purchasing automation).
            utilities_cost_per_hour: Per-hour overhead that applies regardless
                of automation.
            automation_cost: One-time cost to eliminate labor for the rest
                of the game.
            ad_sqrt_scale: Goodwill earned per sqrt(spend / 100).
            ad_mult_cap: Maximum demand multiplier above 1.0 (saturation cap).
            ad_decay: Daily multiplicative decay applied to a campaign's goodwill.
            ad_var_lo, ad_var_hi: Uniform range on goodwill earned per campaign.
            ad_lifetime_days: Each campaign contributes 0 once its age reaches
                this many days (hard cutoff on top of the multiplicative decay).
            loan_cap: Maximum outstanding loan balance.
            loan_rate_lo, loan_rate_hi: Daily interest rate range. Each morning
                a fresh rate is drawn from Uniform(lo, hi).

        """
        self.total_days = days
        self.current_day = 0
        self.starting_cash = to_decimal(starting_cash).quantize(TWOPLACES)
        self.cash = to_decimal(starting_cash).quantize(TWOPLACES)
        self.labor_cost_per_hour = to_decimal(labor_cost_per_hour).quantize(TWOPLACES)
        self.utilities_cost_per_hour = to_decimal(utilities_cost_per_hour).quantize(
            TWOPLACES,
        )
        self.automation_cost = to_decimal(automation_cost).quantize(TWOPLACES)
        self.has_automation: bool = False

        # Advertising state — all hidden from the model.
        # ad_campaigns is a list of (day_purchased, goodwill_at_purchase).
        # Each campaign contributes goodwill * decay^age to current goodwill,
        # and stops contributing once its age reaches ad_lifetime_days.
        self.ad_sqrt_scale = ad_sqrt_scale
        self.ad_mult_cap = ad_mult_cap
        self.ad_decay = ad_decay
        self.ad_var_lo = ad_var_lo
        self.ad_var_hi = ad_var_hi
        self.ad_lifetime_days = ad_lifetime_days
        self.ad_campaigns: list[tuple[int, float]] = []
        self.today_ad_spend: Decimal = to_decimal(0).quantize(TWOPLACES)

        # Loan state. Single revolving balance. today_loan_rate is drawn each
        # morning from Uniform(loan_rate_lo, loan_rate_hi) and exposed to the
        # model. yesterday_interest_charged is shown for transparency.
        self.loan_cap = to_decimal(loan_cap).quantize(TWOPLACES)
        self.loan_rate_lo = to_decimal(loan_rate_lo)
        self.loan_rate_hi = to_decimal(loan_rate_hi)
        self.loan_balance: Decimal = to_decimal(0).quantize(TWOPLACES)
        self.today_loan_rate: Decimal | None = None
        self.yesterday_interest_charged: Decimal = to_decimal(0).quantize(TWOPLACES)
        self.total_interest_charged: Decimal = to_decimal(0).quantize(TWOPLACES)

        # Initialize components
        self.inventory = Inventory()
        self.demand_model = DemandModel()

        # Daily state tracking
        self.today_supply_costs: dict[str, Decimal] = {}
        self.price_set = False
        self.hours_set = False
        self.open_hour: int | None = None
        self.close_hour: int | None = None
        self.price: Decimal | None = None

        # History tracking
        self.history: list[dict[str, Any]] = []
        self.supply_cost_history: list[dict[str, Decimal]] = []

        # Yesterday's profit for display
        self.yesterday_profit: Decimal | None = None

        # Recipe for making lemonade
        self.recipe = LEMONADE_RECIPE.copy()

    @property
    def ad_goodwill(self) -> float:
        """Total active advertising goodwill (sum across live campaigns).

        Each campaign at age d contributes goodwill * decay^d, and is dropped
        once age >= ad_lifetime_days.
        """
        return sum(
            goodwill * (self.ad_decay ** (self.current_day - day))
            for day, goodwill in self.ad_campaigns
            if (self.current_day - day) < self.ad_lifetime_days
        )

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
            variation = to_decimal(random.uniform(0.9, 1.1))
            self.today_supply_costs[item] = (base_cost * variation).quantize(
                to_decimal("0.0001"),
            )

        # Store in history
        self.supply_cost_history.append(
            {"day": self.current_day, **self.today_supply_costs},
        )

        # Reset today's ad-spend bucket. Goodwill from prior campaigns
        # decays by age in simulate_day, no morning decay needed here.
        self.today_ad_spend = to_decimal(0).quantize(TWOPLACES)

        # Draw today's loan rate and charge interest on outstanding balance.
        rate_float = random.uniform(
            float(self.loan_rate_lo),
            float(self.loan_rate_hi),
        )
        self.today_loan_rate = to_decimal(rate_float).quantize(to_decimal("0.0001"))
        if self.loan_balance > 0:
            interest = (self.loan_balance * self.today_loan_rate).quantize(TWOPLACES)
            if self.cash >= interest:
                self.cash = (self.cash - interest).quantize(TWOPLACES)
            else:
                # Pay what we can; rest compounds onto the balance.
                short = interest - self.cash
                self.cash = to_decimal(0).quantize(TWOPLACES)
                self.loan_balance = (self.loan_balance + short).quantize(TWOPLACES)
            self.yesterday_interest_charged = interest
            self.total_interest_charged = (
                self.total_interest_charged + interest
            ).quantize(TWOPLACES)
        else:
            self.yesterday_interest_charged = to_decimal(0).quantize(TWOPLACES)

        # Reset daily state
        self.price_set = False
        self.hours_set = False
        self.open_hour = None
        self.close_hour = None
        self.price = None

        return {"day": self.current_day, "expired_items": expired, "cash": self.cash}

    def check_morning_prices(self) -> dict[str, Decimal]:
        """Check today's supply costs.

        Returns:
            Dictionary of supply costs

        """
        return self.today_supply_costs.copy()

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
        self,
        cups: int = 0,
        lemons: int = 0,
        sugar: int = 0,
        water: int = 0,
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
        # Calculate total cost
        total_cost = (
            to_decimal(cups) * self.today_supply_costs["cups"]
            + to_decimal(lemons) * self.today_supply_costs["lemons"]
            + to_decimal(sugar) * self.today_supply_costs["sugar"]
            + to_decimal(water) * self.today_supply_costs["water"]
        )

        # Check if enough cash
        if total_cost > self.cash:
            return {
                "success": False,
                "error": (
                    f"Insufficient funds. Cost: ${total_cost:.2f}, "
                    f"Available: ${self.cash:.2f}"
                ),
            }

        # Process order
        self.cash = (self.cash - total_cost).quantize(TWOPLACES)

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
        if close_hour <= open_hour:
            return {
                "success": False,
                "error": (
                    f"Close hour ({close_hour}) must be after open hour ({open_hour})."
                ),
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

    def set_price(self, price: float) -> Decimal:
        """Set today's lemonade price.

        Args:
            price: Price per lemonade (must be >= 0)

        Returns:
            Confirmed price

        """
        self.price = to_decimal(price).quantize(TWOPLACES)
        self.price_set = True

        return self.price

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
            "message": (
                f"Ready to open! Hours: {self.open_hour}-{self.close_hour}, "
                f"Price: ${self.price:.2f}. The stand is now open for business "
                f"and the day will play out automatically."
            ),
        }

    def purchase_automation(self) -> dict[str, Any]:
        """Purchase automation to eliminate labor costs for the rest of the game.

        One-time purchase. Effective immediately (today's operating cost will
        only include utilities). Utilities continue to accrue.

        Returns:
            Dict with success status and either confirmation or error details.

        """
        if self.has_automation:
            return {
                "success": False,
                "error": "Automation has already been purchased.",
            }
        if self.cash < self.automation_cost:
            return {
                "success": False,
                "error": (
                    f"Insufficient cash. Need ${self.automation_cost:.2f}, "
                    f"have ${self.cash:.2f}."
                ),
            }

        self.cash = (self.cash - self.automation_cost).quantize(TWOPLACES)
        self.has_automation = True
        return {
            "success": True,
            "message": (
                f"Automation purchased for ${self.automation_cost:.2f}. "
                f"Labor cost is now $0/hr (utilities still apply at "
                f"${self.utilities_cost_per_hour:.2f}/hr). "
                f"Cash remaining: ${self.cash:.2f}."
            ),
        }

    def purchase_advertising(self, spend: int) -> dict[str, Any]:
        """Spend cash on an ad campaign for today.

        Multiple calls on the same day aggregate (their dollar amounts sum
        before the diminishing-returns curve is applied).

        Args:
            spend: Dollars to spend (positive integer).

        Returns:
            Dict with success status; effectiveness must be inferred from
            subsequent customer counts (no internals exposed).

        """
        if spend <= 0:
            return {
                "success": False,
                "error": "Advertising spend must be a positive integer.",
            }
        spend_dec = to_decimal(spend).quantize(TWOPLACES)
        if self.cash < spend_dec:
            return {
                "success": False,
                "error": (
                    f"Insufficient cash. Need ${spend_dec:.2f}, have ${self.cash:.2f}."
                ),
            }

        self.cash = (self.cash - spend_dec).quantize(TWOPLACES)
        self.today_ad_spend = (self.today_ad_spend + spend_dec).quantize(TWOPLACES)
        return {
            "success": True,
            "message": (
                f"Spent ${spend_dec:.2f} on advertising today "
                f"(today's total ad spend: ${self.today_ad_spend:.2f}). "
                f"Cash remaining: ${self.cash:.2f}."
            ),
        }

    def take_loan(self, amount: int) -> dict[str, Any]:
        """Borrow cash. Increases both cash and loan_balance.

        Args:
            amount: Dollars to borrow (positive integer).

        Returns:
            Dict with success status; rejects if non-positive or if borrowing
            would push outstanding balance over the cap.

        """
        if amount <= 0:
            return {
                "success": False,
                "error": "Loan amount must be a positive integer.",
            }
        amt = to_decimal(amount).quantize(TWOPLACES)
        new_balance = self.loan_balance + amt
        if new_balance > self.loan_cap:
            return {
                "success": False,
                "error": (
                    f"Loan cap is ${self.loan_cap:.2f}. "
                    f"Outstanding balance is ${self.loan_balance:.2f}; "
                    f"can borrow at most ${(self.loan_cap - self.loan_balance):.2f} "
                    f"more."
                ),
            }
        self.cash = (self.cash + amt).quantize(TWOPLACES)
        self.loan_balance = new_balance.quantize(TWOPLACES)
        return {
            "success": True,
            "message": (
                f"Borrowed ${amt:.2f}. Outstanding balance: "
                f"${self.loan_balance:.2f}. Cash: ${self.cash:.2f}."
            ),
        }

    def repay_loan(self, amount: int) -> dict[str, Any]:
        """Repay loan from cash. Decreases both cash and loan_balance.

        Args:
            amount: Dollars to repay (positive integer).

        Returns:
            Dict with success status; rejects if non-positive, if cash is
            insufficient, or if repayment would exceed the outstanding balance.

        """
        if amount <= 0:
            return {
                "success": False,
                "error": "Repayment amount must be a positive integer.",
            }
        amt = to_decimal(amount).quantize(TWOPLACES)
        if amt > self.loan_balance:
            return {
                "success": False,
                "error": (
                    f"Repayment ${amt:.2f} exceeds outstanding balance "
                    f"${self.loan_balance:.2f}."
                ),
            }
        if amt > self.cash:
            return {
                "success": False,
                "error": (
                    f"Insufficient cash. Need ${amt:.2f}, have ${self.cash:.2f}."
                ),
            }
        self.cash = (self.cash - amt).quantize(TWOPLACES)
        self.loan_balance = (self.loan_balance - amt).quantize(TWOPLACES)
        return {
            "success": True,
            "message": (
                f"Repaid ${amt:.2f}. Outstanding balance: "
                f"${self.loan_balance:.2f}. Cash: ${self.cash:.2f}."
            ),
        }

    def simulate_day(self) -> dict[str, Any]:
        """Simulate the day's business after all decisions are made.

        Returns:
            Day's results

        """
        # Roll today's ad spend into a fresh campaign (sqrt scaling +
        # variability), then compute total active goodwill across campaigns
        # within the lifetime window. Each campaign at age d contributes
        # goodwill * decay^d, and contributes 0 once age >= lifetime.
        ad_spend_today = self.today_ad_spend
        if ad_spend_today > 0:
            new_goodwill = (
                math.sqrt(float(ad_spend_today) / 100)
                * self.ad_sqrt_scale
                * random.uniform(self.ad_var_lo, self.ad_var_hi)
            )
            self.ad_campaigns.append((self.current_day, new_goodwill))

        total_goodwill = sum(
            goodwill * (self.ad_decay ** (self.current_day - day))
            for day, goodwill in self.ad_campaigns
            if (self.current_day - day) < self.ad_lifetime_days
        )

        # Demand multiplier saturates at (1 + ad_mult_cap).
        ad_multiplier = 1 + self.ad_mult_cap * (1 - math.exp(-total_goodwill))

        # Calculate customers for each hour, scaled by today's ad multiplier
        hourly_customers = self.demand_model.calculate_daily_customers(
            float(self.price),
            self.open_hour,
            self.close_hour,
        )
        if ad_multiplier != 1.0:
            hourly_customers = {
                h: round(c * ad_multiplier) for h, c in hourly_customers.items()
            }

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
        revenue = to_decimal(total_customers_served) * self.price
        operating_hours = to_decimal(self.close_hour - self.open_hour)
        effective_labor = (
            to_decimal(0) if self.has_automation else self.labor_cost_per_hour
        )
        operating_cost = operating_hours * (
            effective_labor + self.utilities_cost_per_hour
        )
        profit = revenue - operating_cost

        # Update cash
        self.cash = (self.cash + profit).quantize(TWOPLACES)
        self.yesterday_profit = profit.quantize(TWOPLACES)

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
            "ad_spend": ad_spend_today,
        }

        # Store in history
        self.history.append(day_result)

        return day_result

    def get_historical_supply_costs(self) -> list[dict[str, Decimal]]:
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

    def get_system_instructions(self) -> str:
        """Get the system instructions (sent once at conversation start).

        These are the rules and mechanics that stay constant throughout the game.
        Used with Responses API's instructions parameter.

        Returns:
            System instructions string

        """
        labor = self.labor_cost_per_hour
        utilities = self.utilities_cost_per_hour
        total = labor + utilities
        auto_cost = self.automation_cost
        loan_cap = self.loan_cap
        rate_lo_pct = self.loan_rate_lo * to_decimal(100)
        rate_hi_pct = self.loan_rate_hi * to_decimal(100)
        return f"""You run a lemonade stand for {self.total_days} days. \
Your final score is cash on hand minus outstanding loan balance after \
{self.total_days} days.

BUSINESS MECHANICS:
- Starting capital: $1000
- Operating cost per hour the stand is open: ${total:.2f} \
(${labor:.2f} labor + ${utilities:.2f} utilities)
- Automation: pay ${auto_cost:.2f} once to permanently eliminate the labor \
portion. Utilities still apply.
- Advertising: spend cash on ads to boost demand. Effects are temporary, \
have diminishing returns, and ROI is uncertain — you'll need to experiment.
- Loans: borrow up to ${loan_cap:.2f} outstanding. Daily interest rate is \
drawn from {rate_lo_pct:.1f}%–{rate_hi_pct:.1f}% and is shown each morning. \
Interest is charged each morning on the outstanding balance. If cash is \
insufficient, the unpaid interest compounds onto the balance. \
Final score is cash minus outstanding balance, so a loan you don't repay \
fully reduces your score one-for-one.
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
2. Decisions: Order supplies, set price and operating hours, optionally \
purchase automation, optionally buy advertising, optionally take or repay loans
3. IMPORTANT: Call open_for_business() after setting price and hours
4. Evening: Review profit/loss and customer data

AVAILABLE TOOLS:
- check_inventory(): View current stock and expiration dates
- check_morning_prices(): See today's supply costs
- get_historical_supply_costs(): Analyze supply price trends
- order_supplies(cups, lemons, sugar, water): Purchase supplies
- set_price(price): Set today's lemonade price
- set_operating_hours(open_hour, close_hour): Set today's operating hours
- purchase_automation(): One-time purchase that eliminates labor cost \
for the rest of the game
- purchase_advertising(spend): Buy ads (positive integer dollars). \
Affects today's demand. Multiple calls in a single day stack in dollar amount.
- take_loan(amount): Borrow cash. Outstanding balance capped at \
${loan_cap:.2f}. Subject to daily interest at today's rate.
- repay_loan(amount): Pay down loan from cash. Reduces balance one-for-one.
- open_for_business(): REQUIRED - Open the stand after setting price and hours

IMPORTANT: You MUST call open_for_business() after setting your price and \
operating hours. The stand will not operate until you do this."""

    def get_day_summary(self, *, is_first_attempt: bool = True) -> str:
        """Get the current day summary (sent each turn as input).

        This contains only the current state that changes each turn.
        Used with Responses API's input parameter.

        Args:
            is_first_attempt: If True, include tool reminder. If False, only show state.

        Returns:
            Current day summary string

        """
        profit_msg = (
            f" You made ${self.yesterday_profit:.2f} yesterday."
            if self.yesterday_profit is not None
            else ""
        )
        automation_status = "Yes" if self.has_automation else "No"
        rate_pct = (
            self.today_loan_rate * to_decimal(100)
            if self.today_loan_rate is not None
            else to_decimal(0)
        )
        loan_line = (
            f"Loan: ${self.loan_balance:.2f} outstanding "
            f"@ {rate_pct:.2f}%/day today "
            f"(interest yesterday: ${self.yesterday_interest_charged:.2f})"
        )
        summary = f"""Day {self.current_day} of {self.total_days}.{profit_msg}
Current cash: ${self.cash:.2f}
Automation: {automation_status}
{loan_line}
{self._get_historical_table()}"""

        # Only include tool reminder on first attempt of the day
        if is_first_attempt:
            summary += (
                "\nUse the available tools to check inventory, set prices, "
                "order supplies, and open for business. Continue making "
                "decisions until you call open_for_business()."
            )

        return summary

    def _get_historical_table(self) -> str:
        """Generate a table of complete performance history.

        Returns:
            Formatted table string showing all days

        """
        if not self.history:
            return ""

        table = "\nHISTORICAL PERFORMANCE:\n"
        table += (
            "Day | Price | Profit     | Customers | Hours Open | Ad Spend | Ran Out\n"
        )
        table += (
            "----|-------|------------|-----------|------------|----------|--------\n"
        )

        # Show ALL days
        for day in self.history:
            ran_out = "Yes" if day["customers_lost"] > 0 else "No"
            hours = f"{day['open_hour']}-{day['close_hour']}"
            ad_spend = day.get("ad_spend", to_decimal(0))
            table += (
                f"{day['day']:3} | ${day['price']:5.2f} | "
                f"${day['profit']:9.2f} | {day['customers_served']:9} | "
                f"{hours:^10} | ${ad_spend:7.2f} | {ran_out:^7}\n"
            )

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
        total_revenue = sum((day["revenue"] for day in self.history), to_decimal("0"))
        total_operating_cost = sum(
            (day["operating_cost"] for day in self.history),
            to_decimal("0"),
        )
        total_customers = sum(day["customers_served"] for day in self.history)
        total_lost_sales = sum(day["customers_lost"] for day in self.history)

        # Calculate average daily profit
        if self.current_day > 0:
            avg_daily_profit = (self.cash - self.starting_cash) / to_decimal(
                self.current_day,
            )
        else:
            avg_daily_profit = to_decimal("0")

        net_value = (self.cash - self.loan_balance).quantize(TWOPLACES)
        return {
            "days_played": self.current_day,
            "final_cash": self.cash,
            "loan_balance": self.loan_balance,
            "net_value": net_value,  # cash - debt; the score
            "total_interest_charged": self.total_interest_charged,
            "total_profit": net_value - self.starting_cash,
            "total_revenue": total_revenue,
            "total_operating_cost": total_operating_cost,
            "total_customers": total_customers,
            "total_lost_sales": total_lost_sales,
            "average_daily_profit": avg_daily_profit,
            "inventory_value": self.inventory.get_total_value(),
        }
