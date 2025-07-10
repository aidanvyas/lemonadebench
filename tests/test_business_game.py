"""Tests for the business game engine."""

import pytest

from src.lemonade_stand.business_game import BusinessGame


class TestBusinessGame:
    """Test cases for BusinessGame class."""

    def test_init(self):
        """Test game initialization."""
        game = BusinessGame(days=100, starting_cash=100, seed=42)

        assert game.total_days == 100
        assert game.current_day == 0
        assert game.cash == 100
        assert game.hourly_operating_cost == 5
        assert game.yesterday_profit is None

    def test_start_new_day(self):
        """Test starting a new day."""
        game = BusinessGame(seed=42)

        # Start day 1
        day_info = game.start_new_day()

        assert game.current_day == 1
        assert day_info["day"] == 1
        assert day_info["cash"] == 100
        assert "expired_items" in day_info

        # Check supply costs were generated
        assert len(game.today_supply_costs) == 4
        assert 0.04 <= game.today_supply_costs["cups"] <= 0.06  # 0.05 ± 10%
        assert 0.18 <= game.today_supply_costs["lemons"] <= 0.22  # 0.20 ± 10%

    def test_check_morning_prices(self):
        """Test checking morning prices."""
        game = BusinessGame(seed=42)

        # Should fail before day starts
        with pytest.raises(RuntimeError, match="Day hasn't started"):
            game.check_morning_prices()

        # Start day and check
        game.start_new_day()
        prices = game.check_morning_prices()

        assert "cups" in prices
        assert "lemons" in prices
        assert "sugar" in prices
        assert "water" in prices
        assert all(isinstance(p, float) for p in prices.values())

    def test_order_supplies_success(self):
        """Test ordering supplies with sufficient funds."""
        game = BusinessGame(seed=42)
        game.start_new_day()

        # Order some supplies
        result = game.order_supplies(cups=100, lemons=50, sugar=50, water=100)

        assert result["success"] is True
        assert result["remaining_cash"] < 100  # Cash decreased
        assert game.inventory.get_available("cups") == 100
        assert game.inventory.get_available("lemons") == 50

    def test_order_supplies_insufficient_funds(self):
        """Test ordering supplies without enough money."""
        game = BusinessGame(starting_cash=10, seed=42)
        game.start_new_day()

        # Try to order too much
        result = game.order_supplies(cups=1000, lemons=1000)

        assert result["success"] is False
        assert "Insufficient funds" in result["error"]
        assert game.cash == 10  # Cash unchanged
        assert game.inventory.get_available("cups") == 0  # No items added

    def test_set_operating_hours(self):
        """Test setting operating hours."""
        game = BusinessGame()

        # Valid hours
        result = game.set_operating_hours(9, 17)
        assert result["success"] is True
        assert game.open_hour == 9
        assert game.close_hour == 17
        assert game.hours_set is True

        # Invalid hours
        result = game.set_operating_hours(5, 17)  # Too early
        assert result["success"] is False
        assert "Invalid open hour" in result["error"]

        result = game.set_operating_hours(9, 22)  # Too late
        assert result["success"] is False
        assert "Invalid close hour" in result["error"]

        result = game.set_operating_hours(17, 9)  # Backwards
        assert result["success"] is False
        assert "must be after open hour" in result["error"]

    def test_set_price(self):
        """Test setting price."""
        game = BusinessGame()

        # Valid price
        result = game.set_price(2.50)
        assert result["success"] is True
        assert game.price == 2.50
        assert game.price_set is True

        # Negative price
        result = game.set_price(-1)
        assert result["success"] is False
        assert "cannot be negative" in result["error"]

    def test_simulate_day_validations(self):
        """Test day simulation validations."""
        game = BusinessGame()
        game.start_new_day()

        # Should fail without price
        with pytest.raises(RuntimeError, match="price not set"):
            game.simulate_day()

        # Set price, should still fail without hours
        game.set_price(2.0)
        with pytest.raises(RuntimeError, match="hours not set"):
            game.simulate_day()

    def test_simulate_day_success(self):
        """Test successful day simulation."""
        game = BusinessGame(seed=42)
        game.start_new_day()

        # Order supplies
        game.order_supplies(cups=100, lemons=100, sugar=100, water=100)

        # Set up day
        game.set_operating_hours(10, 14)  # 4 hours
        game.set_price(2.0)

        # Simulate
        result = game.simulate_day()

        assert result["day"] == 1
        assert result["price"] == 2.0
        assert result["hours_open"] == 4
        assert result["customers_served"] >= 0
        assert result["operating_cost"] == 20  # 4 hours * $5
        assert "profit" in result
        assert "hourly_sales" in result

        # Check inventory was used
        assert game.inventory.get_available("cups") < 100

    def test_simulate_day_stockout(self):
        """Test simulation with insufficient inventory."""
        game = BusinessGame(seed=42)
        game.start_new_day()

        # Order minimal supplies
        game.order_supplies(cups=5, lemons=5, sugar=5, water=5)

        # Set up for high demand
        game.set_operating_hours(11, 14)  # Peak hours
        game.set_price(1.0)  # Low price

        result = game.simulate_day()

        # Should have lost sales
        assert result["customers_lost"] > 0
        assert result["customers_served"] == 5  # Only what we could make

        # All inventory used
        assert game.inventory.can_make_lemonade() == 0

    def test_historical_data(self):
        """Test getting historical data."""
        game = BusinessGame(seed=42)

        # No history initially
        assert game.get_historical_data() == []

        # Play a day
        game.start_new_day()
        game.order_supplies(cups=50, lemons=50, sugar=50, water=50)
        game.set_operating_hours(9, 17)
        game.set_price(2.5)
        game.simulate_day()

        # Check history
        history = game.get_historical_data()
        assert len(history) == 1
        assert history[0]["day"] == 1
        assert history[0]["price"] == 2.5
        assert history[0]["hours"] == "9-17"
        assert "customers" in history[0]
        assert "profit" in history[0]

    def test_supply_cost_history(self):
        """Test tracking supply cost history."""
        game = BusinessGame(seed=42)

        # Play multiple days
        for _ in range(3):
            game.start_new_day()
            game.set_price(2.0)
            game.set_operating_hours(9, 17)
            game.simulate_day()

        # Check cost history
        cost_history = game.get_historical_supply_costs()
        assert len(cost_history) == 3

        # Each day should have all items
        for day_costs in cost_history:
            assert "day" in day_costs
            assert "cups" in day_costs
            assert "lemons" in day_costs
            assert "sugar" in day_costs
            assert "water" in day_costs

    def test_check_ready_for_next_day(self):
        """Test checking if ready for next day."""
        game = BusinessGame()
        game.start_new_day()

        # Initially not ready
        ready, missing = game.check_ready_for_next_day()
        assert ready is False
        assert len(missing) == 2

        # Set price
        game.set_price(2.0)
        ready, missing = game.check_ready_for_next_day()
        assert ready is False
        assert len(missing) == 1

        # Set hours
        game.set_operating_hours(9, 17)
        ready, missing = game.check_ready_for_next_day()
        assert ready is True
        assert len(missing) == 0

    def test_expiration_handling(self):
        """Test that expired items are removed."""
        game = BusinessGame(seed=42)

        # Day 1: Buy lots of lemons and minimal other supplies
        game.start_new_day()
        game.order_supplies(cups=5, lemons=50, sugar=5, water=5)
        game.set_price(10.0)  # High price to minimize sales
        game.set_operating_hours(9, 10)  # Minimal hours
        game.simulate_day()

        # Check we still have lemons
        remaining_lemons = game.inventory.get_available("lemons")
        assert remaining_lemons > 40  # Should have most lemons left

        # Fast forward to day 7 (lemons still good)
        for _ in range(2, 8):
            game.start_new_day()
            game.set_price(10.0)  # Keep high price
            game.set_operating_hours(9, 10)
            game.simulate_day()

        # Day 8: Lemons should expire (bought day 1, shelf life 7)
        day_info = game.start_new_day()
        assert "lemons" in day_info["expired_items"]
        assert day_info["expired_items"]["lemons"] > 40
        assert game.inventory.get_available("lemons") == 0

    def test_bankruptcy(self):
        """Test game over when bankrupt."""
        game = BusinessGame(starting_cash=10, seed=42)
        game.start_new_day()

        # Set high operating cost day
        game.set_price(5.0)  # High price = no customers
        game.set_operating_hours(6, 21)  # 15 hours = $75 cost
        game.simulate_day()

        # Should be bankrupt
        assert game.cash < 0
        assert game.is_game_over() is True

    def test_turn_prompts(self):
        """Test turn prompt generation."""
        game = BusinessGame()

        # First turn should have full prompt
        prompt = game.get_turn_prompt()
        assert "You run a lemonade stand" in prompt
        assert "100 days" in prompt
        assert "AVAILABLE TOOLS" in prompt

        # Start day 1
        game.start_new_day()
        game.set_price(2.0)
        game.set_operating_hours(9, 17)
        game.simulate_day()

        # Day 2 should have minimal prompt
        game.start_new_day()
        prompt = game.get_turn_prompt()
        assert "Day 2 of 100" in prompt
        assert "You made $" in prompt
        assert "AVAILABLE TOOLS" not in prompt  # No system prompt

    def test_final_results(self):
        """Test getting final game results."""
        game = BusinessGame(seed=42)

        # Play a few days
        for _ in range(1, 4):
            game.start_new_day()
            game.order_supplies(cups=20, lemons=20, sugar=20, water=20)
            game.set_price(2.0)
            game.set_operating_hours(10, 14)
            game.simulate_day()

        # Get final results
        results = game.get_final_results()

        assert results["days_played"] == 3
        assert results["final_cash"] == game.cash
        assert results["total_profit"] == game.cash - 100
        assert results["total_customers"] >= 0
        assert "average_daily_profit" in results
        assert "inventory_value" in results
