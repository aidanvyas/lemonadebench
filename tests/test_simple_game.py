"""Tests for the simplified price-only game."""

import sys
from pathlib import Path

# Add repository root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.lemonade_stand.simple_game import SimpleLemonadeGame


def test_game_initialization():
    """Test game initializes correctly."""
    game = SimpleLemonadeGame(days=50)

    assert game.days == 50
    assert game.current_day == 1
    assert game.cash == 100.0
    assert not game.game_over
    assert game.demand_intercept == 100
    assert game.demand_slope == 25
    assert game.optimal_price == 2.00
    assert game.suggested_starting_price is None


def test_demand_calculation():
    """Test demand curve calculation."""
    game = SimpleLemonadeGame()

    # Linear demand: customers = max(0, 100 - 25*price)
    assert game.calculate_demand(1.00) == 75  # 100 - 25*1
    assert game.calculate_demand(2.00) == 50  # 100 - 25*2 (optimal)
    assert game.calculate_demand(3.00) == 25  # 100 - 25*3
    assert game.calculate_demand(4.00) == 0  # max(0, 100 - 100)
    assert game.calculate_demand(5.00) == 0  # max(0, negative)

    # Edge cases
    assert game.calculate_demand(0) == 100  # Max customers at $0
    assert game.calculate_demand(-1.00) == 0  # Negative price returns 0


def test_custom_demand_parameters():
    """Test game with custom demand parameters."""
    # Test inverse demand: Q = 50 - 25p, optimal at $1
    game = SimpleLemonadeGame(demand_intercept=50, demand_slope=25)
    
    assert game.demand_intercept == 50
    assert game.demand_slope == 25
    assert game.optimal_price == 1.00  # 50 / (2 * 25)
    
    # Test demand calculation
    assert game.calculate_demand(0.50) == 37  # 50 - 25*0.5 = 37.5 -> 37
    assert game.calculate_demand(1.00) == 25  # 50 - 25*1 (optimal)
    assert game.calculate_demand(2.00) == 0   # 50 - 50 = 0


def test_play_turn():
    """Test playing a single turn."""
    game = SimpleLemonadeGame()

    result = game.play_turn(price=1.00)

    assert result["day"] == 1
    assert result["price"] == 1.00
    assert result["customers"] == 75  # 100 - 25*1
    assert result["revenue"] == 75.0  # 75 * $1
    assert result["costs"] == 0.0  # No costs
    # Profit should be 75 ± 10%
    assert 67.5 <= result["profit"] <= 82.5
    # Cash should be $100 start + profit
    assert 167.5 <= result["cash"] <= 182.5
    assert game.current_day == 2


def test_profit_calculation():
    """Test profit calculations at different prices."""
    game = SimpleLemonadeGame()

    # At suggested price
    result = game.play_turn(price=1.00)
    # 75 customers * $1 = $75 base profit
    assert result["customers"] == 75  # 100 - 25*1
    assert 67.5 <= result["profit"] <= 82.5  # $75 ± 10%

    # Reset game
    game = SimpleLemonadeGame()

    # At optimal price
    result = game.play_turn(price=2.00)
    # 50 customers * $2.00 = $100 base profit
    assert result["customers"] == 50  # 100 - 25*2
    assert 90.0 <= result["profit"] <= 110.0  # $100 ± 10%

    # High price - fewer customers
    game = SimpleLemonadeGame()
    result = game.play_turn(price=3.00)
    assert result["customers"] == 25  # 100 - 25*3
    assert 67.5 <= result["profit"] <= 82.5  # $75 ± 10%

    # Price = 0 case (special: should always be 0)
    game = SimpleLemonadeGame()
    result = game.play_turn(price=0.00)
    assert result["customers"] == 100  # Max customers
    assert result["revenue"] == 0.0  # No revenue when price is 0
    assert result["costs"] == 0.0  # No costs
    assert result["profit"] == 0.0  # No profit (0 * any random factor = 0)


def test_game_over_conditions():
    """Test game over conditions."""
    # Test day limit
    game = SimpleLemonadeGame(days=3)
    game.play_turn(price=2.00)
    game.play_turn(price=2.00)
    game.play_turn(price=2.00)
    assert game.game_over

    # With no costs, cash can't go negative from operations
    # Game only ends from day limit


def test_get_state():
    """Test getting game state."""
    game = SimpleLemonadeGame(days=10)

    state = game.get_state()
    assert state["day"] == 1
    assert state["cash"] == 100.0
    assert state["days_remaining"] == 10
    assert not state["game_over"]
    assert state["last_result"] is None

    # Play a turn
    game.play_turn(price=1.00)
    state = game.get_state()

    assert state["day"] == 2
    assert state["days_remaining"] == 9
    assert state["last_result"] is not None
    assert state["last_result"]["price"] == 1.00
