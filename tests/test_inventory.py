"""Tests for the inventory management system."""

import pytest
from decimal import Decimal

from src.lemonade_stand.business_game import Inventory


class TestInventory:
    """Test cases for Inventory class."""

    def test_init(self):
        """Test inventory initialization."""
        inv = Inventory()

        # Check all item types are initialized
        assert "cups" in inv.items
        assert "lemons" in inv.items
        assert "sugar" in inv.items
        assert "water" in inv.items

        # Check all start empty
        assert inv.get_available("cups") == 0
        assert inv.get_available("lemons") == 0
        assert inv.get_available("sugar") == 0
        assert inv.get_available("water") == 0

    def test_add_items(self):
        """Test adding items to inventory."""
        inv = Inventory()

        # Add some items
        inv.add_items("cups", 100, current_day=1)
        inv.add_items("lemons", 50, current_day=1)

        assert inv.get_available("cups") == 100
        assert inv.get_available("lemons") == 50

        # Add more of same type
        inv.add_items("cups", 50, current_day=2)
        assert inv.get_available("cups") == 150

    def test_add_invalid_item_type(self):
        """Test adding invalid item type raises error."""
        inv = Inventory()

        with pytest.raises(ValueError, match="Unknown item type"):
            inv.add_items("invalid_item", 10, current_day=1)

    def test_use_items_success(self):
        """Test using items when sufficient inventory."""
        inv = Inventory()

        # Add items
        inv.add_items("cups", 10, current_day=1)
        inv.add_items("lemons", 10, current_day=1)
        inv.add_items("sugar", 10, current_day=1)
        inv.add_items("water", 10, current_day=1)

        # Use some items
        recipe = {"cups": 5, "lemons": 5, "sugar": 5, "water": 5}
        result = inv.use_items(recipe)

        assert result is True
        assert inv.get_available("cups") == 5
        assert inv.get_available("lemons") == 5
        assert inv.get_available("sugar") == 5
        assert inv.get_available("water") == 5

    def test_use_items_insufficient(self):
        """Test using items when insufficient inventory."""
        inv = Inventory()

        # Add items (not enough lemons)
        inv.add_items("cups", 10, current_day=1)
        inv.add_items("lemons", 2, current_day=1)
        inv.add_items("sugar", 10, current_day=1)
        inv.add_items("water", 10, current_day=1)

        # Try to use more than available
        recipe = {"cups": 5, "lemons": 5, "sugar": 5, "water": 5}
        result = inv.use_items(recipe)

        assert result is False
        # Inventory should be unchanged
        assert inv.get_available("cups") == 10
        assert inv.get_available("lemons") == 2

    def test_fifo_usage(self):
        """Test that items are used FIFO (first in, first out)."""
        inv = Inventory()

        # Add items on different days
        inv.add_items("lemons", 5, current_day=1)  # Expires day 8
        inv.add_items("lemons", 5, current_day=3)  # Expires day 10

        # Use some items
        recipe = {"lemons": 7}
        inv.use_items(recipe)

        # Should have used all 5 from day 1 and 2 from day 3
        assert inv.get_available("lemons") == 3

        # Check remaining are from the later batch
        details = inv.get_inventory_details()
        assert len(details["lemons"]) == 1
        assert details["lemons"][0]["quantity"] == 3
        assert details["lemons"][0]["expires_day"] == 10

    def test_expiration(self):
        """Test removing expired items."""
        inv = Inventory()

        # Add items with different expiry
        inv.add_items("cups", 100, current_day=1)  # Expires day 31
        inv.add_items("lemons", 50, current_day=1)  # Expires day 8
        inv.add_items("sugar", 75, current_day=1)  # Expires day 61
        inv.add_items("water", 200, current_day=1)  # Never expires

        # Move to day 9 (lemons should expire)
        expired = inv.remove_expired(current_day=9)

        assert "lemons" in expired
        assert expired["lemons"] == 50
        assert inv.get_available("lemons") == 0
        assert inv.get_available("cups") == 100  # Still good
        assert inv.get_available("water") == 200  # Never expires

    def test_water_never_expires(self):
        """Test that water never expires."""
        inv = Inventory()

        inv.add_items("water", 100, current_day=1)

        # Even after many days, water should remain
        expired = inv.remove_expired(current_day=10000)

        assert "water" not in expired
        assert inv.get_available("water") == 100

    def test_can_make_lemonade(self):
        """Test calculating how many lemonades can be made."""
        inv = Inventory()

        # Add uneven amounts
        inv.add_items("cups", 10, current_day=1)
        inv.add_items("lemons", 5, current_day=1)
        inv.add_items("sugar", 20, current_day=1)
        inv.add_items("water", 100, current_day=1)

        # Limited by lemons (only 5)
        assert inv.can_make_lemonade() == 5

        # Use up all lemons
        recipe = {"cups": 5, "lemons": 5, "sugar": 5, "water": 5}
        inv.use_items(recipe)

        # Now can't make any
        assert inv.can_make_lemonade() == 0

    def test_get_inventory_details(self):
        """Test getting detailed inventory information."""
        inv = Inventory()

        # Add items on different days
        inv.add_items("cups", 50, current_day=1)
        inv.add_items("cups", 30, current_day=5)
        inv.add_items("water", 100, current_day=1)

        details = inv.get_inventory_details()

        # Check cups have two batches
        assert len(details["cups"]) == 2
        assert details["cups"][0]["quantity"] == 50
        assert details["cups"][0]["expires_day"] == 31
        assert details["cups"][1]["quantity"] == 30
        assert details["cups"][1]["expires_day"] == 35

        # Check water shows 'never' for expiry
        assert len(details["water"]) == 1
        assert details["water"][0]["expires_day"] == "never"

    def test_get_total_value(self):
        """Test calculating total inventory value."""
        inv = Inventory()

        # Add items
        inv.add_items("cups", 100, current_day=1)  # 100 * 0.05 = 5.00
        inv.add_items("lemons", 50, current_day=1)  # 50 * 0.20 = 10.00
        inv.add_items("sugar", 75, current_day=1)  # 75 * 0.10 = 7.50
        inv.add_items("water", 200, current_day=1)  # 200 * 0.02 = 4.00

        assert inv.get_total_value() == Decimal("26.50")
