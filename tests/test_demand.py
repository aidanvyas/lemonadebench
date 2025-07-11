"""Tests for the demand modeling system."""

from src.lemonade_stand.business_game import DemandModel


class TestDemandModel:
    """Test cases for DemandModel class."""

    def test_init(self):
        """Test demand model initialization."""
        model = DemandModel()

        assert model.base_demand_intercept == 50
        assert model.price_sensitivity == 10

        # Test with custom parameters
        custom_model = DemandModel(base_demand_intercept=100, price_sensitivity=20)
        assert custom_model.base_demand_intercept == 100
        assert custom_model.price_sensitivity == 20

    def test_calculate_base_demand(self):
        """Test base demand calculation."""
        model = DemandModel(base_demand_intercept=50, price_sensitivity=10)

        # At price 0, demand should be intercept
        assert model.calculate_base_demand(0) == 50

        # At price 2, demand = 50 - 10*2 = 30
        assert model.calculate_base_demand(2) == 30

        # At price 5, demand = 50 - 10*5 = 0
        assert model.calculate_base_demand(5) == 0

        # At price 10, demand should be 0 (not negative)
        assert model.calculate_base_demand(10) == 0

    def test_hour_multipliers(self):
        """Test getting hour multipliers."""
        model = DemandModel()

        # Test known hours
        assert model.get_hour_multiplier(6) == 0.3  # Early morning
        assert model.get_hour_multiplier(12) == 1.5  # Lunch peak
        assert model.get_hour_multiplier(20) == 0.4  # Late evening

        # Test hour outside operating hours
        assert model.get_hour_multiplier(3) == 0.0  # 3am - closed
        assert model.get_hour_multiplier(22) == 0.0  # 10pm - closed

    def test_calculate_customers_no_randomness(self):
        """Test customer calculation without random variation."""
        model = DemandModel()

        # Price $2, noon (peak), no randomness
        # Base = 50 - 10*2 = 30
        # With multiplier = 30 * 1.5 = 45
        customers = model.calculate_customers(
            price=2.0, hour=12, random_variation=False
        )
        assert customers == 45

        # Price $2, early morning, no randomness
        # Base = 30, multiplier = 0.3, result = 9
        customers = model.calculate_customers(price=2.0, hour=6, random_variation=False)
        assert customers == 9

        # Price $5 (zero base demand), any hour
        customers = model.calculate_customers(
            price=5.0, hour=12, random_variation=False
        )
        assert customers == 0

    def test_calculate_customers_with_randomness(self):
        """Test customer calculation with random variation."""
        model = DemandModel()
        model.set_random_seed(42)  # For reproducibility

        # Run multiple times to check variation
        results = []
        for _ in range(10):
            customers = model.calculate_customers(
                price=2.0, hour=12, random_variation=True
            )
            results.append(customers)

        # Should have some variation
        assert len(set(results)) > 1

        # All should be within ±10% of base (45)
        for result in results:
            assert 40 <= result <= 50  # 45 ± 10% rounded

    def test_calculate_daily_customers(self):
        """Test calculating customers for full day."""
        model = DemandModel()
        model.set_random_seed(42)

        # Open 9am-5pm
        customers = model.calculate_daily_customers(
            price=2.0, open_hour=9, close_hour=17, random_variation=False
        )

        # Should have 8 hours of data
        assert len(customers) == 8
        assert 9 in customers
        assert 16 in customers
        assert 17 not in customers  # Close hour is exclusive
        assert 8 not in customers  # Before open

        # Check peak hour has most customers
        assert customers[12] > customers[9]  # Lunch > morning

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        model = DemandModel()

        # Negative price should still work (theoretical)
        base = model.calculate_base_demand(-1)
        assert base == 60  # 50 - 10*(-1) = 60

        # Very high price
        customers = model.calculate_customers(
            price=100, hour=12, random_variation=False
        )
        assert customers == 0

        # Operating outside normal hours
        customers = model.calculate_customers(price=2, hour=23, random_variation=False)
        assert customers == 0  # Closed

    def test_random_seed_consistency(self):
        """Test that setting seed gives consistent results."""
        model1 = DemandModel()
        model1.set_random_seed(123)

        model2 = DemandModel()
        model2.set_random_seed(123)

        # Should get same results
        results1 = [model1.calculate_customers(2.0, 12) for _ in range(5)]
        results2 = [model2.calculate_customers(2.0, 12) for _ in range(5)]

        assert results1 == results2

    def test_get_peak_hours(self):
        """Test getting peak hours."""
        model = DemandModel()
        peak_hours = model.get_peak_hours()
        
        assert peak_hours == [11, 12, 13, 14]
        assert all(isinstance(h, int) for h in peak_hours)
