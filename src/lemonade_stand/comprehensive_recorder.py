"""Comprehensive recording of all model interactions for analysis."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import statistics
from collections import defaultdict
import json


@dataclass
class DailyMetrics:
    """Metrics for a single day of business."""

    day: int
    survived: bool
    cash_start: float
    cash_end: float
    revenue: float
    operating_cost: float
    supply_cost: float
    profit: float

    # Customer metrics
    customers_wanted: int
    customers_served: int
    customers_lost: int
    service_rate: float

    # Inventory metrics
    inventory_purchased: Dict[str, int]
    inventory_used: Dict[str, int]
    inventory_expired: Dict[str, int]
    inventory_value_end: float

    # Pricing metrics
    price_set: float
    hours_open: int

    # Peak vs off-peak
    peak_customers_wanted: int
    peak_customers_served: int
    off_peak_customers_wanted: int
    off_peak_customers_served: int

    # Tool usage
    tool_calls: Dict[str, int]
    turn_attempts: int

    # Economic efficiency
    revenue_per_customer: float
    cost_per_customer_served: float
    opportunity_cost: float  # Lost revenue from stockouts


@dataclass
class GameMetrics:
    """Comprehensive metrics for an entire game."""

    model: str
    game_number: int
    starting_cash: float
    days_target: int

    # Survival metrics
    days_survived: int
    survival_rate: float  # days_survived / days_target
    final_cash: float
    went_bankrupt: bool
    bankruptcy_day: Optional[int]

    # Economic metrics
    total_revenue: float
    total_costs: float
    total_profit: float
    burn_rate: float  # Average daily loss
    days_to_bankruptcy: float  # Projected based on burn rate

    # Customer service metrics
    total_customers_wanted: int
    total_customers_served: int
    total_customers_lost: int
    overall_service_rate: float
    stockout_days: int
    stockout_rate: float

    # Inventory efficiency
    total_inventory_purchased: Dict[str, int]
    total_inventory_used: Dict[str, int]
    total_inventory_expired: Dict[str, int]
    inventory_efficiency: Dict[str, float]  # used / purchased
    expired_value: float

    # Pricing strategy
    unique_prices: List[float]
    price_changes: int
    average_price: float
    price_variance: float
    optimal_price_discovery: bool  # Did they find ~$2.69?

    # Operational metrics
    average_hours_open: float
    total_operating_hours: int
    revenue_per_hour: float

    # Peak optimization
    peak_focus_rate: float  # % of served customers during peak hours
    peak_revenue_share: float  # % of revenue from peak hours

    # Tool usage patterns
    total_tool_calls: Dict[str, int]
    tool_calls_per_day: Dict[str, float]
    average_turn_attempts: float

    # Learning metrics
    performance_trend: str  # "improving", "declining", "stable"
    learning_rate: float  # Rate of improvement in key metrics

    # Token usage
    total_tokens: int
    tokens_per_day: float
    total_cost: float
    cost_per_day: float

    # Daily history for analysis
    daily_metrics: List[DailyMetrics] = field(default_factory=list)


class MetricsAnalyzer:
    """Analyze game results and compute comprehensive metrics."""

    def __init__(self):
        self.peak_hours = [11, 12, 13, 14]  # Peak demand hours
        self.optimal_price = 2.69  # Theoretical optimum
        self.optimal_daily_profit = 625.54

    def analyze_game(self, game_result: Dict[str, Any]) -> GameMetrics:
        """Analyze a single game and compute all metrics."""
        # Basic info
        metrics = GameMetrics(
            model=game_result["model"],
            game_number=game_result["game_number"],
            starting_cash=1000,  # Default, update if in data
            days_target=game_result.get("days_target", 100),
            days_survived=game_result["days_played"],
            survival_rate=game_result["days_played"]
            / game_result.get("days_target", 100),
            final_cash=game_result["final_cash"],
            went_bankrupt=game_result["final_cash"] < 0,
            bankruptcy_day=game_result["days_played"]
            if game_result["final_cash"] < 0
            else None,
            total_revenue=game_result["total_revenue"],
            total_costs=game_result["total_operating_cost"]
            + self._calculate_supply_costs(game_result),
            total_profit=game_result["total_profit"],
            burn_rate=-game_result["average_daily_profit"]
            if game_result["average_daily_profit"] < 0
            else 0,
            days_to_bankruptcy=0,  # Calculate below
            total_customers_wanted=game_result["total_customers"]
            + game_result["total_customers_lost"],
            total_customers_served=game_result["total_customers"],
            total_customers_lost=game_result["total_customers_lost"],
            overall_service_rate=game_result["total_customers"]
            / (game_result["total_customers"] + game_result["total_customers_lost"])
            if (game_result["total_customers"] + game_result["total_customers_lost"])
            > 0
            else 0,
            stockout_days=game_result["days_with_stockouts"],
            stockout_rate=game_result["stockout_rate"],
            total_inventory_purchased=self._sum_inventory_purchases(game_result),
            total_inventory_used=self._calculate_inventory_used(game_result),
            total_inventory_expired=game_result["total_expired_items"],
            inventory_efficiency={},  # Calculate below
            expired_value=game_result["total_expired_value"],
            unique_prices=[],  # Extract below
            price_changes=0,  # Calculate below
            average_price=0,  # Calculate below
            price_variance=0,  # Calculate below
            optimal_price_discovery=False,  # Check below
            average_hours_open=15,  # Default, update if variable
            total_operating_hours=15 * game_result["days_played"],
            revenue_per_hour=game_result["total_revenue"]
            / (15 * game_result["days_played"])
            if game_result["days_played"] > 0
            else 0,
            peak_focus_rate=game_result["peak_customer_ratio"],
            peak_revenue_share=0,  # Calculate if we have hourly data
            total_tool_calls={},  # Extract from history
            tool_calls_per_day={},
            average_turn_attempts=game_result.get("average_turn_attempts", 1),
            performance_trend="stable",  # Analyze trend
            learning_rate=0,  # Calculate improvement rate
            total_tokens=game_result["token_usage"]["total_tokens"],
            tokens_per_day=game_result["token_usage"]["total_tokens"]
            / game_result["days_played"]
            if game_result["days_played"] > 0
            else 0,
            total_cost=game_result["cost_info"]["total_cost"],
            cost_per_day=game_result["cost_info"]["total_cost"]
            / game_result["days_played"]
            if game_result["days_played"] > 0
            else 0,
        )

        # Calculate days to bankruptcy
        if metrics.burn_rate > 0:
            metrics.days_to_bankruptcy = metrics.starting_cash / metrics.burn_rate

        # Extract pricing strategy
        prices = []
        price_changes = 0
        last_price = None

        for day_data in game_result["game_history"]:
            price = day_data["price"]
            prices.append(price)
            if last_price is not None and price != last_price:
                price_changes += 1
            last_price = price

            # Check for optimal price discovery
            if abs(price - self.optimal_price) < 0.20:  # Within $0.20 of optimal
                metrics.optimal_price_discovery = True

        metrics.unique_prices = sorted(list(set(prices)))
        metrics.price_changes = price_changes
        metrics.average_price = statistics.mean(prices) if prices else 0
        metrics.price_variance = statistics.variance(prices) if len(prices) > 1 else 0

        # Calculate inventory efficiency
        for item in ["cups", "lemons", "sugar", "water"]:
            purchased = metrics.total_inventory_purchased.get(item, 0)
            used = metrics.total_inventory_used.get(item, 0)
            if purchased > 0:
                metrics.inventory_efficiency[item] = used / purchased
            else:
                metrics.inventory_efficiency[item] = 0

        # Analyze performance trend
        metrics.performance_trend = self._analyze_trend(game_result)
        metrics.learning_rate = self._calculate_learning_rate(game_result)

        # Process daily metrics
        metrics.daily_metrics = self._extract_daily_metrics(game_result)

        return metrics

    def _calculate_supply_costs(self, game_result: Dict[str, Any]) -> float:
        """Calculate total supply costs from history."""
        # This would need the actual purchase history
        # For now, estimate from revenue and profit
        return (
            game_result["total_revenue"]
            - game_result["total_profit"]
            - game_result["total_operating_cost"]
        )

    def _sum_inventory_purchases(self, game_result: Dict[str, Any]) -> Dict[str, int]:
        """Sum all inventory purchases from game history."""
        # This would need detailed purchase logs
        # Placeholder for now
        return {"cups": 0, "lemons": 0, "sugar": 0, "water": 0}

    def _calculate_inventory_used(self, game_result: Dict[str, Any]) -> Dict[str, int]:
        """Calculate inventory actually used to serve customers."""
        # Estimate based on customers served
        # Each lemonade needs specific ingredients
        customers = game_result["total_customers"]
        return {
            "cups": customers,
            "lemons": customers * 2,  # Assuming 2 lemons per cup
            "sugar": customers * 20,  # 20g sugar per cup
            "water": customers * 200,  # 200ml water per cup
        }

    def _analyze_trend(self, game_result: Dict[str, Any]) -> str:
        """Analyze if performance is improving, declining, or stable."""
        if len(game_result["game_history"]) < 5:
            return "stable"

        # Look at profit trend over time
        profits = [day["profit"] for day in game_result["game_history"]]
        first_half = statistics.mean(profits[: len(profits) // 2])
        second_half = statistics.mean(profits[len(profits) // 2 :])

        if second_half > first_half * 1.1:
            return "improving"
        elif second_half < first_half * 0.9:
            return "declining"
        else:
            return "stable"

    def _calculate_learning_rate(self, game_result: Dict[str, Any]) -> float:
        """Calculate rate of improvement in key metrics."""
        if len(game_result["game_history"]) < 3:
            return 0.0

        # Compare service rate improvement
        history = game_result["game_history"]
        early_days = history[:3]
        late_days = history[-3:]

        early_service_rate = sum(d["customers_served"] for d in early_days) / sum(
            d["customers_served"] + d["customers_lost"] for d in early_days
        )
        late_service_rate = sum(d["customers_served"] for d in late_days) / sum(
            d["customers_served"] + d["customers_lost"] for d in late_days
        )

        if early_service_rate > 0:
            return (late_service_rate - early_service_rate) / early_service_rate
        return 0.0

    def _extract_daily_metrics(self, game_result: Dict[str, Any]) -> List[DailyMetrics]:
        """Extract detailed metrics for each day."""
        daily_metrics = []
        cash_history = game_result.get("daily_cash_history", [])

        for i, day_data in enumerate(game_result["game_history"]):
            # Calculate peak vs off-peak
            peak_wanted = 0
            peak_served = 0
            off_peak_wanted = 0
            off_peak_served = 0

            for hour, sales in day_data["hourly_sales"].items():
                if int(hour) in self.peak_hours:
                    peak_wanted += sales["customers_wanted"]
                    peak_served += sales["customers_served"]
                else:
                    off_peak_wanted += sales["customers_wanted"]
                    off_peak_served += sales["customers_served"]

            # Create daily metric
            daily_metric = DailyMetrics(
                day=day_data["day"],
                survived=True,  # If we have data, they survived the day
                cash_start=cash_history[i] if i < len(cash_history) else 0,
                cash_end=day_data["cash"],
                revenue=day_data["revenue"],
                operating_cost=day_data["operating_cost"],
                supply_cost=day_data["revenue"]
                - day_data["profit"]
                - day_data["operating_cost"],
                profit=day_data["profit"],
                customers_wanted=day_data["customers_served"]
                + day_data["customers_lost"],
                customers_served=day_data["customers_served"],
                customers_lost=day_data["customers_lost"],
                service_rate=day_data["customers_served"]
                / (day_data["customers_served"] + day_data["customers_lost"])
                if (day_data["customers_served"] + day_data["customers_lost"]) > 0
                else 0,
                inventory_purchased={},  # Would need purchase logs
                inventory_used={},  # Calculate from customers served
                inventory_expired={},  # Would need expiry logs
                inventory_value_end=0,  # Would need inventory tracking
                price_set=day_data["price"],
                hours_open=day_data["hours_open"],
                peak_customers_wanted=peak_wanted,
                peak_customers_served=peak_served,
                off_peak_customers_wanted=off_peak_wanted,
                off_peak_customers_served=off_peak_served,
                tool_calls={},  # Would need tool call logs
                turn_attempts=1,  # Default, update if available
                revenue_per_customer=day_data["revenue"] / day_data["customers_served"]
                if day_data["customers_served"] > 0
                else 0,
                cost_per_customer_served=(
                    day_data["operating_cost"]
                    + (
                        day_data["revenue"]
                        - day_data["profit"]
                        - day_data["operating_cost"]
                    )
                )
                / day_data["customers_served"]
                if day_data["customers_served"] > 0
                else 0,
                opportunity_cost=day_data["customers_lost"] * day_data["price"],
            )

            daily_metrics.append(daily_metric)

        return daily_metrics


class ComprehensiveRecorder:
    """Records absolutely everything about model interactions."""

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records = []

    def record_request(self, day: int, request_data: dict):
        """Record the full request sent to the API."""
        self.records.append(
            {
                "type": "request",
                "day": day,
                "timestamp": datetime.now().isoformat(),
                "data": request_data,
            }
        )

    def record_response(self, day: int, response: Any):
        """Record the full response from the API."""
        # Convert response object to dict
        response_data = {
            "id": response.id if hasattr(response, "id") else None,
            "model": response.model if hasattr(response, "model") else None,
            "created_at": response.created_at
            if hasattr(response, "created_at")
            else None,
            "status": response.status if hasattr(response, "status") else None,
            "output": [],
        }

        # Extract all output items
        if hasattr(response, "output"):
            for item in response.output:
                output_item = {
                    "type": item.type if hasattr(item, "type") else None,
                    "id": item.id if hasattr(item, "id") else None,
                }

                # Handle different output types
                if item.type == "message":
                    output_item["role"] = item.role if hasattr(item, "role") else None
                    output_item["content"] = []
                    if hasattr(item, "content"):
                        for content in item.content:
                            output_item["content"].append(
                                {
                                    "type": content.type
                                    if hasattr(content, "type")
                                    else None,
                                    "text": content.text
                                    if hasattr(content, "text")
                                    else None,
                                }
                            )

                elif item.type == "function_call":
                    output_item["name"] = item.name if hasattr(item, "name") else None
                    output_item["arguments"] = (
                        item.arguments if hasattr(item, "arguments") else None
                    )
                    output_item["call_id"] = (
                        item.call_id if hasattr(item, "call_id") else None
                    )

                elif item.type == "reasoning":
                    output_item["content"] = (
                        item.content if hasattr(item, "content") else None
                    )

                response_data["output"].append(output_item)

        # Extract usage data
        if hasattr(response, "usage"):
            usage = response.usage
            response_data["usage"] = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
                "reasoning_tokens": 0,
            }

            if hasattr(usage, "prompt_tokens_details"):
                response_data["usage"]["prompt_tokens_details"] = {
                    "cached_tokens": getattr(
                        usage.prompt_tokens_details, "cached_tokens", 0
                    )
                }

            if hasattr(usage, "output_tokens_details"):
                response_data["usage"]["reasoning_tokens"] = getattr(
                    usage.output_tokens_details, "reasoning_tokens", 0
                )

        # Extract reasoning data
        if hasattr(response, "reasoning"):
            response_data["reasoning"] = {
                "effort": getattr(response.reasoning, "effort", None),
                "summary": getattr(response.reasoning, "summary", None),
            }

        self.records.append(
            {
                "type": "response",
                "day": day,
                "timestamp": datetime.now().isoformat(),
                "data": response_data,
            }
        )

    def record_tool_execution(
        self, day: int, tool_name: str, arguments: dict, result: str
    ):
        """Record tool execution details."""
        self.records.append(
            {
                "type": "tool_execution",
                "day": day,
                "timestamp": datetime.now().isoformat(),
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )

    def record_game_state(self, day: int, price: float, customers: int, profit: float):
        """Record game state after each turn."""
        self.records.append(
            {
                "type": "game_state",
                "day": day,
                "timestamp": datetime.now().isoformat(),
                "price": price,
                "customers": customers,
                "profit": profit,
            }
        )

    def record_error(self, day: int, error: Exception):
        """Record any errors that occur."""
        self.records.append(
            {
                "type": "error",
                "day": day,
                "timestamp": datetime.now().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )

    def get_recording_data(self, test_name: str, model_name: str) -> dict:
        """Get the recording data as a dictionary."""
        return {
            "session_id": self.session_id,
            "test_name": test_name,
            "model_name": model_name,
            "start_time": self.records[0]["timestamp"] if self.records else None,
            "end_time": self.records[-1]["timestamp"] if self.records else None,
            "total_records": len(self.records),
            "records": self.records,
        }


def generate_metrics_report(game_metrics: List[GameMetrics]) -> Dict[str, Any]:
    """Generate a comprehensive report from multiple game metrics."""
    report = {
        "summary": {
            "total_games": len(game_metrics),
            "models_tested": list(set(m.model for m in game_metrics)),
        },
        "survival_analysis": {
            "average_days_survived": statistics.mean(
                m.days_survived for m in game_metrics
            ),
            "survival_rate": statistics.mean(m.survival_rate for m in game_metrics),
            "bankruptcy_rate": sum(1 for m in game_metrics if m.went_bankrupt)
            / len(game_metrics),
            "average_days_to_bankruptcy": statistics.mean(
                m.days_to_bankruptcy for m in game_metrics if m.days_to_bankruptcy > 0
            )
            if any(m.days_to_bankruptcy > 0 for m in game_metrics)
            else 0,
        },
        "economic_performance": {
            "average_burn_rate": statistics.mean(m.burn_rate for m in game_metrics),
            "average_total_profit": statistics.mean(
                m.total_profit for m in game_metrics
            ),
            "profit_variance": statistics.variance(
                [m.total_profit for m in game_metrics]
            )
            if len(game_metrics) > 1
            else 0,
            "best_profit": max(m.total_profit for m in game_metrics),
            "worst_profit": min(m.total_profit for m in game_metrics),
        },
        "customer_service": {
            "average_service_rate": statistics.mean(
                m.overall_service_rate for m in game_metrics
            ),
            "average_stockout_rate": statistics.mean(
                m.stockout_rate for m in game_metrics
            ),
            "total_customers_lost": sum(m.total_customers_lost for m in game_metrics),
            "total_opportunity_cost": sum(
                m.total_customers_lost * m.average_price for m in game_metrics
            ),
        },
        "inventory_management": {
            "average_efficiency": {
                item: statistics.mean(
                    m.inventory_efficiency.get(item, 0) for m in game_metrics
                )
                for item in ["cups", "lemons", "sugar", "water"]
            },
            "total_expired_value": sum(m.expired_value for m in game_metrics),
        },
        "pricing_strategy": {
            "price_discovery_rate": sum(
                1 for m in game_metrics if m.optimal_price_discovery
            )
            / len(game_metrics),
            "average_price_changes": statistics.mean(
                m.price_changes for m in game_metrics
            ),
            "average_price": statistics.mean(m.average_price for m in game_metrics),
            "price_variance": statistics.mean(m.price_variance for m in game_metrics),
        },
        "operational_efficiency": {
            "average_revenue_per_hour": statistics.mean(
                m.revenue_per_hour for m in game_metrics
            ),
            "peak_focus_rate": statistics.mean(m.peak_focus_rate for m in game_metrics),
        },
        "learning_metrics": {
            "improving_games": sum(
                1 for m in game_metrics if m.performance_trend == "improving"
            )
            / len(game_metrics),
            "average_learning_rate": statistics.mean(
                m.learning_rate for m in game_metrics
            ),
        },
        "cost_analysis": {
            "average_tokens_per_day": statistics.mean(
                m.tokens_per_day for m in game_metrics
            ),
            "average_cost_per_day": statistics.mean(
                m.cost_per_day for m in game_metrics
            ),
            "total_cost": sum(m.total_cost for m in game_metrics),
        },
    }

    # Add model-specific breakdowns
    model_metrics = defaultdict(list)
    for m in game_metrics:
        model_metrics[m.model].append(m)

    report["model_comparison"] = {}
    for model, metrics in model_metrics.items():
        report["model_comparison"][model] = {
            "games": len(metrics),
            "avg_days_survived": statistics.mean(m.days_survived for m in metrics),
            "avg_profit": statistics.mean(m.total_profit for m in metrics),
            "avg_service_rate": statistics.mean(
                m.overall_service_rate for m in metrics
            ),
            "price_discovery_rate": sum(1 for m in metrics if m.optimal_price_discovery)
            / len(metrics),
            "avg_cost_per_day": statistics.mean(m.cost_per_day for m in metrics),
        }

    return report


def save_metrics_report(report: Dict[str, Any], filename: str):
    """Save metrics report to JSON file."""
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)


def print_metrics_summary(report: Dict[str, Any]):
    """Print a formatted summary of the metrics report."""
    print("\n" + "=" * 80)
    print("LEMONADEBENCH v0.5 - COMPREHENSIVE METRICS REPORT")
    print("=" * 80)

    print(f"\nTotal Games Analyzed: {report['summary']['total_games']}")
    print(f"Models Tested: {', '.join(report['summary']['models_tested'])}")

    print("\n--- SURVIVAL METRICS ---")
    print(
        f"Average Days Survived: {report['survival_analysis']['average_days_survived']:.1f}"
    )
    print(f"Survival Rate: {report['survival_analysis']['survival_rate']:.1%}")
    print(f"Bankruptcy Rate: {report['survival_analysis']['bankruptcy_rate']:.1%}")
    print(
        f"Average Days to Bankruptcy: {report['survival_analysis']['average_days_to_bankruptcy']:.1f}"
    )

    print("\n--- ECONOMIC PERFORMANCE ---")
    print(
        f"Average Burn Rate: ${report['economic_performance']['average_burn_rate']:.2f}/day"
    )
    print(
        f"Average Total Profit: ${report['economic_performance']['average_total_profit']:.2f}"
    )
    print(f"Best Performance: ${report['economic_performance']['best_profit']:.2f}")
    print(f"Worst Performance: ${report['economic_performance']['worst_profit']:.2f}")

    print("\n--- CUSTOMER SERVICE ---")
    print(
        f"Average Service Rate: {report['customer_service']['average_service_rate']:.1%}"
    )
    print(
        f"Average Stockout Rate: {report['customer_service']['average_stockout_rate']:.1%}"
    )
    print(
        f"Total Customers Lost: {report['customer_service']['total_customers_lost']:,}"
    )
    print(
        f"Total Opportunity Cost: ${report['customer_service']['total_opportunity_cost']:.2f}"
    )

    print("\n--- INVENTORY EFFICIENCY ---")
    for item, efficiency in report["inventory_management"][
        "average_efficiency"
    ].items():
        print(f"{item.capitalize()} Efficiency: {efficiency:.1%}")
    print(
        f"Total Expired Value: ${report['inventory_management']['total_expired_value']:.2f}"
    )

    print("\n--- PRICING STRATEGY ---")
    print(
        f"Optimal Price Discovery Rate: {report['pricing_strategy']['price_discovery_rate']:.1%}"
    )
    print(
        f"Average Price Changes: {report['pricing_strategy']['average_price_changes']:.1f}"
    )
    print(f"Average Price Set: ${report['pricing_strategy']['average_price']:.2f}")

    print("\n--- LEARNING & ADAPTATION ---")
    print(
        f"Games Showing Improvement: {report['learning_metrics']['improving_games']:.1%}"
    )
    print(
        f"Average Learning Rate: {report['learning_metrics']['average_learning_rate']:.1%}"
    )

    print("\n--- COST ANALYSIS ---")
    print(
        f"Average Tokens/Day: {report['cost_analysis']['average_tokens_per_day']:.0f}"
    )
    print(f"Average Cost/Day: ${report['cost_analysis']['average_cost_per_day']:.4f}")
    print(f"Total Cost: ${report['cost_analysis']['total_cost']:.4f}")

    if "model_comparison" in report:
        print("\n--- MODEL COMPARISON ---")
        print(
            f"{'Model':<20} {'Days':<8} {'Profit':<12} {'Service':<10} {'Discovery':<10} {'Cost/Day':<10}"
        )
        print("-" * 70)
        for model, stats in report["model_comparison"].items():
            print(
                f"{model:<20} {stats['avg_days_survived']:<8.1f} ${stats['avg_profit']:<11.2f} "
                f"{stats['avg_service_rate']:<9.1%} {stats['price_discovery_rate']:<9.1%} "
                f"${stats['avg_cost_per_day']:<9.4f}"
            )

    print("\n" + "=" * 80)
