#!/usr/bin/env python3
"""Main benchmark runner for LemonadeBench with header-based rate limiting."""

import argparse
import json
import logging
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.lemonade_stand.responses_ai_player import ResponsesAIPlayer
from src.lemonade_stand.simple_game import SimpleLemonadeGame

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_game(
    days: int = 100,
    demand_intercept: float = 100,
    demand_slope: float = 25,
    suggested_price: float = None,
) -> SimpleLemonadeGame:
    """Create game instance with specified parameters.

    Args:
        days: Number of days to simulate
        demand_intercept: 'a' in demand function Q = a - b*p
        demand_slope: 'b' in demand function Q = a - b*p
        suggested_price: Suggested starting price (None for no suggestion)

    Returns:
        Configured game instance
    """
    game = SimpleLemonadeGame(
        days=days, demand_intercept=demand_intercept, demand_slope=demand_slope
    )
    game.suggested_starting_price = suggested_price
    return game


class HeaderBasedRateLimiter:
    """Rate limiter with header support and exponential backoff."""

    def __init__(self):
        # Rate limit info from headers
        self.limit_requests = None
        self.limit_tokens = None
        self.remaining_requests = None
        self.remaining_tokens = None
        self.reset_requests = None
        self.reset_tokens = None

        # Stats
        self.total_requests = 0
        self.successful_requests = 0
        self.rate_limit_errors = 0
        self.rate_limit_waits = 0
        self.total_wait_time = 0

        # Track actual usage
        self.request_times = []  # Track request timestamps
        self.token_counts = []  # Track token usage per request

        # Exponential backoff parameters for errors
        self.base_wait = 1.0
        self.max_wait = 120.0
        self.retry_count = 0

        # Simple rate limiting without headers
        self.requests_per_minute = 60  # Conservative default
        self.last_request_time = 0

    def parse_duration(self, duration_str: str) -> float:
        """Parse duration string like '17ms' or '6m0s' to seconds."""
        if not duration_str:
            return 60.0  # Default to 1 minute

        # Handle different time units
        multipliers = {"ms": 1 / 1000, "s": 1, "m": 60, "h": 3600, "d": 86400}

        # Extract number and unit from strings like "17ms", "6m0s", "1s"
        import re

        # Handle compound formats like "6m0s"
        total_seconds = 0
        parts = re.findall(r"(\d+\.?\d*)([a-z]+)", duration_str)

        for num_str, unit in parts:
            num = float(num_str)
            if unit in multipliers:
                total_seconds += num * multipliers[unit]

        return max(total_seconds, 0.001)  # At least 1ms

    def update_from_headers(self, headers: dict) -> None:
        """Update rate limit info from API response headers."""
        # Extract rate limit info
        self.limit_requests = int(headers.get("x-ratelimit-limit-requests", 60))
        self.limit_tokens = int(headers.get("x-ratelimit-limit-tokens", 150000))
        self.remaining_requests = int(headers.get("x-ratelimit-remaining-requests", 0))
        self.remaining_tokens = int(headers.get("x-ratelimit-remaining-tokens", 0))

        # Parse reset times
        self.reset_requests = self.parse_duration(
            headers.get("x-ratelimit-reset-requests", "60s")
        )
        self.reset_tokens = self.parse_duration(
            headers.get("x-ratelimit-reset-tokens", "60s")
        )

    def should_wait(self, min_tokens_buffer: int = 5000) -> tuple[bool, float]:
        """Check if we should wait before making a request.

        Returns:
            (should_wait, wait_seconds)
        """
        if self.remaining_requests is None:
            # No header info yet, don't wait
            return False, 0

        # Check if we're out of requests
        if self.remaining_requests <= 0:
            return True, self.reset_requests + 0.5  # Add small buffer

        # Check if we're low on tokens
        if self.remaining_tokens < min_tokens_buffer:
            return True, self.reset_tokens + 0.5  # Add small buffer

        return False, 0

    def wait_if_needed(self, min_tokens_buffer: int = 5000) -> None:
        """Wait if rate limits require it."""
        # If we have header info, use it
        if self.remaining_requests is not None:
            should_wait, wait_time = self.should_wait(min_tokens_buffer)

            if should_wait:
                self.rate_limit_waits += 1
                self.total_wait_time += wait_time

                logger.info(
                    f"Rate limit reached (req: {self.remaining_requests}/{self.limit_requests}, "
                    f"tokens: {self.remaining_tokens}/{self.limit_tokens}). "
                    f"Waiting {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
        else:
            # Simple rate limiting without headers
            now = time.time()
            time_since_last = now - self.last_request_time
            min_interval = 60.0 / self.requests_per_minute

            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                time.sleep(wait_time)
                self.total_wait_time += wait_time

    def record_request_start(self) -> None:
        """Record that we're about to make a request."""
        self.total_requests += 1
        self.last_request_time = time.time()
        self.request_times.append(self.last_request_time)

    def record_success(self, tokens_used: int, headers: dict = None) -> None:
        """Record a successful request."""
        self.successful_requests += 1
        self.token_counts.append(tokens_used)
        # Reset backoff on success
        self.retry_count = 0

        # Update rate limit info if headers provided
        if headers:
            self.update_from_headers(headers)

    def handle_rate_limit_error(self) -> float:
        """Handle a rate limit error and return wait time."""
        self.rate_limit_errors += 1
        self.retry_count += 1

        # Exponential backoff with jitter
        wait_time = min(self.base_wait * (2**self.retry_count), self.max_wait)
        wait_time += random.uniform(0, wait_time * 0.1)  # Add 10% jitter

        self.total_wait_time += wait_time
        logger.warning(
            f"Rate limit hit. Waiting {wait_time:.1f}s (retry #{self.retry_count})"
        )

        return wait_time

    def get_current_rate(self) -> tuple[float, float]:
        """Get current request and token rates (per minute)."""
        now = time.time()
        recent_cutoff = now - 60  # Last minute

        recent_requests = sum(1 for t in self.request_times if t > recent_cutoff)
        recent_tokens = sum(
            tokens
            for t, tokens in zip(self.request_times, self.token_counts, strict=False)
            if t > recent_cutoff
            and len(self.token_counts) > self.request_times.index(t)
        )

        return recent_requests, recent_tokens

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        avg_tokens = (
            sum(self.token_counts) / len(self.token_counts) if self.token_counts else 0
        )
        req_rate, token_rate = self.get_current_rate()

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "rate_limit_errors": self.rate_limit_errors,
            "rate_limit_waits": self.rate_limit_waits,
            "total_wait_time": self.total_wait_time,
            "average_tokens_per_request": avg_tokens,
            "current_request_rate": req_rate,
            "current_token_rate": token_rate,
            "success_rate": self.successful_requests / self.total_requests
            if self.total_requests > 0
            else 0,
            "current_limits": {
                "requests": self.limit_requests,
                "tokens": self.limit_tokens,
            },
            "remaining": {
                "requests": self.remaining_requests,
                "tokens": self.remaining_tokens,
            },
        }


def run_single_game(
    test_name: str,
    run_number: int,
    game_config: dict,
    use_suggested: bool,
    use_exploration: bool,
    days: int = 30,
    model: str = "gpt-4.1-nano",
    rate_limiter: HeaderBasedRateLimiter = None,
) -> dict:
    """Run a single game instance."""
    logger.info(f"Starting: {test_name} - Run {run_number}")

    # Create game instance
    game = create_game(days=days, **game_config)

    # Configure game
    game._use_suggested_price = use_suggested
    game._use_exploration_hint = use_exploration

    # Create player with custom make_decision wrapper to handle headers
    player = ResponsesAIPlayer(model_name=model, include_calculator=True)

    # Always enable comprehensive recording
    player.enable_recording()

    # Run game
    prices = []
    profits = []
    start_time = time.time()
    errors = []

    for day in range(1, days + 1):
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                # Wait if rate limited (before making request)
                if rate_limiter:
                    rate_limiter.wait_if_needed()
                    rate_limiter.record_request_start()

                price = player.make_decision(game)
                result = game.play_turn(price)
                prices.append(price)
                profits.append(result["profit"])

                # Record success and token usage
                if rate_limiter:
                    # Get token count and headers from the last API call
                    tokens_used = 0
                    if hasattr(player, "last_token_usage"):
                        tokens_used = player.last_token_usage.get("total_tokens", 0)

                    # Get headers if available
                    headers = getattr(player, "last_headers", None)
                    rate_limiter.record_success(tokens_used, headers)

                # Success - break out of retry loop
                break

            except Exception as e:
                error_msg = str(e)
                retry_count += 1

                # Check if it's a rate limit error
                if "rate_limit_exceeded" in error_msg.lower() or "429" in error_msg:
                    if rate_limiter and retry_count < max_retries:
                        # Wait with exponential backoff
                        wait_time = rate_limiter.handle_rate_limit_error()
                        time.sleep(wait_time)
                        continue  # Retry
                    else:
                        # Max retries exceeded
                        logger.error(
                            f"Max retries exceeded for {test_name} Run {run_number} Day {day}"
                        )
                        errors.append(
                            {"day": day, "error": "Rate limit - max retries exceeded"}
                        )
                        prices.append(
                            game.suggested_starting_price if use_suggested else 1.00
                        )
                        profits.append(0)
                        break
                else:
                    # Non-rate-limit error
                    logger.error(
                        f"Error in {test_name} Run {run_number} Day {day}: {error_msg[:100]}"
                    )
                    errors.append({"day": day, "error": error_msg[:200]})
                    prices.append(
                        game.suggested_starting_price if use_suggested else 1.00
                    )
                    profits.append(0)
                    break

    duration = time.time() - start_time

    # Calculate results
    total_profit = sum(profits)
    unique_prices = sorted({p for p in prices if p > 0})
    avg_price = sum(prices) / len(prices) if prices else 0

    # Calculate optimal values for this game based on demand function
    # For Q = a - bp, optimal price = a/(2b), giving Q* = a/2
    # Revenue = p* × Q* = (a/2b) × (a/2) = a²/(4b)
    optimal_price = game.optimal_price
    # optimal_quantity = game.demand_intercept / 2  # Not used currently
    optimal_profit = (game.demand_intercept**2) / (4 * game.demand_slope)

    days_at_optimal = sum(1 for p in prices if abs(p - optimal_price) < 0.01)

    # Calculate tool call breakdown
    tool_call_breakdown = {}
    for day_record in player.tool_call_history:
        for tool in day_record["tools"]:
            tool_call_breakdown[tool] = tool_call_breakdown.get(tool, 0) + 1

    # Calculate cost
    cost_info = player.calculate_cost()

    logger.info(
        f"Completed: {test_name} - Run {run_number} "
        f"(${total_profit:.2f}, {duration:.1f}s, {len(errors)} errors)"
    )

    # Get comprehensive recording data
    comprehensive_data = None
    if player.recorder:
        comprehensive_data = player.recorder.records

    return {
        "test_name": test_name,
        "run_number": run_number,
        "total_profit": total_profit,
        "average_price": avg_price,
        "unique_prices": unique_prices,
        "days_at_optimal": days_at_optimal,
        "optimal_price": optimal_price,
        "optimal_daily_profit": optimal_profit,
        "prices": prices,
        "profits": profits,
        "token_usage": player.total_token_usage,
        "tool_calls": player.tool_call_count,
        "tool_call_breakdown": tool_call_breakdown,
        "calculator_history": player.calculator_history,  # Add calculator expressions
        "cost_info": cost_info,
        "duration_seconds": duration,
        "efficiency": total_profit / (optimal_profit * days)
        if optimal_profit > 0
        else 0,
        "errors": errors,
        "comprehensive_recording": comprehensive_data,  # Full API interaction details
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """Aggregate results from multiple runs of the same test."""
    test_name = runs[0]["test_name"]

    # Extract key metrics
    efficiencies = [r["efficiency"] for r in runs]
    total_profits = [r["total_profit"] for r in runs]
    days_at_optimal = [r["days_at_optimal"] for r in runs]
    unique_prices_counts = [len(r["unique_prices"]) for r in runs]
    durations = [r["duration_seconds"] for r in runs]
    total_tokens = sum(r["token_usage"]["total_tokens"] for r in runs)
    total_errors = sum(len(r.get("errors", [])) for r in runs)

    return {
        "test_name": test_name,
        "num_runs": len(runs),
        "efficiency": {
            "mean": statistics.mean(efficiencies),
            "std": statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0,
            "min": min(efficiencies),
            "max": max(efficiencies),
            "values": efficiencies,
        },
        "total_profit": {
            "mean": statistics.mean(total_profits),
            "std": statistics.stdev(total_profits) if len(total_profits) > 1 else 0,
            "values": total_profits,
        },
        "days_at_optimal": {
            "mean": statistics.mean(days_at_optimal),
            "values": days_at_optimal,
        },
        "unique_prices": {
            "mean": statistics.mean(unique_prices_counts),
            "values": unique_prices_counts,
        },
        "duration_seconds": {
            "mean": statistics.mean(durations),
            "total": sum(durations),
        },
        "total_tokens": total_tokens,
        "total_errors": total_errors,
        "optimal_daily_profit": runs[0]["optimal_daily_profit"],
        "individual_runs": runs,
    }


def main():
    """Run LemonadeBench with header-based rate limiting."""
    parser = argparse.ArgumentParser(description="Run LemonadeBench experiments")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per test (5 runs provides sufficient statistical power given low variability)",
    )
    parser.add_argument("--days", type=int, default=30, help="Number of days per game")
    parser.add_argument(
        "--model", type=str, help="Single model to use (deprecated, use --models)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4.1-nano"],
        help="Models to test (e.g., gpt-4.1-nano gpt-4.1-mini o4-mini)",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["all"],
        choices=["all", "suggested", "no-guidance", "exploration", "inverse"],
        help="Which tests to run",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate comparison plots after running experiments",
    )
    args = parser.parse_args()

    # Handle backward compatibility: --model overrides --models
    if args.model:
        logger.warning("--model is deprecated, use --models instead")
        models = [args.model]
    else:
        models = args.models

    logger.info("LEMONADEBENCH - Header-Based Rate Limiting")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Runs per test: {args.runs}")
    logger.info(f"Tests: {args.tests}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define test configurations
    # Each config: (name, game_params, use_suggested, use_exploration)
    all_test_configs = {
        "suggested": (
            "Suggested Price",
            {"demand_intercept": 100, "demand_slope": 25, "suggested_price": 1.00},
            True,
            False,
        ),
        "no-guidance": (
            "No Guidance",
            {"demand_intercept": 100, "demand_slope": 25, "suggested_price": None},
            False,
            False,
        ),
        "exploration": (
            "Exploration Hint",
            {"demand_intercept": 100, "demand_slope": 25, "suggested_price": None},
            False,
            True,
        ),
        "inverse": (
            "Inverse Demand",
            {"demand_intercept": 50, "demand_slope": 25, "suggested_price": 2.00},
            True,
            True,
        ),
    }

    # Select tests to run
    if "all" in args.tests:
        test_configs = list(all_test_configs.values())
    else:
        test_configs = [
            all_test_configs[test] for test in args.tests if test in all_test_configs
        ]

    # Calculate expected workload
    total_games = args.runs * len(test_configs) * len(models)
    total_api_calls = total_games * args.days
    logger.info(f"Total games: {total_games}")
    logger.info(f"Total API calls: {total_api_calls:,}")
    logger.info("")

    # Initialize rate limiter
    rate_limiter = HeaderBasedRateLimiter()

    # Run all tests sequentially
    all_results = []
    start_time = datetime.now()
    completed_count = 0
    total_count = args.runs * len(test_configs) * len(models)

    # Sequential execution - simpler and respects rate limits properly
    for model in models:
        logger.info(f"\nTesting model: {model}")

        # We need to patch the ResponsesAIPlayer to expose headers
        # For now, we'll work with the existing implementation

        for test_name, game_config, use_suggested, use_exploration in test_configs:
            logger.info(f"\nRunning test: {test_name}")

            for run_num in range(1, args.runs + 1):
                try:
                    result = run_single_game(
                        test_name,
                        run_num,
                        game_config,
                        use_suggested,
                        use_exploration,
                        args.days,
                        model,
                        rate_limiter,
                    )
                    result["model"] = model  # Add model to result
                    all_results.append(result)
                    completed_count += 1

                    # Progress update
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = completed_count / elapsed * 60 if elapsed > 0 else 0
                    eta = (total_count - completed_count) / rate if rate > 0 else 0
                    api_calls_made = completed_count * args.days
                    api_rate = api_calls_made / elapsed * 60 if elapsed > 0 else 0

                    # Get rate limiter stats
                    rl_stats = rate_limiter.get_stats()

                    if completed_count % 5 == 0 or completed_count == total_count:
                        logger.info(
                            f"\n[PROGRESS] {completed_count}/{total_count} games "
                            f"({completed_count / total_count * 100:.0f}%)"
                        )
                        logger.info(f"  Elapsed: {elapsed / 60:.1f} minutes")
                        logger.info(f"  Game rate: {rate:.1f}/min")
                        logger.info(f"  API rate: {api_rate:.0f}/min")
                        logger.info(
                            f"  Rate limit errors: {rl_stats['rate_limit_errors']}"
                        )
                        logger.info(
                            f"  Rate limit waits: {rl_stats['rate_limit_waits']}"
                        )
                        logger.info(
                            f"  Total wait time: {rl_stats['total_wait_time']:.1f}s"
                        )

                        if (
                            rl_stats.get("remaining")
                            and rl_stats["remaining"].get("requests") is not None
                        ):
                            logger.info(
                                f"  Remaining: {rl_stats['remaining']['requests']} requests, "
                                f"{rl_stats['remaining'].get('tokens', 0) / 1000:.0f}k tokens"
                            )

                        logger.info(
                            f"  Current rates: {rl_stats['current_request_rate']:.0f} req/min, "
                            f"{rl_stats['current_token_rate'] / 1000:.0f}k tokens/min"
                        )

                        if eta > 0:
                            logger.info(f"  ETA: {eta / 60:.1f} minutes")

                except Exception as e:
                    logger.error(f"\n[FATAL ERROR] {test_name} - Run {run_num}:")
                    logger.error(f"  {str(e)}")
                    import traceback

                    traceback.print_exc()
                    # Continue with next run
                    continue

    total_duration = (datetime.now() - start_time).total_seconds()

    # Group results by model and test
    grouped_results = {}
    for result in all_results:
        model = result.get("model", "unknown")
        test_name = result["test_name"]
        key = (model, test_name)
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)

    # Aggregate results
    aggregated_results = []
    for model in models:
        for test_name, _, _, _ in test_configs:
            key = (model, test_name)
            if key in grouped_results:
                aggregated = aggregate_runs(grouped_results[key])
                aggregated["model"] = model
                aggregated_results.append(aggregated)

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("SUMMARY OF ALL TESTS")
    logger.info(f"{'=' * 70}\n")

    total_tokens = 0
    current_model = None
    for r in aggregated_results:
        # Add model header when switching models
        if r["model"] != current_model:
            current_model = r["model"]
            logger.info(f"\n--- {current_model} ---")

        logger.info(f"{r['test_name']} ({r['num_runs']} runs):")
        logger.info(
            f"  Efficiency: {r['efficiency']['mean']:.1%} ± {r['efficiency']['std']:.1%}"
        )
        logger.info(
            f"    Range: {r['efficiency']['min']:.1%} - {r['efficiency']['max']:.1%}"
        )
        logger.info(
            f"  Total profit: ${r['total_profit']['mean']:.2f} ± ${r['total_profit']['std']:.2f}"
        )
        logger.info(f"  Days at optimal: {r['days_at_optimal']['mean']:.1f}/30")
        logger.info(f"  Unique prices: {r['unique_prices']['mean']:.1f}")
        logger.info(f"  Errors: {r['total_errors']}")
        logger.info(f"  Tokens per run: {r['total_tokens'] / r['num_runs']:,.0f}")
        logger.info("")
        total_tokens += r["total_tokens"]

    # Final stats
    rl_stats = rate_limiter.get_stats()
    logger.info(f"Total games completed: {completed_count}/{total_count}")
    logger.info(f"Total tokens used: {total_tokens:,}")
    # Note: This is rough estimate - actual costs vary by model
    logger.info(
        f"Estimated cost (assuming nano pricing): ${total_tokens * 0.15 / 1_000_000:.2f}"
    )
    logger.info(f"Total duration: {total_duration / 60:.1f} minutes")
    logger.info(f"Average API rate: {total_api_calls / (total_duration / 60):.0f}/min")
    logger.info(f"Rate limit errors: {rl_stats['rate_limit_errors']}")
    logger.info(f"Rate limit waits: {rl_stats['rate_limit_waits']}")
    logger.info(f"Total wait time: {rl_stats['total_wait_time']:.1f}s")
    logger.info(f"Success rate: {rl_stats['success_rate']:.1%}")
    if rl_stats["current_limits"]["requests"]:
        logger.info(
            f"API Limits: {rl_stats['current_limits']['requests']} RPM, "
            f"{rl_stats['current_limits']['tokens'] / 1000:.0f}k TPM"
        )
    if rl_stats["average_tokens_per_request"] > 0:
        logger.info(
            f"Average tokens per request: {rl_stats['average_tokens_per_request']:.0f}"
        )

    # Save results with descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_names = "-".join(args.tests) if "all" not in args.tests else "all"
    model_names = "-".join(models) if len(models) > 1 else models[0]
    filename = f"results/json/{model_names}_{test_names}_{args.runs}runs_{args.days}days_{timestamp}.json"
    Path("results/json").mkdir(parents=True, exist_ok=True)

    with open(filename, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "models": models,
                "days": args.days,
                "runs_per_test": args.runs,
                "tests_run": [tc[0] for tc in test_configs],
                "aggregated_results": aggregated_results,
                "total_duration_seconds": total_duration,
                "total_api_calls": total_api_calls,
                "rate_limiter_stats": rl_stats,
                "execution_mode": "sequential_with_headers",
                "completed_games": completed_count,
                "total_tokens": total_tokens,
            },
            f,
            indent=2,
        )

    logger.info(f"\nResults saved to: {filename}")

    # Print efficiency summary
    logger.info(f"\n{'=' * 70}")
    logger.info("EFFICIENCY SUMMARY (Key Finding)")
    logger.info(f"{'=' * 70}")
    for r in aggregated_results:
        model_label = f"{r['model']}/" if len(models) > 1 else ""
        logger.info(f"{model_label}{r['test_name']:25} {r['efficiency']['mean']:6.1%}")

    # Generate plots if requested
    if args.plots:
        logger.info(
            "\nNote: Use 'python analysis/analyze_results.py --latest --plots' to generate plots"
        )


# Plot generation removed - now in analyze_results.py


if __name__ == "__main__":
    main()
