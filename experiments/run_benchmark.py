#!/usr/bin/env python3
"""Main benchmark runner for LemonadeBench with adaptive rate limiting."""

import argparse
import json
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock

# Plotting imports removed - now in analyze_results.py

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.lemonade_stand.responses_ai_player import ResponsesAIPlayer
from src.lemonade_stand.simple_game import SimpleLemonadeGame

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_game(days: int = 100, demand_intercept: float = 100, demand_slope: float = 25, 
                suggested_price: float = None) -> SimpleLemonadeGame:
    """Create game instance with specified parameters.
    
    Args:
        days: Number of days to simulate
        demand_intercept: 'a' in demand function Q = a - b*p
        demand_slope: 'b' in demand function Q = a - b*p
        suggested_price: Suggested starting price (None for no suggestion)
        
    Returns:
        Configured game instance
    """
    game = SimpleLemonadeGame(days=days, demand_intercept=demand_intercept, demand_slope=demand_slope)
    game.suggested_starting_price = suggested_price
    return game


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on API responses."""

    def __init__(self, initial_rpm=100, initial_tpm=150000):
        """Initialize with conservative defaults."""
        self.lock = Lock()

        # Request rate limiting
        self.requests_per_minute = initial_rpm
        self.request_tokens = initial_rpm
        self.request_rate = initial_rpm / 60.0
        self.last_request_update = time.time()

        # Token rate limiting
        self.tokens_per_minute = initial_tpm
        self.token_window = []  # List of (timestamp, token_count) tuples
        self.window_duration = 60  # 1 minute window

        # Adaptive parameters
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.last_rate_limit_time = 0
        self.rate_limit_events = []  # Track rate limit events

        # Stats
        self.total_requests = 0
        self.successful_requests = 0
        self.rate_limit_errors = 0

    def acquire(self, expected_tokens=2000):
        """Wait until we can safely make a request."""
        with self.lock:
            while True:
                now = time.time()

                # Update request tokens
                elapsed = now - self.last_request_update
                self.request_tokens = min(self.requests_per_minute,
                                        self.request_tokens + elapsed * self.request_rate)
                self.last_request_update = now

                # Clean old token entries
                cutoff = now - self.window_duration
                self.token_window = [(ts, tokens) for ts, tokens in self.token_window if ts > cutoff]

                # Calculate current token usage
                current_token_usage = sum(tokens for _, tokens in self.token_window)

                # Check if we have capacity
                if (self.request_tokens >= 1 and
                    current_token_usage + expected_tokens < self.tokens_per_minute):
                    self.request_tokens -= 1
                    self.token_window.append((now, expected_tokens))
                    self.total_requests += 1
                    return

                # Calculate wait time
                if self.request_tokens < 1:
                    request_wait = (1 - self.request_tokens) / self.request_rate
                else:
                    request_wait = 0

                if current_token_usage + expected_tokens >= self.tokens_per_minute:
                    # Wait for some tokens to expire
                    token_wait = 0.5
                else:
                    token_wait = 0

                wait_time = max(request_wait, token_wait, 0.1)
                time.sleep(wait_time)

    def report_success(self):
        """Report a successful request."""
        with self.lock:
            self.successful_requests += 1
            self.consecutive_successes += 1
            self.consecutive_failures = 0

            # Gradually increase limits after consecutive successes
            if self.consecutive_successes >= 10 and time.time() - self.last_rate_limit_time > 60:
                self.requests_per_minute = min(int(self.requests_per_minute * 1.1), 400)
                self.request_rate = self.requests_per_minute / 60.0
                self.tokens_per_minute = min(int(self.tokens_per_minute * 1.1), 180000)
                self.consecutive_successes = 0
                logger.info(f"Increased limits: {self.requests_per_minute} RPM, {self.tokens_per_minute/1000:.0f}k TPM")

    def report_rate_limit(self, error_msg):
        """Report a rate limit error and adjust limits."""
        with self.lock:
            self.rate_limit_errors += 1
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.last_rate_limit_time = time.time()
            self.rate_limit_events.append((time.time(), error_msg))

            # Reduce limits based on error type
            if "TPM" in error_msg or "tokens" in error_msg:
                self.tokens_per_minute = int(self.tokens_per_minute * 0.7)
                logger.warning(f"Token limit hit, reducing to {self.tokens_per_minute/1000:.0f}k TPM")
            else:
                self.requests_per_minute = int(self.requests_per_minute * 0.7)
                self.request_rate = self.requests_per_minute / 60.0
                logger.warning(f"Request limit hit, reducing to {self.requests_per_minute} RPM")

            # Wait before continuing
            time.sleep(5.0)

    def get_stats(self):
        """Get current rate limiter statistics."""
        with self.lock:
            success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 0
            return {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'rate_limit_errors': self.rate_limit_errors,
                'success_rate': success_rate,
                'current_rpm_limit': self.requests_per_minute,
                'current_tpm_limit': self.tokens_per_minute,
            }


# Global rate limiter
rate_limiter = AdaptiveRateLimiter()


def run_single_game(test_name: str, run_number: int, game_config: dict, use_suggested: bool,
                    use_exploration: bool, days: int = 30, model: str = "gpt-4.1-nano",
                    inter_day_delay: float = 0.5) -> dict:
    """Run a single game instance."""
    logger.info(f"Starting: {test_name} - Run {run_number}")

    # Create game instance
    game = create_game(days=days, **game_config)

    # Configure game
    game._use_suggested_price = use_suggested
    game._use_exploration_hint = use_exploration

    # Create player
    player = ResponsesAIPlayer(
        model_name=model,
        include_calculator=True
    )
    
    # Always enable comprehensive recording
    player.enable_recording()

    # Run game
    prices = []
    profits = []
    start_time = time.time()
    errors = []

    for day in range(1, days + 1):
        # Add delay between days to smooth out API usage
        if day > 1 and inter_day_delay > 0:
            time.sleep(inter_day_delay)

        try:
            # Estimate tokens for this request
            estimated_tokens = 3000 if day == 1 else 2000

            # Acquire rate limit slot
            rate_limiter.acquire(estimated_tokens)

            price = player.make_decision(game)
            result = game.play_turn(price)
            prices.append(price)
            profits.append(result['profit'])

            # Report success
            rate_limiter.report_success()

        except Exception as e:
            error_msg = str(e)
            errors.append({'day': day, 'error': error_msg[:200]})

            # Handle rate limit errors
            if "rate_limit_exceeded" in error_msg:
                rate_limiter.report_rate_limit(error_msg)

                # Retry once after rate limit adjustment
                try:
                    rate_limiter.acquire(estimated_tokens)
                    price = player.make_decision(game)
                    result = game.play_turn(price)
                    prices.append(price)
                    profits.append(result['profit'])
                    rate_limiter.report_success()
                except Exception as retry_error:
                    logger.error(f"Retry failed for {test_name} Run {run_number} Day {day}: {str(retry_error)[:100]}")
                    prices.append(game.suggested_starting_price if use_suggested else 1.00)
                    profits.append(0)
            else:
                # For other errors, use default price
                logger.error(f"Error in {test_name} Run {run_number} Day {day}: {error_msg[:100]}")
                prices.append(game.suggested_starting_price if use_suggested else 1.00)
                profits.append(0)

    duration = time.time() - start_time

    # Calculate results
    total_profit = sum(profits)
    unique_prices = sorted({p for p in prices if p > 0})
    avg_price = sum(prices) / len(prices) if prices else 0

    # Calculate optimal values for this game based on demand function
    # For Q = a - bp, optimal price = a/(2b), giving Q* = a/2
    # Revenue = p* × Q* = (a/2b) × (a/2) = a²/(4b)
    optimal_price = game.optimal_price
    optimal_quantity = game.demand_intercept / 2
    optimal_profit = (game.demand_intercept ** 2) / (4 * game.demand_slope)

    days_at_optimal = sum(1 for p in prices if abs(p - optimal_price) < 0.01)

    # Calculate tool call breakdown
    tool_call_breakdown = {}
    for day_record in player.tool_call_history:
        for tool in day_record['tools']:
            tool_call_breakdown[tool] = tool_call_breakdown.get(tool, 0) + 1

    # Calculate cost
    cost_info = player.calculate_cost()

    logger.info(f"Completed: {test_name} - Run {run_number} "
                f"(${total_profit:.2f}, {duration:.1f}s, {len(errors)} errors)")

    # Get comprehensive recording data
    comprehensive_data = None
    if player.recorder:
        comprehensive_data = player.recorder.records
    
    return {
        'test_name': test_name,
        'run_number': run_number,
        'total_profit': total_profit,
        'average_price': avg_price,
        'unique_prices': unique_prices,
        'days_at_optimal': days_at_optimal,
        'optimal_price': optimal_price,
        'optimal_daily_profit': optimal_profit,
        'prices': prices,
        'profits': profits,
        'token_usage': player.total_token_usage,
        'tool_calls': player.tool_call_count,
        'tool_call_breakdown': tool_call_breakdown,
        'calculator_history': player.calculator_history,  # Add calculator expressions
        'cost_info': cost_info,
        'duration_seconds': duration,
        'efficiency': total_profit / (optimal_profit * days) if optimal_profit > 0 else 0,
        'errors': errors,
        'comprehensive_recording': comprehensive_data  # Full API interaction details
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """Aggregate results from multiple runs of the same test."""
    test_name = runs[0]['test_name']

    # Extract key metrics
    efficiencies = [r['efficiency'] for r in runs]
    total_profits = [r['total_profit'] for r in runs]
    days_at_optimal = [r['days_at_optimal'] for r in runs]
    unique_prices_counts = [len(r['unique_prices']) for r in runs]
    durations = [r['duration_seconds'] for r in runs]
    total_tokens = sum(r['token_usage']['total_tokens'] for r in runs)
    total_errors = sum(len(r.get('errors', [])) for r in runs)

    return {
        'test_name': test_name,
        'num_runs': len(runs),
        'efficiency': {
            'mean': statistics.mean(efficiencies),
            'std': statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0,
            'min': min(efficiencies),
            'max': max(efficiencies),
            'values': efficiencies
        },
        'total_profit': {
            'mean': statistics.mean(total_profits),
            'std': statistics.stdev(total_profits) if len(total_profits) > 1 else 0,
            'values': total_profits
        },
        'days_at_optimal': {
            'mean': statistics.mean(days_at_optimal),
            'values': days_at_optimal
        },
        'unique_prices': {
            'mean': statistics.mean(unique_prices_counts),
            'values': unique_prices_counts
        },
        'duration_seconds': {
            'mean': statistics.mean(durations),
            'total': sum(durations)
        },
        'total_tokens': total_tokens,
        'total_errors': total_errors,
        'optimal_daily_profit': runs[0]['optimal_daily_profit'],
        'individual_runs': runs
    }


def main():
    """Run LemonadeBench with adaptive rate limiting.

    Default of 5 runs per test is sufficient due to very low variability in results:
    - Standard deviation ~1% across runs
    - Clear failure mode (75% vs 100% efficiency)
    - Cost-effective: ~$0.17 for full benchmark
    - Time-efficient: ~30 minutes total
    """
    parser = argparse.ArgumentParser(description='Run LemonadeBench experiments')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per test (5 runs provides sufficient statistical power given low variability)')
    parser.add_argument('--days', type=int, default=30, help='Number of days per game')
    parser.add_argument('--model', type=str, help='Single model to use (deprecated, use --models)')
    parser.add_argument('--models', nargs='+', default=['gpt-4.1-nano'], help='Models to test (e.g., gpt-4.1-nano gpt-4.1-mini o4-mini)')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between days in seconds')
    parser.add_argument('--tests', nargs='+', default=['all'],
                       choices=['all', 'suggested', 'no-guidance', 'exploration', 'inverse'],
                       help='Which tests to run')
    parser.add_argument('--plots', action='store_true', help='Generate comparison plots after running experiments')
    args = parser.parse_args()
    
    # Handle backward compatibility: --model overrides --models
    if args.model:
        logger.warning("--model is deprecated, use --models instead")
        models = [args.model]
    else:
        models = args.models

    logger.info("LEMONADEBENCH - Adaptive Rate Limiting")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Runs per test: {args.runs}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Inter-day delay: {args.delay}s")
    logger.info(f"Tests: {args.tests}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define test configurations
    # Each config: (name, game_params, use_suggested, use_exploration)
    all_test_configs = {
        'suggested': (
            "Suggested Price", 
            {"demand_intercept": 100, "demand_slope": 25, "suggested_price": 1.00},
            True, False
        ),
        'no-guidance': (
            "No Guidance",
            {"demand_intercept": 100, "demand_slope": 25, "suggested_price": None},
            False, False
        ),
        'exploration': (
            "Exploration Hint",
            {"demand_intercept": 100, "demand_slope": 25, "suggested_price": None},
            False, True
        ),
        'inverse': (
            "Inverse Demand",
            {"demand_intercept": 50, "demand_slope": 25, "suggested_price": 2.00},
            True, True
        ),
    }

    # Select tests to run
    if 'all' in args.tests:
        test_configs = list(all_test_configs.values())
    else:
        test_configs = [all_test_configs[test] for test in args.tests if test in all_test_configs]

    # Calculate expected workload
    total_games = args.runs * len(test_configs) * len(models)
    total_api_calls = total_games * args.days
    logger.info(f"Total games: {total_games}")
    logger.info(f"Total API calls: {total_api_calls:,}")
    logger.info("")

    # Initialize rate limiter
    global rate_limiter
    rate_limiter = AdaptiveRateLimiter()

    # Run all tests and runs in parallel
    all_results = []
    start_time = datetime.now()
    completed_count = 0
    total_count = args.runs * len(test_configs) * len(models)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = []
        for model in models:
            for test_name, game_config, use_suggested, use_exploration in test_configs:
                for run_num in range(1, args.runs + 1):
                    future = executor.submit(
                        run_single_game,
                        test_name, run_num, game_config,
                        use_suggested, use_exploration,
                        args.days, model, args.delay
                    )
                    futures.append((future, model, test_name, run_num))

        # Collect results as they complete
        for future, model, test_name, run_num in futures:
            try:
                result = future.result()
                all_results.append(result)
                completed_count += 1

                # Progress update every 5 completions
                if completed_count % 5 == 0 or completed_count == total_count:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = completed_count / elapsed * 60 if elapsed > 0 else 0
                    eta = (total_count - completed_count) / rate * 60 if rate > 0 else 0
                    api_rate = completed_count * args.days / elapsed * 60  # API calls per minute

                    # Get rate limiter stats
                    rl_stats = rate_limiter.get_stats()

                    logger.info(f"\n[PROGRESS] {completed_count}/{total_count} games "
                               f"({completed_count/total_count*100:.0f}%)")
                    logger.info(f"  Game rate: {rate:.1f}/min")
                    logger.info(f"  API rate: {api_rate:.0f}/min")
                    logger.info(f"  Current limits: {rl_stats['current_rpm_limit']} RPM, "
                               f"{rl_stats['current_tpm_limit']/1000:.0f}k TPM")
                    logger.info(f"  Success rate: {rl_stats['success_rate']:.1%}")
                    if eta > 0:
                        logger.info(f"  ETA: {eta/60:.1f} minutes")

            except Exception as e:
                logger.error(f"\n[FATAL ERROR] {test_name} - Run {run_num}:")
                logger.error(f"  {str(e)}")
                import traceback
                traceback.print_exc()

    total_duration = (datetime.now() - start_time).total_seconds()

    # Group results by model and test
    grouped_results = {}
    for result in all_results:
        model = result['model']
        test_name = result['test_name']
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
                aggregated['model'] = model
                aggregated_results.append(aggregated)

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY OF ALL TESTS")
    logger.info(f"{'='*70}\n")

    total_tokens = 0
    current_model = None
    for r in aggregated_results:
        # Add model header when switching models
        if r['model'] != current_model:
            current_model = r['model']
            logger.info(f"\n--- {current_model} ---")
        
        logger.info(f"{r['test_name']} ({r['num_runs']} runs):")
        logger.info(f"  Efficiency: {r['efficiency']['mean']:.1%} ± {r['efficiency']['std']:.1%}")
        logger.info(f"    Range: {r['efficiency']['min']:.1%} - {r['efficiency']['max']:.1%}")
        logger.info(f"  Total profit: ${r['total_profit']['mean']:.2f} ± ${r['total_profit']['std']:.2f}")
        logger.info(f"  Days at optimal: {r['days_at_optimal']['mean']:.1f}/30")
        logger.info(f"  Unique prices: {r['unique_prices']['mean']:.1f}")
        logger.info(f"  Errors: {r['total_errors']}")
        logger.info(f"  Tokens per run: {r['total_tokens'] / r['num_runs']:,.0f}")
        logger.info("")
        total_tokens += r['total_tokens']

    # Final stats
    rl_stats = rate_limiter.get_stats()
    logger.info(f"Total games completed: {completed_count}/{total_count}")
    logger.info(f"Total tokens used: {total_tokens:,}")
    # Note: This is rough estimate - actual costs vary by model
    logger.info(f"Estimated cost (assuming nano pricing): ${total_tokens * 0.15 / 1_000_000:.2f}")
    logger.info(f"Total duration: {total_duration/60:.1f} minutes")
    logger.info(f"Average API rate: {total_api_calls / (total_duration/60):.0f}/min")
    logger.info(f"Rate limit errors: {rl_stats['rate_limit_errors']}")
    logger.info(f"Final limits: {rl_stats['current_rpm_limit']} RPM, {rl_stats['current_tpm_limit']/1000:.0f}k TPM")

    # Save results with descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_names = "-".join(args.tests) if 'all' not in args.tests else "all"
    model_names = "-".join(models) if len(models) > 1 else models[0]
    filename = f"results/json/{model_names}_{test_names}_{args.runs}runs_{args.days}days_{args.workers}workers_{timestamp}.json"
    Path("results/json").mkdir(parents=True, exist_ok=True)

    with open(filename, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'models': models,
            'days': args.days,
            'runs_per_test': args.runs,
            'workers': args.workers,
            'inter_day_delay': args.delay,
            'tests_run': [tc[0] for tc in test_configs],
            'aggregated_results': aggregated_results,
            'total_duration_seconds': total_duration,
            'total_api_calls': total_api_calls,
            'rate_limiter_stats': rl_stats,
            'execution_mode': 'adaptive_parallel',
            'completed_games': completed_count,
            'total_tokens': total_tokens
        }, f, indent=2)

    logger.info(f"\nResults saved to: {filename}")

    # Print efficiency summary
    logger.info(f"\n{'='*70}")
    logger.info("EFFICIENCY SUMMARY (Key Finding)")
    logger.info(f"{'='*70}")
    for r in aggregated_results:
        model_label = f"{r['model']}/" if len(models) > 1 else ""
        logger.info(f"{model_label}{r['test_name']:25} {r['efficiency']['mean']:6.1%}")
    
    # Generate plots if requested
    if args.plots:
        logger.info("\nNote: Use 'python analysis/analyze_results.py --latest --plots' to generate plots")


# Plot generation removed - now in analyze_results.py


if __name__ == "__main__":
    main()
