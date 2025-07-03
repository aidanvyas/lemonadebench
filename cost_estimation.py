#!/usr/bin/env python3
"""Calculate cost estimates for running LemonadeBench experiments."""


# Model pricing (per 1M tokens)
MODEL_PRICING = {
    'gpt-4.1-nano': {
        'name': 'GPT-4.1 nano',
        'input': 0.10,
        'cached_input': 0.025,
        'output': 0.40
    },
    'gpt-4.1-mini': {
        'name': 'GPT-4.1 mini',
        'input': 0.40,
        'cached_input': 0.10,
        'output': 1.60
    },
    'gpt-4.1': {
        'name': 'GPT-4.1',
        'input': 2.00,
        'cached_input': 0.50,
        'output': 8.00
    },
    'o3': {
        'name': 'OpenAI o3',
        'input': 2.00,
        'cached_input': 0.50,
        'output': 8.00
    },
    'o4-mini': {
        'name': 'OpenAI o4-mini',
        'input': 1.10,
        'cached_input': 0.275,
        'output': 4.40
    }
}

def estimate_tokens_per_test(days):
    """Estimate tokens for a single test based on observed patterns."""
    # Based on 30-day test data:
    # - Suggested Price: 63,536 tokens (2,118 per day)
    # - No Guidance: 37,583 tokens (1,253 per day)
    # - Exploration: 35,477 tokens (1,183 per day)
    # - Inverse: 34,522 tokens (1,151 per day)

    # Average: ~1,426 tokens per day
    # But tokens accumulate due to conversation history

    if days <= 30:
        # Linear approximation for short runs
        base_tokens = 300  # Initial system prompt
        per_day_tokens = 1400
        total_input = base_tokens + (per_day_tokens * days)
        total_output = 20 * days  # ~20 output tokens per day
    else:
        # For longer runs, growth slows due to context limits
        # First 30 days: standard growth
        first_30_input = 300 + (1400 * 30)
        first_30_output = 20 * 30

        # After 30 days: reduced growth (context window effects)
        remaining_days = days - 30
        additional_input = remaining_days * 2000  # Higher per-day due to history
        additional_output = remaining_days * 20

        total_input = first_30_input + additional_input
        total_output = first_30_output + additional_output

    return total_input, total_output

def calculate_cost(model_key, input_tokens, output_tokens, cache_ratio=0):
    """Calculate cost for given tokens with optional caching."""
    pricing = MODEL_PRICING[model_key]

    cached_input_tokens = int(input_tokens * cache_ratio)
    non_cached_input_tokens = input_tokens - cached_input_tokens

    input_cost = (non_cached_input_tokens / 1_000_000) * pricing['input']
    cached_cost = (cached_input_tokens / 1_000_000) * pricing['cached_input']
    output_cost = (output_tokens / 1_000_000) * pricing['output']

    total_cost = input_cost + cached_cost + output_cost

    return {
        'total_cost': total_cost,
        'input_cost': input_cost,
        'cached_cost': cached_cost,
        'output_cost': output_cost
    }

def estimate_experiment_costs(days, runs, tests=4):
    """Estimate costs for full experiment across all models."""
    print(f"\nCOST ESTIMATION: {runs} runs × {tests} tests × {days} days")
    print("=" * 80)

    # Estimate tokens per test
    input_per_test, output_per_test = estimate_tokens_per_test(days)
    total_input = input_per_test * tests * runs
    total_output = output_per_test * tests * runs

    print("\nToken estimates:")
    print(f"  Per test: {input_per_test:,} input, {output_per_test:,} output")
    print(f"  Total ({runs} runs × {tests} tests): {total_input:,} input, {total_output:,} output")

    # Calculate costs for each model
    print(f"\n{'Model':<20} {'No Cache':<15} {'75% Cache':<15} {'Savings':<15}")
    print("-" * 65)

    for model_key, pricing in MODEL_PRICING.items():
        # No caching
        cost_no_cache = calculate_cost(model_key, total_input, total_output, 0)

        # With 75% caching (realistic for conversation continuity)
        cost_cached = calculate_cost(model_key, total_input, total_output, 0.75)

        savings = cost_no_cache['total_cost'] - cost_cached['total_cost']
        savings_pct = (savings / cost_no_cache['total_cost']) * 100 if cost_no_cache['total_cost'] > 0 else 0

        print(f"{pricing['name']:<20} ${cost_no_cache['total_cost']:>13.2f} ${cost_cached['total_cost']:>13.2f} ${savings:>10.2f} ({savings_pct:>4.1f}%)")

def main():
    """Run cost estimations for different scenarios."""

    print("\n" + "="*80)
    print("LEMONADEBENCH COST ESTIMATION")
    print("="*80)

    # Scenario 1: 30 runs of 30 days
    estimate_experiment_costs(days=30, runs=30)

    # Scenario 2: 30 runs of 100 days
    estimate_experiment_costs(days=100, runs=30)

    # Additional insights
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. GPT-4.1-nano is 20x cheaper than GPT-4.1")
    print("2. Caching can save up to 75% on input costs")
    print("3. 100-day tests use ~3x more tokens than 30-day tests")
    print("4. Output tokens are minimal (~20 per day)")
    print("\nNOTE: Actual caching depends on reaching 1024+ token prompts")
    print("      With conversation continuity, this happens around day 10-15")

if __name__ == "__main__":
    main()
