import json
from pathlib import Path

# Collect token usage from all 30-day benchmarks
results_dir = Path("results/json")
model_data = {}

# Find all 30-day benchmark files
for file in results_dir.glob("*_1games_30days_*.json"):
    if "_full" not in str(file):  # Skip the full files for now
        try:
            with open(file) as f:
                data = json.load(f)

            # Extract token usage for each model
            if "results" in data:
                for model_name, model_results in data["results"].items():
                    if (
                        "individual_games" in model_results
                        and model_results["individual_games"]
                    ):
                        game = model_results["individual_games"][0]
                        if "token_usage" in game:
                            tokens = game["token_usage"]

                            # Also get performance metrics
                            model_data[model_name] = {
                                "input_tokens": tokens.get("input_tokens", 0),
                                "output_tokens": tokens.get("output_tokens", 0),
                                "cached_input_tokens": tokens.get(
                                    "cached_input_tokens", 0
                                ),
                                "reasoning_tokens": tokens.get("reasoning_tokens", 0),
                                "total_tokens": tokens.get("total_tokens", 0),
                                "total_profit": float(game.get("total_profit", 0)),
                                "total_cost": model_results.get("total_cost", 0),
                                "days_played": game.get("days_played", 30),
                            }
        except Exception:
            continue

# Sort by total tokens for readability
sorted_models = sorted(
    model_data.items(), key=lambda x: x[1]["total_tokens"], reverse=True
)

print("=" * 120)
print("TOKEN USAGE COMPARISON ACROSS ALL MODELS (30-DAY BENCHMARK)")
print("=" * 120)
print()
print(
    f"{'Model':<22} {'Input':>10} {'Cached':>10} {'Cache%':>7} {'Output':>10} {'Reasoning':>10} {'Total':>10} {'Profit':>10} {'API Cost':>10}"
)
print("-" * 120)

for model_name, data in sorted_models:
    input_tokens = data["input_tokens"]
    cached_tokens = data["cached_input_tokens"]
    cache_pct = (
        (cached_tokens / input_tokens * 100)
        if input_tokens > 0 and cached_tokens > 0
        else 0
    )

    # Format with K/M suffixes for readability
    def format_tokens(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.0f}K"
        else:
            return str(n)

    print(
        f"{model_name:<22} {format_tokens(input_tokens):>10} {format_tokens(cached_tokens):>10} {cache_pct:>6.1f}% {format_tokens(data['output_tokens']):>10} {format_tokens(data['reasoning_tokens']):>10} {format_tokens(data['total_tokens']):>10} ${data['total_profit']:>9.0f} ${data['total_cost']:>9.4f}"
    )

print("-" * 120)

# Calculate some insights
print("\n" + "=" * 120)
print("KEY INSIGHTS")
print("-" * 120)

# Find most and least efficient
most_efficient = min(sorted_models, key=lambda x: x[1]["total_tokens"])
least_efficient = max(sorted_models, key=lambda x: x[1]["total_tokens"])

print("\n1. TOKEN EFFICIENCY:")
print(
    f"   Most efficient: {most_efficient[0]} with {most_efficient[1]['total_tokens']:,} tokens"
)
print(
    f"   Least efficient: {least_efficient[0]} with {least_efficient[1]['total_tokens']:,} tokens"
)
print(
    f"   Difference: {least_efficient[1]['total_tokens'] / most_efficient[1]['total_tokens']:.1f}x"
)

# Analyze caching
cached_models = [
    (name, data) for name, data in sorted_models if data["cached_input_tokens"] > 0
]
if cached_models:
    print("\n2. CACHING USAGE:")
    for name, data in cached_models[:5]:
        cache_pct = data["cached_input_tokens"] / data["input_tokens"] * 100
        print(
            f"   {name}: {cache_pct:.1f}% cached ({data['cached_input_tokens']:,} of {data['input_tokens']:,} input tokens)"
        )

# Reasoning tokens (o1/o3 models)
reasoning_models = [
    (name, data) for name, data in sorted_models if data["reasoning_tokens"] > 0
]
if reasoning_models:
    print("\n3. REASONING TOKENS (O-SERIES):")
    for name, data in reasoning_models:
        reasoning_pct = data["reasoning_tokens"] / data["total_tokens"] * 100
        print(
            f"   {name}: {data['reasoning_tokens']:,} reasoning tokens ({reasoning_pct:.1f}% of total)"
        )

# Correlation between tokens and performance
print("\n4. TOKENS VS PERFORMANCE:")
profitable_models = [
    (name, data) for name, data in sorted_models if data["total_profit"] > 10000
]
print(
    f"   High performers (>$10K profit) average: {sum(d['total_tokens'] for _, d in profitable_models) / len(profitable_models):,.0f} tokens"
)

low_token_models = [
    (name, data) for name, data in sorted_models if data["total_tokens"] < 200000
]
if low_token_models:
    avg_profit = sum(d["total_profit"] for _, d in low_token_models) / len(
        low_token_models
    )
    print(f"   Low token users (<200K) average profit: ${avg_profit:,.0f}")

# Token cost efficiency
print("\n5. TOKEN COST EFFICIENCY (profit per 1K tokens):")
efficiency_data = [
    (name, data["total_profit"] / (data["total_tokens"] / 1000))
    for name, data in sorted_models
    if data["total_tokens"] > 0
]
efficiency_data.sort(key=lambda x: x[1], reverse=True)
for name, efficiency in efficiency_data[:5]:
    print(f"   {name}: ${efficiency:.2f} profit per 1K tokens")
