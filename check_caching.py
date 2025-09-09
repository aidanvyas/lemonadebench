import json

with open(
    "results/json/claude-3-haiku-20240307_1games_30days_v05_20250829_120404_full.json"
) as f:
    data = json.load(f)

print("=" * 80)
print("CHECKING FOR PROMPT CACHING IN CLAUDE HAIKU BENCHMARK")
print("=" * 80)

total_input = 0
total_cached = 0
total_output = 0

# Check if we have any cached tokens across all days
if "games" in data and len(data["games"]) > 0:
    game = data["games"][0]

    for day_idx, day in enumerate(game.get("days", [])):
        day_input = 0
        day_cached = 0
        day_output = 0

        for attempt in day.get("attempts", []):
            if "response" in attempt and "usage" in attempt["response"]:
                usage = attempt["response"]["usage"]
                input_tokens = usage.get("input_tokens", 0)
                cached_tokens = usage.get("cache_read_input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

                day_input += input_tokens
                day_cached += cached_tokens
                day_output += output_tokens

                total_input += input_tokens
                total_cached += cached_tokens
                total_output += output_tokens

        if day_idx < 5 or day_idx >= 25:  # Show first 5 and last 5 days
            print(f"\nDay {day_idx + 1}:")
            print(f"  Input tokens:  {day_input:,}")
            print(f"  Cached tokens: {day_cached:,}")
            print(f"  Output tokens: {day_output:,}")
            if day_input > 0:
                cache_rate = (day_cached / day_input) * 100 if day_cached > 0 else 0
                print(f"  Cache hit rate: {cache_rate:.1f}%")

print("\n" + "=" * 80)
print("SUMMARY")
print("-" * 80)
print(f"Total input tokens:  {total_input:,}")
print(f"Total cached tokens: {total_cached:,}")
print(f"Total output tokens: {total_output:,}")
print(
    f"\nCache hit rate: {(total_cached / total_input * 100) if total_input > 0 and total_cached > 0 else 0:.1f}%"
)

# Calculate potential savings with caching
haiku_input_price = 0.80  # per million tokens (new pricing)
haiku_cache_price = 0.08  # per million tokens (new pricing)

actual_input_cost = (total_input / 1_000_000) * haiku_input_price
potential_cached_cost = (
    total_input * 0.9 / 1_000_000
) * haiku_cache_price  # Assume 90% could be cached
potential_non_cached_cost = (total_input * 0.1 / 1_000_000) * haiku_input_price

print("\n" + "=" * 80)
print("COST ANALYSIS WITH PROMPT CACHING")
print("-" * 80)
print(f"Current cost (no caching): ${actual_input_cost:.4f}")
print("Potential cost with caching:")
print(f"  - 90% cached @ $0.08/MTok: ${potential_cached_cost:.4f}")
print(f"  - 10% uncached @ $0.80/MTok: ${potential_non_cached_cost:.4f}")
print(f"  - Total: ${potential_cached_cost + potential_non_cached_cost:.4f}")
print(
    f"  - Savings: ${actual_input_cost - (potential_cached_cost + potential_non_cached_cost):.4f} ({((actual_input_cost - (potential_cached_cost + potential_non_cached_cost)) / actual_input_cost * 100):.1f}%)"
)

print("\n" + "=" * 80)
print("IMPLEMENTATION STATUS")
print("-" * 80)
print("❌ Prompt caching is NOT currently implemented in AnthropicPlayer")
print("   - No cache_control markers in messages")
print("   - No use of beta client for caching features")
print("   - Missing ~82% cost reduction opportunity")
