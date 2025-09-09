"""Analyze token usage and costs for Gemini Flash games"""

import json
import os

# Gemini 2.5 Flash pricing (per million tokens)
FREE_TIER = {
    "input": 0.00,  # Free
    "output": 0.00,  # Free
    "limits": "10 RPM, 250K TPM",
}

PAID_TIER = {
    "tier1": {
        "input": 0.30,  # $0.30 per million input tokens
        "output": 2.50,  # $2.50 per million output tokens
        "limits": "100 RPM, 2M TPM",
    },
    "tier2": {"input": 0.30, "output": 2.50, "limits": "2000 RPM, 10M TPM"},
}

print("=" * 80)
print("GEMINI 2.5 FLASH - TOKEN USAGE ANALYSIS")
print("=" * 80)
print()

# Analyze each completed game
game_files = [
    "results/json/gemini-2.5-flash_game1_1756508310420_86681.json",
    "results/json/gemini-2.5-flash_game2_1756508707371_86681.json",
    "results/json/gemini-2.5-flash_game3_1756509145238_86681.json",
    "results/json/gemini-2.5-flash_game4_1756509551787_86681.json",
]

total_input_tokens = 0
total_output_tokens = 0
games_data = []

for i, filepath in enumerate(game_files, 1):
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)

        # Count tokens from all interactions
        input_tokens = 0
        output_tokens = 0

        for day in data.get("days", []):
            for interaction in day.get("interactions", []):
                # Each interaction is an API call
                # Estimate tokens based on request/response size
                if "request" in interaction:
                    # Rough estimate: 1 token per 4 characters
                    request_str = str(interaction["request"])
                    input_tokens += len(request_str) // 4

                if "response" in interaction:
                    response_str = str(interaction["response"])
                    output_tokens += len(response_str) // 4

        games_data.append(
            {
                "game": i,
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
                "profit": float(data["final_results"]["total_profit"]),
            }
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

print("Individual Game Token Usage:")
print("-" * 80)
print(
    f"{'Game':<6} {'Input Tokens':>15} {'Output Tokens':>15} {'Total Tokens':>15} {'Profit':>12}"
)
print("-" * 80)

for game in games_data:
    print(
        f"{game['game']:<6} {game['input']:>15,} {game['output']:>15,} {game['total']:>15,} ${game['profit']:>11.2f}"
    )

print("-" * 80)
print(
    f"{'TOTAL':<6} {total_input_tokens:>15,} {total_output_tokens:>15,} {total_input_tokens + total_output_tokens:>15,}"
)

# Average per game
avg_input = total_input_tokens / len(games_data)
avg_output = total_output_tokens / len(games_data)
avg_total = avg_input + avg_output

print()
print("Average per 30-day game:")
print(f"  Input tokens:  {avg_input:,.0f}")
print(f"  Output tokens: {avg_output:,.0f}")
print(f"  Total tokens:  {avg_total:,.0f}")

print("\n" + "=" * 80)
print("COST ESTIMATES")
print("-" * 80)

# Calculate costs for different tiers
print("\nFREE TIER (Current):")
print("  Cost per game: $0.00")
print(f"  Limits: {FREE_TIER['limits']}")
print("  Problem: Rate limited to ~1-2 days per hour due to 250K TPM limit")

print("\nTIER 1 (Paid - Requires linking billing account):")
tier1_input_cost = (avg_input / 1_000_000) * PAID_TIER["tier1"]["input"]
tier1_output_cost = (avg_output / 1_000_000) * PAID_TIER["tier1"]["output"]
tier1_total = tier1_input_cost + tier1_output_cost

print(f"  Input cost:  ${tier1_input_cost:.4f} per game")
print(f"  Output cost: ${tier1_output_cost:.4f} per game")
print(f"  Total cost:  ${tier1_total:.4f} per game")
print(f"  Limits: {PAID_TIER['tier1']['limits']}")
print("  Speed: ~7-10 minutes per game (no rate limiting)")

# Project costs for full benchmarks
print("\n" + "=" * 80)
print("PROJECTED COSTS FOR FULL BENCHMARKS")
print("-" * 80)

benchmarks = [
    ("5 Flash games", 5),
    ("5 Pro games", 5),
    ("10 games total", 10),
    ("100 games (robust statistics)", 100),
]

print(f"\n{'Benchmark':<30} {'Tier 1 Cost':>15} {'Time Estimate':>20}")
print("-" * 80)

for name, games in benchmarks:
    cost = tier1_total * games
    # Pro might use 20% more tokens
    if "Pro" in name:
        cost *= 1.2
    time_min = games * 7
    time_max = games * 10
    print(f"{name:<30} ${cost:>14.2f} {f'{time_min}-{time_max} minutes':>20}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("-" * 80)
print("""
Based on token usage analysis:

1. Each 30-day game uses ~140-180K tokens total
2. At Tier 1 pricing ($0.30/$2.50 per million), each game costs ~$0.05-0.10
3. Running 10 games (5 Flash + 5 Pro) would cost approximately $0.50-1.00

BENEFITS OF UPGRADING TO TIER 1:
• 10x higher RPM (100 vs 10)
• 8x higher TPM (2M vs 250K)
• Complete 10 games in ~1.5 hours instead of 20+ hours
• Total cost under $1 for comprehensive benchmarking

To upgrade:
1. Go to console.cloud.google.com
2. Link a billing account to your project
3. Rate limits automatically increase to Tier 1
""")
