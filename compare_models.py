import json

# GPT-5 family results
with open(
    "results/json/gpt-5-gpt-5-mini-gpt-5-nano_1games_30days_v05_20250810_010501.json"
) as f:
    gpt5_data = json.load(f)

# GPT-4.1 family results
with open(
    "results/json/gpt-4.1-nano-gpt-4.1-mini-gpt-4.1-o4-mini-o3_1games_30days_v05_20250713_220015.json"
) as f:
    gpt4_data = json.load(f)

# Claude Haiku results
with open(
    "results/json/claude-3-haiku-20240307_1games_30days_v05_20250829_120404.json"
) as f:
    haiku_data = json.load(f)

print("Model Performance Comparison (30 days)")
print("=" * 70)
print(
    f"{'Model':<20} {'Profit':>10} {'Customers':>10} {'Stockout%':>10} {'API Cost':>10}"
)
print("-" * 70)

# Extract GPT-5 family
for model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
    if model in gpt5_data["results"]:
        r = gpt5_data["results"][model]
        print(
            f"{model:<20} ${r['total_profit']['mean']:>9} {r['total_customers']['mean']:>10} {r['stockout_rate']['mean'] * 100:>9.1f}% ${r['total_cost']:>9.4f}"
        )

# Extract GPT-4.1 family
for model in ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]:
    if model in gpt4_data["results"]:
        r = gpt4_data["results"][model]
        print(
            f"{model:<20} ${r['total_profit']['mean']:>9} {r['total_customers']['mean']:>10} {r['stockout_rate']['mean'] * 100:>9.1f}% ${r['total_cost']:>9.4f}"
        )

# Extract O-series
for model in ["o4-mini", "o3"]:
    if model in gpt4_data["results"]:
        r = gpt4_data["results"][model]
        print(
            f"{model:<20} ${r['total_profit']['mean']:>9} {r['total_customers']['mean']:>10} {r['stockout_rate']['mean'] * 100:>9.1f}% ${r['total_cost']:>9.4f}"
        )

print("-" * 70)

# Claude Haiku
r = haiku_data["results"]["claude-3-haiku-20240307"]
print(
    f"{'claude-3-haiku':<20} ${r['total_profit']['mean']:>9} {r['total_customers']['mean']:>10} {r['stockout_rate']['mean'] * 100:>9.1f}% ${r['total_cost']:>9.4f}"
)

print("\n" + "=" * 70)
print("\nKey Insights:")
print("-" * 70)

# Calculate relative performance
haiku_profit = float(
    haiku_data["results"]["claude-3-haiku-20240307"]["total_profit"]["mean"]
)
haiku_cost = haiku_data["results"]["claude-3-haiku-20240307"]["total_cost"]

# Compare to best and worst performers
gpt5_profit = float(gpt5_data["results"]["gpt-5"]["total_profit"]["mean"])
gpt5_cost = gpt5_data["results"]["gpt-5"]["total_cost"]

gpt4_nano_profit = float(gpt4_data["results"]["gpt-4.1-nano"]["total_profit"]["mean"])
gpt4_nano_cost = gpt4_data["results"]["gpt-4.1-nano"]["total_cost"]

print(f"\n1. Claude Haiku achieved ${haiku_profit:.2f} profit vs:")
print(
    f"   - GPT-5: ${gpt5_profit:.2f} ({(haiku_profit / gpt5_profit) * 100:.1f}% of GPT-5 performance)"
)
print(
    f"   - GPT-4.1-nano: ${gpt4_nano_profit:.2f} ({(haiku_profit / gpt4_nano_profit):.1f}x better)"
)

print("\n2. Cost efficiency (profit per dollar of API cost):")
print(f"   - Claude Haiku: ${haiku_profit / haiku_cost:.2f} profit per $1 API cost")
print(f"   - GPT-5: ${gpt5_profit / gpt5_cost:.2f} profit per $1 API cost")
print(
    f"   - GPT-4.1-nano: ${gpt4_nano_profit / gpt4_nano_cost:.2f} profit per $1 API cost"
)

print("\n3. Claude Haiku positioned as mid-tier performer:")
print(
    f"   - Outperforms budget models (GPT-4.1-nano) by {(haiku_profit / gpt4_nano_profit):.1f}x"
)
print(f"   - Costs {(haiku_cost / gpt4_nano_cost):.1f}x more than GPT-4.1-nano")
print(
    f"   - Achieves {(haiku_profit / gpt5_profit) * 100:.1f}% of GPT-5 performance at {(haiku_cost / gpt5_cost) * 100:.1f}% of the cost"
)
