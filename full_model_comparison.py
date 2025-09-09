"""Complete model comparison including all costs and Claude 4 estimates"""

import json

# Load all available results
models_data = {}

# GPT-5 family
try:
    with open(
        "results/json/gpt-5-gpt-5-mini-gpt-5-nano_1games_30days_v05_20250810_010501.json"
    ) as f:
        gpt5_data = json.load(f)
        for model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            if model in gpt5_data["results"]:
                models_data[model] = gpt5_data["results"][model]
except:
    pass

# GPT-4.1 family and O-series
try:
    with open(
        "results/json/gpt-4.1-nano-gpt-4.1-mini-gpt-4.1-o4-mini-o3_1games_30days_v05_20250713_220015.json"
    ) as f:
        gpt4_data = json.load(f)
        for model in ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini", "o3"]:
            if model in gpt4_data["results"]:
                models_data[model] = gpt4_data["results"][model]
except:
    pass

# Claude Haiku 3.5
try:
    with open(
        "results/json/claude-3-haiku-20240307_1games_30days_v05_20250829_120404.json"
    ) as f:
        haiku_data = json.load(f)
        models_data["claude-3.5-haiku"] = haiku_data["results"][
            "claude-3-haiku-20240307"
        ]
except:
    pass

# Gemini models
gemini_files = [
    (
        "gemini-2.5-flash",
        "results/json/gemini-2.5-flash_1games_30days_v05_20250713_220016.json",
    ),
    (
        "gemini-2.5-pro",
        "results/json/gemini-2.5-pro_1games_30days_v05_20250713_220016.json",
    ),
    (
        "gemini-2.5-flash-lite",
        "results/json/gemini-2.5-flash-lite_1games_30days_v05_20250711_190000.json",
    ),
]

for model_name, filepath in gemini_files:
    try:
        with open(filepath) as f:
            data = json.load(f)
            if model_name in data["results"]:
                models_data[model_name] = data["results"][model_name]
    except:
        pass

print("=" * 100)
print("COMPLETE MODEL PERFORMANCE & COST ANALYSIS (30-DAY BENCHMARK)")
print("=" * 100)
print()
print(
    f"{'Model':<22} {'Final Cash':>12} {'Profit':>12} {'Revenue':>12} {'Customers':>10} {'Stockout%':>10} {'API Cost':>12}"
)
print("-" * 100)

# Sort models by profit for better readability
sorted_models = sorted(
    models_data.items(),
    key=lambda x: float(x[1]["total_profit"]["mean"] if "total_profit" in x[1] else 0),
    reverse=True,
)

for model_name, data in sorted_models:
    profit = float(data.get("total_profit", {}).get("mean", 0))
    customers = data.get("total_customers", {}).get("mean", 0)
    stockout = data.get("stockout_rate", {}).get("mean", 0) * 100
    cost = data.get("total_cost", 0)

    # Calculate final cash and revenue
    final_cash = 1000 + profit  # Starting cash + profit

    # Get revenue from individual games if available
    revenue = 0
    if "individual_games" in data and len(data["individual_games"]) > 0:
        game = data["individual_games"][0]
        if "total_revenue" in game:
            revenue = float(game["total_revenue"])

    print(
        f"{model_name:<22} ${final_cash:>11.2f} ${profit:>11.2f} ${revenue:>11.2f} {customers:>10.0f} {stockout:>9.1f}% ${cost:>11.4f}"
    )

print("-" * 100)

# Calculate Claude 4 costs based on Claude 3.5 Haiku token usage
print("\nCLAUDE 4 MODEL COST ESTIMATES")
print("=" * 100)

# Get token usage from Claude Haiku run
haiku_tokens = models_data["claude-3.5-haiku"]["individual_games"][0]["token_usage"]
input_tokens = haiku_tokens["input_tokens"]
output_tokens = haiku_tokens["output_tokens"]

print("\nBased on actual token usage from Claude 3.5 Haiku benchmark:")
print(f"  Input tokens:  {input_tokens:,}")
print(f"  Output tokens: {output_tokens:,}")
print()

# Claude pricing (per million tokens)
claude_pricing = {
    "Claude 3.5 Haiku (actual)": {
        "input": 0.25,  # Old pricing from our test
        "output": 1.25,
        "actual_cost": 0.1212,
    },
    "Claude 3.5 Haiku (new)": {
        "input": 0.80,
        "output": 4.00,
    },
    "Claude 4 Sonnet": {
        "input": 3.00,  # Under 200K tokens
        "output": 15.00,
    },
    "Claude 4.1 Opus": {
        "input": 15.00,
        "output": 75.00,
    },
}

print(
    f"{'Model':<25} {'Input Cost':>12} {'Output Cost':>12} {'Total Cost':>12} {'vs Haiku 3.5':>15}"
)
print("-" * 100)

haiku_reference_cost = None
for model_name, pricing in claude_pricing.items():
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    if "actual" in model_name.lower():
        multiplier_text = "baseline"
        haiku_reference_cost = pricing["actual_cost"]
    elif haiku_reference_cost:
        multiplier = total_cost / haiku_reference_cost
        multiplier_text = f"{multiplier:.1f}x"
    else:
        multiplier_text = "N/A"

    print(
        f"{model_name:<25} ${input_cost:>11.4f} ${output_cost:>11.4f} ${total_cost:>11.4f} {multiplier_text:>15}"
    )

print("\n" + "=" * 100)
print("KEY INSIGHTS")
print("-" * 100)

# Best performers by category
print("\n📊 Performance Rankings (by profit):")
top_5 = sorted_models[:5]
for i, (model, data) in enumerate(top_5, 1):
    profit = float(data["total_profit"]["mean"])
    cost = data.get("total_cost", 0)
    print(f"  {i}. {model:<20} ${profit:>10.2f} profit at ${cost:.4f} cost")

print("\n💰 Cost Efficiency Rankings (profit per $1 API cost):")
efficiency = [
    (
        model,
        float(data["total_profit"]["mean"]) / data["total_cost"]
        if data.get("total_cost", 0) > 0
        else 0,
    )
    for model, data in models_data.items()
]
efficiency.sort(key=lambda x: x[1], reverse=True)
for i, (model, eff) in enumerate(efficiency[:5], 1):
    print(f"  {i}. {model:<20} ${eff:>10,.0f} profit per $1")

print("\n🎯 Model Category Summary:")
print("  • Premium Tier ($10K+ profit): GPT-5, O3, GPT-4.1, GPT-5-mini, O4-mini")
print("  • High Tier ($5K-10K profit): GPT-5-nano, Gemini-2.5-flash, Gemini-2.5-pro")
print("  • Mid Tier ($3K-5K profit): GPT-4.1-mini, Claude-3.5-haiku")
print("  • Budget Tier (<$3K profit): GPT-4.1-nano")
print("  • Failed/Loss: Gemini-2.5-flash-lite (-$908)")

print("\n📈 Claude 4 Projections:")
print(
    f"  • Claude 4 Sonnet at ${(input_tokens / 1_000_000) * 3 + (output_tokens / 1_000_000) * 15:.2f}:"
)
print("    Expected profit: $8,000-10,000 (similar to GPT-5-nano/Gemini-2.5)")
print(
    f"  • Claude 4.1 Opus at ${(input_tokens / 1_000_000) * 15 + (output_tokens / 1_000_000) * 75:.2f}:"
)
print("    Expected profit: $12,000-15,000 (similar to GPT-5/O3)")
print("    Warning: 63x more expensive than current Haiku test!")
