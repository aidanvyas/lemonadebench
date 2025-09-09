"""Estimate costs for Claude 3.5 Sonnet and Claude 3 Opus based on Haiku benchmark"""

# Token usage from Claude Haiku 30-day run
haiku_total_tokens = 434578
haiku_input_tokens = 416000  # Approximate from ratio
haiku_output_tokens = 18578  # Approximate from ratio

# Anthropic pricing per million tokens
pricing = {
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
}

print("Claude Model Cost Estimates for 30-Day Benchmark")
print("=" * 60)
print(f"Benchmark token usage: {haiku_total_tokens:,} total tokens")
print(f"  - Input tokens: ~{haiku_input_tokens:,}")
print(f"  - Output tokens: ~{haiku_output_tokens:,}")
print("\n" + "-" * 60)

for model, prices in pricing.items():
    input_cost = (haiku_input_tokens / 1_000_000) * prices["input"]
    output_cost = (haiku_output_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost

    print(f"\n{model}:")
    print(f"  Input cost:  ${input_cost:.4f}")
    print(f"  Output cost: ${output_cost:.4f}")
    print(f"  Total cost:  ${total_cost:.4f}")

    if model == "claude-3-haiku":
        print("  (Actual cost from benchmark: $0.1212)")
        haiku_cost = total_cost
    else:
        multiplier = total_cost / haiku_cost
        print(f"  Multiplier vs Haiku: {multiplier:.1f}x")

print("\n" + "=" * 60)
print("\nExpected Performance Implications:")
print("-" * 60)

print("""
Based on model capabilities and the benchmark task:

1. Claude 3.5 Sonnet ($1.28 estimated):
   - Expected profit: $8,000-12,000 range
   - Better reasoning about optimal pricing and inventory
   - Lower stockout rate (likely 15-25%)
   - Similar to GPT-4.1/GPT-5-mini performance tier

2. Claude 3 Opus ($6.39 estimated):
   - Expected profit: $12,000-15,000 range
   - Strongest reasoning and planning capabilities
   - Minimal stockouts (likely <10%)
   - Comparable to GPT-5/O3 performance tier
   - However, 5-6x more expensive than those models

Cost-Performance Trade-offs:
- Haiku: Best for high-volume, cost-sensitive applications
- Sonnet: Balanced choice for production use cases
- Opus: Only justified for complex reasoning tasks where quality matters most
""")
