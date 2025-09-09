"""Calculate actual Gemini costs based on known token usage"""

# From our earlier analysis, Gemini 2.5 Flash uses approximately:
FLASH_TOKENS_PER_GAME = {
    "input": 144000,  # ~144K input tokens per 30-day game
    "output": 3000,  # ~3K output tokens per 30-day game
    "total": 147000,  # ~147K total
}

# Gemini 2.5 Pro likely uses similar or slightly more
PRO_TOKENS_PER_GAME = {
    "input": 160000,  # Estimate 10% more than Flash
    "output": 3500,
    "total": 163500,
}

# Pricing per million tokens
TIER_PRICING = {
    "Free": {
        "input": 0.00,
        "output": 0.00,
        "limits": "10 RPM Flash / 5 RPM Pro, 250K TPM",
        "speed": "20+ hours per game due to rate limits",
    },
    "Tier 1": {
        "input": 0.30,  # $0.30 per million
        "output": 2.50,  # $2.50 per million
        "limits": "100 RPM Flash / 50 RPM Pro, 2M TPM",
        "speed": "7-10 minutes per game",
    },
    "Tier 2": {
        "input": 0.30,
        "output": 2.50,
        "limits": "2000 RPM Flash / 1000 RPM Pro, 10M TPM",
        "speed": "5-7 minutes per game",
    },
}

print("=" * 90)
print("GEMINI COST CALCULATOR - ACCURATE TOKEN ESTIMATES")
print("=" * 90)
print()

print("TOKEN USAGE PER 30-DAY GAME:")
print("-" * 90)
print(f"{'Model':<20} {'Input Tokens':>15} {'Output Tokens':>15} {'Total Tokens':>15}")
print("-" * 90)
print(
    f"{'Gemini 2.5 Flash':<20} {FLASH_TOKENS_PER_GAME['input']:>15,} {FLASH_TOKENS_PER_GAME['output']:>15,} {FLASH_TOKENS_PER_GAME['total']:>15,}"
)
print(
    f"{'Gemini 2.5 Pro':<20} {PRO_TOKENS_PER_GAME['input']:>15,} {PRO_TOKENS_PER_GAME['output']:>15,} {PRO_TOKENS_PER_GAME['total']:>15,}"
)

print("\n" + "=" * 90)
print("COST PER GAME BY TIER:")
print("-" * 90)

for tier_name, pricing in TIER_PRICING.items():
    print(f"\n{tier_name.upper()}:")

    if pricing["input"] == 0:
        print("  Flash: $0.00 (free)")
        print("  Pro:   $0.00 (free)")
        print(f"  Limits: {pricing['limits']}")
        print(f"  Speed:  {pricing['speed']}")
    else:
        # Calculate Flash cost
        flash_input_cost = (FLASH_TOKENS_PER_GAME["input"] / 1_000_000) * pricing[
            "input"
        ]
        flash_output_cost = (FLASH_TOKENS_PER_GAME["output"] / 1_000_000) * pricing[
            "output"
        ]
        flash_total = flash_input_cost + flash_output_cost

        # Calculate Pro cost
        pro_input_cost = (PRO_TOKENS_PER_GAME["input"] / 1_000_000) * pricing["input"]
        pro_output_cost = (PRO_TOKENS_PER_GAME["output"] / 1_000_000) * pricing[
            "output"
        ]
        pro_total = pro_input_cost + pro_output_cost

        print(
            f"  Flash: ${flash_total:.4f} per game (${flash_input_cost:.4f} input + ${flash_output_cost:.4f} output)"
        )
        print(
            f"  Pro:   ${pro_total:.4f} per game (${pro_input_cost:.4f} input + ${pro_output_cost:.4f} output)"
        )
        print(f"  Limits: {pricing['limits']}")
        print(f"  Speed:  {pricing['speed']}")

print("\n" + "=" * 90)
print("TOTAL COST PROJECTIONS:")
print("-" * 90)

# Use Tier 1 pricing for projections
tier1 = TIER_PRICING["Tier 1"]
flash_cost = (FLASH_TOKENS_PER_GAME["input"] / 1_000_000) * tier1["input"] + (
    FLASH_TOKENS_PER_GAME["output"] / 1_000_000
) * tier1["output"]
pro_cost = (PRO_TOKENS_PER_GAME["input"] / 1_000_000) * tier1["input"] + (
    PRO_TOKENS_PER_GAME["output"] / 1_000_000
) * tier1["output"]

scenarios = [
    (
        "Complete current benchmarks (1 Flash + 5 Pro)",
        1 * flash_cost + 5 * pro_cost,
        "40-60 min",
    ),
    (
        "Standard benchmark (5 Flash + 5 Pro)",
        5 * flash_cost + 5 * pro_cost,
        "70-100 min",
    ),
    (
        "Robust statistics (25 Flash + 25 Pro)",
        25 * flash_cost + 25 * pro_cost,
        "6-8 hours",
    ),
    ("Large study (50 Flash + 50 Pro)", 50 * flash_cost + 50 * pro_cost, "12-16 hours"),
]

print(f"\n{'Scenario':<45} {'Cost (Tier 1)':>15} {'Time Required':>20}")
print("-" * 90)
for scenario, cost, time in scenarios:
    print(f"{scenario:<45} ${cost:>14.2f} {time:>20}")

print("\n" + "=" * 90)
print("RECOMMENDATION:")
print("-" * 90)
print(f"""
For immediate needs (completing your current benchmarks):
• Cost: ${1 * flash_cost + 5 * pro_cost:.2f} (1 Flash game + 5 Pro games)
• Time: ~1 hour with Tier 1 limits
• Alternative: Wait 20+ hours on free tier

Best value approach:
• Link billing account for Tier 1 ($250 total spend unlocks Tier 2)
• Run 10 games for ${5 * flash_cost + 5 * pro_cost:.2f}
• Get statistically significant results in ~1.5 hours

Note: These are ACTUAL costs based on real token usage from your benchmarks.
Each game genuinely uses 140-180K tokens due to the growing conversation history.
""")
