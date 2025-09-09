"""Analyze Gemini 2.5 Flash performance across completed games"""

import statistics

# Game results from the 4 completed games
games = [
    {
        "game": 1,
        "profit": 6127.57,
        "customers": 3771,
        "lost_sales": 3535,
        "revenue": 9427.50,
    },
    {
        "game": 2,
        "profit": 11789.44,
        "customers": 5803,
        "lost_sales": 1627,
        "revenue": 15883.25,
    },
    {
        "game": 3,
        "profit": 10186.45,
        "customers": 3748,
        "lost_sales": 156,
        "revenue": 13118.00,
    },
    {
        "game": 4,
        "profit": 5671.90,
        "customers": 3377,
        "lost_sales": 2869,
        "revenue": 8442.50,
    },
]

profits = [g["profit"] for g in games]
customers = [g["customers"] for g in games]
lost_sales = [g["lost_sales"] for g in games]
revenues = [g["revenue"] for g in games]

print("=" * 80)
print("GEMINI 2.5 FLASH - 4 GAME ANALYSIS (30 days each)")
print("=" * 80)
print()

print("Individual Game Results:")
print("-" * 80)
print(
    f"{'Game':<6} {'Profit':>10} {'Revenue':>10} {'Customers':>10} {'Lost Sales':>12} {'Stockout %':>12}"
)
print("-" * 80)

for g in games:
    stockout_pct = (
        (g["lost_sales"] / (g["customers"] + g["lost_sales"])) * 100
        if (g["customers"] + g["lost_sales"]) > 0
        else 0
    )
    print(
        f"{g['game']:<6} ${g['profit']:>9.2f} ${g['revenue']:>9.2f} {g['customers']:>10} {g['lost_sales']:>12} {stockout_pct:>11.1f}%"
    )

print("-" * 80)
print()

print("Statistical Summary:")
print("-" * 80)

# Profit statistics
print("PROFIT:")
print(f"  Average: ${statistics.mean(profits):,.2f}")
print(f"  Std Dev: ${statistics.stdev(profits):,.2f}")
print(f"  Min:     ${min(profits):,.2f}")
print(f"  Max:     ${max(profits):,.2f}")
print(f"  Range:   ${max(profits) - min(profits):,.2f}")
print(f"  CV:      {(statistics.stdev(profits) / statistics.mean(profits)) * 100:.1f}%")

print("\nCUSTOMERS:")
print(f"  Average: {statistics.mean(customers):,.0f}")
print(f"  Std Dev: {statistics.stdev(customers):,.0f}")
print(f"  Min:     {min(customers):,}")
print(f"  Max:     {max(customers):,}")

print("\nLOST SALES (Stockouts):")
print(f"  Average: {statistics.mean(lost_sales):,.0f}")
print(f"  Std Dev: {statistics.stdev(lost_sales):,.0f}")
print(f"  Min:     {min(lost_sales):,}")
print(f"  Max:     {max(lost_sales):,}")

# Calculate overall stockout rate
total_customers = sum(customers)
total_lost = sum(lost_sales)
overall_stockout = (total_lost / (total_customers + total_lost)) * 100

print("\nOVERALL METRICS (across all 4 games):")
print(f"  Total customers served: {total_customers:,}")
print(f"  Total lost sales:       {total_lost:,}")
print(f"  Overall stockout rate:  {overall_stockout:.1f}%")
print(f"  Total revenue:          ${sum(revenues):,.2f}")
print(f"  Average daily profit:   ${statistics.mean(profits) / 30:.2f}")

print()
print("=" * 80)
print("KEY INSIGHTS:")
print("-" * 80)

# Coefficient of variation
cv = (statistics.stdev(profits) / statistics.mean(profits)) * 100

if cv > 40:
    print(
        f"⚠️  HIGH VARIANCE: {cv:.1f}% coefficient of variation indicates inconsistent performance"
    )
elif cv > 20:
    print(
        f"📊 MODERATE VARIANCE: {cv:.1f}% coefficient of variation shows some inconsistency"
    )
else:
    print(
        f"✅ LOW VARIANCE: {cv:.1f}% coefficient of variation indicates consistent performance"
    )

# Best vs worst game
best_idx = profits.index(max(profits))
worst_idx = profits.index(min(profits))
print(
    f"\n📈 Best game (#{games[best_idx]['game']}):  ${games[best_idx]['profit']:,.2f} profit with {games[best_idx]['lost_sales']:,} lost sales"
)
print(
    f"📉 Worst game (#{games[worst_idx]['game']}): ${games[worst_idx]['profit']:,.2f} profit with {games[worst_idx]['lost_sales']:,} lost sales"
)

performance_ratio = max(profits) / min(profits)
print(
    f"\n🎰 Performance ratio: {performance_ratio:.1f}x difference between best and worst game"
)

# Compare to previous single-game result
print("\n📊 Comparison to previous single-game benchmark:")
print("   Previous: $8,473 profit (single game)")
print(f"   Current average: ${statistics.mean(profits):,.2f} (4 games)")
print(
    f"   Difference: ${statistics.mean(profits) - 8473:.2f} ({((statistics.mean(profits) / 8473) - 1) * 100:+.1f}%)"
)
