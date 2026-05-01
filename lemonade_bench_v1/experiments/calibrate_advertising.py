#!/usr/bin/env python3
"""Calibration script for the proposed advertising mechanic.

Runs Monte Carlo trials of "spend $X on ads on day 1, run optimal-price game
for N days" and reports the distribution of cumulative profit deltas vs a
no-ad baseline. Use this to tune (sqrt_scale, mult_cap, decay, variability)
before locking the design in.

Example:
    .venv/bin/python experiments/calibrate_advertising.py
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

# --- Baseline game constants (mirror business_game.py) ---
HOURLY_MULTIPLIERS: dict[int, float] = {
    6: 0.3,
    7: 0.5,
    8: 0.7,
    9: 0.8,
    10: 1.0,
    11: 1.2,
    12: 1.5,
    13: 1.3,
    14: 0.9,
    15: 0.8,
    16: 0.9,
    17: 1.1,
    18: 1.0,
    19: 0.7,
    20: 0.4,
}
PRICE = 2.69
INGREDIENT_COST = 0.37
HOURS_OPEN = list(range(6, 21))  # 6am–8pm, 15 hours
HOUR_VARIATION = 0.10  # ±10% noise per hour, matches DemandModel default
DEMAND_INTERCEPT = 50
DEMAND_SLOPE = 10


@dataclass
class AdParams:
    """Parameters controlling the advertising mechanic."""

    sqrt_scale: float = 10.0  # goodwill = sqrt(spend/100) * sqrt_scale * variability
    mult_cap: float = 0.20  # max multiplier above 1.0 (saturation cap)
    decay: float = 0.80  # daily decay of goodwill
    var_lo: float = 0.90  # variability range on goodwill earned
    var_hi: float = 1.10


def baseline_daily_customers() -> int:
    """One realization of a day's customer count at optimal price + hours."""
    base = max(0, DEMAND_INTERCEPT - DEMAND_SLOPE * PRICE)
    total = 0
    for hour in HOURS_OPEN:
        noise = random.uniform(1 - HOUR_VARIATION, 1 + HOUR_VARIATION)
        total += round(base * HOURLY_MULTIPLIERS[hour] * noise)
    return total


def simulate_one_trial(spend: float, params: AdParams, n_days: int) -> float:
    """Return cumulative profit delta (with-ad − without-ad − spend)."""
    goodwill = (
        math.sqrt(spend / 100)
        * params.sqrt_scale
        * random.uniform(
            params.var_lo,
            params.var_hi,
        )
    )
    profit_delta = 0.0
    margin_per_cup = PRICE - INGREDIENT_COST
    for _ in range(n_days):
        baseline_cust = baseline_daily_customers()
        mult = 1 + params.mult_cap * (1 - math.exp(-goodwill))
        ad_cust = round(baseline_cust * mult)
        # Operating costs (labor + utilities) are identical with or without
        # ads, so they cancel when computing the delta.
        profit_delta += (ad_cust - baseline_cust) * margin_per_cup
        goodwill *= params.decay
    return profit_delta - spend


def simulate_spend(
    spend: float,
    params: AdParams,
    n_days: int = 30,
    n_trials: int = 5000,
) -> dict[str, float]:
    """Run n_trials of an ad campaign, summarize the profit-delta distribution."""
    deltas = [simulate_one_trial(spend, params, n_days) for _ in range(n_trials)]
    deltas.sort()
    return {
        "spend": spend,
        "mean": statistics.mean(deltas),
        "median": deltas[n_trials // 2],
        "std": statistics.stdev(deltas),
        "p10": deltas[int(n_trials * 0.10)],
        "p90": deltas[int(n_trials * 0.90)],
        "pct_profitable": sum(1 for d in deltas if d > 0) / n_trials,
        "mean_roi": statistics.mean(deltas) / spend if spend > 0 else 0.0,
    }


def print_sweep(spends: list[float], params: AdParams, n_days: int) -> None:
    """Print a calibration table across spend amounts."""
    print(
        f"\nAd calibration — n_days={n_days}, params="
        f"sqrt_scale={params.sqrt_scale}, mult_cap={params.mult_cap}, "
        f"decay={params.decay}, var=({params.var_lo},{params.var_hi})",
    )
    print(
        f"{'spend':>8} {'mean Δ':>10} {'median Δ':>10} {'p10':>10} {'p90':>10} "
        f"{'std':>8} {'%profit':>8} {'ROI':>7}",
    )
    print("-" * 80)
    for spend in spends:
        r = simulate_spend(spend, params, n_days=n_days)
        print(
            f"${r['spend']:>7.0f} "
            f"{r['mean']:>+10.0f} "
            f"{r['median']:>+10.0f} "
            f"{r['p10']:>+10.0f} "
            f"{r['p90']:>+10.0f} "
            f"{r['std']:>8.0f} "
            f"{r['pct_profitable']:>8.1%} "
            f"{r['mean_roi']:>+7.2f}x",
        )


def main() -> None:
    spends = [50, 100, 200, 500, 1000, 2500, 5000, 10000]

    print("=" * 80)
    print("CURRENT PROPOSAL")
    print("=" * 80)
    print_sweep(spends, AdParams(), n_days=30)

    print("\n" + "=" * 80)
    print("WEAKER MEAN BOOST (mult_cap=0.10)")
    print("=" * 80)
    print_sweep(spends, AdParams(mult_cap=0.10), n_days=30)

    print("\n" + "=" * 80)
    print("HIGHER VARIANCE (var=0.5–1.5)")
    print("=" * 80)
    print_sweep(spends, AdParams(var_lo=0.5, var_hi=1.5), n_days=30)

    print("\n" + "=" * 80)
    print("FASTER DECAY (decay=0.6)")
    print("=" * 80)
    print_sweep(spends, AdParams(decay=0.6), n_days=30)

    print("\n" + "=" * 80)
    print("HEAVY DIMINISHING RETURNS (sqrt_scale=5, mult_cap=0.10)")
    print("=" * 80)
    print_sweep(spends, AdParams(sqrt_scale=5.0, mult_cap=0.10), n_days=30)


if __name__ == "__main__":
    main()
