#!/usr/bin/env python3
"""Check OpenAI API costs using the Costs API."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import argparse
from datetime import datetime, timedelta

from src.lemonade_stand.costs_tracker import CostsTracker


def main():
    parser = argparse.ArgumentParser(description="Check OpenAI API costs")
    parser.add_argument("--days", type=int, default=7, help="Number of days to look back")
    parser.add_argument("--hourly", action="store_true", help="Show hourly breakdown for today")
    parser.add_argument("--models", nargs="+", help="Filter by specific models")

    args = parser.parse_args()

    # Initialize tracker
    try:
        tracker = CostsTracker()
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure OPENAI_API_KEY or OPENAI_ADMIN_KEY is set")
        return

    if args.hourly:
        # Show hourly breakdown for today
        print("\nHourly Costs (Last 24 hours)")
        print("=" * 60)

        start_time = datetime.now() - timedelta(hours=24)
        data = tracker.get_costs(
            start_time=start_time,
            bucket_width="1h",
            group_by=["model"] if not args.models else None,
            models=args.models
        )

        if "error" in data:
            print(f"Error: {data['error']}")
            print("\nNote: The Costs API may require admin permissions.")
            print("Regular API keys might not have access to organization-wide costs.")
            return

        for bucket in data.get("data", []):
            hour = datetime.fromtimestamp(bucket["start_time"]).strftime("%Y-%m-%d %H:00")
            total = 0

            for result in bucket.get("results", []):
                amount = result.get("amount", {})
                if amount.get("currency") == "usd":
                    total += amount.get("value", 0)

            if total > 0:
                print(f"{hour}: ${total:.6f}")

    else:
        # Show daily summary
        tracker.print_cost_summary(days=args.days)

    # Also show recent experiment cost estimate
    print("\n" + "-" * 60)
    print("Recent Experiment Cost Estimate (last 2 hours):")
    recent_cost = tracker.get_recent_costs(hours=2)
    print(f"  ${recent_cost:.6f}")

    # Compare with our tracked estimates
    print("\nNote: Compare with cost estimates from run_four_tests.py")
    print("to verify our token-based calculations are accurate.")

if __name__ == "__main__":
    main()
