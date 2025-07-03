#!/usr/bin/env python3
"""Analyze comprehensive recordings to understand model behavior."""

import json
from collections import defaultdict
from pathlib import Path


def analyze_recording(filepath: Path):
    """Analyze a single recording file."""
    with open(filepath) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"Analysis of: {filepath.name}")
    print(f"{'='*70}")
    print(f"Model: {data['model_name']}")
    print(f"Test: {data['test_name']}")
    print(f"Duration: {data['start_time']} to {data['end_time']}")
    print(f"Total records: {data['total_records']}")

    # Analyze records by type
    record_types = defaultdict(int)
    tool_calls = defaultdict(int)
    errors = []
    prices_set = []
    calculations = []
    reasoning_tokens = 0

    for record in data['records']:
        record_types[record['type']] += 1

        if record['type'] == 'tool_execution':
            tool_calls[record['tool_name']] += 1

            if record['tool_name'] == 'set_price':
                prices_set.append({
                    'day': record['day'],
                    'price': record['arguments'].get('price', 0)
                })
            elif record['tool_name'] == 'calculate':
                calculations.append({
                    'day': record['day'],
                    'expression': record['arguments'].get('expression', ''),
                    'result': record['result']
                })

        elif record['type'] == 'error':
            errors.append({
                'day': record['day'],
                'error': record['error_message']
            })

        elif record['type'] == 'response':
            # Extract reasoning tokens if available
            usage = record['data'].get('usage', {})
            reasoning_tokens += usage.get('reasoning_tokens', 0)

            # Check for reasoning output
            for output in record['data'].get('output', []):
                if output['type'] == 'reasoning':
                    print(f"\nReasoning on day {record['day']}: {output.get('content', 'No content')}")

    print("\nRecord types:")
    for rtype, count in record_types.items():
        print(f"  {rtype}: {count}")

    print("\nTool usage:")
    for tool, count in tool_calls.items():
        print(f"  {tool}: {count}")

    print("\nPrices explicitly set:")
    for p in prices_set:
        print(f"  Day {p['day']}: ${p['price']:.2f}")

    if calculations:
        print("\nCalculations performed:")
        for c in calculations[:5]:  # Show first 5
            print(f"  Day {c['day']}: {c['expression']} = {c['result']}")
        if len(calculations) > 5:
            print(f"  ... and {len(calculations) - 5} more")

    if errors:
        print("\nErrors encountered:")
        for e in errors:
            print(f"  Day {e['day']}: {e['error'][:100]}...")

    if reasoning_tokens > 0:
        print(f"\nTotal reasoning tokens: {reasoning_tokens:,}")

    # Extract game performance
    game_states = [r for r in data['records'] if r['type'] == 'game_state']
    if game_states:
        total_profit = sum(r['profit'] for r in game_states)
        prices = [r['price'] for r in game_states]
        print("\nGame performance:")
        print(f"  Total profit: ${total_profit:.2f}")
        print(f"  Unique prices tried: {sorted(set(prices))}")
        print(f"  Days played: {len(game_states)}")

def main():
    """Analyze all recordings in the raw_data directory."""
    raw_data_dir = Path("analysis/raw_data")

    if not raw_data_dir.exists():
        print("No recordings found. Run test_with_recording.py first.")
        return

    recordings = list(raw_data_dir.glob("*.json"))
    print(f"Found {len(recordings)} recordings to analyze")

    for filepath in sorted(recordings):
        analyze_recording(filepath)

    print(f"\n{'='*70}")
    print("Key Insights:")
    print("1. Models often skip set_price tool despite it being required")
    print("2. Calculator usage varies significantly between models")
    print("3. Reasoning tokens are only used by o-series models")
    print("4. All models stick to suboptimal prices (anchoring bias)")

if __name__ == "__main__":
    main()
