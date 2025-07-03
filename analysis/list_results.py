#!/usr/bin/env python3
"""List and summarize saved results."""

import json
from pathlib import Path
from datetime import datetime
import argparse


def summarize_results(filepath: Path) -> dict:
    """Load and summarize a results file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    models = data.get('models', [])
    conditions = data.get('conditions', [])
    runs = data.get('runs_per_model', 0)
    days = data.get('days', 0)
    total_results = len(data.get('results', []))
    
    # Calculate total tokens used
    total_tokens = 0
    for result in data.get('results', []):
        if 'token_usage' in result:
            total_tokens += result['token_usage'].get('total_tokens', 0)
    
    return {
        'file': filepath.name,
        'timestamp': data.get('timestamp', 'unknown'),
        'models': models,
        'conditions': conditions,
        'runs_per_condition': runs,
        'days': days,
        'total_results': total_results,
        'total_tokens': total_tokens
    }


def main():
    """List all results files."""
    parser = argparse.ArgumentParser(description="List saved experiment results")
    parser.add_argument(
        "--recent",
        type=int,
        default=10,
        help="Show N most recent results (default: 10)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full details for each result"
    )
    
    args = parser.parse_args()
    
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found.")
        return 1
    
    # Find all JSON files
    result_files = sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not result_files:
        print("No result files found.")
        return 1
    
    print(f"\nFound {len(result_files)} result files. Showing {min(args.recent, len(result_files))} most recent:\n")
    print("=" * 100)
    
    for i, filepath in enumerate(result_files[:args.recent]):
        try:
            summary = summarize_results(filepath)
            
            print(f"\n{i+1}. {summary['file']}")
            print(f"   Timestamp: {summary['timestamp']}")
            print(f"   Models: {', '.join(summary['models'])}")
            print(f"   Conditions: {', '.join(summary['conditions'])}")
            print(f"   Configuration: {summary['runs_per_condition']} runs × {summary['days']} days")
            print(f"   Total experiments: {summary['total_results']}")
            print(f"   Total tokens used: {summary['total_tokens']:,}")
            
            if args.full:
                # Load and show more details
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Group results by model and condition
                results_summary = {}
                for result in data.get('results', []):
                    key = f"{result['model']}_{result.get('condition', 'default')}"
                    if key not in results_summary:
                        results_summary[key] = {
                            'profits': [],
                            'avg_prices': [],
                            'tokens': []
                        }
                    results_summary[key]['profits'].append(result['total_profit'])
                    results_summary[key]['avg_prices'].append(result['avg_price'])
                    if 'token_usage' in result:
                        results_summary[key]['tokens'].append(result['token_usage']['total_tokens'])
                
                print("\n   Detailed Results:")
                for key, data in sorted(results_summary.items()):
                    avg_profit = sum(data['profits']) / len(data['profits'])
                    avg_price = sum(data['avg_prices']) / len(data['avg_prices'])
                    avg_tokens = sum(data['tokens']) / len(data['tokens']) if data['tokens'] else 0
                    
                    print(f"   - {key}:")
                    print(f"     Avg Profit: ${avg_profit:.2f}")
                    print(f"     Avg Price: ${avg_price:.2f}")
                    print(f"     Avg Tokens: {avg_tokens:.0f}")
                
        except Exception as e:
            print(f"   Error reading file: {e}")
    
    print("\n" + "=" * 100)
    print(f"\nTo generate plots from a result file, use:")
    print(f"  python generate_plots.py results/<filename>")
    print(f"\nTo view a specific result in detail:")
    print(f"  python list_results.py --full --recent 1")
    
    return 0


if __name__ == "__main__":
    exit(main())