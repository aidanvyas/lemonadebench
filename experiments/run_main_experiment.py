#!/usr/bin/env python3
"""Run the main LemonadeBench experiments."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_experiment(name: str, models: list[str], conditions: list[str], runs: int = 30, days: int = 30):
    """Run an experiment and save results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*80}")
    print(f"Running {name} - {timestamp}")
    print(f"Models: {', '.join(models)}")
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Configuration: {runs} runs × {days} days")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable,
        "compare_models.py",
        "--models", *models,
        "--conditions", *conditions,
        "--runs", str(runs),
        "--days", str(days)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ {name} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {name} failed with error: {e}")
        return False

    return True


def main():
    """Run all main experiments."""
    print("LEMONADEBENCH MAIN EXPERIMENTS")
    print("=" * 80)

    # Test configuration (reduced for testing)
    TEST_MODE = True
    if TEST_MODE:
        print("⚠️  Running in TEST MODE with reduced parameters")
        runs = 2
        days = 10
        models = ["gpt-4.1-nano"]
    else:
        runs = 30
        days = 30
        models = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "o4-mini"]

    experiments = [
        {
            "name": "Main Test - Core Conditions",
            "models": models,
            "conditions": ["suggested", "no_guidance"],
        },
        {
            "name": "Exploration Test",
            "models": models,
            "conditions": ["exploration"],
        },
    ]

    # Track results
    results_dir = Path("results")
    initial_files = set(results_dir.glob("*.json")) if results_dir.exists() else set()

    # Run experiments
    for exp in experiments:
        success = run_experiment(
            exp["name"],
            exp["models"],
            exp["conditions"],
            runs=runs,
            days=days
        )
        if not success:
            print("Stopping due to error")
            return 1

    # Find new result files
    final_files = set(results_dir.glob("*.json")) if results_dir.exists() else set()
    new_files = final_files - initial_files

    if new_files:
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"\nGenerated {len(new_files)} result files:")
        for f in sorted(new_files):
            print(f"  - {f.name}")

        print("\nNext steps:")
        print("1. Review results: python list_results.py --full")
        print("2. Generate plots: python generate_plots.py results/<filename>")
        print("3. For inverse demand test, see test_inverse_demand.py")

    return 0


if __name__ == "__main__":
    exit(main())
