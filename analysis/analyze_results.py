#!/usr/bin/env python3
"""Analyze LemonadeBench results and generate tables and plots in various formats."""

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def calculate_stats(values: list[float]) -> tuple[float, float, float, float]:
    """Calculate mean, std, min, max for a list of values."""
    if not values:
        return 0, 0, 0, 0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0
    return mean, std, min(values), max(values)


def generate_text_tables(results: list[dict[str, Any]], model: str, runs_per_test: int, days: int) -> str:
    """Generate plain text summary tables."""
    output = []

    output.append("LEMONADEBENCH RESULTS")
    output.append(f"Model: {model}")
    output.append(f"Games per condition: {runs_per_test}")
    output.append(f"Days per game: {days}")
    output.append("=" * 80)

    # Table 1: Efficiency Summary
    output.append("\nTable 1: Efficiency by Test Condition")
    output.append("-" * 60)
    output.append(f"{'Condition':<30} {'Mean':<10} {'Std Dev':<10} {'Range'}")
    output.append("-" * 60)

    for r in results:
        mean = r['efficiency']['mean'] * 100
        std = r['efficiency']['std'] * 100
        min_val = r['efficiency']['min'] * 100
        max_val = r['efficiency']['max'] * 100
        output.append(f"{r['test_name']:<30} {mean:>6.1f}%    {std:>6.1f}%    {min_val:.1f}-{max_val:.1f}%")

    # Table 2: Price Exploration
    output.append("\n\nTable 2: Price Exploration Metrics")
    output.append("-" * 60)
    output.append(f"{'Condition':<30} {'Unique Prices':<15} {'Days at Optimal'}")
    output.append("-" * 60)

    for r in results:
        unique = r['unique_prices']['mean']
        optimal = r['days_at_optimal']['mean']
        output.append(f"{r['test_name']:<30} {unique:>6.1f}         {optimal:>6.1f}/{days}")

    # Table 3: Economic Performance
    output.append("\n\nTable 3: Economic Performance")
    output.append("-" * 60)
    output.append(f"{'Condition':<30} {'Avg Profit':<15} {'Optimal':<15} {'Gap'}")
    output.append("-" * 60)

    for r in results:
        avg_profit = r['total_profit']['mean']
        optimal_profit = r['optimal_daily_profit'] * days
        gap = optimal_profit - avg_profit
        output.append(f"{r['test_name']:<30} ${avg_profit:>7.2f}       ${optimal_profit:>7.2f}       ${gap:>6.2f}")

    # Table 4: Token Usage and Cost
    output.append("\n\nTable 4: Token Usage and Cost")
    output.append("-" * 60)
    output.append(f"{'Condition':<30} {'Tokens/Run':<15} {'Total Tokens':<15} {'Cost'}")
    output.append("-" * 60)

    # Determine pricing based on model
    pricing_per_million = {
        'gpt-4.1-nano': 0.15,
        'gpt-4.1-mini': 2.00,
        'gpt-4.1': 10.00,
        'o4-mini': 5.50,
        'o3': 10.00
    }
    token_price = pricing_per_million.get(model, 0.15)

    for r in results:
        tokens_per_run = r['total_tokens'] / r['num_runs']
        total_tokens = r['total_tokens']
        cost = total_tokens * token_price / 1_000_000
        output.append(f"{r['test_name']:<30} {tokens_per_run:>10,.0f}     {total_tokens:>12,}     ${cost:>5.2f}")

    # Overall summary
    total_tokens = sum(r['total_tokens'] for r in results)
    total_cost = total_tokens * token_price / 1_000_000
    output.append("-" * 60)
    output.append(f"{'TOTAL':<30} {'':<15} {total_tokens:>12,}     ${total_cost:>5.2f}")

    # Key finding
    avg_efficiency = statistics.mean(r['efficiency']['mean'] for r in results) * 100
    output.append("\n" + "=" * 80)
    output.append(f"\nKEY FINDING: Despite perfect information and tools, {model} achieved")
    output.append(f"only ~{avg_efficiency:.0f}% efficiency, demonstrating fundamental failure at economic reasoning.")

    return "\n".join(output)


def generate_detailed_text(results: list[dict[str, Any]], model: str, runs_per_test: int, days: int) -> str:
    """Generate detailed text output with tool usage breakdown."""
    output = []

    output.append("LEMONADEBENCH DETAILED RESULTS")
    output.append(f"Model: {model}")
    output.append(f"Games per condition: {runs_per_test}")
    output.append(f"Days per game: {days}")
    output.append("=" * 100)

    for idx, test_result in enumerate(results):
        output.append(f"\n\nTest {idx + 1}: {test_result['test_name']}")
        output.append("-" * 80)

        individual_runs = test_result['individual_runs']

        # Calculate detailed statistics
        profits = [r['total_profit'] for r in individual_runs]
        set_price_calls = [r['tool_call_breakdown'].get('set_price', 0) for r in individual_runs]
        calculate_calls = [r['tool_call_breakdown'].get('calculate', 0) for r in individual_runs]
        get_history_calls = [r['tool_call_breakdown'].get('get_historical_data', 0) for r in individual_runs]

        # Token breakdown
        input_tokens = [r['token_usage']['input_tokens'] for r in individual_runs]
        output_tokens = [r['token_usage']['output_tokens'] for r in individual_runs]
        reasoning_tokens = [r['token_usage'].get('reasoning_tokens', 0) for r in individual_runs]
        cached_tokens = [r['token_usage'].get('cached_input_tokens', 0) for r in individual_runs]

        # Performance metrics
        efficiencies = [r['efficiency'] for r in individual_runs]
        unique_prices = [len(r['unique_prices']) for r in individual_runs]
        days_at_optimal = [r['days_at_optimal'] for r in individual_runs]

        output.append("\nPerformance Metrics:")
        output.append(f"  Efficiency: {statistics.mean(efficiencies)*100:.1f}% ± {statistics.stdev(efficiencies)*100 if len(efficiencies) > 1 else 0:.1f}%")
        output.append(f"  Total Profit: ${statistics.mean(profits):.2f} ± ${statistics.stdev(profits) if len(profits) > 1 else 0:.2f}")
        output.append(f"  Days at Optimal: {statistics.mean(days_at_optimal):.1f} / {days}")
        output.append(f"  Unique Prices Tried: {statistics.mean(unique_prices):.1f}")

        output.append("\nTool Usage (per game):")
        output.append(f"  set_price: {statistics.mean(set_price_calls):.1f} calls")
        output.append(f"  calculate: {statistics.mean(calculate_calls):.1f} calls")
        output.append(f"  get_historical_data: {statistics.mean(get_history_calls):.1f} calls")

        output.append("\nToken Usage (per game):")
        output.append(f"  Input: {statistics.mean(input_tokens):,.0f} tokens")
        output.append(f"  Output: {statistics.mean(output_tokens):,.0f} tokens")
        output.append(f"  Reasoning: {statistics.mean(reasoning_tokens):,.0f} tokens")
        output.append(f"  Cached: {statistics.mean(cached_tokens):,.0f} tokens")
        output.append(f"  Total: {statistics.mean([r['token_usage']['total_tokens'] for r in individual_runs]):,.0f} tokens")

    return "\n".join(output)


def generate_latex_tables(results: list[dict[str, Any]], model: str, runs_per_test: int, days: int) -> tuple[str, str]:
    """Generate LaTeX tables (detailed and simple versions)."""
    latex_content = []
    latex_content.append("% LEMONADEBENCH DETAILED RESULTS")
    latex_content.append(f"% Model: {model}")
    latex_content.append(f"% Games per condition: {runs_per_test}")
    latex_content.append(f"% Days per game: {days}")
    latex_content.append("")

    # Process each test condition
    for idx, test_result in enumerate(results):
        test_name = test_result['test_name']
        individual_runs = test_result['individual_runs']

        # Calculate statistics for each metric
        profits = [r['total_profit'] for r in individual_runs]
        set_price_calls = [r['tool_call_breakdown'].get('set_price', 0) for r in individual_runs]
        calculate_calls = [r['tool_call_breakdown'].get('calculate', 0) for r in individual_runs]
        get_history_calls = [r['tool_call_breakdown'].get('get_historical_data', 0) for r in individual_runs]

        input_tokens = [r['token_usage']['input_tokens'] for r in individual_runs]
        output_tokens = [r['token_usage']['output_tokens'] for r in individual_runs]
        reasoning_tokens = [r['token_usage'].get('reasoning_tokens', 0) for r in individual_runs]

        costs = [r['cost_info']['total_cost'] for r in individual_runs]
        durations = [r['duration_seconds'] for r in individual_runs]

        # Calculate statistics
        profit_mean, profit_std, profit_min, profit_max = calculate_stats(profits)

        # Determine default profit based on test type
        if test_name == "Suggested Price":
            default_profit = "\\$2,250"  # $1.00 * 75 customers * 30 days
        elif test_name == "Inverse Demand (100-50p)":
            default_profit = "\\$1,125"  # $1.50 * 25 customers * 30 days
        else:
            default_profit = "N/A"

        # Determine optimal profit
        optimal_profit = f"\\${test_result['optimal_daily_profit'] * days:,.0f}"

        # Start table
        latex_content.append(f"% Table {idx + 1}: {test_name}")
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\small")
        latex_content.append(f"\\caption{{{test_name}}}")
        latex_content.append("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}")
        latex_content.append("\\hline")

        # Header rows
        latex_content.append("\\multirow{2}{*}{Model} & \\multirow{2}{*}{\\begin{tabular}[c]{@{}c@{}}Default\\\\Profit\\end{tabular}} & \\multirow{2}{*}{\\begin{tabular}[c]{@{}c@{}}Optimal\\\\Profit\\end{tabular}} & \\multicolumn{4}{c|}{Profit} & \\multicolumn{3}{c|}{Tool Calls} & \\multicolumn{3}{c|}{Tokens} & \\multirow{2}{*}{\\begin{tabular}[c]{@{}c@{}}Avg\\\\Cost\\end{tabular}} & \\multirow{2}{*}{\\begin{tabular}[c]{@{}c@{}}Avg\\\\Time\\end{tabular}} \\\\")
        latex_content.append("\\cline{4-13}")
        latex_content.append("& & & Mean & Std & Min & Max & set\\_price & calculate & get\\_history & Input & Reason & Output & & \\\\")
        latex_content.append("\\hline")

        # Data row
        row = []
        row.append(model)
        row.append(default_profit)
        row.append(optimal_profit)
        row.append(f"\\${profit_mean:,.0f}")
        row.append(f"\\${profit_std:,.0f}")
        row.append(f"\\${profit_min:,.0f}")
        row.append(f"\\${profit_max:,.0f}")
        row.append(f"{statistics.mean(set_price_calls):.1f}")
        row.append(f"{statistics.mean(calculate_calls):.1f}")
        row.append(f"{statistics.mean(get_history_calls):.1f}")
        row.append(f"{statistics.mean(input_tokens)/1000:.1f}k")
        row.append(f"{statistics.mean(reasoning_tokens):.0f}")
        row.append(f"{statistics.mean(output_tokens):,.0f}")
        row.append(f"\\${statistics.mean(costs):.4f}")
        row.append(f"{statistics.mean(durations):.1f}s")

        latex_content.append(" & ".join(row) + " \\\\")
        latex_content.append("\\hline")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        latex_content.append("")

    # Create simplified version
    latex_simple = []
    latex_simple.append("% Simplified summary table")
    latex_simple.append("\\begin{table}[h]")
    latex_simple.append("\\centering")
    latex_simple.append(f"\\caption{{LemonadeBench Results Summary ({model}, {runs_per_test} runs per condition)}}")
    latex_simple.append("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    latex_simple.append("\\hline")
    latex_simple.append("Condition & \\begin{tabular}[c]{@{}c@{}}Avg\\\\Profit\\end{tabular} & \\begin{tabular}[c]{@{}c@{}}Std\\\\Dev\\end{tabular} & \\begin{tabular}[c]{@{}c@{}}Efficiency\\\\(\\%)\\end{tabular} & \\begin{tabular}[c]{@{}c@{}}Unique\\\\Prices\\end{tabular} & \\begin{tabular}[c]{@{}c@{}}Days at\\\\Optimal\\end{tabular} & \\begin{tabular}[c]{@{}c@{}}Total\\\\Tokens\\end{tabular} \\\\")
    latex_simple.append("\\hline")

    for r in results:
        row = []
        row.append(r['test_name'].replace("(100-50p)", ""))
        row.append(f"\\${r['total_profit']['mean']:,.0f}")
        row.append(f"\\${r['total_profit']['std']:,.0f}")
        row.append(f"{r['efficiency']['mean']*100:.1f}\\%")
        row.append(f"{r['unique_prices']['mean']:.1f}")
        row.append(f"{r['days_at_optimal']['mean']:.1f}")
        row.append(f"{r['total_tokens']/1000:.0f}k")
        latex_simple.append(" & ".join(row) + " \\\\")

    latex_simple.append("\\hline")
    latex_simple.append("\\end{tabular}")
    latex_simple.append("\\end{table}")

    return "\n".join(latex_content), "\n".join(latex_simple)


def generate_plots(data: dict, output_dir: Path = None) -> Path:
    """Generate plots from experiment results."""
    # Extract metadata
    models = data.get('models', [data.get('model', 'gpt-4.1-nano')])
    if not isinstance(models, list):
        models = [models]
    
    # Get aggregated results
    if 'aggregated_results' in data:
        results = data['aggregated_results']
    else:
        # Fallback for older format
        results = data.get('tests', [])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"LemonadeBench Results - {', '.join(models)}", fontsize=16)
    
    # Prepare data structure
    model_conditions = {}
    for result in results:
        model = result.get('model', models[0] if models else 'unknown')
        condition = result['test_name']
        if model not in model_conditions:
            model_conditions[model] = {}
        model_conditions[model][condition] = result
    
    # Plot 1: Average profit by model and condition
    ax1 = axes[0, 0]
    conditions = sorted(set(r['test_name'] for r in results))
    x = range(len(models))
    width = 0.8 / len(conditions) if conditions else 0.8
    
    for i, condition in enumerate(conditions):
        profits = []
        for model in models:
            if model in model_conditions and condition in model_conditions[model]:
                avg_profit = model_conditions[model][condition]['total_profit']['mean']
                profits.append(avg_profit)
            else:
                profits.append(0)
        ax1.bar([xi + i*width for xi in x], profits, width, label=condition)
    
    ax1.set_title("Average Profit by Model and Condition")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Average Profit ($)")
    ax1.set_xticks([xi + width*len(conditions)/2 for xi in x])
    ax1.set_xticklabels(models, rotation=45 if len(models) > 3 else 0)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Price evolution for suggested condition (first run)
    ax2 = axes[0, 1]
    for result in results:
        if result['test_name'] == "Suggested Price" and 'individual_runs' in result:
            model = result.get('model', models[0] if models else 'unknown')
            # Get first run's prices
            if result['individual_runs']:
                first_run = result['individual_runs'][0]
                if 'prices' in first_run:
                    prices = first_run['prices']
                    days = list(range(1, len(prices) + 1))
                    ax2.plot(days, prices, label=model, alpha=0.7)
    
    ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Optimal ($2.00)')
    ax2.set_title("Price Evolution - Suggested Price Condition")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Token usage by model
    ax3 = axes[1, 0]
    model_tokens = {}
    for result in results:
        model = result.get('model', models[0] if models else 'unknown')
        if model not in model_tokens:
            model_tokens[model] = 0
        model_tokens[model] += result.get('total_tokens', 0)
    
    # Average tokens per game (divide by number of conditions)
    avg_tokens = [model_tokens.get(m, 0) / len(conditions) if conditions else 0 for m in models]
    ax3.bar(models, avg_tokens)
    ax3.set_title("Average Token Usage per Game")
    ax3.set_xlabel("Model")
    ax3.set_ylabel("Tokens")
    ax3.set_xticklabels(models, rotation=45 if len(models) > 3 else 0)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Efficiency distribution
    ax4 = axes[1, 1]
    efficiency_data = []
    labels = []
    for model in models:
        if model in model_conditions:
            for condition in sorted(model_conditions[model].keys()):
                result = model_conditions[model][condition]
                if 'efficiency' in result and 'values' in result['efficiency']:
                    efficiencies = [e * 100 for e in result['efficiency']['values']]
                    if efficiencies:
                        efficiency_data.append(efficiencies)
                        label = f"{model[:10]}/{condition[:10]}" if len(models) > 1 else condition
                        labels.append(label)
    
    if efficiency_data:
        ax4.boxplot(efficiency_data, labels=labels)
        ax4.set_title("Efficiency Distribution")
        ax4.set_xlabel("Model/Condition")
        ax4.set_ylabel("Efficiency (%)")
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    # Use timestamp from results or current time
    timestamp = data.get("timestamp", datetime.now().isoformat())
    timestamp_str = timestamp.replace(":", "").replace("-", "").replace(".", "_")[:15]
    filename = f"lemonadebench_{timestamp_str}.png"
    filepath = output_dir / filename
    
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    
    return filepath


def list_results(recent: int = 10, full: bool = False) -> None:
    """List and summarize all saved results."""
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found.")
        return

    # Find all JSON files
    result_files = sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not result_files:
        print("No result files found.")
        return

    print(f"\nFound {len(result_files)} result files. Showing {min(recent, len(result_files))} most recent:\n")
    print("=" * 100)

    for i, filepath in enumerate(result_files[:recent]):
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Extract metadata
            models = data.get('models', [data.get('model', 'unknown')])
            if not isinstance(models, list):
                models = [models]
            days = data.get('days', 30)
            runs_per_test = data.get('runs_per_test', 1)
            timestamp = data.get('timestamp', 'unknown')
            
            # Get test conditions
            if 'aggregated_results' in data:
                conditions = list(set(r['test_name'] for r in data['aggregated_results']))
                total_tokens = sum(r.get('total_tokens', 0) for r in data['aggregated_results'])
            else:
                conditions = data.get('tests_run', [])
                total_tokens = data.get('total_tokens', 0)

            print(f"\n{i+1}. {filepath.name}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Models: {', '.join(models)}")
            print(f"   Conditions: {', '.join(conditions) if conditions else 'N/A'}")
            print(f"   Configuration: {runs_per_test} runs × {days} days")
            print(f"   Total tokens used: {total_tokens:,}")

            if full and 'aggregated_results' in data:
                print("\n   Detailed Results:")
                for result in data['aggregated_results']:
                    model = result.get('model', models[0] if models else 'unknown')
                    test_name = result['test_name']
                    efficiency = result['efficiency']['mean'] * 100
                    profit = result['total_profit']['mean']
                    tokens = result.get('total_tokens', 0)
                    
                    print(f"   - {model} / {test_name}:")
                    print(f"     Efficiency: {efficiency:.1f}%")
                    print(f"     Avg Profit: ${profit:.2f}")
                    print(f"     Total Tokens: {tokens:,}")

        except Exception as e:
            print(f"   Error reading file: {e}")

    print("\n" + "=" * 100)
    print("\nTo analyze a specific result in detail:")
    print("  python analyze_results.py results/<filename>")
    print("\nTo analyze the latest result with plots:")
    print("  python analyze_results.py --latest --plots")


def analyze_recording(filepath: Path) -> None:
    """Analyze a single recording file for behavioral insights."""
    with open(filepath) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"Behavioral Analysis: {filepath.name}")
    print(f"{'='*70}")
    print(f"Model: {data['model_name']}")
    print(f"Test: {data['test_name']}")
    print(f"Total records: {data['total_records']}")

    # Analyze records by type
    record_types = {}
    tool_calls = {}
    prices_set = []
    calculations = []
    reasoning_tokens = 0
    errors = []

    for record in data['records']:
        record_types[record['type']] = record_types.get(record['type'], 0) + 1

        if record['type'] == 'tool_execution':
            tool_name = record['tool_name']
            tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1

            if tool_name == 'set_price':
                prices_set.append({
                    'day': record['day'],
                    'price': record['arguments'].get('price', 0)
                })
            elif tool_name == 'calculate':
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
            usage = record['data'].get('usage', {})
            reasoning_tokens += usage.get('reasoning_tokens', 0)

    print("\nTool usage:")
    for tool, count in sorted(tool_calls.items()):
        print(f"  {tool}: {count} calls")

    if prices_set:
        print("\nPrice decisions:")
        unique_prices = sorted(set(p['price'] for p in prices_set))
        print(f"  Unique prices tried: {unique_prices}")
        print(f"  Days with explicit set_price: {len(prices_set)}/30")

    if calculations:
        print(f"\nCalculations performed: {len(calculations)}")
        for c in calculations[:3]:  # Show first 3
            print(f"  Day {c['day']}: {c['expression']} = {c['result']}")
        if len(calculations) > 3:
            print(f"  ... and {len(calculations) - 3} more")

    if errors:
        print(f"\nErrors encountered: {len(errors)}")
        for e in errors[:2]:
            print(f"  Day {e['day']}: {e['error'][:80]}...")

    if reasoning_tokens > 0:
        print(f"\nReasoning tokens used: {reasoning_tokens:,}")

    # Extract game performance
    game_states = [r for r in data['records'] if r['type'] == 'game_state']
    if game_states:
        total_profit = sum(r['profit'] for r in game_states)
        final_cash = game_states[-1]['cash'] if game_states else 0
        print("\nGame performance:")
        print(f"  Total profit: ${total_profit:.2f}")
        print(f"  Final cash: ${final_cash:.2f}")


def analyze_recordings(recordings_dir: Path = None) -> None:
    """Analyze all recordings in the raw_data directory."""
    if recordings_dir is None:
        recordings_dir = Path("analysis/raw_data")

    if not recordings_dir.exists():
        print("No recordings found. Run experiments with recording enabled.")
        return

    recordings = list(recordings_dir.glob("*.json"))
    print(f"\nFound {len(recordings)} recordings to analyze")

    # Group by model and test
    grouped = {}
    for filepath in sorted(recordings):
        try:
            with open(filepath) as f:
                data = json.load(f)
            key = f"{data['model_name']}_{data['test_name']}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(filepath)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # Analyze each group
    for key, filepaths in sorted(grouped.items()):
        print(f"\n\n{'='*80}")
        print(f"Analyzing {key} ({len(filepaths)} recordings)")
        print(f"{'='*80}")
        
        # Analyze the most recent recording for this model/test combo
        latest = max(filepaths, key=lambda p: p.stat().st_mtime)
        analyze_recording(latest)

    print(f"\n{'='*70}")
    print("Key Behavioral Insights:")
    print("1. Models often skip set_price tool despite it being required")
    print("2. Calculator usage varies significantly between models")
    print("3. Reasoning tokens are only used by o-series models")
    print("4. All models exhibit strong anchoring bias")


def main():
    """Main entry point for analyzing results."""
    parser = argparse.ArgumentParser(description='Analyze LemonadeBench experiment results')
    parser.add_argument('input_file', nargs='?', help='JSON results file to analyze')
    parser.add_argument('--format', choices=['text', 'detailed', 'latex', 'all'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--output', help='Output file (default: stdout for text, filename.tex for latex)')
    parser.add_argument('--latest', action='store_true', help='Use the most recent results file')
    parser.add_argument('--plots', action='store_true', help='Generate comparison plots')
    parser.add_argument('--plot-dir', type=Path, default=None, help='Output directory for plots (default: plots/)')
    parser.add_argument('--show-plots', action='store_true', help='Show plots after generating')
    parser.add_argument('--list', action='store_true', help='List all saved results')
    parser.add_argument('--recent', type=int, default=10, help='With --list, show N most recent results (default: 10)')
    parser.add_argument('--full', action='store_true', help='With --list, show full details for each result')
    parser.add_argument('--recordings', action='store_true', help='Analyze raw API recordings for behavioral insights')
    parser.add_argument('--recordings-dir', type=Path, default=None, help='Directory containing recordings (default: analysis/raw_data)')

    args = parser.parse_args()

    # Handle --list mode
    if args.list:
        list_results(args.recent, args.full)
        return
    
    # Handle --recordings mode
    if args.recordings:
        analyze_recordings(args.recordings_dir)
        return

    # Determine input file
    if args.latest or not args.input_file:
        # Find the most recent results file
        results_dir = Path("results")
        json_files = sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not json_files:
            print("Error: No results files found in results/")
            sys.exit(1)

        input_file = json_files[0]
        print(f"Using latest results file: {input_file}")
    else:
        input_file = Path(args.input_file)
        if not input_file.exists():
            print(f"Error: File not found: {input_file}")
            sys.exit(1)

    # Load data
    with open(input_file) as f:
        data = json.load(f)

    # Extract metadata
    models = data.get('models', [data.get('model', 'gpt-4.1-nano')])
    if not isinstance(models, list):
        models = [models]
    model = models[0] if models else 'gpt-4.1-nano'  # For text output, use first model
    days = data.get('days', 30)
    runs_per_test = data.get('runs_per_test', 1)

    # Get aggregated results
    if 'aggregated_results' in data:
        results = data['aggregated_results']
    else:
        # Fallback for older format
        print("Warning: Using older results format")
        results = data.get('tests', [])

    # Generate output based on format
    if args.format == 'text' or args.format == 'all':
        output = generate_text_tables(results, model, runs_per_test, days)
        if args.output and args.format == 'text':
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Text tables written to: {args.output}")
        else:
            print(output)

    if args.format == 'detailed' or args.format == 'all':
        output = generate_detailed_text(results, model, runs_per_test, days)
        if args.output and args.format == 'detailed':
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Detailed analysis written to: {args.output}")
        else:
            if args.format == 'all':
                print("\n\n" + "="*100 + "\n")
            print(output)

    if args.format == 'latex' or args.format == 'all':
        detailed_latex, simple_latex = generate_latex_tables(results, model, runs_per_test, days)

        if args.output:
            output_path = Path(args.output)
        else:
            # Default output path for LaTeX
            output_path = Path("paper/tables/lemonadebench_results.tex")
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write detailed version
        with open(output_path, 'w') as f:
            f.write(detailed_latex)
        print(f"LaTeX tables written to: {output_path}")

        # Write simple version
        simple_path = output_path.with_name(output_path.stem + "_simple.tex")
        with open(simple_path, 'w') as f:
            f.write(simple_latex)
        print(f"Simplified LaTeX table written to: {simple_path}")

        if args.format == 'all' and not args.output:
            print("\n\n" + "="*100 + "\n")
            print("SIMPLIFIED LATEX TABLE:")
            print(simple_latex)
    
    # Generate plots if requested
    if args.plots:
        print(f"\nGenerating plots...")
        plot_path = generate_plots(data, args.plot_dir)
        print(f"Plots saved to: {plot_path}")
        
        if args.show_plots:
            plt.show()


if __name__ == "__main__":
    main()
