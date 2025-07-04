#!/usr/bin/env python3
"""Analyze LemonadeBench results and generate tables in various formats."""

import argparse
import json
import sys
import statistics
from pathlib import Path
from typing import List, Dict, Any


def calculate_stats(values: List[float]) -> tuple[float, float, float, float]:
    """Calculate mean, std, min, max for a list of values."""
    if not values:
        return 0, 0, 0, 0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0
    return mean, std, min(values), max(values)


def generate_text_tables(results: List[Dict[str, Any]], model: str, runs_per_test: int, days: int) -> str:
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


def generate_detailed_text(results: List[Dict[str, Any]], model: str, runs_per_test: int, days: int) -> str:
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


def generate_latex_tables(results: List[Dict[str, Any]], model: str, runs_per_test: int, days: int) -> tuple[str, str]:
    """Generate LaTeX tables (detailed and simple versions)."""
    latex_content = []
    latex_content.append(f"% LEMONADEBENCH DETAILED RESULTS")
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


def main():
    """Main entry point for analyzing results."""
    parser = argparse.ArgumentParser(description='Analyze LemonadeBench experiment results')
    parser.add_argument('input_file', nargs='?', help='JSON results file to analyze')
    parser.add_argument('--format', choices=['text', 'detailed', 'latex', 'all'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--output', help='Output file (default: stdout for text, filename.tex for latex)')
    parser.add_argument('--latest', action='store_true', help='Use the most recent results file')
    
    args = parser.parse_args()
    
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
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    model = data.get('model', 'gpt-4.1-nano')
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


if __name__ == "__main__":
    main()