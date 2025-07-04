# LemonadeBench: Economic Reasoning Benchmark for LLMs

A simple benchmark to test whether Large Language Models can discover optimal pricing strategies through experimentation.

## Key Finding

LLMs fail at basic economic reasoning. Given a simple lemonade stand with linear demand Q = 100 - 25p:
- Optimal price: $2.00 (50 customers, $100 profit/day)
- Models anchor to suggested price ($1.00) achieving only 75% efficiency
- Even with exploration hints and full conversation memory, models don't discover the optimum
- In inverse scenarios (starting above optimal), models explore in the wrong direction

## Project Structure

```
lemonade_stand/
├── src/
│   └── lemonade_stand/
│       ├── __init__.py
│       ├── simple_game.py          # Core game engine
│       └── responses_ai_player.py  # AI player using OpenAI Responses API
├── experiments/
│   ├── run_benchmark.py          # Main benchmark runner (5 runs per test by default)
│   ├── compare_models.py          # Compare multiple models
│   └── test_inverse_demand.py     # Test inverse demand scenarios
├── analysis/
│   ├── generate_plots.py          # Generate plots from results
│   └── list_results.py            # List and summarize results
├── tests/
│   └── test_simple_game.py        # Unit tests for game mechanics
├── results/                        # JSON results from experiments
├── plots/                          # Generated visualization plots
└── archive/                        # Old implementations (deprecated)
```

## Quick Start

1. Install dependencies using [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

3. Run the four main test conditions (5 runs each by default):
```bash
uv run python experiments/run_benchmark.py
```

## Test Conditions

1. **Suggested Price**: Model starts with $1.00 suggestion (suboptimal)
2. **No Guidance**: Model starts with no price suggestion
3. **Exploration Hint**: Model encouraged to "try different prices"
4. **Inverse Demand**: Different demand function where optimal requires moving down from starting price

## Results Summary

With GPT-4.1-nano over 30 days:
- **Suggested Price**: Anchored at $1.00 (75.0% efficiency)
- **No Guidance**: Stuck at $1.00 (75.0% efficiency)
- **Exploration Hint**: Brief exploration to $1.25, then back to $1.00 (75.3% efficiency)
- **Inverse Demand**: Explored wrong direction ($1.50→$1.65→$1.82→$4.17), never found $1.00 optimal (70.5% efficiency)

Total cost: ~$0.02 for all tests with conversation memory enabled.

## Key Components

### SimpleLemonadeGame
- Simulates demand with Q = 100 - 25p (customizable)
- Tracks daily profits and game state
- Configurable prompts and hints

### ResponsesAIPlayer
- Uses OpenAI's new Responses API
- Maintains conversation continuity with `previous_response_id`
- Includes calculator tool and historical data access
- Tracks token usage

## Analysis Tools

View saved results:
```bash
python analysis/list_results.py
```

Generate plots from results:
```bash
python analysis/generate_plots.py results/four_tests_YYYYMMDD_HHMMSS.json
```

## Paper

This benchmark demonstrates that current LLMs:
1. Exhibit strong anchoring bias to suggested prices
2. Fail to infer demand curves from profit feedback
3. Don't engage in systematic price exploration
4. Lack basic economic intuition about price-demand relationships

Even with perfect memory and computational tools, models achieve only 70-75% of optimal performance on this trivial economic task.