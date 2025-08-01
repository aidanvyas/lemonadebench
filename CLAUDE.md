# LemonadeBench Codebase Guide

## Common Commands
```bash
# Run benchmark (automatically generates analysis)
uv run python experiments/run_benchmark.py --days 5                                        # Simple test
uv run python experiments/run_benchmark.py --days 10 --models gpt-4.1-nano o4-mini      # Research run
uv run python experiments/run_benchmark.py --days 30 --models gpt-4.1-nano gpt-4.1-mini gpt-4.1 o4-mini o3  # Full benchmark

# Analyze results manually
uv run python analysis/analyze_results.py --latest     # Analyze most recent results
uv run python analysis/analyze_results.py --file results/json/[filename]_full.json

# Development
uv run python -m pytest                # Run tests
uv run ruff check && uv run ruff format    # Lint and format
```

## Core Files
- `src/lemonade_stand/business_game.py` - Main game engine with inventory management
- `src/lemonade_stand/openai_player.py` - AI player using OpenAI Responses API  
- `src/lemonade_stand/game_recorder.py` - Records all interactions, API calls, and game states
- `experiments/run_benchmark.py` - Benchmark runner that orchestrates games and analysis
- `analysis/analyze_results.py` - Generates metrics, LaTeX tables, and plots from recordings

## Recording System
The benchmark now captures EVERYTHING for complete reproducibility:
- Every API request/response with full details
- All reasoning traces (for o1/o3 models)
- All tool calls and their results
- Complete game state before/after each turn
- Token usage breakdown (input/output/reasoning/cached)
- Timing information for API calls

Two JSON files are saved:
- `[timestamp].json` - Summary results for backwards compatibility
- `[timestamp]_full.json` - Complete recording with all interactions

## Game Mechanics
- Demand curve: Q = 50 - 10p (optimal price: $2.50)
- Recipe: 1 cup + 1 lemon + 1 sugar + 1 water = 1 lemonade
- Inventory expires: Lemons (7 days), Cups (30 days), Sugar (60 days), Water (never)
- Operating cost: $5/hour
- Operating hours: Any hours 0-23 (24-hour operation allowed)
- Goal: Maximize profit over 30 days (default)

## Project Guidelines
- Use `uv add <package>` for dependencies (never pip)
- AI must call `open_for_business()` after setting price/hours each day
- Default test model is `gpt-4.1-nano` (cheapest/fastest)
- Benchmarks automatically run analysis and generate LaTeX/plots
- All recordings use comprehensive format for full reproducibility