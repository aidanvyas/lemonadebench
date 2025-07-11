# LemonadeBench Codebase Guide

## Common Commands
```bash
# Run benchmark
uv run python experiments/run_benchmark.py --games 5    # Run 5 games
uv run python experiments/run_benchmark.py --games 30 --models gpt-4.1-mini o3

# Analyze results  
uv run python analysis/analyze_results.py --latest
uv run python analysis/analyze_results.py --latest --latex --plots

# Development
uv run python -m pytest                # Run tests
uv run ruff check && uv run ruff format    # Lint and format
```

## Core Files
- `src/lemonade_stand/business_game.py` - Main game engine with inventory management
- `src/lemonade_stand/openai_player.py` - AI player using OpenAI Responses API
- `experiments/run_benchmark.py` - Benchmark runner with rate limiting
- `analysis/analyze_results.py` - Results analysis and visualization

## Game Mechanics
- Demand curve: Q = 50 - 10p (optimal price: $2.50)
- Recipe: 1 cup + 1 lemon + 1 sugar + 1 water = 1 lemonade
- Inventory expires: Lemons (7 days), Cups (30 days), Sugar (60 days), Water (never)
- Operating cost: $5/hour
- Operating hours: Any hours 0-23 (24-hour operation allowed)
- Goal: Maximize profit over 100 days

## Project Guidelines
- Use `uv add <package>` for dependencies (never pip)
- AI must call `open_for_business()` after setting price/hours each day
- Tests use `assert game.history` to check game state
- Results saved to `results/json/` with timestamps