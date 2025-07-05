# Claude Instructions for LemonadeBench

## Key Commands to Use

```bash
# Run experiments
uv run python experiments/run_benchmark.py                  # Quick test (5 runs)
uv run python experiments/run_benchmark.py --runs 30       # Full benchmark
uv run python experiments/run_benchmark.py --models gpt-4.1-nano claude-3-haiku

# Analyze results
uv run python analysis/analyze_results.py --list           # See all results
uv run python analysis/analyze_results.py --latest         # Analyze most recent
uv run python analysis/analyze_results.py --latest --latex # Generate LaTeX tables
uv run python analysis/analyze_results.py --latest --plots # Generate plots

# Code quality
uv run ruff check                                          # Check linting
uv run ruff format                                         # Auto-format
uv run python -m pytest                                    # Run tests
```

## Core Files to Know

- `src/lemonade_stand/simple_game.py` - Game mechanics (demand: Q = 100 - 25p, optimal: $2)
- `src/lemonade_stand/responses_ai_player.py` - AI player with OpenAI Responses API
- `experiments/run_benchmark.py` - Benchmark runner with rate limiting
- `analysis/analyze_results.py` - Result analysis and visualization
- `ROADMAP.md` - Future features and versions

## Development Rules

### ALWAYS use uv for packages
- `uv add <package>` to add dependencies
- `uv run <command>` to run any Python command
- Never use pip directly

### Before committing
1. Run `uv run ruff check` and fix any issues
2. Run `uv run ruff format` to format code
3. Run tests if you changed game logic
4. Write clear commit messages

### Code style
- Use type hints for all functions
- Keep functions under 50 lines
- Use f-strings for formatting
- Follow existing patterns in the codebase

## Testing Approach

When adding features:
1. Write tests first in `tests/`
2. Run `uv run python -m pytest -v` to see failures
3. Implement the feature
4. Verify tests pass

## Important Implementation Notes

- Game is deterministic (no randomness) for reproducible benchmarking
- Rate limiting uses OpenAI response headers (x-ratelimit-remaining-*)
- Results organized in `results/`:
  - `json/` - Full experiment results with API interactions
  - `plots/` - Generated visualizations  
  - `tex/` - LaTeX tables
- Always use absolute paths in file operations

## Benchmark Conditions

The benchmark tests 4 scenarios:
1. **suggested**: $1.00 starting price suggested
2. **no-guidance**: No price guidance given  
3. **exploration**: Hint to "try different prices"
4. **inverse**: Different demand curve (Q = 50 - 25p) with $2.00 suggested

## Common Tasks

### Adding a new model
1. Add to `model_pricing` dict in `responses_ai_player.py`
2. Test with `--models your-model --runs 1`
3. Add to default models list if stable

### Debugging rate limits
- Check `x-ratelimit-*` headers in logs
- Use `--runs 1` for testing
- Monitor "Rate limit waits" in output

### Generating paper figures
1. Run full benchmark first
2. Use `--latex` for tables
3. Use `--plots` for visualizations
4. Output in `results/tex/` and `results/plots/`