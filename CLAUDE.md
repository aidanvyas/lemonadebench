# Claude Context for LemonadeBench

## Project Overview

LemonadeBench is an economic reasoning benchmark for Large Language Models. It tests whether LLMs can discover optimal pricing strategies through experimentation in a simple lemonade stand simulation.

**Key Finding**: LLMs fail at basic economic reasoning. They exhibit strong anchoring bias to suggested prices and lack economic intuition about price-demand relationships.

## Development Guidelines

### Package Management
- **Use `uv`** for all dependency management and virtual environment operations
- Run `uv add <package>` to add dependencies
- Use `uv run <command>` to run scripts in the virtual environment
- The project uses `pyproject.toml` for configuration

### Code Quality
- **Use `ruff`** for linting and formatting
- Run `uv run ruff check` before committing
- Run `uv run ruff format` to auto-format code
- Follow type hints throughout the codebase

### Testing
- **Write tests** for all new functionality using pytest
- Run tests with `uv run pytest`
- Maintain test coverage for core game mechanics
- Tests are located in the `tests/` directory

### Voice and Style
- Maintain a **consistent engineering voice** like an Anthropic engineer
- Be precise and technical in documentation
- Focus on reproducible results and clear methodology
- Use proper academic language when describing findings

## Project Structure

```
lemonade_stand/
├── src/lemonade_stand/          # Core game engine and AI player
├── experiments/                 # Benchmark experiments
├── analysis/                   # Result analysis and plotting
├── tests/                      # Unit tests
├── results/                    # JSON experiment results
└── paper/                      # LaTeX paper drafts
```

## Core Components

1. **SimpleLemonadeGame**: Linear demand simulation (Q = 100 - 25p)
2. **ResponsesAIPlayer**: OpenAI Responses API integration with conversation memory
3. **Four Test Conditions**: Suggested price, no guidance, exploration hint, inverse demand

## Key Commands

```bash
# Setup
uv sync

# Run main benchmark
uv run python experiments/run_four_tests.py

# Run tests and linting
uv run pytest
uv run ruff check
uv run ruff format

# Generate analysis
uv run python analysis/generate_plots.py results/latest.json
```

## Research Focus

This benchmark demonstrates that current LLMs:
- Exhibit strong anchoring bias to suggested prices
- Fail to infer demand curves from profit feedback  
- Don't engage in systematic price exploration
- Lack basic economic intuition about price-demand relationships

Even with perfect memory and computational tools, models achieve only 70-75% of optimal performance on this trivial economic task.

## Token Usage Tracking

Always track and report:
- Input tokens
- Output tokens  
- Reasoning tokens (for reasoning models)
- Total tokens
- API duration
- Tool call counts

This data is crucial for understanding the computational cost of economic reasoning failures.