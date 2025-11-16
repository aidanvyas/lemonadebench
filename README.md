# LemonadeBench

## Quick Start

### 1. Clone and Install
```bash
gh repo clone aidanvyas/lemonadebench
cd lemonadebench
uv sync  # Install dependencies with uv
```

### 2. Set OpenAI API Key
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 3. Run Benchmark
```bash
# Simple test (5 days, gpt-5-nano)
uv run python experiments/run_benchmark.py --days 5

# Research run (10 days, nano vs mini)
uv run python experiments/run_benchmark.py --days 10 --models gpt-5-nano gpt-5-mini

# Full benchmark (30 days, multiple models)
uv run python experiments/run_benchmark.py --days 30 --models gpt-5-nano gpt-5-mini gpt-5 gpt-5.1
```

**Note**: Benchmarks automatically generate analysis (LaTeX tables + plots). Use `--no-analysis` to skip.

## Game Mechanics

### Business Operations
- **Starting capital**: $1,000
- **Game length**: 30 days (default)
- **Operating cost**: $5/hour while open
- **Operating window**: 24 hours (choose any hours)
- **Demand function**: Q = 50 - 10p with hourly variations

### Inventory Management
- **Cups**: 30-day shelf life
- **Lemons**: 7-day shelf life (spoil quickly!)
- **Sugar**: 60-day shelf life  
- **Water**: Never expires
- **Daily price variations**: ±10% on all supplies

### Available Actions
- `check_inventory`: See available supplies and expiration dates
- `check_morning_prices`: View today's supply costs (varies ±10% daily)
- `get_historical_supply_costs`: Review past supply price trends
- `order_supplies`: Purchase supplies (instant delivery)
- `set_price`: Set lemonade price per cup
- `set_operating_hours`: Choose when to open (any hours 0-23)
- `open_for_business`: Start selling for the day

## Project Structure

```
lemonadebench/
├── pyproject.toml             # Project config & dependencies
├── uv.lock                    # Locked dependency versions
├── .python-version            # Python 3.12 (used by uv)
├── src/lemonadebench/         # Core implementation
│   ├── business_game.py       # Game mechanics + inventory + demand
│   ├── openai_player.py       # OpenAI-based AI player
│   └── game_recorder.py       # Comprehensive interaction recording
├── experiments/               # Benchmark runners
│   └── run_benchmark.py       # Orchestrates games + recording + analysis
├── analysis/                  # Business efficiency analysis
│   └── analyze_results.py     # Generates LaTeX tables and plots
├── tests/                     # Unit tests
└── results/                   # Experiment outputs
    ├── json/                  # Raw results + comprehensive recordings
    ├── latex/                 # Generated LaTeX tables
    └── plots/                 # Profit trajectory visualizations
```

## Analysis

Analysis is **automatic** when running benchmarks. For manual analysis:

```bash
# Analyze most recent results
uv run python analysis/analyze_results.py --latest

# Analyze specific comprehensive recording
uv run python analysis/analyze_results.py --file results/json/[filename]_full.json
```

## Running Tests

```bash
uv run python -m pytest
```

## Roadmap

- **v0.5** (current): Basic inventory management and price discovery
- **v1.0** (in development): Comprehensive economic decision making over a decade
- **v2.0** (planned): Multi-agent markets to test strategic decision making and AI alignment

## Citation

If you use LemonadeBench in your research:
```bibtex
@misc{lemonadebench2025,
  title={LemonadeBench: Evaluating the Economic Intuition of Large Language Models in Simple Markets},
  author={Vyas, Aidan},
  year={2025},
  url={https://github.com/aidanvyas/lemonadebench}
}
```

## License

MIT License - see LICENSE file for details.