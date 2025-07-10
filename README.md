# LemonadeBench

## Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/aidanvyas/lemonadebench.git
cd lemonadebench
uv sync  # Install dependencies with uv
```

### 2. Set OpenAI API Key
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 3. Run Benchmark
```bash
# Quick test (1 model, 10 days)
uv run python experiments/run_benchmark.py --models gpt-4.1-mini --days 10

# Full benchmark (5 models, 30 days - default)
uv run python experiments/run_benchmark.py --models gpt-4.1-nano gpt-4.1-mini gpt-4.1 o4-mini o3
```

## Game Mechanics

### Business Operations
- **Starting capital**: $1,000
- **Operating cost**: $5/hour while open
- **Operating window**: 6am-9pm (choose your hours)
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
- `set_operating_hours`: Choose when to open (6am-9pm available)
- `open_for_business`: Start selling for the day

## Project Structure

```
lemonade_stand/
├── pyproject.toml             # Project config & dependencies
├── uv.lock                    # Locked dependency versions
├── .python-version            # Python version for pyenv
├── src/lemonade_stand/        # Core implementation
│   ├── business_game.py       # Game mechanics + inventory + demand
│   ├── openai_player.py       # AI player using OpenAI Responses API
│   └── comprehensive_recorder.py # Metrics tracking and analysis
├── experiments/               # Benchmark runners
│   └── run_benchmark.py       # Main benchmark with rate limiting
├── analysis/                  # Results analysis
│   └── analyze_results.py     # Comprehensive metrics report
├── tests/                     # Unit tests
└── results/                   # Experiment outputs
    └── sample/                # Key results from paper
```

## Analysis

View results:
```bash
# Analyze specific results file (shows summary statistics)
uv run python analysis/analyze_results.py results/json/your_results.json

# Save detailed metrics report (exports comprehensive metrics to JSON)
uv run python analysis/analyze_results.py results/json/your_results.json --save-report metrics.json

# Show detailed model comparison (displays best/worst games per model)
uv run python analysis/analyze_results.py results/json/your_results.json --compare-models

# List all available result files
uv run python analysis/analyze_results.py --list

# Analyze the most recent results
uv run python analysis/analyze_results.py --latest

# Generate LaTeX table for paper
uv run python analysis/analyze_results.py --latest --latex results/tex/benchmark.tex

# Generate profit-over-time plots
uv run python analysis/analyze_results.py --latest --plots results/plots/
```

## Roadmap

- **v0.5** (current): Basic inventory management and price discovery
- **v1.0** (in development): Comprehensive economic decision making over a decade
- **v2.0** (planned): Multi-agent markets to test strategic decision making and AI alignment

See [ROADMAP.md](ROADMAP.md) for detailed plans.

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