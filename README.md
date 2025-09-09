# LemonadeBench

## Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/aidanvyas/lemonadebench.git
cd lemonadebench
uv sync  # Install dependencies with uv
```

### 2. Set API Keys
```bash
echo "OPENAI_API_KEY=your_openai_key" > .env           # OpenAI (gpt-4.1/5, o3/o4)
echo "GEMINI_API_KEY=your_gemini_key" >> .env          # Google Gemini
# Optional provider keys (scaffolds in place; implementer-provided SDKs)
echo "ANTHROPIC_API_KEY=your_anthropic_key" >> .env    # Anthropic Claude
echo "XAI_API_KEY=your_xai_key" >> .env                # xAI Grok
echo "DEEPSEEK_API_KEY=your_deepseek_key" >> .env      # DeepSeek
```

### 3. Run Benchmark
```bash
# Simple test (5 days, nano)
uv run python experiments/run_benchmark.py --days 5

# Research run (10 days, nano vs o4-mini)
uv run python experiments/run_benchmark.py --days 10 --models gpt-4.1-nano o4-mini

# Full benchmark (30 days, multiple models)
uv run python experiments/run_benchmark.py --days 30 --models gpt-4.1-nano gpt-4.1-mini gpt-4.1 o4-mini o3
```

**Note**: Benchmarks automatically generate analysis (LaTeX tables + plots). Use `--no-analysis` to skip.

## Running Tests

Before running the unit tests, make sure the `openai` package is installed. The
tests import the `OpenAIPlayer` class, which depends on this library. If it is
missing you'll see an error like:

```
ModuleNotFoundError: No module named 'openai'
```

Install dependencies with `uv sync` or install `openai` directly:

```bash
uv pip install openai
```

## Multi-Provider Support

The benchmark supports multiple AI providers through a unified factory pattern:

### Supported Providers
- **OpenAI**: GPT-4.1 family (nano/mini/full), GPT-5 family, O1/O3/O4 models
- **Google Gemini**: Gemini 1.0/1.5/2.0/2.5 models  
- **Anthropic**: Claude models (3.5-sonnet, etc.)
- **xAI**: Grok models (scaffold ready)
- **DeepSeek**: DeepSeek chat/reasoner models (scaffold ready)

### Usage
```bash
# Single provider
uv run python experiments/run_benchmark.py --models gpt-4.1-nano

# Multiple providers in one benchmark
uv run python experiments/run_benchmark.py --models gpt-4.1-nano gemini-2.0-flash-exp claude-3.5-sonnet

# The PlayerFactory automatically routes to the correct provider based on model prefix
```

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
lemonade_stand/
├── pyproject.toml             # Project config & dependencies
├── uv.lock                    # Locked dependency versions
├── .python-version            # Python version for pyenv
├── CLAUDE.md                  # Development guide and common commands
├── src/lemonade_stand/        # Core implementation
│   ├── business_game.py       # Game mechanics + inventory + demand
│   ├── base_player.py         # Abstract player with shared tools
│   ├── openai_player.py       # OpenAI-based AI player
│   ├── gemini_player.py       # Gemini-based AI player
│   ├── anthropic_player.py    # Claude player (scaffold)
│   ├── xai_player.py          # Grok player (scaffold)
│   ├── deepseek_player.py     # DeepSeek player (scaffold)
│   ├── player_factory.py      # Provider routing by model prefix
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


## Roadmap

- **v0.5** (current): Basic inventory management and price discovery
  - Providers: OpenAI, Gemini (complete); Anthropic, xAI, DeepSeek (scaffolded)
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
