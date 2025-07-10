# LemonadeBench: Economic Reasoning Benchmark for LLMs

A benchmark revealing that state-of-the-art Large Language Models fail at basic economic reasoning, unable to discover optimal pricing through experimentation or manage simple business operations effectively.

## Version 0.5 - Business Simulation

LemonadeBench v0.5 tests economic intuition, long-term planning, and decision-making under uncertainty through a 30-day lemonade stand simulation with inventory management.

### Key Findings

Testing 5 OpenAI models on 30-day simulations revealed:
- **61.5x performance gap** between worst (gpt-4.1-nano: $213) and best (o4-mini: $13,092)
- **O3 Paradox**: Achieved 96.4% service rate but lower profit than O4-mini due to overordering
- **No price arbitrage**: Models fail to exploit daily price variations by stockpiling
- **Optimal profit**: $20,065 over 30 days; best model achieved only 65%

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
# Quick test (1 game, 10 days)
uv run python experiments/run_benchmark.py --models gpt-4.1-mini --games 1 --days 10

# Full benchmark (5 games, 30 days)
uv run python experiments/run_benchmark.py --models gpt-4.1-mini gpt-4.1 o4-mini --games 5
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
- Check morning prices and inventory
- Order supplies (instant delivery)
- Set operating hours
- Set lemonade price
- Open for business

## Project Structure

```
lemonade_stand/
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
# Analyze specific results file
uv run python analysis/analyze_results.py results/json/your_results.json

# Save detailed metrics report
uv run python analysis/analyze_results.py results/json/your_results.json --save-report metrics.json

# Show model comparison
uv run python analysis/analyze_results.py results/json/your_results.json --compare-models
```

## Roadmap

- **v0.5** (current): Basic inventory management and price discovery
- **v1.0** (planned): 10-year simulation with financial markets, loans, and marketing
- **v2.0** (future): Multi-agent markets to test AI collusion

See [ROADMAP.md](ROADMAP.md) for detailed plans.

## Citation

If you use LemonadeBench in your research:
```bibtex
@misc{lemonadebench2025,
  title={LemonadeBench: A Simple Test Reveals Economic Reasoning Gaps in State-of-the-Art LLMs},
  author={Vyas, Aidan},
  year={2025},
  url={https://github.com/aidanvyas/lemonadebench}
}
```

## License

MIT License - see LICENSE file for details.