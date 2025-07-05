# LemonadeBench: Economic Reasoning Benchmark for LLMs

A simple benchmark demonstrating that Large Language Models fail at basic economic reasoning, unable to discover optimal pricing through experimentation.

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
uv run python experiments/run_benchmark.py  # Runs 5 iterations of 4 test conditions
```

## Test Conditions

1. **Suggested Price**: Model starts with $1.00 suggestion (tests anchoring bias)
2. **No Guidance**: No price suggestion (tests autonomous exploration)
3. **Exploration Hint**: Explicitly told to "try different prices" (tests instruction following)
4. **Inverse Demand**: Q = 50 - 25p with optimal at $1.00 (tests exploration direction)

## Project Structure

```
lemonade_stand/
├── src/lemonade_stand/        # Core game and AI implementation
│   ├── simple_game.py         # Economic simulation (Q = 100 - 25p)
│   ├── responses_ai_player.py # OpenAI Responses API integration
│   └── comprehensive_recorder.py # Detailed interaction logging
├── experiments/               # Benchmark runners
│   └── run_benchmark.py       # Main benchmark with rate limiting
├── analysis/                  # Results analysis
│   └── analyze_results.py     # Statistical analysis and visualization
├── tests/                     # Unit tests
└── results/                   # Experiment outputs
    ├── json/                  # Full experiment results
    ├── plots/                 # Generated visualizations
    └── tex/                   # LaTeX tables
```

## Analysis

View results:
```bash
uv run python analysis/analyze_results.py --latest         # Text summary
uv run python analysis/analyze_results.py --latest --latex # LaTeX tables
uv run python analysis/analyze_results.py --latest --plots # Generate visualizations
```

## Citation

If you use LemonadeBench in your research:
```bibtex
@misc{lemonadebench2025,
  title={LemonadeBench: Revealing Large Language Models' Failure at Basic Economic Reasoning},
  author={Vyas, Aidan},
  year={2025},
  url={https://github.com/aidanvyas/lemonadebench}
}
```

## License

MIT License - see LICENSE file for details.