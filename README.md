# LemonadeBench: Economic Reasoning Benchmark for LLMs

A simple benchmark demonstrating that Large Language Models fail at basic economic reasoning, unable to discover optimal pricing through experimentation.

## Key Finding

**LLMs achieve only 75% efficiency on a trivial economic task.** Given a lemonade stand with linear demand Q = 100 - 25p:
- **Optimal strategy**: Price at $2.00 → 50 customers → $100 profit/day
- **What LLMs do**: Anchor to $1.00 → 75 customers → $75 profit/day
- **The failure**: Despite having perfect memory, calculator access, and profit feedback, models never discover the optimal price through experimentation

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
4. **Inverse Demand**: Q = 100 - 50p with optimal at $1.00 (tests exploration direction)

## Results Summary

Across all models and conditions:
- **Efficiency**: ~74-76% (vs 100% optimal)
- **Price exploration**: 1-2 unique prices tried (out of infinite possibilities)
- **Days at optimal**: 0/30 in almost all cases
- **Pattern**: Strong anchoring to initial prices, no systematic exploration

## Cost Estimates

Full benchmark (5 runs × 4 conditions × 30 days):
- **gpt-4.1-nano**: ~$0.17
- **gpt-4.1-mini**: ~$2.00
- **gpt-4.1**: ~$10.00
- **o4-mini**: ~$5.50

## Project Structure

```
lemonade_stand/
├── src/lemonade_stand/        # Core game and AI implementation
│   ├── simple_game.py         # Economic simulation (Q = 100 - 25p)
│   ├── responses_ai_player.py # OpenAI Responses API integration
│   └── comprehensive_recorder.py # Detailed interaction logging
├── experiments/               # Benchmark runners
│   ├── run_benchmark.py       # Main benchmark (adaptive rate limiting)
│   └── compare_models.py      # Multi-model comparison
├── analysis/                  # Results analysis
│   └── analyze_results.py     # Statistical analysis, tables, and visualization
├── paper/                     # LaTeX paper and tables
└── results/paper/            # Publication-ready results
```

## Analysis

View results:
```bash
uv run python analysis/analyze_results.py --latest                # Text summary
uv run python analysis/analyze_results.py --latest --format latex  # LaTeX tables
uv run python analysis/analyze_results.py --latest --plots         # Generate visualizations
```

## Technical Details

- **Game Engine**: Configurable demand function with daily profit tracking
- **AI Integration**: OpenAI Responses API with conversation memory (`previous_response_id`)
- **Tools Available**: `get_historical_data`, `set_price`, `calculate`
- **Recording**: Comprehensive capture of all API interactions and calculations
- **Rate Limiting**: Adaptive system to handle API limits gracefully

## Key Insights

This benchmark reveals that current LLMs:
1. **Lack economic intuition**: Don't understand price-demand relationships
2. **Exhibit strong anchoring**: Stick to suggested prices despite poor performance
3. **Fail to explore**: Even when told to experiment, exploration is minimal
4. **Can't learn from feedback**: Have all data needed but don't infer the pattern

The failure is universal across model sizes and types, suggesting a fundamental limitation in economic reasoning capabilities.

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