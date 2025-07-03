# Quick Start Guide

## Setup

1. **Clone the repository**:
```bash
git clone https://github.com/aidanvyas/lemonadebench.git
cd lemonadebench
```

2. **Install dependencies with uv**:
```bash
uv sync
```

3. **Set your OpenAI API key**:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Run the Main Benchmark

### Test all 4 conditions with GPT-4.1-nano (recommended for cost):
```bash
uv run python experiments/run_four_tests.py
```

This runs:
1. **Suggested Price**: Model starts with $1.00 suggestion
2. **No Guidance**: Model starts with no price guidance  
3. **Exploration Hint**: Model encouraged to explore prices
4. **Inverse Demand**: Different demand curve, optimal at $1.00

### Compare multiple models:
```bash
uv run python experiments/compare_models.py --models gpt-4.1-nano gpt-4.1-mini --days 30 --runs 3
```

## Understanding the Game

- **Demand function**: Q = 100 - 25p (standard conditions)
- **Optimal price**: $2.00 (gives 50 customers, $100 profit/day)
- **Models typically stick to**: $1.00 (75 customers, $75 profit/day)
- **Efficiency**: Models achieve only 75% of optimal profits

## Key Findings

Models consistently fail to discover optimal pricing:
- Strong anchoring bias to suggested prices
- No systematic exploration of price-demand relationships
- Even reasoning models (o4-mini) explore in wrong direction
- More compute/reasoning doesn't improve economic intuition

## Analyze Results

View saved results:
```bash
uv run python analysis/list_results.py
```

Generate plots from results:
```bash
uv run python analysis/generate_plots.py results/four_tests_YYYYMMDD_HHMMSS.json
```

## Cost Estimates

Running the full 4-test suite costs approximately:
- GPT-4.1-nano: ~$0.02
- GPT-4.1-mini: ~$0.08  
- GPT-4.1: ~$0.40
- o4-mini: ~$0.20

## Next Steps

See `FEATURES.md` for planned enhancements and research directions.