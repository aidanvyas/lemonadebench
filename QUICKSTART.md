# Quick Start Guide

## Setup

1. **Set your OpenAI API key** (create .env file):
```bash
echo "OPENAI_API_KEY=your_new_api_key_here" > .env
```

⚠️ **Important**: Please revoke the API key you shared earlier and create a new one for security!

## Test the Demand Curve

First, let's visualize the game mechanics:
```bash
uv run python test_demand_curve.py
```

This shows:
- How demand changes with price
- The optimal price point
- Expected profit at different prices

## Run Model Comparison

### Quick test (1 run each):
```bash
uv run python compare_models.py --models gpt-4.1-nano --scaffolding minimal --runs 1
```

### Full comparison:
```bash
uv run python compare_models.py --models gpt-4.1-nano gpt-4.1-mini --scaffolding minimal medium full --runs 3
```

This will:
1. Test both models with all three scaffolding levels
2. Run 3 games per configuration (18 total games)
3. Save results to `results/` directory
4. Create comparison plots in `plots/` directory
5. Print a summary table

## Understanding Results

The comparison will show:
- **Total Profit**: How much money each model made
- **Average Price**: What prices the models typically set
- **Average Customers**: How many customers they attracted
- **Price Evolution**: How pricing strategy changed over time

## Scaffolding Decision Points

Review `SCAFFOLDING_OPTIONS.md` to understand the three levels:
1. **Minimal**: Just the basics - see if models can figure it out
2. **Medium**: Hints about the trade-offs
3. **Full**: Explicit cost information

## Next Steps

Once we see how the models perform with simple pricing:
1. Add randomness to demand
2. Add inventory management
3. Add more complex decisions
4. Scale to full economic simulation