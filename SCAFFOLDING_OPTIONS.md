# Lemonade Stand Game Design

## Overview
A simplified pricing game to test AI models' ability to discover optimal economic strategies.

## Game Mechanics
- **Demand function**: customers = max(0, 50 - 10 * price)
- **No costs**: Profit = Revenue = Price × Customers
- **Suggested starting price**: $2.00
- **Optimal strategy**: Price at $2.50 maximizes profit
- **Demand examples**:
  - $0.50 → 45 customers → $22.50 revenue
  - $1.00 → 40 customers → $40.00 revenue
  - $2.00 → 30 customers → $60.00 revenue
  - $2.50 → 25 customers → $62.50 revenue (optimal)
  - $3.00 → 20 customers → $60.00 revenue
  - $5.00 → 0 customers → $0 revenue

## AI Interface

**System Prompt:**
```
You run a lemonade stand. Your goal is to maximize total profit over 100 days.

Each morning, you set the price for lemonade.
Each evening, you'll see your profit for that day.

Suggested starting price: $2.00

Respond with a JSON object: {"price": <non-negative number with up to 2 decimal places>}
```

**Turn Prompts:**
- Day 1: `Day 1. Starting cash: $100.00. Set your price.`
- Later days: `Day X. Cash: $Y. Yesterday: Price=$P, Profit=$Q. Set today's price.`

## Expected Behaviors

### Optimal Strategy
With demand = 50 - 10 * price and no costs:
- Revenue = price * customers = price * (50 - 10 * price)
- To find optimal: d(revenue)/d(price) = 50 - 20 * price = 0
- Optimal price = $2.50

**Key insight**: There's a clear optimal price at $2.50 that yields $62.50 profit!

This creates an interesting test:
- Will models discover the optimal $2.50 price?
- Will they get stuck at the suggested $2.00 price?
- Will they explore higher prices despite lower suggested price?
- Can they do basic price optimization?

### Model Differences
- **GPT-4.1-nano**: Faster, cheaper, but might need more scaffolding
- **GPT-4.1-mini**: Better reasoning, might discover optimal price with less help

## Questions for Testing
1. How quickly do models converge to optimal pricing?
2. Do they explore the price space or get stuck?
3. How much does scaffolding help each model?
4. Do they understand the profit calculation?
5. Can they recover from bad initial prices?

## Running the Comparison
```bash
# Test both models with all scaffolding levels
python compare_models.py --models gpt-4.1-nano gpt-4.1-mini --scaffolding minimal medium full --runs 3

# Test just one configuration
python compare_models.py --models gpt-4.1-nano --scaffolding minimal --runs 1
```