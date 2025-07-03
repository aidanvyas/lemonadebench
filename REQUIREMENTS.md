# Requirements and Future Features

## Current Implementation
- ✅ Basic lemonade stand game with price-only mechanics
- ✅ AI player interface with OpenAI API integration
- ✅ Model comparison framework
- ✅ Performance visualization and analysis
- ✅ Configurable game length (days parameter)
- ✅ Full conversation history for AI context
- ✅ API response time tracking

## Future Requirements

### Tool Use Design Decisions
- **When to force tool use**: Determine which tools should be mandatory vs optional
  - Currently: `set_price` is enforced, `get_historical_data` is optional
  - Future: `set_price` may become optional with intelligent defaults
  - Consider: Should some analysis be required before decisions?
  - Balance: Autonomy vs ensuring game progression
- **Default behaviors**: What happens when tools aren't used
  - Current: Falls back to suggested price ($5.00)
  - Alternative: Use last price? Adaptive defaults?
  - Edge cases: First day, after failures, timeouts
- **Tool discovery**: How models learn about available tools
  - Explicit in prompt vs letting them discover
  - Tool descriptions and when to make them more/less detailed
- **Tool call patterns**: Optimize for different model behaviors
  - Some models (o4-mini) are selective, others (gpt-4.1-mini) use every tool
  - Should we reward efficiency or thoroughness?
- **Tool organization**: Alphabetize tool listings for ease of use
  - Consistent ordering helps models and humans parse available tools
  - Future: May want categories when tool count grows

### Cost Tracking
- **Token costs**: Track tokens used per decision and total cost per game
- **Model pricing**: Store cost per 1K tokens for each model
- **Cost efficiency metrics**: Profit per dollar spent on API
- **Budget limits**: Option to cap spending per experiment
- **Cost comparison**: Which model gives best ROI?

### Advanced Metrics
- **Decision quality score**: How close to optimal pricing
- **Learning rate**: How quickly models improve strategy
- **Exploration vs exploitation**: Track pricing variance
- **Strategy persistence**: How long models stick with a strategy

### Technical Improvements
- **Async API calls**: Parallel model testing for faster experiments
- **Caching**: Store results to avoid re-running identical experiments
- **Resume functionality**: Continue interrupted experiments
- **Rate limit handling**: Automatic backoff and retry
- **Export formats**: CSV, Excel, LaTeX tables for papers

### Benchmark Extensions
- **Difficulty levels**: Different demand curves
- **Multi-product**: Lemonade, cookies, etc.
- **Competition mode**: Models compete against each other
- **Human baseline**: Interface for human players
- **Transfer learning**: Test if models learned from one game transfer to variants

### Analysis Tools
- **Statistical significance**: T-tests between model performances
- **Confidence intervals**: Error bars on all metrics
- **Strategy clustering**: Identify common pricing patterns
- **Failure analysis**: Why models don't find optimal strategies

## Model Cost Information (Placeholder)
```python
# To be updated with actual pricing
MODEL_COSTS = {
    "gpt-4.1-nano": {"input": 0.00, "output": 0.00},  # $/1K tokens
    "gpt-4.1-mini": {"input": 0.00, "output": 0.00},
    "o4-mini": {"input": 0.00, "output": 0.00},
}
```

## Usage Monitoring
- Track cumulative API costs across all experiments
- Alert when approaching budget limits
- Generate cost reports by model, experiment type, and date
- Optimize prompts to reduce token usage while maintaining quality