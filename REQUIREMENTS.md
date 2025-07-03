# Requirements and Implementation Status

## Current Implementation
- ✅ Lemonade stand game with linear demand (Q = 100 - 25p)
- ✅ OpenAI Responses API integration with conversation memory
- ✅ Four test conditions (suggested price, no guidance, exploration, inverse)
- ✅ Model comparison framework with multiple models
- ✅ Comprehensive recording of all API interactions
- ✅ Token usage and cost tracking
- ✅ Tool usage analysis (set_price, calculate, get_historical_data)
- ✅ Performance visualization and analysis tools

## Model Pricing (per 1M tokens)
```python
MODEL_PRICING = {
    'gpt-4.1-nano': {'input': 0.10, 'cached_input': 0.025, 'output': 0.40},
    'gpt-4.1-mini': {'input': 0.40, 'cached_input': 0.10, 'output': 1.60},
    'gpt-4.1': {'input': 2.00, 'cached_input': 0.50, 'output': 8.00},
    'o3': {'input': 2.00, 'cached_input': 0.50, 'output': 8.00},
    'o4-mini': {'input': 1.10, 'cached_input': 0.275, 'output': 4.40},
}
```

## Key Findings
- Models exhibit strong anchoring bias to suggested prices
- No systematic exploration of price-demand relationships
- Tool usage is inconsistent (models skip set_price when possible)
- Reasoning models (o4-mini) perform worse, exploring wrong direction
- Conversation memory via previous_response_id prevents traditional caching
- Models achieve only 75% of optimal profits consistently

## Technical Architecture
- **Game Engine**: Simple linear demand with configurable parameters
- **AI Integration**: Responses API with previous_response_id for state
- **Recording**: Comprehensive capture of all interactions for analysis
- **Cost Tracking**: Per-call token tracking with cost calculation
- **Analysis**: Automated result processing and visualization

## Future Enhancements
See `FEATURES.md` for detailed roadmap including:
- Non-linear demand functions
- Multi-agent competition
- Human baseline comparisons
- Statistical significance testing
- Extended economic scenarios

## Usage Guidelines
- Use gpt-4.1-nano for initial experiments (lowest cost)
- Run at least 30 days to see behavioral patterns
- Enable comprehensive recording for detailed analysis
- Compare multiple runs for statistical validity
- Monitor costs via token usage tracking