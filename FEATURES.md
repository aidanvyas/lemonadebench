# LemonadeBench Features & Roadmap

## Current Features (v0.1.0)

### Core Game Engine
- [x] Linear demand function simulation (Q = 100 - 25p)
- [x] Daily profit tracking
- [x] Configurable game length
- [x] Multiple prompt conditions (suggested price, no guidance, exploration hints)

### AI Integration
- [x] OpenAI Responses API integration with conversation memory
- [x] Tool support (calculator, historical data access)
- [x] Token usage tracking
- [x] Multiple model support (GPT-4.1-nano, GPT-4.1-mini, o4-mini, etc.)

### Analysis & Benchmarking
- [x] Four core test conditions
- [x] Results saving (JSON format)
- [x] Plot generation from results
- [x] Multi-model comparison framework
- [x] Token cost tracking

## Demonstrated Findings

- [x] **Anchoring Bias**: Models stick to suggested prices even when suboptimal
- [x] **No Economic Intuition**: Models don't infer demand curves from profit feedback
- [x] **Wrong Direction Exploration**: When starting above optimal, models explore upward instead of downward
- [x] **Memory Doesn't Help**: Full conversation history doesn't improve economic reasoning

## Planned Features (Future Versions)

### Enhanced Economic Scenarios
- [ ] Non-linear demand functions (quadratic, exponential)
- [ ] Time-varying demand (seasonal effects, trends)
- [ ] Stochastic demand with noise
- [ ] Competition scenarios (multiple agents)
- [ ] Supply constraints and inventory management

### Advanced AI Testing
- [ ] Reasoning model comparison (o1, o3, o4 series)
- [ ] Different prompting strategies (chain-of-thought, few-shot examples)
- [ ] Multi-turn price negotiations
- [ ] Learning from human demonstrations

### Expanded Benchmark Suite
- [ ] Multiple difficulty levels
- [ ] Different business scenarios (restaurant pricing, ride-sharing, etc.)
- [ ] Cross-domain transfer testing
- [ ] Human baseline comparison

### Analysis & Evaluation
- [ ] Statistical significance testing
- [ ] Confidence intervals for results
- [ ] Learning curve analysis
- [ ] Error pattern classification
- [ ] Publication-ready result tables

### Technical Improvements
- [ ] Configuration file system
- [ ] Automated experiment scheduling
- [ ] Parallel model testing
- [ ] Result database integration
- [ ] Web interface for human testing
- [ ] OpenAI Costs API integration (requires admin API keys for actual cost verification)

## Research Applications

### Potential Papers
- [ ] "LLMs Lack Basic Economic Intuition: Evidence from Simple Pricing Tasks"
- [ ] "Anchoring Bias in Large Language Models: A Lemonade Stand Study"
- [ ] "Why AI Can't Run a Lemonade Stand: Economic Reasoning Failures"

### Extensions
- [ ] Multi-agent economic simulations
- [ ] Game theory scenarios
- [ ] Market mechanism design testing
- [ ] Behavioral economics comparisons

## Technical Debt
- [ ] Add comprehensive unit tests
- [ ] Error handling improvements
- [ ] Code documentation
- [ ] Type hints throughout
- [ ] Performance optimization for large-scale experiments

## Community Contributions Welcome

We're particularly interested in:
- Additional economic scenarios
- New model integrations
- Statistical analysis improvements
- Visualization enhancements
- Documentation improvements

## Version History

### v0.1.0 (Current)
- Basic lemonade stand simulation
- Four core test conditions
- Responses API integration
- Initial findings on LLM economic reasoning failures