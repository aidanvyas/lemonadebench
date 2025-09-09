# LemonadeBench NanoEval Integration Guide

## Overview

This guide explains how to run LemonadeBench with OpenAI's NanoEval framework, providing a clean, minimal, and highly performant evaluation system.

## Architecture

The integration follows NanoEval's three core principles:

### 1. Minimal Indirection (~200 lines)
- Core evaluation logic in `lemonade_nanoeval.py`
- Clean separation between task generation and solving
- No unnecessary abstractions

### 2. Separation of Concerns
```
GameTask → LemonadeSolver → TaskResult → SQLite
   ↓            ↓              ↓           ↓
(Day setup) (AI decisions) (Metrics)  (Storage)
```

### 3. High Performance
- Async/await for parallel execution
- SQLite for efficient result tracking
- Controlled parallelism to respect rate limits

## Key Components

### GameTask
Represents a single day in the game:
```python
@dataclass
class GameTask:
    task_id: str      # Unique identifier
    day: int          # Day number (1-30)
    game: BusinessGame # Game instance
    context: Dict     # Optional context
```

### LemonadeSolver
Async solver that interfaces with OpenAI:
- Manages API calls with retry logic
- Tracks token usage
- Executes game tools
- Returns structured results

### TaskResult
Captures all metrics from a game day:
- Success/failure status
- Profit, revenue, costs
- AI decisions made
- Token usage
- Execution time

## Usage

### Basic Example
```python
import asyncio
from lemonade_nanoeval import LemonadeBenchEval

async def run_eval():
    eval = LemonadeBenchEval(
        model="gpt-4.1-nano",
        days=30,
        parallel=5  # Run 5 days concurrently
    )
    results = await eval.run()
    print(f"Total Profit: ${results['total_profit']:.2f}")

asyncio.run(run_eval())
```

### Advanced Configuration
```python
eval = LemonadeBenchEval(
    model="gpt-4.1",
    days=30,
    parallel=10,  # Higher parallelism for better models
    db_path=Path("results/gpt4_benchmark.db")
)
```

### Comparing Multiple Models
```python
async def compare_models():
    models = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
    results = {}
    
    for model in models:
        eval = LemonadeBenchEval(model=model, days=30)
        results[model] = await eval.run()
    
    # Compare profits
    for model, result in results.items():
        print(f"{model}: ${result['total_profit']:.2f}")
```

## Database Schema

Results are stored in SQLite with this schema:
```sql
CREATE TABLE results (
    task_id TEXT PRIMARY KEY,
    day INTEGER,
    success BOOLEAN,
    profit REAL,
    revenue REAL,
    costs REAL,
    decisions TEXT,      -- JSON string
    token_usage TEXT,    -- JSON string
    duration_ms INTEGER,
    error TEXT,
    timestamp DATETIME
);
```

## Query Examples

### Get daily profits
```sql
SELECT day, profit 
FROM results 
WHERE success = 1 
ORDER BY day;
```

### Calculate average decision metrics
```sql
SELECT 
    AVG(json_extract(decisions, '$.set_price.price')) as avg_price,
    AVG(json_extract(decisions, '$.set_operating_hours.hours')) as avg_hours
FROM results
WHERE success = 1;
```

### Token usage analysis
```sql
SELECT 
    SUM(json_extract(token_usage, '$.input')) as total_input,
    SUM(json_extract(token_usage, '$.output')) as total_output
FROM results;
```

## Performance Characteristics

### Concurrency
- Default: 5 parallel tasks
- Adjustable based on API rate limits
- Automatic retry with exponential backoff

### Token Usage (30-day game)
| Model | Input Tokens | Output Tokens | Total |
|-------|-------------|---------------|-------|
| gpt-4.1-nano | ~50K | ~10K | ~60K |
| gpt-4.1-mini | ~50K | ~12K | ~62K |
| gpt-4.1 | ~50K | ~15K | ~65K |

### Execution Time
- Serial: ~5-10 minutes for 30 days
- Parallel (5): ~1-2 minutes
- Parallel (10): ~30-60 seconds

## Integration with NanoEval

When NanoEval is released, update the imports:

```python
# Current (standalone)
from lemonade_nanoeval import LemonadeBenchEval

# Future (with NanoEval)
from nanoeval import Eval, EvalSpec
from lemonade_nanoeval import LemonadeSolver, GameTask

class LemonadeBenchEval(Eval):
    def __init__(self, model: str):
        spec = EvalSpec(
            name="lemonade_bench",
            model=model,
            solver=LemonadeSolver(model),
            tasks=self._generate_tasks()
        )
        super().__init__(spec)
```

## Debugging

### Enable detailed logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect database
```bash
sqlite3 lemonade_gpt-4.1-nano_*.db
.headers on
.mode column
SELECT * FROM results LIMIT 10;
```

### Profile with py-spy
```bash
py-spy top -- python lemonade_nanoeval.py
```

## Best Practices

1. **Batch Processing**: Use parallel execution for faster runs
2. **Error Handling**: Check `success` field before using metrics
3. **Token Tracking**: Monitor token usage to estimate costs
4. **Database Queries**: Use JSON functions for decision analysis
5. **Rate Limiting**: Adjust `parallel` based on your API tier

## Comparison with Original Implementation

| Aspect | Original | NanoEval |
|--------|----------|----------|
| Lines of Code | ~500 | ~200 |
| Execution Model | Sequential | Async/Parallel |
| Result Storage | JSON files | SQLite |
| Task Granularity | Full game | Per day |
| Retry Logic | Per attempt | Per task |
| Token Tracking | In-memory | Database |

## Future Enhancements

1. **Streaming Support**: Add streaming for o1/o3 models
2. **Caching**: Implement prompt caching for repeated evaluations
3. **Metrics**: Add more sophisticated performance metrics
4. **Visualization**: Build dashboard for result analysis
5. **Multi-Provider**: Extend to support Anthropic, Gemini

## Conclusion

The NanoEval integration provides a cleaner, faster, and more maintainable way to run LemonadeBench evaluations. The async architecture and SQLite storage make it easy to run large-scale comparisons and analyze results programmatically.