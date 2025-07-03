# Codebase Cleanup Analysis

## Current Structure

### Core Game Components
- `src/lemonade_stand/simple_game.py` - Main game engine
- `src/lemonade_stand/responses_ai_player.py` - AI player using Responses API

### Deprecated/Old Components
- `src/lemonade_stand/simple_ai_player.py` - Old Chat Completions implementation
- `src/lemonade_stand/tools_ai_player.py` - Old implementation with tools

### Test Scripts (Many!)
- `test_simple_game.py` - Basic game testing
- `test_simple_ai_player.py` - Tests for old AI
- `test_tools_ai_player.py` - Tests for old tools AI
- `test_responses_api.py` - Basic Responses API test
- `test_responses_minimal.py` - Minimal API test
- `test_three_conditions.py` - Old three conditions test
- `test_three_conditions_responses.py` - Updated three conditions
- `test_conversation_memory.py` - Memory testing
- `test_memory_comparison.py` - Memory comparison
- `test_inverse_demand.py` - First inverse demand test
- `test_conditions_comparison.py` - Conditions comparison wrapper

### Main Scripts
- `compare_models.py` - Main comparison framework
- `run_four_tests.py` - Run the 4 main tests
- `run_main_experiment.py` - Main experiment runner

### Utilities
- `generate_plots.py` - Generate plots from results
- `list_results.py` - List and summarize results

## Proposed Cleanup

### 1. Remove Deprecated Files
- Old AI implementations (simple_ai_player.py, tools_ai_player.py)
- Their associated tests
- Redundant test scripts

### 2. Organize into Clear Directories
```
lemonade_stand/
├── src/
│   └── lemonade_stand/
│       ├── __init__.py
│       ├── simple_game.py          # Core game
│       └── responses_ai_player.py  # AI player
├── experiments/
│   ├── run_four_tests.py          # Main 4 tests
│   ├── compare_models.py          # Model comparison
│   └── test_inverse_demand.py     # Special tests
├── analysis/
│   ├── generate_plots.py          # Plot generation
│   └── list_results.py            # Results summary
├── tests/
│   └── test_simple_game.py        # Unit tests
├── results/                        # Results storage
├── plots/                          # Generated plots
└── README.md                       # Documentation
```

### 3. Configuration Management
- Move demand functions to config
- Centralize experiment parameters
- Clean up hardcoded values