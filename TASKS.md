# LemonadeBench Tasks

> Actionable tasks extracted from ROADMAP.md  
> Last Updated: 2025-09-09

## 🔥 Current Sprint (This Week)
_Focus: API Integration & Model Support_

- [ ] **Claude Support** - Add Anthropic player implementation
  - Branch: `task/add-claude-support`
  - Files: `src/lemonade_stand/anthropic_player.py`
  - Test with: `uv run python experiments/run_benchmark.py --models claude-3.5-haiku --days 5`

- [ ] **Fix o1/o3 Token Counting** - Reasoning tokens not being tracked
  - Branch: `fix/o1-token-counting`
  - Issue: Reasoning tokens show as 0 in results
  
- [ ] **Add Gemini Flash 2.0** - Update pricing and model config
  - Branch: `task/gemini-flash-2`
  - Files: `src/lemonade_stand/gemini_player.py`

## 📦 Next Up (Backlog)
### High Priority
- [ ] OpenRouter integration for unified model access
- [ ] Implement caching for repeated API calls
- [ ] Add retry logic with exponential backoff (check if already done)

### Medium Priority  
- [ ] Web interface for human players
- [ ] Leaderboard system design
- [ ] Batch processing for parallel games

### Low Priority / Nice to Have
- [ ] Performance analytics dashboard
- [ ] Extended 10-year simulation support
- [ ] Multi-location expansion mechanics

## 🐛 Bug Fixes
- [ ] Token usage incorrect for streaming responses
- [ ] Game recorder doesn't capture tool call details for some models
- [ ] Memory leak in long-running benchmarks (>100 games)

## 🧹 Technical Debt
- [ ] Refactor player factory pattern
- [ ] Add comprehensive type hints
- [ ] Improve test coverage (currently 67%)
- [ ] Document API response formats

## ✅ Completed (This Month)
- [x] Complete ruff compliance for src/ directory - 2025-09-09
- [x] Maintain 100% test pass rate (39/39 tests) - 2025-09-09
- [x] Add GPT-5 model family support - 2025-08-29
- [x] Implement Decimal type for currency - 2025-08-27
- [x] Add comprehensive game recording - 2025-08-25

---

## Task Guidelines

### Creating a Task Branch
```bash
git checkout main && git pull
git checkout -b [type]/[description]
# types: task|fix|docs|refactor|test
```

### Task Lifecycle
1. Pick task from "Current Sprint"
2. Create branch
3. Implement & test
4. Create PR with link to this task
5. Move to "Completed" after merge

### Priority Levels
- 🔥 **Current Sprint**: Do this week
- 📦 **Next Up**: Next 2-4 weeks  
- 🐛 **Bug Fixes**: As encountered
- 🧹 **Technical Debt**: When touching related code