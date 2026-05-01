## Tooling
- [ ] Fix all ruff issues completely
- [ ] Fix all ty issues completely

## v1.0 (from paper §Research Roadmap)
- [x] Decade-long simulations (now 100-day default)
- [ ] 30 runs per model for statistical significance
- [ ] Seasonal demand patterns (holidays, weather)
- [ ] Marketing initiatives with uncertain ROI
- [x] Automation vs human labor decisions
- [ ] Multi-location expansion strategies
- [ ] Capital structure: debt issuance, stock buybacks, interest
- [ ] Vertical integration opportunities
- [ ] Human baseline performance

## v2.0 (multi-agent, from paper §Research Roadmap)
- [ ] Global demand function — multiple AI-operated stands compete directly
- [ ] Condition 1: baseline (independent operation, shared demand)
- [ ] Condition 2: communication-enabled (observe spontaneous cartels / tacit collusion)
- [ ] Condition 3: legally-constrained (anti-competitive prohibitions; profit vs compliance tension)

## Other paper-mentioned future work
- [ ] Continuous conversation mode (preserve reasoning chains across days, vs current stateless design)
- [ ] Calibrate efficiency-loss magnitudes against real small-business data
- [ ] Balanced scoring system that equalizes potential impact of each efficiency dimension

## Not in paper but worth considering
- [ ] Multi-provider support (Anthropic, Google, xAI in addition to OpenAI)

## Out of scope (TODO follow-ups from automation PR)
- [ ] Update analysis/analyze_results.py to handle automation in efficiency calculations (still hardcodes $5/hr)
- [ ] Context-window pruning for OpenAI reasoning models on 100-day runs (o3/o4-mini have 200K limits)
- [ ] Re-run v1 baselines with the new defaults so we have comparable post-feature numbers
