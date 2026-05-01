# LemonadeBench v1 — Game Mechanics

## Game length and starting state

- **Default horizon**: 100 days (was 30 in v0.5)
- **Starting cash**: $1,000
- **Starting inventory**: 0 of each ingredient
- **Game over**: cash < 0 (bankrupt) or day count exhausted

## Daily decisions

Each day the model makes the following decisions, in any order, before opening the stand:
1. Order supplies (cups, lemons, sugar, water)
2. Set price per lemonade
3. Set operating hours
4. Optionally purchase automation
5. Call `open_for_business()` to commit

Then the simulator runs the day's sales, and the model sees the results next morning.

## Demand

Customer demand per hour:
$$Q(p, h) = (50 - 10p) \cdot m_h \cdot \epsilon_h$$

- $p$ = price per cup
- $m_h$ = hourly multiplier (foot traffic curve)
- $\epsilon_h \sim U(0.9, 1.1)$ = independent ±10% noise per hour

If $50 - 10p \le 0$ (price ≥ $5), demand is zero.

### Hourly multipliers

| Hour | $m_h$ | Hour | $m_h$ | Hour | $m_h$ |
|---|---:|---|---:|---|---:|
| 6 AM | 0.3 | 11 AM | 1.2 | 4 PM | 0.9 |
| 7 AM | 0.5 | 12 PM | 1.5 | 5 PM | 1.1 |
| 8 AM | 0.7 | 1 PM | 1.3 | 6 PM | 1.0 |
| 9 AM | 0.8 | 2 PM | 0.9 | 7 PM | 0.7 |
| 10 AM | 1.0 | 3 PM | 0.8 | 8 PM | 0.4 |

Sum across all 15 hours (6 AM–8 PM): $\sum m_h = 13.1$. Hours outside 6 AM–8 PM have $m_h = 0$.

## Ingredients (recipe + economics)

One cup of lemonade requires **all four** ingredients in stock:

| Item | Base cost | Shelf life |
|---|---:|---:|
| Cup | $0.05 | 30 days |
| Lemon | $0.20 | 7 days |
| Sugar | $0.10 | 60 days |
| Water | $0.02 | never |
| **Total per cup** | **$0.37** | |

- Daily prices vary ±10% around base.
- Inventory uses **FIFO** — oldest items consumed first.
- Items purchased on day $N$ with shelf life $S$ expire on the morning of day $N + S$.
- Expired items are discarded automatically each morning.
- Supplies are delivered instantly when ordered.

## Operating costs

Per hour the stand is open:

| Component | Cost | Eliminated by |
|---|---:|---|
| Labor | $3.00/hr | automation |
| Utilities | $2.00/hr | — |
| **Total (default)** | **$5.00/hr** | |

## Automation (added v1.0)

- One-time purchase: **$1,000**
- Effect: labor drops to $0/hr permanently
- Same-day effect (today's hours bill at the new rate)
- Can only be purchased once
- Rejects if `cash < 1000`
- Utilities ($2/hr) continue regardless

Payback at full operation: $1,000 ÷ ($3 × 15 hrs) ≈ 22 days.

## Theoretical optimal (no automation)

Max profit per cup:
$$\pi(p) = (50 - 10p)(p - 0.37)$$
$$p^* = \$2.69 \quad Q(p^*) = 23.1 \text{ cust/hr}$$

Daily profit at optimal:
$$\pi_{\text{day}} = 53.59 \times 13.1 - 5 \times 15 = \$702.03 - \$75 = \$627.03$$

100-day theoretical max (no automation): **$62,703**.

With automation purchased on day 1: same daily revenue, operating cost drops from $75 to $30 → +$45/day × 100 days = +$4,500. Less the $1,000 cost = **net +$3,500** over no-automation baseline. New theoretical max: **~$66,200**.

## Advertising

A `purchase_advertising(spend)` tool that lets the model spend cash on ads, boosting demand for the following days with diminishing returns and noisy outcomes.

### Mechanic

1. **Spend → goodwill**: each campaign adds to a hidden "advertising goodwill" stock:
   $$\Delta \text{goodwill} = \sqrt{\text{spend}/100} \cdot s \cdot u, \quad u \sim U(0.9, 1.1)$$
   where $s$ is `sqrt_scale` (parameter, currently 10).
2. **Goodwill → demand multiplier** each day:
   $$\text{multiplier} = 1 + c \cdot (1 - e^{-\text{goodwill}})$$
   where $c$ is `mult_cap` — the maximum boost. Saturates at $1+c$ no matter how much goodwill accumulates.
3. **Per-campaign decay + 7-day cutoff**: each campaign at age $d$ contributes $\text{goodwill} \cdot 0.80^d$ to current goodwill. Once $d \ge 7$, the campaign drops out completely (zero contribution).
4. **Effect on demand**: that day's customer counts (after the existing per-hour noise) are scaled by the multiplier derived from total active goodwill across all live campaigns.

### What the model sees vs. doesn't see

- Sees: cash deducted, confirmation of spend.
- Does NOT see: goodwill, multiplier, decay rate, sqrt scaling, variability range. Must infer effectiveness from observed sales over subsequent days.

### Same-day aggregation

Multiple `purchase_advertising` calls in the same day **stack in dollar amount**, then the diminishing-returns curve is applied once at end of day. So ten $20 calls behaves identically to one $200 call. This prevents exploiting square-root concavity by splitting one budget into many small calls.

### 7-day campaign lifetime

Each $purchase_advertising$ call creates a new campaign tracked individually. A campaign at age $d$ days contributes $\text{goodwill} \cdot \text{decay}^d$ to current total goodwill, but **stops contributing entirely once age reaches 7 days**. So a single ad spend has zero residual effect by day 8.

Multiple campaigns from different days sum their contributions into total goodwill, which then maps through the saturation curve to today's multiplier.

### Parameters (current shipped values)

| Parameter | Value | What it controls |
|---|---:|---|
| `sqrt_scale` | 10.0 | How quickly goodwill saturates per dollar spent |
| `mult_cap` | 0.20 | Maximum demand boost — caps multiplier at 1.20× |
| `decay` | 0.80 | Per-campaign daily decay (geometric until cutoff) |
| `lifetime_days` | 7 | Each campaign contributes 0 once age reaches this |
| `var_lo`, `var_hi` | 0.9, 1.1 | ±10% variability on goodwill earned |

Calibration sweep showed marginal ROI hits zero around $200 spend; peak total profit at $200 is roughly $1,900 over a 30-day window. See `experiments/calibrate_advertising.py` to re-tune.

## Loans

A revolving loan facility for short-term liquidity. The score function is **`cash − loan_balance`**, so unpaid debt cuts directly into profit.

### Mechanic

- `take_loan(amount)`: borrow `amount` dollars. Cash and outstanding balance both increase by `amount`. Rejected if it would push outstanding above the cap.
- `repay_loan(amount)`: pay down balance from cash. Both decrease by `amount`. Rejected if `amount > balance` or `amount > cash`.
- **Single revolving balance** — multiple `take_loan` calls accumulate to one balance.
- **Cap**: $10,000 outstanding.

### Daily interest

- Each morning, today's daily rate is drawn from `Uniform(0.9%, 1.1%)`.
- The rate is **shown to the model** in the day summary.
- Interest = `balance × today_rate` is charged at start of day (before any model decisions):
  - If `cash ≥ interest`: paid from cash.
  - If `cash < interest`: pay what we can; the unpaid portion **compounds** onto the balance.

### What this tests

- **Cash-flow management under leverage** — daily interest forces continuous accounting
- **Short-term thinking** — at ~1%/day, holding debt long is ruinous; loans are best for fast-payback investments (e.g. accelerating automation, ad campaigns)
- **Score discipline** — `cash − debt` as the score forces models to actually repay, not just borrow and spend

### Final score

`net_value = cash - loan_balance` (was just `cash` pre-loans).

## Tools available to the model

| Tool | Purpose |
|---|---|
| `check_inventory()` | View stock + expiration dates |
| `check_morning_prices()` | Today's supply prices |
| `get_historical_supply_costs()` | Past supply prices for trend analysis |
| `order_supplies(cups, lemons, sugar, water)` | Buy supplies |
| `set_price(price)` | Today's lemonade price |
| `set_operating_hours(open_hour, close_hour)` | Today's hours |
| `purchase_automation()` | One-time labor elimination ($1,000) |
| `purchase_advertising(spend)` | Buy ads; uncertain ROI, diminishing returns, same-day aggregation |
| `take_loan(amount)` | Borrow cash; capped at $10K; daily variable interest |
| `repay_loan(amount)` | Pay down loan balance from cash |
| `open_for_business()` | Commit decisions; required to start the day |

The model does not have direct access to the demand function, hourly multipliers, or noise distribution — these must be inferred from observed sales.

## What the model sees each day

- Day number, total days
- Cash on hand
- Profit yesterday
- Automation status (Yes/No)
- Historical performance table (every prior day's price, profit, customers served, hours, stockouts)

## Other roadmap items (not yet designed)

- Marketing features beyond simple ad spend (brand investment, promotions, quality upgrades).
- Stock buybacks and dividend payouts (the rest of "capital structure" — loans are done).
- Multi-location, vertical integration — paper roadmap.

See `TODO.md` for the full v1.0/v2.0 backlog.
