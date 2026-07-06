# Adaptive Paper Trader

An AI paper-trading system that analyzes crypto markets 24/7, simulates trades,
and **learns from every completed trade**. It never places real orders — the
exchange client is constructed without credentials, so live trading is
impossible by construction.

## How it decides (no fixed indicator rules)

```
market data ──► features (measurements) ──► regime HMM ──► changepoint (BOCPD)
                                                 │                │
                                                 ▼                ▼
                              online learner (calibrated prob) ── veto gates
                                                 │
                          similar-trade memory (kNN retrieval) blends in
                                                 │
                                                 ▼
                            bias / confidence / entry / SL / TP / RR
                                                 │
                              risk sizing (4 adaptive multipliers
                               × fixed circuit breakers) ──► paper trade
```

- **Regime layer** — a sticky Gaussian HMM discovers unlabeled market regimes
  (e.g. "low-vol up-drift"); Bayesian Online Changepoint Detection raises an
  "assumptions no longer valid" alarm that vetoes entries and collapses size.
- **Prediction layer** — an online logistic learner predicts P(TP before SL),
  shrunk toward 0.5 whenever its recent calibration (rolling Brier score) is
  poor. Confidence must be *earned*.
- **Memory layer** — before every analysis, the k nearest historical trades
  (z-scored feature distance) are retrieved; their empirical win rate blends
  into confidence and their lessons feed the reasoning.
- **Risk layer** — fractional-Kelly base risk × volatility targeting ×
  confidence × drawdown decay × changepoint collapse, inside deliberately
  dumb fixed circuit breakers (max positions, hard drawdown, max daily loss).
- **Journal** — every close produces a structured review (entry timing from
  MAE/MFE, SL/TP assessment, lessons) stored in the learning database.

The learner and calibration state persist to `data/learner_state.json`;
all trades/analyses/reviews/equity live in SQLite (`data/paper_trader.db`),
plain SQL schema, easy to migrate to PostgreSQL.

## Quick start (local)

Requires Python 3.11+.

```bash
cd adaptive-paper-trader
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # fill in TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (optional)
# edit config/config.yaml   # symbols, timeframe, thresholds, risk

python main.py
```

Dashboard: http://localhost:8787 (HTML) · `/metrics` (JSON) · `/trades` (JSON).

Runs fine with Telegram unset (notifications become no-ops).

### Optional: LLM-written reasoning & reviews

Set `llm.enabled: true` in `config/config.yaml` and `ANTHROPIC_API_KEY` in
`.env`. The LLM only *explains* — every number and decision still comes from
the quantitative engine, with a deterministic template as fallback.

## Docker

```bash
cp .env.example .env   # fill in secrets
docker compose up -d --build
docker compose logs -f
```

SQLite DB + learner state persist in `./data` (mounted volume).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

18 tests cover: feature finiteness & trend detection, HMM regime separation,
BOCPD alarm behavior, learner convergence, calibration shrinkage, risk
multipliers, circuit breakers, trade PnL math (long & short), retrieval +
confidence blending, journal reviews, and metrics computation.

## Configuration highlights (`config/config.yaml`)

| Key | Meaning |
|---|---|
| `exchange.symbols` / `timeframe` | markets analyzed (any ccxt exchange, public data) |
| `strategy.min_confidence` | calibrated confidence floor to trade (default 0.60) |
| `strategy.min_rr` | minimum risk/reward (default 1.5) |
| `strategy.sl_sigma_mult` / `tp_rr` | SL = 1.6σ, TP = 2× stop distance |
| `changepoint.alert_threshold` | changepoint prob that vetoes entries / triggers exits |
| `risk.base_risk_pct`, `kelly_fraction` | per-trade risk at full multipliers |
| `risk.drawdown_hard_pct`, `max_daily_loss_pct` | fixed circuit breakers |
| `retrieval.k` | similar historical trades retrieved per analysis |

## Project layout

```
main.py                     orchestrator: analyzer loops, monitor loop, dashboard
config/config.yaml          all tunables (secrets stay in .env)
app/
  data/collector.py         ccxt async, read-only, retries (no credentials ever)
  features/engine.py        measurements: returns, vol, flow, structure
  regime/hmm.py             sticky Gaussian HMM (numpy, hand-rolled, tested)
  regime/bocpd.py           Bayesian online changepoint detection
  analysis/model.py         online logistic + calibration tracker (persisted)
  analysis/engine.py        bias/confidence/zones + vetoes + learning updates
  analysis/reasoning.py     template reasoning, optional Claude narrative
  trading/risk.py           adaptive sizing + fixed circuit breakers
  trading/paper_engine.py   simulated entries, PnL math
  trading/monitor.py        TP/SL/adaptive/time exits, MAE/MFE tracking
  journal/review.py         structured post-trade reviews + lessons
  db/database.py            SQLite schema & queries (PostgreSQL-ready SQL)
  db/retrieval.py           kNN similar-trade memory + confidence blend
  telegram/notifier.py      Bot API events: analysis / opened / closed / system
  dashboard/metrics.py      win rate, R, PF, drawdown, Sharpe, calibration…
  dashboard/server.py       aiohttp dashboard
tests/                      18 unit tests
```

## Safety properties

- **Dry-run by default**: the market-data client has no API keys. Real orders
  (Bybit, via `app/trading/executor.py`) require BOTH `live.enabled: true` in
  config AND `BYBIT_API_KEY`/`BYBIT_API_SECRET` in `.env`; until then every
  order intent is only logged (`live_dry_run_*`), never sent. A per-order
  notional cap (`live.max_notional_usd`) applies even when live.
- **Fixed circuit breakers** are outside the adaptive system and cannot be
  loosened by learning.
- Paper equity, sizes, and PnL are simulations for research only — not
  financial advice, and results will differ from live trading (no slippage
  or fee modeling by default).
