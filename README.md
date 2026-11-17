# tradingbot

Research code for a daily-bar ETF trading system: a trend-switch **core** that holds most of the money,
plus short-term **sleeves** (dip-buy and breakout), each of which has to pass a fixed set of gates
before it is trusted.

The rules come from a public video course on building a trading bot; the permutation test comes from a
separate video on strategy validation. Nothing here is investment advice, and the current results
(below) do not support trading these rules with real money.

## Status

Built and tested: data download, backtester with self-checks, core rule, sleeve rules, gates,
permutation tests, account combiner.

Not built yet: the live side (broker connection, order placement, risk limits, alerts, reviews).

## Rules

- **Core:** hold QQQ while it closes above its 200-day average, otherwise hold GLD.
- **Dip-buy sleeve:** above the 200-day average, buy when RSI(2) closes below 10; sell on a close above
  the 5-day average or after 10 closes; 8% stop.
- **Breakout sleeve:** above the 200-day average, buy a 50-day high close; sell on a 20-day low close
  or a close below the 200-day average; 8% stop.
- Every signal is read at a close and filled at the next open. Costs are 0.05% per side.

All settings live in `config/system.yaml`.

## Run it

```
pip install -e ".[dev]"
python -m pytest
python -m tradingbot.cli fetch      # Yahoo Finance bars and VIX from FRED into data/
python -m tradingbot.cli check      # the backtester must pass on fake data first
python -m tradingbot.cli study      # label how each ticker moves (training data)
python -m tradingbot.cli core       # core rule, neighbouring settings, doubled costs
python -m tradingbot.cli sleeves    # walk every ticker x family through the gates
python -m tradingbot.cli mcpt       # permutation tests (training data)
python -m tradingbot.cli account    # combine the core with the sleeves that survive
```

Add `--locked` to `core`, `sleeves` or `account` to run the locked test (2025 onward). Each locked
result is written to `results/locked_ledger.json` and is never recomputed, so the test stays a
one-time check. Do not change a rule after looking at it.

## Validation gates

1. The backtester recovers a planted 1% edge, shows about zero on noise, and rejects a signal that
   reads tomorrow's price.
2. Sleeves: fits the ticker's label, profitable in training, enough trades, profit factor of 1.3 or
   more, profitable at neighbouring settings.
3. Permutation test: the rule must beat 99% of runs on shuffled prices (p of 0.01 or less).
4. Locked test, once.

## Results so far (training window 2022-01-03 to 2024-12-31, data fetched 2026-11-17)

- Backtester checks: planted edge +0.84% per trade, noise -0.10%, cheating signal stopped. All pass.
- Core: 13.5% a year, 28.4% maximum drawdown. All four neighbouring settings are profitable.
- Sleeves: 7 of 30 candidates pass the training gates.
- Permutation tests: **nothing passes.** The core's p-value is 0.50 (its return equals the median on
  shuffled prices) and the sleeves range from 0.23 to 0.44.
- The locked test has not been run.

The shuffle keeps each ticker's overall rise, so the test asks whether the timing rule adds anything
beyond holding the assets. Over these three years it did not, to a level that can be told from luck.

## Layout

```
config/system.yaml        every setting
tradingbot/data.py        downloads and loads bars
tradingbot/indicators.py  SMA, EMA, RSI, rolling highs and lows
tradingbot/signals.py     trend switch, dip-buy, breakout
tradingbot/backtester.py  next-open fills, costs, stops, look-ahead guard, stats
tradingbot/checks.py      fake-data checks of the backtester
tradingbot/labels.py      snaps-back / keeps-going labels
tradingbot/gates.py       sleeve gates and the one-time locked test
tradingbot/permutation.py bar shuffling and the permutation test
tradingbot/account.py     sleeve selection and the combined account
tradingbot/cli.py         the commands above
```
