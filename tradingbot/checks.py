"""Check the backtester on fake data before any real result goes through it.

Three checks with pass ranges fixed in advance:
- planted edge: a known +1% per trade must come back as roughly +0.9% after costs;
- pure noise: no edge must come back as roughly zero minus costs;
- cheating signal: a signal that reads tomorrow's close must be stopped by the look-ahead guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtester import LookAheadError, lookahead_guard, run_long_flat, stats

N_DAYS = 1500
RANGES = {"planted_edge": (0.006, 0.012), "pure_noise": (-0.002, 0.001)}


def fake_bars(seed: int, n: int = N_DAYS) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    open_moves = rng.normal(0, 0.005, n)
    close_moves = rng.normal(0, 0.008, n)
    flags = rng.random(n) < 0.10
    opens, closes = np.empty(n), np.empty(n)
    prev_close = 300.0
    for i in range(n):
        opens[i] = prev_close * (1 + open_moves[i])
        closes[i] = opens[i] * (1 + close_moves[i])
        prev_close = closes[i]
    idx = pd.bdate_range("2015-01-01", periods=n)
    bars = pd.DataFrame({"open": opens, "close": closes}, index=idx)
    bars["high"] = bars[["open", "close"]].max(axis=1) * 1.002
    bars["low"] = bars[["open", "close"]].min(axis=1) * 0.998
    return bars[["open", "high", "low", "close"]], pd.Series(flags, index=idx)


def plant_edge(bars: pd.DataFrame, flags: pd.Series, edge: float = 0.01) -> pd.DataFrame:
    """After a flagged close on day t, lift every price from the open of day t+2 onward by `edge`."""
    lifts = np.ones(len(bars))
    for t in np.flatnonzero(flags.to_numpy()):
        if t + 2 < len(bars):
            lifts[t + 2:] *= 1 + edge
    return bars.mul(lifts, axis=0)


def flag_signal(flags: pd.Series) -> pd.DataFrame:
    """Buy the open after every flagged close, sell the open after that."""
    return pd.DataFrame({"entry": flags.to_numpy(bool), "exit": True}, index=flags.index)


def cheating_signal(bars: pd.DataFrame) -> pd.DataFrame:
    """Deliberately wrong: buys when tomorrow's close is higher than today's."""
    peek = bars["close"].shift(-1) > bars["close"]
    return pd.DataFrame({"entry": peek.fillna(False), "exit": True})


def run_checks(cost: float = 0.0005) -> dict:
    out = {}
    bars, flags = fake_bars(7)
    planted = plant_edge(bars, flags)
    avg = stats(run_long_flat(planted, flag_signal(flags), cost=cost))["avg_trade"]
    lo, hi = RANGES["planted_edge"]
    out["planted_edge"] = {"avg_trade": avg, "range": [lo, hi], "pass": bool(lo <= avg <= hi)}

    noise_bars, noise_flags = fake_bars(8)
    avg = stats(run_long_flat(noise_bars, flag_signal(noise_flags), cost=cost))["avg_trade"]
    lo, hi = RANGES["pure_noise"]
    out["pure_noise"] = {"avg_trade": avg, "range": [lo, hi], "pass": bool(lo <= avg <= hi)}

    try:
        lookahead_guard(cheating_signal, planted)
        caught = False
    except LookAheadError:
        caught = True
    out["cheating_signal"] = {"stopped": caught, "pass": caught}
    out["all_pass"] = all(v["pass"] for v in out.values())
    return out
