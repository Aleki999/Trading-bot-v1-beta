"""Backtest engine: signals are read at a close and filled at the next open, after costs.

Two runners share the same rules:
- run_long_flat: one ticker, long or flat, with an optional stop held at the broker and a time exit.
- run_switch: always invested in one of two tickers (the core's trend switch).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


class LookAheadError(Exception):
    """Raised when a signal's value on a day changes once later bars are removed."""


@dataclass
class Trade:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    ret: float            # after costs on both sides
    reason: str           # signal | time | stop | stop_gap | end


@dataclass
class Result:
    daily: pd.Series                       # daily return of the strategy when it uses all its capital
    trades: list[Trade] = field(default_factory=list)
    trade_id: pd.Series | None = None      # index into trades for the trade open that day, -1 when flat

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.trades])


def lookahead_guard(signal_fn, bars: pd.DataFrame, n_days: int = 30, seed: int = 7, warmup: int = 250) -> None:
    """Recompute the signal with the data cut off at random days; it must match the full-data value."""
    full = signal_fn(bars)
    first = min(warmup, max(len(bars) - 2, 1))
    candidates = np.arange(first, len(bars) - 1)
    if len(candidates) == 0:
        return
    rng = np.random.default_rng(seed)
    days = rng.choice(candidates, size=min(n_days, len(candidates)), replace=False)
    for i in days:
        cut = signal_fn(bars.iloc[: i + 1])
        a = np.atleast_1d(np.asarray(cut.iloc[-1], dtype=float))
        b = np.atleast_1d(np.asarray(full.iloc[i], dtype=float))
        if not np.array_equal(a, b, equal_nan=True):
            raise LookAheadError("signal uses future data")


def _bounds(index: pd.DatetimeIndex, start, end) -> tuple[int, int]:
    lo = 0 if start is None else int(index.searchsorted(pd.Timestamp(start), side="left"))
    hi = len(index) - 1 if end is None else int(index.searchsorted(pd.Timestamp(end), side="right")) - 1
    return max(lo, 1), hi


def run_long_flat(bars: pd.DataFrame, signal: pd.DataFrame, cost: float = 0.0005, stop_pct: float | None = None,
                  max_hold: int | None = None, start=None, end=None) -> Result:
    """Long/flat backtest of one ticker. Entries are only taken inside [start, end].

    A signal at the close of day d-1 trades at the open of day d. The stop sits at the broker from
    the entry fill: it fills at the stop price, or at the open when the open is already below it.
    The time exit sells at the open after the `max_hold`-th close in the trade.
    """
    idx = bars.index
    o, low, c = (bars[k].to_numpy(float) for k in ("open", "low", "close"))
    entry = signal["entry"].to_numpy(bool)
    exit_ = signal["exit"].to_numpy(bool)
    lo, hi = _bounds(idx, start, end)

    daily = np.zeros(len(idx))
    trade_id = np.full(len(idx), -1)
    trades: list[Trade] = []
    in_pos = False
    exit_pending = ""
    entry_px = stop_px = 0.0
    entry_i = sig_i = held = 0

    def close_trade(d: int, price: float, reason: str) -> None:
        nonlocal in_pos, exit_pending
        ret = price / entry_px * (1 - cost) ** 2 - 1
        trades.append(Trade(idx[sig_i], idx[entry_i], idx[d], entry_px, price, ret, reason))
        in_pos, exit_pending = False, ""

    for d in range(lo, hi + 1):
        growth = 1.0
        stopped = False
        if in_pos:
            trade_id[d] = len(trades)
            if exit_pending:
                growth *= o[d] / c[d - 1] * (1 - cost)
                close_trade(d, o[d], exit_pending)
            elif stop_pct is not None and o[d] <= stop_px:
                growth *= o[d] / c[d - 1] * (1 - cost)
                close_trade(d, o[d], "stop_gap")
                stopped = True
            elif stop_pct is not None and low[d] <= stop_px:
                growth *= stop_px / c[d - 1] * (1 - cost)
                close_trade(d, stop_px, "stop")
                stopped = True
            else:
                growth *= c[d] / c[d - 1]
        # Flat at the open (or just sold at it on a signal): yesterday's entry signal buys this open.
        if not in_pos and not stopped and entry[d - 1]:
            in_pos, entry_px, entry_i, sig_i, held = True, o[d], d, d - 1, 0
            stop_px = entry_px * (1 - stop_pct) if stop_pct is not None else 0.0
            trade_id[d] = len(trades)
            growth *= 1 - cost
            if stop_pct is not None and low[d] <= stop_px:
                growth *= stop_px / entry_px * (1 - cost)
                close_trade(d, stop_px, "stop")
            else:
                growth *= c[d] / entry_px
        daily[d] = growth - 1
        if in_pos:
            held += 1
            if exit_[d]:
                exit_pending = "signal"
            elif max_hold is not None and held >= max_hold:
                exit_pending = "time"

    if in_pos:  # still open at the end of the window: mark it at the last close, with the exit cost
        ret = c[hi] / entry_px * (1 - cost) ** 2 - 1
        trades.append(Trade(idx[sig_i], idx[entry_i], idx[hi], entry_px, c[hi], ret, "end"))

    sl = slice(lo, hi + 1)
    return Result(pd.Series(daily[sl], index=idx[sl]), trades, pd.Series(trade_id[sl], index=idx[sl]))


def run_switch(engine: pd.DataFrame, safe: pd.DataFrame, in_engine: pd.Series, cost: float = 0.0005,
               start=None, end=None) -> Result:
    """Always hold the engine or the safe asset. `in_engine` at a close sets the holding from the next open."""
    idx = engine.index.intersection(safe.index)
    engine, safe = engine.loc[idx], safe.loc[idx]
    want = in_engine.reindex(idx).fillna(False).to_numpy(bool)
    opens = {True: engine["open"].to_numpy(float), False: safe["open"].to_numpy(float)}
    closes = {True: engine["close"].to_numpy(float), False: safe["close"].to_numpy(float)}
    lo, hi = _bounds(idx, start, end)

    daily = np.zeros(len(idx))
    trades: list[Trade] = []
    holding = None
    entry_i, entry_px = lo, 0.0
    for d in range(lo, hi + 1):
        target = bool(want[d - 1])
        if holding is None:
            daily[d] = (1 - cost) * closes[target][d] / opens[target][d] - 1
            holding, entry_i, entry_px = target, d, opens[target][d]
        elif target != holding:
            out_px = opens[holding][d]
            daily[d] = out_px / closes[holding][d - 1] * (1 - cost) ** 2 * closes[target][d] / opens[target][d] - 1
            trades.append(Trade(idx[d - 1], idx[entry_i], idx[d], entry_px, out_px,
                                out_px / entry_px * (1 - cost) ** 2 - 1, "engine" if holding else "safe"))
            holding, entry_i, entry_px = target, d, opens[target][d]
        else:
            daily[d] = closes[holding][d] / closes[holding][d - 1] - 1
    if holding is not None:
        trades.append(Trade(idx[entry_i], idx[entry_i], idx[hi], entry_px, closes[holding][hi],
                            closes[holding][hi] / entry_px * (1 - cost) ** 2 - 1, "end"))
    sl = slice(lo, hi + 1)
    return Result(pd.Series(daily[sl], index=idx[sl]), trades)


def stats(result: Result, bars_per_year: int = 252) -> dict:
    """Summary numbers for a backtest. Trade numbers leave out a position still open at the end."""
    daily = result.daily
    equity = (1 + daily).cumprod()
    total = float(equity.iloc[-1] - 1) if len(equity) else 0.0
    years = len(daily) / bars_per_year
    annual = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    max_dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    rets = np.array([t.ret for t in result.trades if t.reason != "end"])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    if len(losses) and losses.sum() < 0:
        pf = float(wins.sum() / -losses.sum())
    else:
        pf = float("inf") if len(wins) else 0.0
    return {
        "total_return": total,
        "annual_return": float(annual),
        "max_drawdown": max_dd,
        "return_per_drawdown": float(annual / -max_dd) if max_dd < 0 else float("inf"),
        "trades": int(len(rets)),
        "win_rate": float(len(wins) / len(rets)) if len(rets) else 0.0,
        "profit_factor": pf,
        "avg_trade": float(rets.mean()) if len(rets) else 0.0,
    }
