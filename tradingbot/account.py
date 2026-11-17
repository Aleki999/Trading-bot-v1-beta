"""Combine the core and the chosen sleeves into one account.

This works on daily returns and weights, not share counts: each sleeve trade takes a slot of equity
for as long as it is open, and the core gets the rest (or stays at 100% when margin is on). It is an
approximation of the live account: no whole-share rounding and no intraday cash timing.
"""
from __future__ import annotations

import pandas as pd

from .backtester import Result, stats


def pick_sleeves(candidates: dict[str, Result], cfg: dict) -> list[str]:
    """Best return per drawdown first, one per ticker, skipping any too correlated with one already picked.
    Candidate names look like 'QQQ:dip_buy'."""
    acct = cfg["account"]
    ranked = sorted(candidates, key=lambda k: stats(candidates[k], cfg["bars_per_year"])["return_per_drawdown"],
                    reverse=True)
    chosen: list[str] = []
    for name in ranked:
        if len(chosen) == acct["max_sleeves"]:
            break
        if any(name.split(":")[0] == c.split(":")[0] for c in chosen):
            continue
        daily = candidates[name].daily
        if any(daily.corr(candidates[c].daily) > acct["max_correlation"] for c in chosen):
            continue
        chosen.append(name)
    return chosen


def sleeve_weights(result: Result, cfg: dict, vix: pd.Series | None = None) -> pd.Series:
    """Share of equity the sleeve holds each day: a slot per open trade, halved if VIX was high at the signal."""
    acct = cfg["account"]
    sizes = []
    for trade in result.trades:
        size = acct["slot"]
        if vix is not None:
            seen = vix.loc[:trade.signal_date].dropna()
            if len(seen) and seen.iloc[-1] > acct["vix_half_size_above"]:
                size /= 2
        sizes.append(size)
    ids = result.trade_id
    return ids.map(lambda i: sizes[i] if i >= 0 else 0.0).astype(float)


def combine(core: Result, sleeves: dict[str, Result], cfg: dict, vix: pd.Series | None = None) -> dict:
    acct = cfg["account"]
    idx = core.daily.index
    weights = pd.DataFrame({k: sleeve_weights(r, cfg, vix).reindex(idx).fillna(0.0) for k, r in sleeves.items()},
                           index=idx)
    returns = pd.DataFrame({k: r.daily.reindex(idx).fillna(0.0) for k, r in sleeves.items()}, index=idx)
    in_sleeves = weights.sum(axis=1) if len(sleeves) else pd.Series(0.0, index=idx)
    if acct["margin"]:
        core_weight = pd.Series(1.0, index=idx)
        drag = in_sleeves * acct["margin_rate"] / 360
    else:
        core_weight = 1.0 - in_sleeves
        drag = core_weight.diff().abs().fillna(0.0) * cfg["cost_per_side"]   # trading the core to fund sleeves
    daily = core_weight * core.daily + (weights * returns).sum(axis=1) - drag
    return {
        "daily": daily,
        "stats": stats(Result(daily), cfg["bars_per_year"]),
        "core_weight_avg": float(core_weight.mean()),
        "max_exposure": float((core_weight + in_sleeves).max()),
    }
