"""Label how each ticker moves, using training data only.

- snaps_back: after a sharp dip inside an uptrend, the next few days beat the ticker's usual few days.
- keeps_going: after a new high inside an uptrend, the next few weeks beat the ticker's usual few weeks.
A sleeve family is only tried on tickers that carry its label.
"""
from __future__ import annotations

import pandas as pd

from .signals import breakout, dip_buy


def _edge(bars: pd.DataFrame, events: pd.Series, forward_days: int, start, end) -> tuple[int, float]:
    close = bars["close"]
    forward = close.shift(-forward_days) / close - 1
    in_window = (bars.index >= pd.Timestamp(start)) & (bars.index <= pd.Timestamp(end))
    forward = forward[in_window].dropna()          # only days whose whole forward span is in the window
    hits = events.reindex(forward.index).fillna(False).to_numpy(bool)
    if not hits.any():
        return 0, 0.0
    return int(hits.sum()), float(forward[hits].mean() - forward.mean())


def label_ticker(bars: pd.DataFrame, cfg: dict, start, end) -> dict:
    fam = cfg["sleeve_families"]
    rules = cfg["labels"]
    dip_events = dip_buy(bars, **fam["dip_buy"]["signal"])["entry"]
    high_events = breakout(bars, **fam["breakout"]["signal"])["entry"]
    n_dip, dip_edge = _edge(bars, dip_events, rules["snaps_back"]["forward_days"], start, end)
    n_high, high_edge = _edge(bars, high_events, rules["keeps_going"]["forward_days"], start, end)
    return {
        "snaps_back": bool(n_dip >= rules["snaps_back"]["min_events"] and dip_edge > 0),
        "snaps_back_events": n_dip,
        "snaps_back_edge": dip_edge,
        "keeps_going": bool(n_high >= rules["keeps_going"]["min_events"] and high_edge > 0),
        "keeps_going_events": n_high,
        "keeps_going_edge": high_edge,
    }
