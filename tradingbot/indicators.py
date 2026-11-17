"""Indicators. Every one uses only bars up to and including the current bar."""
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 2) -> pd.Series:
    """Wilder's RSI. 100 when there were no down moves, 50 when there were no moves at all."""
    change = close.diff()
    up = change.clip(lower=0.0)
    down = -change.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    out = 100.0 - 100.0 / (1.0 + avg_up / avg_down)
    out = out.mask((avg_down == 0) & (avg_up > 0), 100.0)
    out = out.mask((avg_down == 0) & (avg_up == 0), 50.0)
    return out


def highest_close(close: pd.Series, n: int) -> pd.Series:
    """Highest close of the last n bars, counting the current bar."""
    return close.rolling(n, min_periods=n).max()


def lowest_close(close: pd.Series, n: int) -> pd.Series:
    """Lowest close of the last n bars, counting the current bar."""
    return close.rolling(n, min_periods=n).min()
