"""Signal rules. Each is decided at a bar's close and traded at the next open.

Sleeve signals return a DataFrame with boolean columns `entry` and `exit`.
The trend switch returns a boolean Series: True = hold the engine, False = hold the safe asset.
"""
from __future__ import annotations

import pandas as pd

from .indicators import ema, highest_close, lowest_close, rsi, sma


def trend_switch(bars: pd.DataFrame, line_days: int = 200, kind: str = "sma") -> pd.Series:
    close = bars["close"]
    line = ema(close, line_days) if kind == "ema" else sma(close, line_days)
    return (close > line).fillna(False)


def dip_buy(bars: pd.DataFrame, rsi_len: int = 2, rsi_below: float = 10, trend_len: int = 200,
            exit_sma: int = 5) -> pd.DataFrame:
    """Buy a sharp pullback inside an uptrend; leave once price is back above its short average."""
    close = bars["close"]
    entry = (rsi(close, rsi_len) < rsi_below) & (close > sma(close, trend_len))
    exit_ = close > sma(close, exit_sma)
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


def breakout(bars: pd.DataFrame, high_len: int = 50, low_len: int = 20, trend_len: int = 200) -> pd.DataFrame:
    """Buy a new high close inside an uptrend; leave on a new low close or a close below the trend line."""
    close = bars["close"]
    trend = sma(close, trend_len)
    entry = (close >= highest_close(close, high_len)) & (close > trend)
    exit_ = (close <= lowest_close(close, low_len)) | (close < trend)
    return pd.DataFrame({"entry": entry.fillna(False), "exit": exit_.fillna(False)})


FAMILIES = {"dip_buy": dip_buy, "breakout": breakout}
