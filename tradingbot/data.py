"""Price data: split- and dividend-adjusted daily bars from Yahoo Finance, and VIX from FRED."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR

PRICES = DATA_DIR / "prices.parquet"
VIX = DATA_DIR / "vix.parquet"
FRED_VIX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


def download_prices(tickers: list[str], start: str, path: Path = PRICES) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    for ticker in tickers:
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
        if raw is None or raw.empty:
            raise RuntimeError(f"no data returned for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frame = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
        frame.insert(0, "ticker", ticker)
        frame.index.name = "date"
        rows.append(frame.reset_index())
    prices = pd.concat(rows, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path, index=False)
    return prices


def load_prices(path: Path = PRICES) -> dict[str, pd.DataFrame]:
    prices = pd.read_parquet(path)
    prices["date"] = pd.to_datetime(prices["date"])
    return {ticker: frame.drop(columns="ticker").set_index("date").sort_index()
            for ticker, frame in prices.groupby("ticker")}


def download_vix(path: Path = VIX) -> pd.Series:
    raw = pd.read_csv(FRED_VIX_URL)
    raw.columns = ["date", "vix"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["vix"] = pd.to_numeric(raw["vix"], errors="coerce")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(path, index=False)
    return raw.set_index("date")["vix"]


def load_vix(path: Path = VIX) -> pd.Series | None:
    if not path.exists():
        return None
    raw = pd.read_parquet(path)
    return raw.set_index("date")["vix"].sort_index()
