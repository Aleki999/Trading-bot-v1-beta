"""The gates a sleeve candidate must pass, in order, and the one-time locked test."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtester import Result, lookahead_guard, run_long_flat, stats
from .config import RESULTS_DIR, window
from .signals import FAMILIES

LEDGER = RESULTS_DIR / "locked_ledger.json"


def run_sleeve(bars: pd.DataFrame, family: str, cfg: dict, start, end, overrides: dict | None = None) -> Result:
    """Backtest one sleeve family on one ticker. `overrides` changes signal or engine settings by name."""
    fam = cfg["sleeve_families"][family]
    signal_kw, engine_kw = dict(fam["signal"]), dict(fam["engine"])
    for name, value in (overrides or {}).items():
        (signal_kw if name in signal_kw else engine_kw)[name] = value
    signal = FAMILIES[family](bars, **signal_kw)
    return run_long_flat(bars, signal, cost=cfg["cost_per_side"], start=start, end=end, **engine_kw)


def _ledger_read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def locked_test(key: str, run, ledger: Path = LEDGER) -> dict:
    """Run a locked test once. A second call for the same key returns the stored result unchanged."""
    book = _ledger_read(ledger)
    if key in book:
        return {**book[key], "rerun": True}
    result = run()
    book[key] = result
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(book, indent=2, default=str))
    return {**result, "rerun": False}


def evaluate_sleeve(ticker: str, family: str, bars: pd.DataFrame, label: dict, cfg: dict,
                    run_locked: bool = False, ledger: Path = LEDGER) -> dict:
    """Walk one candidate through the gates. It stops at the first gate it fails."""
    fam = cfg["sleeve_families"][family]
    train_start, train_end = window(cfg, "train")
    out = {"ticker": ticker, "family": family, "passed": [], "failed": None}

    def gate(name: str, ok: bool) -> bool:
        if ok:
            out["passed"].append(name)
        else:
            out["failed"] = name
        return ok

    lookahead_guard(lambda b: FAMILIES[family](b, **fam["signal"]), bars)

    if not gate("fits", bool(label.get(fam["label"]))):
        return out
    train = stats(run_sleeve(bars, family, cfg, train_start, train_end), cfg["bars_per_year"])
    out["train"] = train
    if not gate("profitable", train["total_return"] > 0):
        return out
    if not gate("enough_trades", train["trades"] >= fam["min_trades"]):
        return out
    if not gate("profit_factor", train["profit_factor"] >= cfg["gates"]["min_profit_factor"]):
        return out
    neighbors = {}
    for name, values in fam["neighbors"].items():
        for value in values:
            res = stats(run_sleeve(bars, family, cfg, train_start, train_end, {name: value}), cfg["bars_per_year"])
            neighbors[f"{name}={value}"] = res["total_return"]
    out["neighbors"] = neighbors
    if not gate("neighbors", all(v > 0 for v in neighbors.values())):
        return out
    if not run_locked:
        out["failed"] = "locked_test_not_run"
        return out
    locked_start, locked_end = window(cfg, "locked")
    locked = locked_test(f"sleeve:{ticker}:{family}",
                         lambda: stats(run_sleeve(bars, family, cfg, locked_start, locked_end), cfg["bars_per_year"]),
                         ledger)
    out["locked"] = locked
    gate("locked_test", locked["total_return"] > 0)
    return out
