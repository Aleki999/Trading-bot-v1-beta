import numpy as np
import pandas as pd
import pytest

from tradingbot.account import combine, pick_sleeves, sleeve_weights
from tradingbot.backtester import LookAheadError, Result, lookahead_guard, run_long_flat, run_switch, stats
from tradingbot.checks import cheating_signal, fake_bars, run_checks
from tradingbot.config import load_config
from tradingbot.gates import locked_test
from tradingbot.indicators import rsi, sma
from tradingbot.permutation import permutation_test, permute_bars
from tradingbot.signals import breakout, dip_buy, trend_switch


def bars_from(opens, closes, lows=None):
    idx = pd.bdate_range("2024-01-01", periods=len(opens))
    opens, closes = np.array(opens, float), np.array(closes, float)
    lows = np.minimum(opens, closes) if lows is None else np.array(lows, float)
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes), "low": lows, "close": closes}, index=idx)


def sig(entry, exit_, index):
    return pd.DataFrame({"entry": entry, "exit": exit_}, index=index)


def test_fills_at_next_open_with_costs():
    bars = bars_from([100, 100, 110, 120, 120], [100, 105, 115, 120, 120])
    signal = sig([True, False, False, False, False], [False, False, True, False, False], bars.index)
    res = run_long_flat(bars, signal, cost=0.001)
    assert len(res.trades) == 1
    trade = res.trades[0]
    assert (trade.entry_price, trade.exit_price, trade.reason) == (100, 120, "signal")
    assert trade.ret == pytest.approx(1.2 * 0.999 ** 2 - 1)
    assert (1 + res.daily).prod() - 1 == pytest.approx(trade.ret)   # daily returns compound to the trade


def test_stop_fills_at_stop_price_or_gap_open():
    bars = bars_from([100, 100, 100, 100], [100, 100, 100, 100], lows=[100, 100, 91, 100])
    signal = sig([True, False, False, False], [False] * 4, bars.index)
    res = run_long_flat(bars, signal, cost=0.0, stop_pct=0.08)
    assert res.trades[0].reason == "stop" and res.trades[0].exit_price == pytest.approx(92)

    gap = bars_from([100, 100, 85, 100], [100, 100, 86, 100], lows=[100, 100, 84, 100])
    res = run_long_flat(gap, signal, cost=0.0, stop_pct=0.08)
    assert res.trades[0].reason == "stop_gap" and res.trades[0].exit_price == 85


def test_time_exit_sells_after_the_nth_close():
    bars = bars_from([100] * 8, [100] * 8)
    signal = sig([True] + [False] * 7, [False] * 8, bars.index)
    res = run_long_flat(bars, signal, cost=0.0, max_hold=3)
    trade = res.trades[0]
    assert trade.reason == "time"
    assert trade.entry_date == bars.index[1] and trade.exit_date == bars.index[4]   # closes on days 1, 2, 3


def test_switch_pays_both_sides_and_uses_next_open():
    engine = bars_from([100, 100, 100, 100], [100, 100, 100, 100])
    safe = bars_from([50, 50, 50, 50], [50, 50, 50, 55])
    want = pd.Series([True, True, False, False], index=engine.index)
    res = run_switch(engine, safe, want, cost=0.001)
    total = (1 + res.daily).prod()
    assert total == pytest.approx(0.999 * 0.999 ** 2 * 1.1)   # enter, then sell + buy, then safe gains 10%


def test_guard_passes_real_signals_and_stops_a_cheat():
    bars, _ = fake_bars(3, 600)
    for fn in (dip_buy, breakout, trend_switch):
        lookahead_guard(fn, bars)
    with pytest.raises(LookAheadError):
        lookahead_guard(cheating_signal, bars)


def test_backtester_checks_pass():
    out = run_checks(0.0005)
    assert out["planted_edge"]["pass"], out["planted_edge"]
    assert out["pure_noise"]["pass"], out["pure_noise"]
    assert out["cheating_signal"]["pass"]


def test_indicators():
    close = pd.Series([1.0, 2, 3, 4, 5])
    assert sma(close, 2).iloc[-1] == 4.5
    assert rsi(close, 2).iloc[-1] == 100
    assert rsi(pd.Series([5.0, 4, 3, 2, 1]), 2).iloc[-1] == 0


def test_permutation_keeps_endpoints_and_changes_the_path():
    bars, _ = fake_bars(5, 400)
    fake = permute_bars([bars], rng=np.random.default_rng(1))[0]
    assert fake["close"].iloc[-1] == pytest.approx(bars["close"].iloc[-1])
    assert fake["open"].iloc[0] == pytest.approx(bars["open"].iloc[0])
    assert not np.allclose(fake["close"].iloc[1:-1], bars["close"].iloc[1:-1])
    assert (fake["high"] >= fake[["open", "close"]].max(axis=1) - 1e-9).all()


def trending_bars(seed, n=600, persist=0.8):
    """Fake bars where an up day is usually followed by another up day, and the same for down days."""
    rng = np.random.default_rng(seed)
    size = np.abs(rng.normal(0, 0.01, n))
    keep = rng.random(n) < persist
    sign = np.ones(n)
    for i in range(1, n):
        sign[i] = sign[i - 1] if keep[i] else -sign[i - 1]
    closes = 100 * np.cumprod(1 + size * sign)
    opens = np.concatenate(([100.0], closes[:-1]))
    return bars_from(opens, closes)


def test_permutation_test_flags_a_real_edge_and_not_noise():
    def momentum_stat(frames):
        b = frames[0]
        up = (b["close"] > b["open"]).to_numpy()
        return stats(run_long_flat(b, sig(up, ~up, b.index), cost=0.0))["total_return"]

    assert permutation_test([trending_bars(2)], momentum_stat, n=100, seed=1)["p_value"] < 0.05
    # On noise the p-value is spread evenly between 0 and 1, so judge several series, not one.
    noise_p = [permutation_test([fake_bars(s, 600)[0]], momentum_stat, n=100, seed=1)["p_value"] for s in range(20, 30)]
    assert np.median(noise_p) > 0.2


def test_locked_test_runs_once(tmp_path):
    calls = []

    def run():
        calls.append(1)
        return {"total_return": 0.1}

    ledger = tmp_path / "ledger.json"
    first = locked_test("k", run, ledger)
    second = locked_test("k", run, ledger)
    assert len(calls) == 1 and not first["rerun"] and second["rerun"]
    assert second["total_return"] == 0.1


def test_account_weights_and_selection():
    cfg = load_config()
    bars = bars_from([100, 100, 100, 110, 110, 110], [100, 100, 110, 110, 110, 110])
    signal = sig([False, True, False, False, False, False], [False, False, True, False, False, False], bars.index)
    sleeve = run_long_flat(bars, signal, cost=0.0)
    weights = sleeve_weights(sleeve, cfg)
    assert weights.tolist() == [0.0, 0.2, 0.2, 0.0, 0.0]        # a 20% slot while the trade is open
    high_vix = pd.Series(30.0, index=bars.index)
    assert sleeve_weights(sleeve, cfg, high_vix).max() == pytest.approx(0.1)

    core = Result(pd.Series(0.0, index=sleeve.daily.index))
    acct = combine(core, {"X:dip_buy": sleeve}, cfg)
    assert acct["max_exposure"] == pytest.approx(1.0)            # no margin: sleeves are funded out of the core
    assert (1 + acct["daily"]).prod() - 1 == pytest.approx(0.2 * 0.1, abs=1e-3)

    same = {"A:dip_buy": sleeve, "B:dip_buy": sleeve}
    assert len(pick_sleeves(same, cfg)) == 1                     # identical returns are too correlated to hold both
