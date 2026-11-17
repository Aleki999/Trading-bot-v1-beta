"""Monte Carlo permutation test: is a result better than the same rule gets on shuffled prices?

Bars are shuffled so each fake series keeps the real one's drift, volatility and overall return but
loses any pattern in the order of its moves. A rule with a real timing edge should beat nearly all of
them. The shuffle breaks volatility clustering, so the test flatters rules that depend on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ["open", "high", "low", "close"]


def permute_bars(frames: list[pd.DataFrame], start_index: int = 0, rng: np.random.Generator | None = None
                 ) -> list[pd.DataFrame]:
    """Shuffle bars after `start_index`. Several tickers passed together get the same shuffle,
    which keeps how they move relative to each other. Frames must share one index."""
    rng = rng or np.random.default_rng()
    n = len(frames[0])
    n_perm = n - start_index - 1
    order_intrabar = rng.permutation(n_perm)
    order_gap = rng.permutation(n_perm)
    out = []
    for bars in frames:
        log = np.log(bars[COLS].to_numpy(float))
        first = start_index + 1
        gap = log[first:, 0] - log[first - 1:-1, 3]            # open versus the previous close
        intrabar = log[first:, 1:] - log[first:, [0]]          # high, low, close versus the bar's open
        gap, intrabar = gap[order_gap], intrabar[order_intrabar]
        new = log.copy()
        closes = log[first - 1, 3] + np.cumsum(gap + intrabar[:, 2])
        prev_closes = np.concatenate(([log[first - 1, 3]], closes[:-1]))
        new[first:, 0] = prev_closes + gap
        new[first:, 1:] = new[first:, [0]] + intrabar
        frame = bars.copy()
        frame[COLS] = np.exp(new)
        out.append(frame)
    return out


def permutation_test(frames: list[pd.DataFrame], statistic, n: int = 1000, seed: int = 11,
                     start_index: int = 0) -> dict:
    """`statistic(frames) -> float`, larger is better. Returns the real value and its quasi p-value."""
    rng = np.random.default_rng(seed)
    real = float(statistic(frames))
    fakes = np.array([float(statistic(permute_bars(frames, start_index, rng))) for _ in range(n)])
    beat = int((fakes >= real).sum())
    return {
        "real": real,
        "p_value": (beat + 1) / (n + 1),
        "n": n,
        "fake_median": float(np.median(fakes)),
        "fake_95th": float(np.quantile(fakes, 0.95)),
    }
