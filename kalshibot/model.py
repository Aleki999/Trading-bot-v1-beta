"""The pricing model and the trade rule.

Tomorrow's high is treated as a bell curve centred on the forecast. Each Kalshi bucket covers whole
degrees, so a bucket of 96 to 97 is the chance the high lands between 95.5 and 97.5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Bucket:
    ticker: str
    label: str
    low: int | None        # lowest whole degree in the bucket, None = open below
    high: int | None       # highest whole degree in the bucket, None = open above
    ask: float | None      # dollars to buy one YES contract, None = nothing offered
    bid: float | None = None


@dataclass(frozen=True)
class Row:
    bucket: Bucket
    fair: float            # model probability, 0 to 1
    edge_points: float | None   # (fair - ask) x 100, None when there is no ask


def _cdf(x: float, mean: float, spread: float) -> float:
    return 0.5 * (1 + math.erf((x - mean) / (spread * math.sqrt(2))))


def bucket_probability(mean: float, spread: float, low: int | None, high: int | None) -> float:
    upper = 1.0 if high is None else _cdf(high + 0.5, mean, spread)
    lower = 0.0 if low is None else _cdf(low - 0.5, mean, spread)
    return upper - lower


def parse_bucket(market: dict) -> Bucket:
    """Turn one Kalshi market into a bucket. `greater 69` means 70 or above; `less 62` means 61 or below."""
    kind = market["strike_type"]
    if kind == "between":
        low, high = int(market["floor_strike"]), int(market["cap_strike"])
    elif kind == "greater":
        low, high = int(market["floor_strike"]) + 1, None
    elif kind == "less":
        low, high = None, int(market["cap_strike"]) - 1
    else:
        raise ValueError(f"unknown strike type {kind!r} on {market['ticker']}")

    def price(name: str) -> float | None:
        raw = market.get(name)
        return float(raw) if raw not in (None, "") else None

    ask = price("yes_ask_dollars")
    offered = float(market.get("yes_ask_size_fp") or 0) > 0
    if ask is None or not offered or not 0 < ask < 1:
        ask = None
    return Bucket(market["ticker"], market.get("yes_sub_title", ""), low, high, ask, price("yes_bid_dollars"))


def check_buckets(buckets: list[Bucket]) -> None:
    """The buckets of one event must cover every temperature exactly once."""
    ordered = sorted(buckets, key=lambda b: -math.inf if b.low is None else b.low)
    if not ordered or ordered[0].low is not None or ordered[-1].high is not None:
        raise ValueError("buckets do not cover both tails")
    for left, right in zip(ordered, ordered[1:]):
        if left.high is None or right.low != left.high + 1:
            raise ValueError(f"gap or overlap between {left.ticker} and {right.ticker}")


def evaluate(buckets: list[Bucket], forecast_high: float, spread: float) -> list[Row]:
    check_buckets(buckets)
    rows = []
    for b in sorted(buckets, key=lambda b: -math.inf if b.low is None else b.low):
        fair = bucket_probability(forecast_high, spread, b.low, b.high)
        rows.append(Row(b, fair, None if b.ask is None else (fair - b.ask) * 100))
    return rows


def decide(rows: list[Row], min_edge_points: float) -> Row | None:
    """Buy the bucket with the largest edge, if that edge reaches the minimum. Otherwise pass."""
    priced = [r for r in rows if r.edge_points is not None]
    if not priced:
        return None
    best = max(priced, key=lambda r: r.edge_points)
    return best if best.edge_points >= min_edge_points else None


def taker_fee(count: int, price: float, rate: float = 0.07) -> float:
    """Kalshi's taker fee for one fill, rounded up to the cent."""
    return math.ceil(round(rate * count * price * (1 - price) * 100, 6)) / 100


def size_order(price: float, max_spend: float, rate: float = 0.07) -> int:
    """Largest whole number of contracts whose cost plus fee fits in the spend cap."""
    count = int(max_spend // price)
    while count > 0 and round(count * price + taker_fee(count, price, rate), 6) > max_spend:
        count -= 1
    return count
