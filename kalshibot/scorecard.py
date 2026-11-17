"""Score past trades once their markets settle: python -m kalshibot.scorecard

Reads the ledger, looks up the result of every bought market, and prints wins, cost and profit for
paper and live trades separately. Settled results are remembered in settled.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from .bot import ROOT, load_config, read_ledger
from .kalshi import KalshiClient


def settle(trades: list[dict], client: KalshiClient, cache_path: Path) -> dict:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for trade in trades:
        ticker = trade["ticker"]
        if ticker not in cache:
            result = client.get_market(ticker).get("result", "")
            if result in ("yes", "no"):
                cache[ticker] = result
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1))
    return cache


def summarize(trades: list[dict], results: dict) -> dict:
    done = [t for t in trades if t["ticker"] in results]
    wins = [t for t in done if results[t["ticker"]] == "yes"]
    cost = sum(t["cost"] for t in done)
    payout = sum(t["count"] for t in wins)
    return {
        "trades": len(trades), "settled": len(done), "wins": len(wins),
        "cost": round(cost, 2), "payout": round(payout, 2), "profit": round(payout - cost, 2),
        "avg_price": round(sum(t["price"] for t in done) / len(done), 3) if done else None,
        "avg_fair": round(sum(t["fair"] for t in done) / len(done), 3) if done else None,
        "win_rate": round(len(wins) / len(done), 3) if done else None,
    }


def main() -> None:
    cfg = load_config()
    data_dir = ROOT / cfg["run"]["data_dir"]
    trades = [r for r in read_ledger(data_dir / "decisions.jsonl") if r["decision"] == "BUY"]
    results = settle(trades, KalshiClient(cfg["api"]["kalshi_base"]), data_dir / "settled.json")
    for mode in ("paper", "live"):
        s = summarize([t for t in trades if t["mode"] == mode], results)
        print(f"{mode:<6} trades {s['trades']}  settled {s['settled']}  wins {s['wins']}  cost ${s['cost']:.2f}  "
              f"payout ${s['payout']:.2f}  profit ${s['profit']:+.2f}")
        if s["settled"]:
            print(f"       win rate {s['win_rate']:.0%} against an average price of {s['avg_price']:.0%} "
                  f"and an average model chance of {s['avg_fair']:.0%}")


if __name__ == "__main__":
    main()
