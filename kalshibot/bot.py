"""Run the Kalshi weather bot: python -m kalshibot.bot [--loop] [--live]

Each run: get the forecast high, get the bucket prices, work out each bucket's fair chance and edge,
and buy the best bucket if its edge is large enough. Without --live nothing is sent to Kalshi; the
trade is only written to the ledger as a paper trade.

Live trading needs all three: `dry_run: false` in config/kalshi.yaml, the --live flag, and the
environment variables KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from .kalshi import KalshiClient, load_private_key
from .model import decide, evaluate, parse_bucket, size_order, taker_fee
from .weather import Weather

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "kalshi.yaml"
log = logging.getLogger("kalshibot")


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def event_ticker(series: str, day: date) -> str:
    return f"{series}-{day.strftime('%y%b%d').upper()}"


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def trades_on(ledger: list[dict], run_day: str, mode: str) -> int:
    return sum(1 for r in ledger if r["decision"] == "BUY" and r["run_day"] == run_day and r["mode"] == mode)


def table(rows) -> str:
    lines = [f"{'bucket':<16}{'fair':>8}{'ask':>8}{'edge':>9}"]
    for r in rows:
        ask = "   --" if r.bucket.ask is None else f"{r.bucket.ask * 100:5.1f}c"
        edge = "     --" if r.edge_points is None else f"{r.edge_points:+7.1f}"
        lines.append(f"{r.bucket.label:<16}{r.fair * 100:7.1f}%{ask:>8}{edge:>9}")
    return "\n".join(lines)


def run_once(cfg: dict, weather: Weather, client: KalshiClient, live: bool, now: datetime | None = None) -> dict:
    """One pass. Returns the record written to the ledger."""
    station, limits, model = cfg["station"], cfg["limits"], cfg["model"]
    zone = ZoneInfo(weather.point(station["lat"], station["lon"])["time_zone"])
    now = now or datetime.now(zone)
    run_day = now.astimezone(zone).date()
    target = run_day + timedelta(days=1 if cfg["target_day"] == "tomorrow" else 0)
    mode = "live" if live else "paper"
    ledger_path = ROOT / cfg["run"]["data_dir"] / "decisions.jsonl"
    record = {"time": now.isoformat(timespec="seconds"), "run_day": str(run_day), "target_day": str(target),
              "mode": mode, "decision": "PASS", "reason": ""}

    def finish(reason: str = "") -> dict:
        record["reason"] = reason
        log.info("%s %s for %s%s", record["decision"], mode, target, f": {reason}" if reason else "")
        append_ledger(ledger_path, record)
        return record

    forecast = weather.forecast_high(station["lat"], station["lon"], target)
    if forecast is None:
        return finish("no daytime forecast for the target day")
    record["forecast_high"] = forecast["high"]
    event = event_ticker(cfg["series_ticker"], target)
    markets = [m for m in client.get_markets(event) if m.get("status") == "active"]
    if not markets:
        return finish(f"no open markets for {event}")
    rows = evaluate([parse_bucket(m) for m in markets], forecast["high"], model["spread_f"])
    record["rows"] = [{"ticker": r.bucket.ticker, "label": r.bucket.label, "fair": round(r.fair, 4),
                       "ask": r.bucket.ask, "edge_points": None if r.edge_points is None else round(r.edge_points, 2)}
                      for r in rows]
    log.info("forecast high %.0fF (%s), spread %.1fF, event %s\n%s", forecast["high"], forecast["period"],
             model["spread_f"], event, table(rows))

    pick = decide(rows, model["min_edge_points"])
    if pick is None:
        return finish(f"no bucket has an edge of {model['min_edge_points']:.0f} points")
    if trades_on(read_ledger(ledger_path), str(run_day), mode) >= limits["max_trades_per_day"]:
        return finish("daily trade limit reached")
    count = size_order(pick.bucket.ask, limits["max_spend"], limits["fee_rate"])
    if count < 1:
        return finish("one contract costs more than the spend cap")
    fee = taker_fee(count, pick.bucket.ask, limits["fee_rate"])
    record.update(decision="BUY", ticker=pick.bucket.ticker, label=pick.bucket.label, count=count,
                  price=pick.bucket.ask, fee=fee, cost=round(count * pick.bucket.ask + fee, 2),
                  fair=round(pick.fair, 4), edge_points=round(pick.edge_points, 2))
    if live:
        order_id = f"wx-{target:%Y%m%d}-{now:%H%M%S}"
        reply = client.buy_yes(pick.bucket.ticker, count, pick.bucket.ask, order_id)
        filled = float(reply.get("fill_count", 0))
        record.update(order_id=reply.get("order_id"), client_order_id=order_id, filled=filled)
        if filled < count:      # the rest was cancelled; record what was really bought
            record.update(count=filled, fee=taker_fee(int(filled), pick.bucket.ask, limits["fee_rate"]) if filled else 0.0)
            record["cost"] = round(filled * pick.bucket.ask + record["fee"], 2)
        if not filled:
            record["decision"] = "PASS"
            return finish("order sent but nothing filled at the ask")
    return finish(f"{record['count']} x {pick.bucket.label} at {pick.bucket.ask * 100:.0f}c, "
                  f"edge {pick.edge_points:+.1f}, cost ${record['cost']:.2f}")


def build_client(cfg: dict, live: bool) -> KalshiClient:
    key_id, key_path = os.environ.get("KALSHI_API_KEY_ID"), os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    private_key = None
    if live:
        if cfg["run"]["dry_run"]:
            raise SystemExit("--live was given but dry_run is true in config/kalshi.yaml")
        if not (key_id and key_path):
            raise SystemExit("live trading needs KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
        private_key = load_private_key(Path(key_path).read_bytes())
    return KalshiClient(cfg["api"]["kalshi_base"], key_id if live else None, private_key)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="place real orders (see the requirements above)")
    parser.add_argument("--loop", action="store_true", help="keep running every poll_minutes")
    args = parser.parse_args(argv)
    cfg = load_config()
    data_dir = ROOT / cfg["run"]["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(data_dir / "bot.log")])
    weather = Weather(cfg["api"]["weather_base"], cfg["api"]["user_agent"])
    client = build_client(cfg, args.live)
    if args.live:
        log.info("LIVE mode. Balance: $%.2f", client.get_balance())
    while True:
        try:
            run_once(cfg, weather, client, args.live)
        except Exception as exc:        # one bad run must not stop the loop; it is logged and retried next time
            log.error("run failed: %s", exc)
            if not args.loop:
                raise
        if not args.loop:
            break
        time.sleep(cfg["run"]["poll_minutes"] * 60)


if __name__ == "__main__":
    main()
