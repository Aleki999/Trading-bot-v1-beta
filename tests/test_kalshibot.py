import base64
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from kalshibot import bot
from kalshibot.kalshi import KalshiClient, sign
from kalshibot.model import (Bucket, bucket_probability, check_buckets, decide, evaluate, parse_bucket, size_order,
                             taker_fee)
from kalshibot.scorecard import summarize
from kalshibot.weather import pick_high

FIXTURES = Path(__file__).parent / "fixtures"
MARKETS = json.loads((FIXTURES / "kalshi_markets.json").read_text())["markets"]
FORECAST = json.loads((FIXTURES / "nws_forecast.json").read_text())


def test_probabilities_match_the_worked_example():
    # Forecast 96F with a 3F spread: the source video shows 25.8% for 96-97 and 1.5% for 89 or below.
    assert bucket_probability(96, 3, 96, 97) == pytest.approx(0.258, abs=0.001)
    assert bucket_probability(96, 3, None, 89) == pytest.approx(0.015, abs=0.001)


def test_live_sample_parses_and_covers_every_temperature():
    buckets = [parse_bucket(m) for m in MARKETS]
    check_buckets(buckets)
    by_label = {b.label: b for b in buckets}
    assert (by_label["70° or above"].low, by_label["70° or above"].high) == (70, None)
    assert (by_label["61° or below"].low, by_label["61° or below"].high) == (None, 61)
    assert (by_label["68° to 69°"].low, by_label["68° to 69°"].high) == (68, 69)
    rows = evaluate(buckets, 65, 3)
    assert sum(r.fair for r in rows) == pytest.approx(1.0)


def test_buckets_with_a_gap_are_refused():
    buckets = [Bucket("a", "", None, 61, 0.1), Bucket("b", "", 64, None, 0.1)]
    with pytest.raises(ValueError):
        check_buckets(buckets)


def test_no_offer_means_no_ask():
    market = dict(MARKETS[0], yes_ask_size_fp="0.00")
    assert parse_bucket(market).ask is None


def test_fee_and_sizing_match_the_worked_example():
    # 7 contracts at 13c: $0.91 plus a $0.06 fee is $0.97; an eighth contract would pass $1.
    assert taker_fee(7, 0.13) == 0.06
    assert size_order(0.13, 1.00) == 7
    assert size_order(0.99, 0.50) == 0


def test_decide_needs_the_minimum_edge_and_takes_the_largest():
    buckets = [Bucket("lo", "", None, 63, 0.05), Bucket("mid", "", 64, 65, 0.09), Bucket("hi", "", 66, None, 0.60)]
    rows = evaluate(buckets, 65, 3)
    pick = decide(rows, 10)
    assert pick.bucket.ticker == "lo" and pick.edge_points > 10
    assert decide(rows, 99) is None


def test_forecast_picks_the_daytime_period_for_the_day():
    assert pick_high(FORECAST, date(2026, 11, 17))["high"] == 65
    assert pick_high(FORECAST, date(2026, 11, 18))["high"] == 69
    assert pick_high(FORECAST, date(2026, 11, 16)) is None       # only "Tonight" is left for that day


def test_signatures_verify_for_both_key_types():
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sig = base64.b64decode(sign(rsa_key, "1700000000000", "get", "/trade-api/v2/portfolio/balance?x=1"))
    rsa_key.public_key().verify(sig, b"1700000000000GET/trade-api/v2/portfolio/balance",
                                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                                hashes.SHA256())
    ed_key = ed25519.Ed25519PrivateKey.generate()
    sig = base64.b64decode(sign(ed_key, "1", "POST", "/trade-api/v2/portfolio/events/orders"))
    ed_key.public_key().verify(sig, b"1POST/trade-api/v2/portfolio/events/orders")


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text, self.headers = payload, status, json.dumps(payload), {}

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return FakeResponse(self.reply)


def test_order_request_shape():
    session = FakeSession({"order_id": "abc", "fill_count": "7.00", "remaining_count": "0.00"})
    client = KalshiClient("https://example.test/trade-api/v2", "key-id", ed25519.Ed25519PrivateKey.generate(),
                          session, clock=lambda: 1700000000.0)
    client.buy_yes("KXHIGHAUS-26NOV17-B64.5", 7, 0.13, "wx-1")
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "https://example.test/trade-api/v2/portfolio/events/orders")
    assert kwargs["json"] == {"ticker": "KXHIGHAUS-26NOV17-B64.5", "client_order_id": "wx-1", "side": "bid",
                              "count": "7.00", "price": "0.1300", "time_in_force": "immediate_or_cancel",
                              "self_trade_prevention_type": "taker_at_cross"}
    assert kwargs["headers"]["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"


class FakeWeather:
    def point(self, lat, lon):
        return {"time_zone": "America/Chicago"}

    def forecast_high(self, lat, lon, day):
        return pick_high(FORECAST, day)


class FakeKalshi:
    def __init__(self):
        self.orders = []

    def get_markets(self, event):
        return MARKETS if event == "KXHIGHAUS-26NOV17" else []

    def buy_yes(self, ticker, count, price, client_order_id):
        self.orders.append((ticker, count, price))
        return {"order_id": "o1", "fill_count": f"{count:.2f}"}


def test_paper_run_logs_a_trade_sends_nothing_and_respects_the_daily_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "ROOT", tmp_path)
    cfg = bot.load_config()
    client = FakeKalshi()
    now = datetime(2026, 11, 16, 21, 0, tzinfo=ZoneInfo("America/Chicago"))
    first = bot.run_once(cfg, FakeWeather(), client, live=False, now=now)
    assert first["decision"] == "BUY" and first["label"] == "64° to 65°"     # forecast 65F against a 9c ask
    assert first["cost"] <= cfg["limits"]["max_spend"] and client.orders == []
    second = bot.run_once(cfg, FakeWeather(), client, live=False, now=now)
    assert second["decision"] == "PASS" and second["reason"] == "daily trade limit reached"
    assert len(bot.read_ledger(tmp_path / cfg["run"]["data_dir"] / "decisions.jsonl")) == 2


def test_live_run_sends_one_order(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "ROOT", tmp_path)
    client = FakeKalshi()
    now = datetime(2026, 11, 16, 21, 0, tzinfo=ZoneInfo("America/Chicago"))
    record = bot.run_once(bot.load_config(), FakeWeather(), client, live=True, now=now)
    assert record["decision"] == "BUY" and len(client.orders) == 1
    ticker, count, price = client.orders[0]
    assert count * price + taker_fee(count, price) <= 1.00


def test_scorecard_profit():
    trades = [{"ticker": "a", "count": 7, "cost": 0.97, "price": 0.13, "fair": 0.23},
              {"ticker": "b", "count": 5, "cost": 0.80, "price": 0.15, "fair": 0.26},
              {"ticker": "c", "count": 5, "cost": 0.80, "price": 0.15, "fair": 0.26}]
    s = summarize(trades, {"a": "yes", "b": "no"})
    assert (s["settled"], s["wins"]) == (2, 1)
    assert s["profit"] == pytest.approx(7 - 0.97 - 0.80)
