"""Small Kalshi API client: public market data, plus signed balance and order calls.

Follows docs.kalshi.com as read on 2026-11-17: prices are dollar strings, counts are fixed-point
strings, and orders go to POST /portfolio/events/orders (the older /portfolio/orders is gone).
"""
from __future__ import annotations

import base64
import time
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding


def load_private_key(pem: bytes):
    return serialization.load_pem_private_key(pem, password=None)


def sign(private_key, timestamp_ms: str, method: str, path: str) -> str:
    """Signature over timestamp + METHOD + path. The path has no query string."""
    message = f"{timestamp_ms}{method.upper()}{path.split('?')[0]}".encode()
    if isinstance(private_key, ed25519.Ed25519PrivateKey):
        raw = private_key.sign(message)
    else:
        raw = private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
    return base64.b64encode(raw).decode()


class KalshiClient:
    def __init__(self, base: str, key_id: str | None = None, private_key=None,
                 session: requests.Session | None = None, clock=time.time):
        self.base = base.rstrip("/")
        self.prefix = urlparse(self.base).path          # e.g. /trade-api/v2, part of what gets signed
        self.key_id, self.private_key = key_id, private_key
        self.session = session or requests.Session()
        self.clock = clock

    @property
    def can_trade(self) -> bool:
        return bool(self.key_id and self.private_key)

    def _auth(self, method: str, path: str) -> dict:
        if not self.can_trade:
            raise RuntimeError("this call needs a Kalshi API key id and private key")
        stamp = str(int(self.clock() * 1000))
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": stamp,
                "KALSHI-ACCESS-SIGNATURE": sign(self.private_key, stamp, method, self.prefix + path)}

    def _request(self, method: str, path: str, signed: bool = False, **kwargs) -> dict:
        for attempt in (1, 2):
            headers = self._auth(method, path) if signed else {}
            resp = self.session.request(method, self.base + path, headers=headers, timeout=30, **kwargs)
            if resp.status_code == 429 and attempt == 1 and method == "GET":   # rate limited: wait once
                time.sleep(float(resp.headers.get("Retry-After", 5)))
                continue
            break
        if resp.status_code >= 400:
            raise RuntimeError(f"Kalshi {method} {path} failed: HTTP {resp.status_code} {resp.text[:300]}")
        return resp.json()

    def get_markets(self, event_ticker: str) -> list[dict]:
        return self._request("GET", "/markets", params={"event_ticker": event_ticker, "limit": 100})["markets"]

    def get_market(self, ticker: str) -> dict:
        return self._request("GET", f"/markets/{ticker}")["market"]

    def get_balance(self) -> float:
        return float(self._request("GET", "/portfolio/balance", signed=True)["balance_dollars"])

    def buy_yes(self, ticker: str, count: int, price: float, client_order_id: str) -> dict:
        """Buy YES at up to `price`, taking only what is offered right now; the rest is cancelled."""
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": "bid",                                   # bid = buy YES
            "count": f"{count:.2f}",
            "price": f"{price:.4f}",
            "time_in_force": "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross",
        }
        return self._request("POST", "/portfolio/events/orders", signed=True, json=body)
