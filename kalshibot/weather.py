"""Forecast high from the National Weather Service API (api.weather.gov). No key needed."""
from __future__ import annotations

from datetime import date, datetime

import requests


class Weather:
    def __init__(self, base: str, user_agent: str, session: requests.Session | None = None):
        self.base = base.rstrip("/")
        self.session = session or requests.Session()
        self.headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}
        self._points: dict[tuple[float, float], dict] = {}

    def _get(self, url: str) -> dict:
        resp = self.session.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def point(self, lat: float, lon: float) -> dict:
        """The forecast link and time zone for a location (looked up once, then remembered)."""
        key = (lat, lon)
        if key not in self._points:
            props = self._get(f"{self.base}/points/{lat},{lon}")["properties"]
            self._points[key] = {"forecast": props["forecast"], "time_zone": props["timeZone"]}
        return self._points[key]

    def forecast_high(self, lat: float, lon: float, day: date) -> dict | None:
        """The forecast high in Fahrenheit for `day`, or None when the forecast has no daytime period for it."""
        return pick_high(self._get(self.point(lat, lon)["forecast"]), day)


def pick_high(forecast: dict, day: date) -> dict | None:
    props = forecast["properties"]
    for period in props["periods"]:
        starts = datetime.fromisoformat(period["startTime"])
        if period["isDaytime"] and starts.date() == day:
            if period["temperatureUnit"] != "F":
                raise ValueError(f"expected Fahrenheit, got {period['temperatureUnit']}")
            return {"high": float(period["temperature"]), "period": period["name"],
                    "generated_at": props.get("generatedAt")}
    return None
