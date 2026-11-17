"""Load config/system.yaml and resolve the test windows."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "system.yaml"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def window(cfg: dict, name: str) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    """Return (start, end) for 'train' or 'locked'. A null end means the latest bar."""
    w = cfg["windows"]
    start = pd.Timestamp(w[f"{name}_start"])
    end = w.get(f"{name}_end")
    return start, (pd.Timestamp(end) if end else None)


def all_tickers(cfg: dict) -> list[str]:
    t = cfg["tickers"]
    names = [t["market"], t["cash"], *t["engine_candidates"], *t["safe_candidates"], *t["sleeves"]]
    return sorted(set(names))
