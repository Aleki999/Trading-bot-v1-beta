"""Command line: python -m tradingbot.cli <step>

Steps, in order: fetch, check, study, core, sleeves, mcpt, account.
The locked test only runs with --locked, and each candidate's locked result is stored and never rerun.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import data
from .account import combine, pick_sleeves
from .backtester import lookahead_guard, run_switch, stats
from .checks import run_checks
from .config import RESULTS_DIR, all_tickers, load_config, window
from .gates import LEDGER, evaluate_sleeve, locked_test, run_sleeve
from .labels import label_ticker
from .permutation import permutation_test
from .signals import trend_switch


def save(name: str, payload) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / name).write_text(json.dumps(payload, indent=2, default=str))


def load(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        raise SystemExit(f"results/{name} is missing: run the earlier step first")
    return json.loads(path.read_text())


def pct(x: float) -> str:
    return f"{x * 100:6.1f}%"


def line(label: str, s: dict) -> str:
    if not s["trades"]:   # a combined account has no trade list of its own
        return f"{label:<24} return/yr {pct(s['annual_return'])}  max drawdown {pct(s['max_drawdown'])}"
    return (f"{label:<24} return/yr {pct(s['annual_return'])}  max drawdown {pct(s['max_drawdown'])}  "
            f"trades {s['trades']:>3}  win rate {pct(s['win_rate'])}  profit factor {s['profit_factor']:.2f}")


def core_run(prices, cfg, start, end, line_days=None, kind="sma", cost=None):
    core = cfg["core"]
    engine, safe = prices[core["engine"]], prices[core["safe"]]
    signal = trend_switch(engine, line_days or core["line_days"], kind)
    return run_switch(engine, safe, signal, cost if cost is not None else cfg["cost_per_side"], start, end)


def cmd_fetch(cfg, args):
    prices = data.download_prices(all_tickers(cfg), cfg["windows"]["warmup_start"])
    print(f"prices: {prices['ticker'].nunique()} tickers, {prices['date'].min().date()} to {prices['date'].max().date()}")
    vix = data.download_vix()
    print(f"vix: {len(vix.dropna())} days, latest {vix.dropna().index[-1].date()}")


def cmd_check(cfg, args):
    out = run_checks(cfg["cost_per_side"])
    save("check.json", out)
    for name in ("planted_edge", "pure_noise"):
        r = out[name]
        print(f"{name:<16} avg trade {r['avg_trade'] * 100:+.2f}%  allowed {r['range'][0] * 100:+.1f}% to "
              f"{r['range'][1] * 100:+.1f}%  {'PASS' if r['pass'] else 'FAIL'}")
    print(f"{'cheating_signal':<16} stopped by the guard: {out['cheating_signal']['stopped']}  "
          f"{'PASS' if out['cheating_signal']['pass'] else 'FAIL'}")
    if not out["all_pass"]:
        raise SystemExit("backtester check failed: do not trust any result until this passes")


def cmd_study(cfg, args):
    prices = data.load_prices()
    start, end = window(cfg, "train")
    labels = {t: label_ticker(prices[t], cfg, start, end) for t in cfg["tickers"]["sleeves"]}
    save("labels.json", labels)
    for t, lab in labels.items():
        print(f"{t:<5} snaps back: {str(lab['snaps_back']):<5} ({lab['snaps_back_events']:>2} events, edge "
              f"{lab['snaps_back_edge'] * 100:+.2f}%)   keeps going: {str(lab['keeps_going']):<5} "
              f"({lab['keeps_going_events']:>2} events, edge {lab['keeps_going_edge'] * 100:+.2f}%)")


def cmd_core(cfg, args):
    prices = data.load_prices()
    core = cfg["core"]
    lookahead_guard(lambda b: trend_switch(b, core["line_days"]), prices[core["engine"]])
    t0, t1 = window(cfg, "train")
    out = {"engine": core["engine"], "safe": core["safe"], "line_days": core["line_days"]}
    out["train"] = stats(core_run(prices, cfg, t0, t1), cfg["bars_per_year"])
    out["neighbors"] = {f"{n['line_days']}-day {n['kind']}":
                        stats(core_run(prices, cfg, t0, t1, n["line_days"], n["kind"]), cfg["bars_per_year"])["total_return"]
                        for n in core["neighbors"]}
    out["train_double_cost"] = stats(core_run(prices, cfg, t0, t1, cost=2 * cfg["cost_per_side"]),
                                     cfg["bars_per_year"])["annual_return"]
    out["neighbors_pass"] = all(v > 0 for v in out["neighbors"].values())
    print(line(f"core {core['engine']}/{core['safe']} train", out["train"]))
    for name, total in out["neighbors"].items():
        print(f"  neighbor {name:<14} total return {pct(total)}")
    print(f"  with costs doubled: return/yr {pct(out['train_double_cost'])}")
    if args.locked:
        if not (out["neighbors_pass"] and out["train"]["total_return"] > 0):
            raise SystemExit("core failed in training: locked test not run")
        l0, l1 = window(cfg, "locked")
        out["locked"] = locked_test(f"core:{core['engine']}:{core['safe']}:{core['line_days']}",
                                    lambda: stats(core_run(prices, cfg, l0, l1), cfg["bars_per_year"]))
        print(line("core locked test", out["locked"]) + ("  (stored result)" if out["locked"]["rerun"] else ""))
    save("core.json", out)


def cmd_sleeves(cfg, args):
    prices = data.load_prices()
    labels = load("labels.json")
    rows = []
    for ticker in cfg["tickers"]["sleeves"]:
        for family in cfg["sleeve_families"]:
            rows.append(evaluate_sleeve(ticker, family, prices[ticker], labels[ticker], cfg, run_locked=args.locked))
    save("sleeves.json", rows)
    order = ["fits", "profitable", "enough_trades", "profit_factor", "neighbors"] + (["locked_test"] if args.locked else [])
    print(f"{len(rows)} candidates")
    for name in order:
        print(f"  pass {name:<14} {sum(name in r['passed'] for r in rows):>3}")
    for r in rows:
        if order[-1] in r["passed"]:
            print(line(f"{r['ticker']} {r['family']} train", r["train"]))
            if "locked" in r:
                print(line(f"{r['ticker']} {r['family']} locked", r["locked"]))


def cmd_mcpt(cfg, args):
    """Permutation tests on training data only, for the core and every sleeve that passed the training gates."""
    prices = data.load_prices()
    perm = cfg["permutation"]
    n = args.n or perm["n"]
    t0, t1 = window(cfg, "train")
    core = cfg["core"]
    out = {}

    def core_stat(frames):
        signal = trend_switch(frames[0], core["line_days"])
        return stats(run_switch(frames[0], frames[1], signal, cfg["cost_per_side"], t0, t1))["total_return"]

    idx = prices[core["engine"]].loc[:t1].index.intersection(prices[core["safe"]].index)
    out["core"] = permutation_test([prices[core["engine"]].loc[idx], prices[core["safe"]].loc[idx]], core_stat,
                                   n, perm["seed"])
    print(f"{'core':<16} real {pct(out['core']['real'])}  shuffled median {pct(out['core']['fake_median'])}  "
          f"p-value {out['core']['p_value']:.3f}")
    for r in load("sleeves.json"):
        if "neighbors" not in r["passed"]:
            continue
        name = f"{r['ticker']}:{r['family']}"

        def sleeve_stat(frames, family=r["family"]):
            return stats(run_sleeve(frames[0], family, cfg, t0, t1))["total_return"]

        out[name] = permutation_test([prices[r["ticker"]].loc[:t1]], sleeve_stat, n, perm["seed"])
        print(f"{name:<16} real {pct(out[name]['real'])}  shuffled median {pct(out[name]['fake_median'])}  "
              f"p-value {out[name]['p_value']:.3f}")
    save("mcpt.json", out)


def cmd_account(cfg, args):
    prices = data.load_prices()
    vix = data.load_vix()
    t0, t1 = window(cfg, "train")
    need = "locked_test" if args.locked else "neighbors"
    passed = [r for r in load("sleeves.json") if need in r["passed"]]
    if not args.skip_mcpt:   # the permutation test is a gate too: keep only sleeves that beat the shuffles
        mcpt, limit = load("mcpt.json"), cfg["permutation"]["max_p_value"]
        kept = [r for r in passed if mcpt.get(f"{r['ticker']}:{r['family']}", {}).get("p_value", 1.0) <= limit]
        print(f"permutation gate (p <= {limit}): {len(kept)} of {len(passed)} sleeves pass")
        passed = kept
    train = {f"{r['ticker']}:{r['family']}": run_sleeve(prices[r["ticker"]], r["family"], cfg, t0, t1) for r in passed}
    chosen = pick_sleeves(train, cfg)
    print(f"sleeves chosen on training data: {chosen or 'none'}")
    out = {"sleeves": chosen}
    core_train = core_run(prices, cfg, t0, t1)
    out["train_core_alone"] = stats(core_train, cfg["bars_per_year"])
    out["train"] = combine(core_train, {k: train[k] for k in chosen}, cfg, vix)["stats"]
    print(line("core alone, train", out["train_core_alone"]))
    print(line("account, train", out["train"]))
    if args.locked:
        l0, l1 = window(cfg, "locked")

        def run():
            core_locked = core_run(prices, cfg, l0, l1)
            sleeves = {k: run_sleeve(prices[k.split(":")[0]], k.split(":")[1], cfg, l0, l1) for k in chosen}
            market = prices[cfg["tickers"]["market"]]["close"].loc[l0:l1 if l1 is not None else None]
            return {"account": combine(core_locked, sleeves, cfg, vix)["stats"],
                    "core_alone": stats(core_locked, cfg["bars_per_year"]),
                    "market_buy_hold_total": float(market.iloc[-1] / market.iloc[0] - 1),
                    "through": str(core_locked.daily.index[-1].date())}

        out["locked"] = locked_test("account:" + ",".join(chosen), run)
        print(line("core alone, locked", out["locked"]["core_alone"]))
        print(line("account, locked", out["locked"]["account"]))
        print(f"{cfg['tickers']['market']} buy and hold over the locked window: total {pct(out['locked']['market_buy_hold_total'])}")
    save("account.json", out)


COMMANDS = {"fetch": cmd_fetch, "check": cmd_check, "study": cmd_study, "core": cmd_core,
            "sleeves": cmd_sleeves, "mcpt": cmd_mcpt, "account": cmd_account}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=COMMANDS)
    parser.add_argument("--locked", action="store_true", help=f"also run the one-time locked test (ledger: {LEDGER.name})")
    parser.add_argument("--n", type=int, default=None, help="number of permutations for mcpt")
    parser.add_argument("--skip-mcpt", action="store_true",
                        help="account: choose sleeves without the permutation gate (the course's original method)")
    args = parser.parse_args(argv)
    pd.set_option("display.width", 160)
    COMMANDS[args.step](load_config(), args)


if __name__ == "__main__":
    main()
