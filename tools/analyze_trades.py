#!/usr/bin/env python3
"""Analyze closed trades from the dashboard's /export/trades.csv.

Answers the three diagnostic questions behind a flat/low win-rate run:
  1. Exit-reason breakdown (tp / sl / time_exit / regime_exit / changepoint_exit)
     -> is realized RR diluted below the nominal tp_rr because trades are
     getting cut by the time-stop before reaching target?
  2. Win rate segmented by symbol class (majors vs the excluded synthetic
     stock/commodity perpetuals) -> was the illiquid subset dragging the
     shared learner's edge down?
  3. Win rate trend over time (first half vs second half of the run) ->
     degrading (overfitting hitting live data) vs stable (edge is just thin).

Usage:
  # against a live dashboard (Basic Auth same as the browser dashboard):
  python3 tools/analyze_trades.py --url https://<app>.up.railway.app --user admin --password "$DASHBOARD_PASS"

  # against a CSV already downloaded (e.g. via curl -u ... .../export/trades.csv):
  python3 tools/analyze_trades.py --csv trades.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from collections import Counter

# Kept in sync with config/config.yaml exchange.excluded_symbols.
SYNTHETIC_SYMBOLS = {
    "XAU/USDT:USDT", "XAG/USDT:USDT", "XAUT/USDT:USDT", "SPCX/USDT:USDT",
    "MSTR/USDT:USDT", "BABA/USDT:USDT", "CL/USDT:USDT", "EWY/USDT:USDT",
    "SKHYNIX/USDT:USDT", "SNDK/USDT:USDT", "BTW/USDT:USDT", "GRAM/USDT:USDT",
    "CRCL/USDT:USDT", "DRAM/USDT:USDT", "BILL/USDT:USDT", "BEAT/USDT:USDT",
    "ES/USDT:USDT",
}


def fetch_csv(url: str, user: str, password: str) -> str:
    req = urllib.request.Request(url.rstrip("/") + "/export/trades.csv")
    if user or password:
        import base64
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode()


def load_rows(text: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    closed = [r for r in rows if r.get("exit_ts")]
    if not closed:
        print("No closed trades found.", file=sys.stderr)
    return closed


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "n/a"


def analyze(rows: list[dict]) -> None:
    n = len(rows)
    print(f"Closed trades: {n}\n")

    # 1) exit-reason breakdown
    reasons = Counter(r.get("exit_reason") or "unknown" for r in rows)
    print("Exit reason breakdown:")
    for reason, c in reasons.most_common():
        wins = sum(1 for r in rows
                   if (r.get("exit_reason") or "unknown") == reason
                   and float(r.get("pnl_usd") or 0) > 0)
        print(f"  {reason:16s} {c:4d} ({pct(c, n):>6s})  win rate {pct(wins, c):>6s}")
    print()

    # 2) majors vs synthetic/illiquid pairs
    majors = [r for r in rows if r["symbol"] not in SYNTHETIC_SYMBOLS]
    synth = [r for r in rows if r["symbol"] in SYNTHETIC_SYMBOLS]
    for label, group in (("Crypto majors", majors), ("Synthetic stock/commodity", synth)):
        if not group:
            continue
        wins = sum(1 for r in group if float(r.get("pnl_usd") or 0) > 0)
        pnl = sum(float(r.get("pnl_usd") or 0) for r in group)
        print(f"{label}: {len(group)} trades, win rate {pct(wins, len(group))}, "
              f"total pnl ${pnl:,.2f}")
    print()

    # 3) win rate over time: first half vs second half
    rows_sorted = sorted(rows, key=lambda r: r.get("exit_ts") or "")
    half = len(rows_sorted) // 2
    first, second = rows_sorted[:half], rows_sorted[half:]
    for label, group in (("First half (by time)", first), ("Second half (by time)", second)):
        if not group:
            continue
        wins = sum(1 for r in group if float(r.get("pnl_usd") or 0) > 0)
        pnl = sum(float(r.get("pnl_usd") or 0) for r in group)
        print(f"{label}: {len(group)} trades, win rate {pct(wins, len(group))}, "
              f"total pnl ${pnl:,.2f}")
    print()

    # overall realized R-multiple stats, if present
    r_multiples = [float(r["r_multiple"]) for r in rows
                  if r.get("r_multiple") not in (None, "")]
    if r_multiples:
        avg_r = sum(r_multiples) / len(r_multiples)
        print(f"Average realized R-multiple: {avg_r:+.3f} "
              f"(nominal target from config is tp_rr, typically 2.0)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="dashboard base URL, e.g. https://x.up.railway.app")
    ap.add_argument("--user", default="admin", help="Basic Auth username")
    ap.add_argument("--password", default="", help="Basic Auth password (DASHBOARD_PASS)")
    ap.add_argument("--csv", help="path to an already-downloaded trades.csv")
    args = ap.parse_args()

    if args.csv:
        text = open(args.csv, encoding="utf-8").read()
    elif args.url:
        text = fetch_csv(args.url, args.user, args.password)
    else:
        ap.error("pass either --url (+ --password) or --csv")

    analyze(load_rows(text))


if __name__ == "__main__":
    main()
