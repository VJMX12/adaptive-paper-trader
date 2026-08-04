"""Fetch historical 1h OHLCV + funding rates from Bybit for R&D backtesting.

Saves per-symbol npz files under research/data/. Resumable: skips symbols
already fetched.
"""
import json
import os
import sys
import time

import ccxt
import numpy as np

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "XRP/USDT:USDT", "BNB/USDT:USDT",
    "SOL/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT",
    "LINK/USDT:USDT", "AVAX/USDT:USDT", "XLM/USDT:USDT", "LTC/USDT:USDT",
    "BCH/USDT:USDT", "DOT/USDT:USDT", "UNI/USDT:USDT", "NEAR/USDT:USDT",
    "ICP/USDT:USDT", "APT/USDT:USDT", "ATOM/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "HBAR/USDT:USDT", "SUI/USDT:USDT", "INJ/USDT:USDT",
    "TAO/USDT:USDT", "XMR/USDT:USDT", "AAVE/USDT:USDT",
]
DAYS = 120
TF = "1h"
TF_MS = 3600_000
OUT = os.path.join(os.path.dirname(__file__), "data")


def fetch_ohlcv_full(ex, sym, since_ms, until_ms):
    rows = []
    cursor = since_ms
    while cursor < until_ms:
        batch = ex.fetch_ohlcv(sym, TF, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + TF_MS
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(ex.rateLimit / 1000.0)
    # dedupe + sort
    seen = {}
    for r in rows:
        seen[r[0]] = r
    return [seen[k] for k in sorted(seen)]


def fetch_funding_full(ex, sym, since_ms):
    rows = []
    cursor = since_ms
    for _ in range(100):
        try:
            batch = ex.fetch_funding_rate_history(sym, since=cursor, limit=200)
        except Exception as e:
            print(f"  funding fetch failed for {sym}: {e}", flush=True)
            break
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1]["timestamp"] + 1
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(ex.rateLimit / 1000.0)
    seen = {}
    for r in rows:
        seen[r["timestamp"]] = (r["timestamp"], float(r["fundingRate"] or 0.0))
    return [seen[k] for k in sorted(seen)]


def main():
    os.makedirs(OUT, exist_ok=True)
    ex = ccxt.bybit({"options": {"defaultType": "swap"}})
    now = ex.milliseconds()
    since = now - DAYS * 24 * 3600_000

    for sym in SYMBOLS:
        safe = sym.split("/")[0]
        path = os.path.join(OUT, f"{safe}.npz")
        if os.path.exists(path):
            print(f"skip {safe} (exists)", flush=True)
            continue
        t0 = time.time()
        ohlcv = fetch_ohlcv_full(ex, sym, since, now)
        funding = fetch_funding_full(ex, sym, since)
        arr = np.array(ohlcv, dtype=float)  # ts,o,h,l,c,v
        farr = np.array(funding, dtype=float) if funding else np.zeros((0, 2))
        np.savez_compressed(path, ohlcv=arr, funding=farr)
        print(f"{safe}: {len(arr)} candles, {len(farr)} funding rows "
              f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
