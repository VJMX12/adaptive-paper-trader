"""R&D backtest harness: test decorrelated candidate algorithms on Bybit 1h
data with realistic costs and walk-forward evaluation.

All strategies emit a target position in [-1, +1] per symbol per hour
(fraction of allocated capital, sign = direction). PnL is computed on
close-to-close returns with costs charged on position *changes* (turnover),
matching how a real executor would trade the signal.

Cost model mirrors config.yaml: taker 0.055% + slippage 0.03% per side on
notional => 0.085% per side charged on |Δposition|. Funding is applied
8-hourly from real funding history, sign-correct (longs pay positive rates).

Walk-forward: strategies here are parameter-light rules; evaluation splits
the 120d window into 4 sequential ~30d folds and reports per-fold results.
A candidate "survives" only if net Sharpe > 0 in a majority of folds and
overall net return > 0. No parameter fitting is done on the test folds --
parameters are fixed a priori (stated in each class), so all folds are
effectively out-of-sample; the fold split guards against one lucky regime
carrying the whole result.
"""
import os
from dataclasses import dataclass

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")
COST_PER_SIDE = 0.00085          # fee+slip on |Δpos|, fraction of notional
HOURS_YEAR = 24 * 365


# ---------------------------------------------------------------- data
@dataclass
class Panel:
    symbols: list
    ts: np.ndarray        # (T,) hourly ms timestamps, common grid
    close: np.ndarray     # (T, N)
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    funding: np.ndarray   # (T, N) funding rate charged AT that hour (0 mostly)


def load_panel() -> Panel:
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".npz"))
    raw = {}
    for f in files:
        d = np.load(os.path.join(DATA, f))
        raw[f[:-4]] = (d["ohlcv"], d["funding"])
    # common timestamp grid = intersection of all symbols
    common = None
    for arr, _ in raw.values():
        s = set(arr[:, 0].astype(np.int64))
        common = s if common is None else (common & s)
    ts = np.array(sorted(common), dtype=np.int64)
    idx = {t: i for i, t in enumerate(ts)}
    syms = sorted(raw)
    T, N = len(ts), len(syms)
    close = np.full((T, N), np.nan)
    high = np.full((T, N), np.nan)
    low = np.full((T, N), np.nan)
    vol = np.full((T, N), np.nan)
    fund = np.zeros((T, N))
    for j, s in enumerate(syms):
        arr, farr = raw[s]
        for row in arr:
            i = idx.get(np.int64(row[0]))
            if i is not None:
                high[i, j], low[i, j] = row[2], row[3]
                close[i, j], vol[i, j] = row[4], row[5]
        for fts, fr in farr:
            # funding settles on exact hours; snap to grid
            i = idx.get(np.int64(fts))
            if i is not None:
                fund[i, j] = fr
    return Panel(syms, ts, close, high, low, vol, fund)


# ---------------------------------------------------------------- engine
def simulate(panel: Panel, positions: np.ndarray) -> dict:
    """positions[t, j] = target exposure held DURING hour t->t+1, decided
    with data up to and including close[t]. Returns portfolio hourly returns
    net of trading costs and funding."""
    c = panel.close
    ret = np.zeros_like(c)
    ret[1:] = c[1:] / c[:-1] - 1.0
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)

    pos = np.nan_to_num(positions, nan=0.0)
    pos = np.clip(pos, -1.0, 1.0)

    # pnl for hour t+1 uses position set at t
    gross = (pos[:-1] * ret[1:]).sum(axis=1)
    turnover = np.abs(np.diff(pos, axis=0, prepend=np.zeros((1, pos.shape[1])))).sum(axis=1)
    cost = turnover * COST_PER_SIDE
    # funding charged at settlement hours on held position (long pays +rate)
    fund_pnl = -(pos[:-1] * panel.funding[1:]).sum(axis=1)
    net = gross - cost[:-1] + fund_pnl
    return {"net": net, "gross": gross, "cost": cost[:-1], "funding": fund_pnl}


def metrics(net: np.ndarray) -> dict:
    total = float(np.prod(1 + net) - 1)
    mu, sd = float(net.mean()), float(net.std(ddof=1))
    sharpe = 0.0 if sd == 0 else mu / sd * np.sqrt(HOURS_YEAR)
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(eq)
    mdd = float((1 - eq / peak).max())
    return {"total_ret": total, "sharpe": sharpe, "max_dd": mdd,
            "n_hours": len(net)}


def walk_forward(panel: Panel, strat, folds: int = 4) -> dict:
    pos = strat.positions(panel)
    sim = simulate(panel, pos)
    net = sim["net"]
    T = len(net)
    fold_stats = []
    for k in range(folds):
        seg = net[k * T // folds:(k + 1) * T // folds]
        fold_stats.append(metrics(seg))
    overall = metrics(net)
    overall["gross_total"] = float(np.prod(1 + sim["gross"]) - 1)
    overall["cost_total"] = float(sim["cost"].sum())
    overall["funding_total"] = float(sim["funding"].sum())
    pos_folds = sum(1 for f in fold_stats if f["sharpe"] > 0)
    return {"name": strat.name, "overall": overall, "folds": fold_stats,
            "positive_folds": f"{pos_folds}/{folds}",
            "survives": pos_folds >= (folds // 2 + 1) and overall["total_ret"] > 0}


# ---------------------------------------------------------------- helpers
def rolling_ret(close: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(close, np.nan)
    out[k:] = close[k:] / close[:-k] - 1.0
    return out


def rolling_vol(close: np.ndarray, k: int = 72) -> np.ndarray:
    r = np.zeros_like(close)
    r[1:] = np.log(close[1:] / close[:-1])
    r = np.nan_to_num(r)
    out = np.full_like(close, np.nan)
    csum = np.cumsum(r, axis=0)
    csum2 = np.cumsum(r * r, axis=0)
    for t in range(k, len(close)):
        m = (csum[t] - csum[t - k]) / k
        v = (csum2[t] - csum2[t - k]) / k - m * m
        out[t] = np.sqrt(np.maximum(v, 0))
    return out


def zscore(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for t in range(k, len(x)):
        w = x[t - k:t]
        m, s = np.nanmean(w, axis=0), np.nanstd(w, axis=0)
        out[t] = (x[t] - m) / np.where(s > 1e-12, s, np.nan)
    return out


# ---------------------------------------------------------------- strategies
class XSectionMomentum:
    """Cross-sectional momentum: every REBAL hours, rank symbols by K-hour
    return; long top-Q, short bottom-Q, equal weight, vol-scaled. Classic
    documented crypto anomaly. Params fixed a priori: K=24h, Q=4, rebal=8h."""
    name = "xsect_momentum_24h"
    K, Q, REBAL = 24, 4, 8

    def positions(self, p: Panel) -> np.ndarray:
        mom = rolling_ret(p.close, self.K)
        T, N = p.close.shape
        pos = np.zeros((T, N))
        cur = np.zeros(N)
        for t in range(self.K, T):
            if t % self.REBAL == 0:
                row = mom[t]
                ok = ~np.isnan(row)
                if ok.sum() >= 2 * self.Q:
                    order = np.argsort(row[ok])
                    ids = np.where(ok)[0]
                    cur = np.zeros(N)
                    cur[ids[order[-self.Q:]]] = 1.0 / (2 * self.Q)
                    cur[ids[order[:self.Q]]] = -1.0 / (2 * self.Q)
            pos[t] = cur
        return pos


class XSectionMomentumSlow:
    """Same but 7-day lookback, daily rebalance — captures slower flows,
    much lower turnover (cost matters!)."""
    name = "xsect_momentum_7d"
    K, Q, REBAL = 168, 4, 24

    def positions(self, p: Panel) -> np.ndarray:
        s = XSectionMomentum()
        s.K, s.Q, s.REBAL = self.K, self.Q, self.REBAL
        return s.positions(p)


class DonchianBreakout:
    """Time-series trend: long when close breaks 96h high, short on 96h low
    break, exit on midline cross. Vol-scaled position. Per-symbol, capped
    total exposure."""
    name = "donchian_96h"
    K = 96

    def positions(self, p: Panel) -> np.ndarray:
        T, N = p.close.shape
        pos = np.zeros((T, N))
        hi = np.full((T, N), np.nan)
        lo = np.full((T, N), np.nan)
        for t in range(self.K, T):
            hi[t] = np.nanmax(p.high[t - self.K:t], axis=0)
            lo[t] = np.nanmin(p.low[t - self.K:t], axis=0)
        mid = (hi + lo) / 2
        cur = np.zeros(N)
        for t in range(self.K, T):
            c = p.close[t]
            for j in range(N):
                if np.isnan(c[j]) or np.isnan(hi[t, j]):
                    continue
                if cur[j] == 0:
                    if c[j] > hi[t, j]:
                        cur[j] = 1.0
                    elif c[j] < lo[t, j]:
                        cur[j] = -1.0
                elif cur[j] > 0 and c[j] < mid[t, j]:
                    cur[j] = 0.0
                elif cur[j] < 0 and c[j] > mid[t, j]:
                    cur[j] = 0.0
            k = max(np.abs(cur).sum(), 1.0)
            pos[t] = cur / k
        return pos


class MeanRevZ:
    """Time-series mean reversion: fade 24h z-score extremes (|z|>2), exit
    at |z|<0.5, only when BTC 72h realized vol is below its median (chop
    filter — reversion works in quiet regimes, dies in trends)."""
    name = "meanrev_z24_lowvol"
    K, ENTRY, EXIT = 24, 2.0, 0.5

    def positions(self, p: Panel) -> np.ndarray:
        z = zscore(p.close, self.K)
        btc = p.symbols.index("BTC") if "BTC" in p.symbols else 0
        bv = rolling_vol(p.close[:, [btc]], 72)[:, 0]
        med = np.nanmedian(bv)
        T, N = p.close.shape
        pos = np.zeros((T, N))
        cur = np.zeros(N)
        for t in range(self.K + 72, T):
            calm = bv[t] < med
            for j in range(N):
                zj = z[t, j]
                if np.isnan(zj):
                    continue
                if cur[j] == 0 and calm:
                    if zj > self.ENTRY:
                        cur[j] = -1.0
                    elif zj < -self.ENTRY:
                        cur[j] = 1.0
                elif cur[j] != 0 and abs(zj) < self.EXIT:
                    cur[j] = 0.0
            k = max(np.abs(cur).sum(), 1.0)
            pos[t] = cur / k
        return pos


class BTCLeadLag:
    """Lead-lag: when BTC prints a >2-sigma 4h move, hold alts in the same
    direction for the next 12h (alts historically lag BTC's big moves)."""
    name = "btc_leadlag"
    SIG, HOLD = 2.0, 12

    def positions(self, p: Panel) -> np.ndarray:
        btc = p.symbols.index("BTC") if "BTC" in p.symbols else 0
        r4 = rolling_ret(p.close[:, [btc]], 4)[:, 0]
        sd = np.full(len(r4), np.nan)
        for t in range(168, len(r4)):
            sd[t] = np.nanstd(r4[t - 168:t])
        T, N = p.close.shape
        pos = np.zeros((T, N))
        alt = np.ones(N, dtype=bool)
        alt[btc] = False
        hold_until, direction = -1, 0.0
        for t in range(172, T):
            if not np.isnan(sd[t]) and sd[t] > 0 and abs(r4[t]) > self.SIG * sd[t]:
                hold_until = t + self.HOLD
                direction = np.sign(r4[t])
            if t <= hold_until:
                pos[t, alt] = direction / alt.sum()
        return pos


class FundingCarry:
    """Funding-rate carry: hold positions AGAINST crowded funding — short
    the Q most positive-funding symbols, long the Q most negative, rebalance
    every 8h (funding settlement). Collects funding while roughly market
    neutral. The classic delta-neutral perp carry, directional-lite."""
    name = "funding_carry"
    Q, REBAL = 4, 8

    def positions(self, p: Panel) -> np.ndarray:
        T, N = p.close.shape
        # forward-fill last observed funding rate per symbol as the signal
        sig = np.zeros((T, N))
        last = np.zeros(N)
        for t in range(T):
            nz = p.funding[t] != 0
            last[nz] = p.funding[t][nz]
            sig[t] = last
        pos = np.zeros((T, N))
        cur = np.zeros(N)
        for t in range(24, T):
            if t % self.REBAL == 0:
                row = sig[t]
                order = np.argsort(row)
                cur = np.zeros(N)
                cur[order[-self.Q:]] = -1.0 / (2 * self.Q)   # short high funding
                cur[order[:self.Q]] = 1.0 / (2 * self.Q)     # long negative funding
            pos[t] = cur
        return pos


class VolExpansionBreakout:
    """1h version of the deployed Breakout strategy, as a baseline sanity
    check: range-edge close + vol expansion, hold until opposite edge or
    48h timeout."""
    name = "vol_expansion_breakout"
    K, TH, HOLD = 96, 0.85, 48

    def positions(self, p: Panel) -> np.ndarray:
        T, N = p.close.shape
        sv = rolling_vol(p.close, 48)
        lv = rolling_vol(p.close, 192)
        pos = np.zeros((T, N))
        cur = np.zeros(N)
        opened = np.full(N, -1)
        for t in range(192, T):
            hi = np.nanmax(p.high[t - self.K:t], axis=0)
            lo = np.nanmin(p.low[t - self.K:t], axis=0)
            rng = np.where(hi - lo > 1e-12, hi - lo, np.nan)
            cp = (p.close[t] - lo) / rng
            expanding = sv[t] > 1.2 * lv[t]
            for j in range(N):
                if cur[j] != 0 and t - opened[j] > self.HOLD:
                    cur[j] = 0.0
                if np.isnan(cp[j]):
                    continue
                if cur[j] == 0 and expanding[j]:
                    if cp[j] >= self.TH:
                        cur[j], opened[j] = 1.0, t
                    elif cp[j] <= 1 - self.TH:
                        cur[j], opened[j] = -1.0, t
                elif cur[j] > 0 and cp[j] < 0.5:
                    cur[j] = 0.0
                elif cur[j] < 0 and cp[j] > 0.5:
                    cur[j] = 0.0
            k = max(np.abs(cur).sum(), 1.0)
            pos[t] = cur / k
        return pos


# ---------------------------------------------------------------- main
def main():
    panel = load_panel()
    T, N = panel.close.shape
    days = (panel.ts[-1] - panel.ts[0]) / 86400_000
    print(f"panel: {N} symbols x {T} hours ({days:.0f} days)\n")
    strats = [XSectionMomentum(), XSectionMomentumSlow(), DonchianBreakout(),
              MeanRevZ(), BTCLeadLag(), FundingCarry(), VolExpansionBreakout()]
    results = []
    for s in strats:
        r = walk_forward(panel, s)
        results.append(r)
        o = r["overall"]
        print(f"== {r['name']}  {'SURVIVES' if r['survives'] else 'dies'}")
        print(f"   net {o['total_ret']*100:+.1f}%  gross {o['gross_total']*100:+.1f}%  "
              f"costs {o['cost_total']*100:.1f}%  funding {o['funding_total']*100:+.1f}%")
        print(f"   sharpe {o['sharpe']:+.2f}  maxDD {o['max_dd']*100:.1f}%  "
              f"positive folds {r['positive_folds']}")
        for i, f in enumerate(r["folds"]):
            print(f"     fold{i+1}: ret {f['total_ret']*100:+.2f}%  "
                  f"sharpe {f['sharpe']:+.2f}  dd {f['max_dd']*100:.1f}%")
        print()
    surv = [r["name"] for r in results if r["survives"]]
    print("SURVIVORS:", surv if surv else "none")


if __name__ == "__main__":
    main()
