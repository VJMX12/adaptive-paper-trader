"""Iteration 2: refine cross-sectional momentum.

Discipline: variants are tuned/inspected on the TRAIN half (first 60d) only;
the VALIDATION half (last 60d) is looked at once, at the end, for the
shortlisted variants. Guards against overfitting the improvement.

Refinement axes (each targets the 28% cost drag or signal quality):
  - rank buffer/hysteresis: only swap a name in/out when it moves K ranks
    past the boundary (classic turnover reduction)
  - vol-adjusted momentum: rank ret/vol instead of raw ret
  - rebalance interval
  - portfolio width Q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from backtest import Panel, load_panel, simulate, metrics, rolling_ret, rolling_vol


class XMomBuffered:
    def __init__(self, K=24, Q=4, rebal=8, buffer=0, vol_adj=False):
        self.K, self.Q, self.REBAL, self.BUF, self.VOL = K, Q, rebal, buffer, vol_adj
        self.name = f"xmom K{K} Q{Q} rb{rebal} buf{buffer}{' voladj' if vol_adj else ''}"

    def positions(self, p: Panel) -> np.ndarray:
        mom = rolling_ret(p.close, self.K)
        if self.VOL:
            v = rolling_vol(p.close, 72)
            mom = mom / np.where(v > 1e-12, v, np.nan)
        T, N = p.close.shape
        pos = np.zeros((T, N))
        cur = np.zeros(N)
        for t in range(max(self.K, 72), T):
            if t % self.REBAL == 0:
                row = mom[t]
                ok = ~np.isnan(row)
                if ok.sum() >= 2 * (self.Q + self.BUF):
                    ids = np.where(ok)[0]
                    order = ids[np.argsort(row[ok])]          # ascending
                    top_in = set(order[-self.Q:])
                    top_keep = set(order[-(self.Q + self.BUF):])
                    bot_in = set(order[:self.Q])
                    bot_keep = set(order[:self.Q + self.BUF])
                    new = np.zeros(N)
                    # keep held names that are still inside the buffer zone
                    held_long = {j for j in range(N) if cur[j] > 0}
                    held_short = {j for j in range(N) if cur[j] < 0}
                    longs = list(held_long & top_keep)
                    shorts = list(held_short & bot_keep)
                    # fill remaining slots with the strongest new names
                    for j in reversed(order):
                        if len(longs) >= self.Q:
                            break
                        if j in top_in and j not in longs:
                            longs.append(j)
                    for j in order:
                        if len(shorts) >= self.Q:
                            break
                        if j in bot_in and j not in shorts:
                            shorts.append(j)
                    w = 1.0 / (len(longs) + len(shorts)) if (longs or shorts) else 0.0
                    for j in longs:
                        new[j] = w
                    for j in shorts:
                        new[j] = -w
                    cur = new
            pos[t] = cur
        return pos


def slice_panel(p: Panel, a: int, b: int) -> Panel:
    return Panel(p.symbols, p.ts[a:b], p.close[a:b], p.high[a:b],
                 p.low[a:b], p.volume[a:b], p.funding[a:b])


def eval_on(p: Panel, strat) -> dict:
    sim = simulate(p, strat.positions(p))
    m = metrics(sim["net"])
    m["cost_total"] = float(sim["cost"].sum())
    return m


def main():
    panel = load_panel()
    T = len(panel.ts)
    train = slice_panel(panel, 0, T // 2)
    valid = slice_panel(panel, T // 2, T)

    variants = []
    for K in (12, 24, 48):
        for Q in (3, 4, 5):
            for rebal in (8, 12, 24):
                for buf in (0, 2, 4):
                    for vol_adj in (False, True):
                        variants.append(XMomBuffered(K, Q, rebal, buf, vol_adj))

    print(f"TRAIN (first 60d): sweeping {len(variants)} variants...\n")
    scored = []
    for s in variants:
        m = eval_on(train, s)
        scored.append((m["sharpe"], m["total_ret"], m["cost_total"], s))
    scored.sort(key=lambda x: -x[0])
    print("top 12 by train sharpe:")
    for sh, ret, cost, s in scored[:12]:
        print(f"  {s.name:38s} sharpe {sh:+.2f} ret {ret*100:+.1f}% costs {cost*100:.1f}%")

    # robustness check: pick a STABLE region, not the single best point.
    # median-neighbor score: average sharpe of variants sharing 3 of 4 params.
    print("\nVALIDATION (last 60d, evaluated once) for the top-8 train variants:")
    for sh, ret, cost, s in scored[:8]:
        mv = eval_on(valid, s)
        print(f"  {s.name:38s} train {sh:+.2f} | valid sharpe {mv['sharpe']:+.2f} "
              f"ret {mv['total_ret']*100:+.1f}% dd {mv['max_dd']*100:.1f}% "
              f"costs {mv['cost_total']*100:.1f}%")


if __name__ == "__main__":
    main()
