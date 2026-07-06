"""Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

Maintains a run-length posterior over "how long since the data-generating
process last changed". The system reads P(recent changepoint) as a live
"my assumptions may no longer be valid" signal and collapses exposure
when it spikes.

Predictive model: Normal observations with unknown mean & variance
(Normal-Inverse-Gamma conjugate prior -> Student-t predictive).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

MAX_RUN = 800


class BOCPD:
    def __init__(self, hazard: float = 250.0,
                 mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1e-4):
        self.h = 1.0 / hazard
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.reset()

    def reset(self) -> None:
        self.run_probs = np.array([1.0])
        self.mu = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta = np.array([self.beta0])
        self.t = 0

    def update(self, x: float) -> None:
        # Student-t predictive for each run length hypothesis
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        pred = stats.t.pdf(x, df=df, loc=self.mu, scale=np.maximum(scale, 1e-12))
        pred = np.maximum(pred, 1e-300)

        growth = self.run_probs * pred * (1.0 - self.h)
        cp = float(np.sum(self.run_probs * pred * self.h))
        new_probs = np.concatenate(([cp], growth))
        s = new_probs.sum()
        new_probs = new_probs / s if s > 0 else np.array([1.0])

        # posterior parameter updates (shift by one run length, prepend prior)
        kappa_n = self.kappa + 1.0
        mu_n = (self.kappa * self.mu + x) / kappa_n
        alpha_n = self.alpha + 0.5
        beta_n = self.beta + 0.5 * self.kappa * (x - self.mu) ** 2 / kappa_n

        self.mu = np.concatenate(([self.mu0], mu_n))
        self.kappa = np.concatenate(([self.kappa0], kappa_n))
        self.alpha = np.concatenate(([self.alpha0], alpha_n))
        self.beta = np.concatenate(([self.beta0], beta_n))
        self.run_probs = new_probs

        if len(self.run_probs) > MAX_RUN:  # truncate tail, renormalize
            self.run_probs = self.run_probs[:MAX_RUN]
            self.run_probs /= self.run_probs.sum()
            self.mu, self.kappa = self.mu[:MAX_RUN], self.kappa[:MAX_RUN]
            self.alpha, self.beta = self.alpha[:MAX_RUN], self.beta[:MAX_RUN]
        self.t += 1

    def update_many(self, xs: np.ndarray) -> None:
        for x in np.asarray(xs, dtype=float):
            self.update(float(x))

    def p_recent_changepoint(self, within: int = 6) -> float:
        """P(run length < `within`) = probability the process changed recently."""
        k = min(within, len(self.run_probs))
        return float(self.run_probs[:k].sum())

    def expected_run_length(self) -> float:
        idx = np.arange(len(self.run_probs), dtype=float)
        return float((idx * self.run_probs).sum())
