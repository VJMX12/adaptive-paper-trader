"""Sticky Gaussian HMM for unsupervised market regime discovery.

Hand-rolled (numpy only) so behavior is fully inspectable:
- diagonal-covariance Gaussian emissions
- EM fitting with a sticky self-transition prior
- forward filtering for the live regime posterior

Regimes are discovered, not labeled. Human-readable descriptions are
derived *after* fitting from each state's return/volatility profile.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-10


class StickyGaussianHMM:
    def __init__(self, n_states: int = 4, sticky: float = 0.97, seed: int = 7):
        self.k = n_states
        self.sticky = sticky
        self.rng = np.random.default_rng(seed)
        self.means: np.ndarray | None = None      # (k, d)
        self.vars: np.ndarray | None = None       # (k, d)
        self.trans: np.ndarray | None = None      # (k, k)
        self.pi: np.ndarray | None = None         # (k,)
        self.fitted = False

    # ---------- emissions ----------
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """log N(x | mu_k, diag(var_k)) for all t,k -> (n, k)."""
        n, d = X.shape
        out = np.empty((n, self.k))
        for k in range(self.k):
            var = np.maximum(self.vars[k], EPS)
            diff = X - self.means[k]
            out[:, k] = -0.5 * (np.sum(diff * diff / var, axis=1)
                                + np.sum(np.log(2 * np.pi * var)))
        return out

    # ---------- fitting ----------
    def fit(self, X: np.ndarray, n_iter: int = 40, tol: float = 1e-4) -> None:
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        # init: quantile-based split on |return| + jitter so states differ
        order = np.argsort(X[:, -1])
        chunks = np.array_split(order, self.k)
        self.means = np.array([X[c].mean(axis=0) for c in chunks])
        self.vars = np.array([X[c].var(axis=0) + 1e-6 for c in chunks])
        self.means += self.rng.normal(0, 1e-6, self.means.shape)
        self.trans = np.full((self.k, self.k), (1 - self.sticky) / (self.k - 1))
        np.fill_diagonal(self.trans, self.sticky)
        self.pi = np.full(self.k, 1.0 / self.k)

        prev_ll = -np.inf
        for _ in range(n_iter):
            log_b = self._log_emission(X)
            log_alpha, log_beta, ll = self._forward_backward(log_b)
            if abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
                break
            prev_ll = ll
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)                       # (n, k)

            # xi: transition responsibilities
            log_trans = np.log(np.maximum(self.trans, EPS))
            xi_sum = np.zeros((self.k, self.k))
            for t in range(n - 1):
                m = (log_alpha[t][:, None] + log_trans
                     + log_b[t + 1][None, :] + log_beta[t + 1][None, :])
                m -= _logsumexp(m)
                xi_sum += np.exp(m)

            # M-step with sticky prior (pseudo-counts on the diagonal)
            prior = np.zeros((self.k, self.k))
            np.fill_diagonal(prior, self.sticky * 50.0)
            trans = xi_sum + prior + (1 - self.sticky)
            self.trans = trans / trans.sum(axis=1, keepdims=True)

            w = gamma.sum(axis=0) + EPS
            self.means = (gamma.T @ X) / w[:, None]
            for k in range(self.k):
                diff = X - self.means[k]
                self.vars[k] = (gamma[:, k][:, None] * diff * diff).sum(axis=0) / w[k] + 1e-8
            self.pi = gamma[0] + EPS
            self.pi /= self.pi.sum()
        self.fitted = True

    def _forward_backward(self, log_b: np.ndarray):
        n, k = log_b.shape
        log_trans = np.log(np.maximum(self.trans, EPS))
        log_alpha = np.empty((n, k))
        log_alpha[0] = np.log(self.pi + EPS) + log_b[0]
        for t in range(1, n):
            log_alpha[t] = log_b[t] + _logsumexp(
                log_alpha[t - 1][:, None] + log_trans, axis=0
            )
        log_beta = np.zeros((n, k))
        for t in range(n - 2, -1, -1):
            log_beta[t] = _logsumexp(
                log_trans + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1
            )
        ll = float(_logsumexp(log_alpha[-1]))
        return log_alpha, log_beta, ll

    # ---------- inference ----------
    def filter_posterior(self, X: np.ndarray) -> np.ndarray:
        """Forward-filtered P(state_t | obs_{1..t}) for the *last* observation."""
        if not self.fitted:
            raise RuntimeError("HMM not fitted")
        log_b = self._log_emission(np.asarray(X, dtype=float))
        log_trans = np.log(np.maximum(self.trans, EPS))
        la = np.log(self.pi + EPS) + log_b[0]
        for t in range(1, len(X)):
            la = log_b[t] + _logsumexp(la[:, None] + log_trans, axis=0)
        la -= _logsumexp(la)
        return np.exp(la)

    def describe_states(self) -> list[str]:
        """Post-hoc human labels from each state's (mean return, mean |return|)."""
        if not self.fitted:
            return [f"state_{i}" for i in range(self.k)]
        mu_ret = self.means[:, 0]
        mu_absr = self.means[:, 1]
        vol_med = float(np.median(mu_absr))
        labels = []
        for k in range(self.k):
            vol = "high-vol" if mu_absr[k] > vol_med else "low-vol"
            thr = 0.15 * max(mu_absr[k], EPS)
            if mu_ret[k] > thr:
                drift = "up-drift"
            elif mu_ret[k] < -thr:
                drift = "down-drift"
            else:
                drift = "flat"
            labels.append(f"{vol} {drift}")
        return labels


def _logsumexp(a: np.ndarray, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + EPS)
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    elif not keepdims:
        out = float(np.squeeze(out))
    return out
