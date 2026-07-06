"""Online probabilistic learner + honesty (calibration) tracking.

- OnlineLogistic: SGD logistic regression on z-scored features predicting
  P(trade hits TP before SL). Updated after every closed trade.
- CalibrationTracker: rolling Brier score + reliability buckets, so the
  system knows whether its stated probabilities have been honest lately.
  Confidence is shrunk toward 0.5 when recent calibration is poor.

State is persisted to disk so learning survives restarts.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

EPS = 1e-9


class OnlineLogistic:
    def __init__(self, n_features: int, lr: float = 0.03, l2: float = 1e-3):
        self.n = n_features
        self.lr = lr
        self.l2 = l2
        self.w = np.zeros(n_features)
        self.b = 0.0
        # running z-score normalization (Welford)
        self.count = 0
        self.mean = np.zeros(n_features)
        self.m2 = np.zeros(n_features)
        self.n_updates = 0

    def _normalize(self, x: np.ndarray, update_stats: bool) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if update_stats:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (x - self.mean)
        if self.count > 1:
            std = np.sqrt(self.m2 / (self.count - 1))
            return (x - self.mean) / np.maximum(std, EPS)
        return x - self.mean

    def predict_proba(self, x: np.ndarray) -> float:
        z = self._normalize(x, update_stats=False)
        s = float(self.w @ z + self.b)
        return float(1.0 / (1.0 + np.exp(-np.clip(s, -30, 30))))

    def update(self, x: np.ndarray, y: int) -> None:
        z = self._normalize(x, update_stats=True)
        p = 1.0 / (1.0 + np.exp(-np.clip(float(self.w @ z + self.b), -30, 30)))
        g = p - float(y)
        self.w -= self.lr * (g * z + self.l2 * self.w)
        self.b -= self.lr * g
        self.n_updates += 1

    def contributions(self, x: np.ndarray, names: list[str], top: int = 5):
        """Signed per-feature contribution to the logit (explainability)."""
        z = self._normalize(x, update_stats=False)
        contrib = self.w * z
        order = np.argsort(-np.abs(contrib))[:top]
        return [(names[i], float(contrib[i])) for i in order]

    # ---------- persistence ----------
    def to_dict(self) -> dict:
        return {
            "w": self.w.tolist(), "b": self.b, "count": self.count,
            "mean": self.mean.tolist(), "m2": self.m2.tolist(),
            "n_updates": self.n_updates, "n": self.n,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OnlineLogistic":
        m = cls(d["n"])
        m.w = np.array(d["w"]); m.b = d["b"]; m.count = d["count"]
        m.mean = np.array(d["mean"]); m.m2 = np.array(d["m2"])
        m.n_updates = d.get("n_updates", 0)
        return m


class CalibrationTracker:
    def __init__(self, window: int = 60):
        self.window = window
        self.records: deque[tuple[float, int]] = deque(maxlen=window)

    def record(self, prob: float, outcome: int) -> None:
        self.records.append((float(prob), int(outcome)))

    def brier(self) -> float | None:
        if len(self.records) < 5:
            return None
        return float(np.mean([(p - y) ** 2 for p, y in self.records]))

    def calibration_score(self) -> float:
        """1.0 = perfectly honest lately, 0.0 = as bad as always saying 0.5-worse.

        Maps rolling Brier onto [0,1]: Brier 0.25 (uninformative) -> ~0.5.
        """
        b = self.brier()
        if b is None:
            return 0.7  # mild optimism until there is history
        return float(np.clip(1.0 - 2.0 * b, 0.0, 1.0))

    def shrink(self, prob: float) -> float:
        """Shrink a raw probability toward 0.5 by recent calibration quality."""
        c = self.calibration_score()
        return 0.5 + (prob - 0.5) * (0.4 + 0.6 * c)

    def reliability_buckets(self, n_buckets: int = 5):
        out = []
        if not self.records:
            return out
        arr = np.array(self.records)
        edges = np.linspace(0, 1, n_buckets + 1)
        for i in range(n_buckets):
            mask = (arr[:, 0] >= edges[i]) & (arr[:, 0] < edges[i + 1] + (i == n_buckets - 1))
            if mask.sum() > 0:
                out.append({
                    "bucket": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                    "n": int(mask.sum()),
                    "avg_predicted": float(arr[mask, 0].mean()),
                    "actual_win_rate": float(arr[mask, 1].mean()),
                })
        return out

    def to_dict(self) -> dict:
        return {"window": self.window, "records": list(self.records)}

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationTracker":
        t = cls(d.get("window", 60))
        for p, y in d.get("records", []):
            t.records.append((float(p), int(y)))
        return t


def save_learner_state(path: str | Path, model: OnlineLogistic, cal: CalibrationTracker) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: refuse to persist NaN/Inf weights. A non-finite weight
    # would otherwise reload verbatim and poison sizing permanently (sticky).
    Path(path).write_text(json.dumps({
        "model": model.to_dict(), "calibration": cal.to_dict(),
    }, allow_nan=False))


def load_learner_state(path: str | Path, n_features: int):
    p = Path(path)
    if not p.exists():
        return OnlineLogistic(n_features), CalibrationTracker()
    try:
        d = json.loads(p.read_text())
        model = OnlineLogistic.from_dict(d["model"])
        if model.n != n_features:  # feature schema changed -> start fresh
            return OnlineLogistic(n_features), CalibrationTracker()
        # Reject a corrupted/poisoned state rather than trade on garbage.
        if not (np.all(np.isfinite(model.w)) and np.isfinite(model.b)
                and np.all(np.isfinite(model.mean)) and np.all(np.isfinite(model.m2))):
            return OnlineLogistic(n_features), CalibrationTracker()
        return model, CalibrationTracker.from_dict(d["calibration"])
    except Exception:
        return OnlineLogistic(n_features), CalibrationTracker()
