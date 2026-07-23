"""Thin adapter so the existing AnalysisEngine (HMM regimes + online logistic
learner) speaks the same interface as the rule-based strategies, letting
main.py drive every strategy through one uniform code path."""
from __future__ import annotations

from app.analysis.engine import AnalysisEngine
from app.analysis.model import save_learner_state

id_ = "adaptive_ml"  # avoid shadowing builtin `id`


class MLStrategyAdapter:
    id = "adaptive_ml"
    label = "Adaptive ML"

    def __init__(self, cfg, model, calibration, state_path: str):
        self.engine = AnalysisEngine(cfg, model, calibration)
        self.model = model
        self.calibration = calibration
        self.state_path = state_path
        self.min_conf = self.engine.min_conf

    def analyze(self, snap):
        return self.engine.analyze(snap)

    def learn_from_shadow(self, feats, direction, won, *, symbol=None, norm_snapshot=None) -> None:
        self.engine.learn_from_shadow(feats, direction, won, symbol=symbol,
                                      norm_snapshot=norm_snapshot)

    def learn_from_trade(self, feats, direction, won, confidence, *,
                         exit_reason=None, symbol=None, norm_snapshot=None) -> None:
        self.engine.learn_from_trade(feats, direction, won, confidence,
                                     exit_reason=exit_reason, symbol=symbol,
                                     norm_snapshot=norm_snapshot)

    @property
    def n_updates(self) -> int:
        return self.model.n_updates

    def save(self) -> None:
        save_learner_state(self.state_path, self.model, self.calibration)

    def snapshot(self, names) -> dict:
        return {"kind": "ml", "id": self.id,
                "model": self.model.snapshot(names),
                "calibration": self.calibration.snapshot()}
