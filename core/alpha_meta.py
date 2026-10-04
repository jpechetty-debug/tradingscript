"""Purged chronological meta models; fitted on closed training outcomes only."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .alpha_feeds import aware


FEATURES = ("relative_strength", "earnings", "flow", "sector", "structure")


def feature_vector(context: dict[str, Any]) -> list[float] | None:
    factors = context.get("alpha_factors", {})
    values = [factors.get(key) for key in FEATURES] + [context.get("breadth_sma50")]
    if any(v is None for v in values):
        return None
    vector = [float(v) for v in values]
    return vector if all(np.isfinite(vector)) else None


class MetaModel:
    def __init__(self, min_samples: int = 40, min_class: int = 10):
        if min_samples < 10 or min_class < 2:
            raise ValueError("Invalid model evidence minimums")
        self.min_samples = min_samples
        self.min_class = min_class
        self.models: dict[str, Any] = {}
        self.trained_until: pd.Timestamp | None = None
        self.report: dict[str, Any] = {"status": "UNFITTED"}

    def fit(self, trades: Sequence[Any], cutoff: pd.Timestamp) -> dict[str, Any]:
        """Caller supplies an embargoed observation cutoff; this method also purges overlapping labels."""
        cutoff = aware(cutoff)
        self.models = {}
        self.trained_until = None
        rows = []
        for trade in trades:
            if getattr(trade, "exit_reason", "") == "DATA_END":
                continue
            context = trade.research_context
            vector = feature_vector(context)
            if vector is None or trade.exit_date is None or not np.isfinite(trade.r_multiple):
                continue
            observed = aware(context["observed_at"])
            closed = pd.Timestamp(trade.exit_date)
            closed = (closed.tz_localize("Asia/Kolkata") if closed.tzinfo is None else closed) + pd.Timedelta(hours=18)
            if closed < observed:
                raise ValueError("Outcome closes before its signal observation")
            if observed < cutoff and closed < cutoff:
                rows.append((observed, closed, vector, int(trade.r_multiple > 0), context.get("adaptive_regime", "UNKNOWN")))
        rows.sort(key=lambda r: r[0])
        self.report = {"status": "INSUFFICIENT_TRAINING_EVIDENCE", "known_closed_samples": len(rows),
                       "cutoff": cutoff.isoformat(), "features": [*FEATURES, "breadth_sma50"],
                       "models": {}, "probability_status": "TRAINING_ONLY_ESTIMATE"}
        if len(rows) < self.min_samples:
            return self.report
        self.trained_until = cutoff
        for name in ["GLOBAL", *sorted({str(r[4]) for r in rows})]:
            cohort = rows if name == "GLOBAL" else [r for r in rows if r[4] == name]
            if len(cohort) < self.min_samples:
                continue
            split = max(1, int(len(cohort) * .8))
            boundary = cohort[split][0]
            training = [r for r in cohort[:split] if r[1] < boundary]
            validation = [r for r in cohort[split:] if r[0] >= boundary]
            if (not validation or min(sum(r[3] == label for r in training) for label in (0, 1)) < self.min_class):
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=20261004))
            model.fit([r[2] for r in training], [r[3] for r in training])
            probabilities = model.predict_proba([r[2] for r in validation])[:, 1]
            self.models[name] = model
            self.report["models"][name] = {"fit_samples": len(training), "holdout_samples": len(validation),
                                           "purged_samples": split - len(training),
                                           "chronological_brier": float(brier_score_loss([r[3] for r in validation], probabilities)),
                                           "fit_observed_through": training[-1][0].isoformat()}
        self.report["status"] = "FITTED_REQUIRES_OUT_OF_SAMPLE_VALIDATION" if "GLOBAL" in self.models else "INSUFFICIENT_TRAINING_EVIDENCE"
        return self.report

    def predict(self, context: dict[str, Any], asof: pd.Timestamp) -> float | None:
        asof = aware(asof)
        if self.trained_until is not None and asof < self.trained_until:
            raise ValueError("Cannot apply a model before its training cutoff")
        vector = feature_vector(context)
        model = self.models.get(str(context.get("adaptive_regime")), self.models.get("GLOBAL"))
        if model is None or vector is None:
            return None
        return float(model.predict_proba([vector])[0, 1])
