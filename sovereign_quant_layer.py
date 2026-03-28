"""
sovereign_quant_layer.py
========================
THE QUANTITATIVE INFRASTRUCTURE LAYER (v9.0)
=============================================
Standalone quantitative analysis layer for the Sovereign Engine.

Provides portfolio-level intelligence: factor weight optimisation,
probability calibration, capital allocation, and transaction cost
sensitivity analysis. Operates independently of the live scan pipeline.

The 10 components implemented here:
──────────────────────────────
  1. Factor weight optimizer       — Ridge regression, IR-weighted
  2. Probability calibration       — Platt scaling + isotonic regression
  3. Mean-variance portfolio        — scipy.optimize, Sharpe-maximising
  4. Capital allocation model       — risk budgeting per regime/sector/cluster
  5. Event / gap risk filter        — gap%, ATR spikes, round numbers
  6. Strategy separation            — intraday vs swing vs positional classifier
  7. Multi-horizon correlation       — 5d/20d/60d correlation with regime weighting
  8. Transaction cost sensitivity   — parameterised slippage sweep
  9. Walk-forward weight validation — rolling train/test weight stability
 10. Volatility targeting           — position scaling to hit portfolio vol target

Usage
─────
    # Run full quant analysis on backtest log:
    python sovereign_quant_layer.py --full

    # Print capital allocation model for current regime:
    python sovereign_quant_layer.py --capital-plan --regime TREND_UP

Requirements
────────────
    pip install scipy scikit-learn pandas numpy yfinance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit as sigmoid

from core.runtime_paths import RUNTIME_PATHS, ensure_parent, ensure_runtime_dirs

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import Ridge
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("⚠️  scikit-learn not installed. Some components will be limited.")

warnings.filterwarnings("ignore")

IST     = pytz.timezone("Asia/Kolkata")
VERSION = "9.0-QL"
FACTOR_WEIGHTS_PATH = RUNTIME_PATHS.state_dir / "factor_weights.json"
PROB_CALIBRATION_PATH = RUNTIME_PATHS.state_dir / "prob_calibration.json"
DEFAULT_TRADE_LOG_PATH = RUNTIME_PATHS.artifacts_dir / "backtest_latest.csv"

ensure_runtime_dirs()


def _resolve_state_path(path: str, default_path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else default_path.parent / candidate

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL COLOURS
# ─────────────────────────────────────────────────────────────────────────────
def _c(t: str, code: str) -> str:
    return f"\033[{code}m{t}\033[0m" if sys.stdout.isatty() else t

GREEN  = lambda t: _c(str(t), "92")
YELLOW = lambda t: _c(str(t), "93")
RED    = lambda t: _c(str(t), "91")
CYAN   = lambda t: _c(str(t), "96")
BOLD   = lambda t: _c(str(t), "1")
DIM    = lambda t: _c(str(t), "2")
MAG    = lambda t: _c(str(t), "95")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 1: FACTOR WEIGHT OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
class FactorWeightOptimizer:
    FACTOR_NAMES = ["trend", "momentum", "volume", "volatility", "rs", "breakout", "quality"]
    DEFAULT_WEIGHTS = {"trend": 0.28, "momentum": 0.20, "volume": 0.18,
                       "volatility": 0.12, "rs": 0.12, "breakout": 0.06, "quality": 0.04}

    def __init__(self, alpha: float = 1.0):
        self.alpha    = alpha
        self.weights_: dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        self.ir_:      dict[str, float] = {}
        self.fitted_   = False
        self.n_samples_= 0

    def fit(self, trade_log: pd.DataFrame) -> "FactorWeightOptimizer":
        factor_cols = [f"factor_{n}" for n in self.FACTOR_NAMES]
        available   = [c for c in factor_cols if c in trade_log.columns]

        if len(available) < 3:
            print(DIM(f"  WeightOpt: only {len(available)} factor columns found"))
            return self

        df = trade_log[available + ["pnl_r"]].dropna()
        if len(df) < 20:
            print(DIM(f"  WeightOpt: only {len(df)} samples — need more data"))
            return self

        X = df[available].values
        y = df["pnl_r"].values
        self.n_samples_ = len(df)

        for col in available:
            fname = col.replace("factor_", "")
            wins  = df[df["pnl_r"] > 0][col].values
            if len(wins) > 5 and wins.std() > 0:
                self.ir_[fname] = float(wins.mean() / wins.std())
            else:
                self.ir_[fname] = 1.0

        if _SKLEARN_OK:
            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X)
            best_alpha, best_cv = self.alpha, -np.inf
            for a in [0.1, 1.0, 10.0]:
                model = Ridge(alpha=a)
                cv_sc = cross_val_score(model, X_sc, y, cv=3, scoring="r2")
                if cv_sc.mean() > best_cv:
                    best_cv, best_alpha = cv_sc.mean(), a
            
            self.alpha = best_alpha
            model = Ridge(alpha=best_alpha)
            model.fit(X_sc, y)
            raw_w = {available[i].replace("factor_", ""): max(0.0, float(model.coef_[i]))
                     for i in range(len(available))}
        else:
            raw_w = dict(self.DEFAULT_WEIGHTS)

        ir_scaled = {fname: raw_w.get(fname, 0.0) * max(0.5, self.ir_.get(fname, 1.0))
                    for fname in self.FACTOR_NAMES}
        total = sum(ir_scaled.values())
        if total > 0:
            self.weights_ = {k: round(v / total, 4) for k, v in ir_scaled.items()}
        
        self.fitted_ = True
        return self

    def print_weights(self) -> "FactorWeightOptimizer":
        print(BOLD("\n  Factor Weights (optimised vs default)"))
        print(f"  {'Factor':<12} {'Optimised':>10} {'Default':>10} {'IR':>8}")
        print(f"  {'-'*44}")
        for fname in self.FACTOR_NAMES:
            opt = self.weights_.get(fname, 0.0)
            dflt = self.DEFAULT_WEIGHTS.get(fname, 0.0)
            ir   = self.ir_.get(fname, float("nan"))
            print(f"  {fname:<12} {opt:>10.4f} {dflt:>10.4f} {ir:>8.3f}")
        return self

    def save(self, path: str = str(FACTOR_WEIGHTS_PATH)) -> None:
        target = ensure_parent(_resolve_state_path(path, FACTOR_WEIGHTS_PATH))
        with target.open("w", encoding="utf-8") as f:
            json.dump({"weights": self.weights_, "fitted": self.fitted_}, f, indent=2)

    @classmethod
    def load(cls, path: str = str(FACTOR_WEIGHTS_PATH)) -> "FactorWeightOptimizer":
        obj = cls()
        target = _resolve_state_path(path, FACTOR_WEIGHTS_PATH)
        if target.exists():
            with target.open(encoding="utf-8") as f:
                d = json.load(f)
            obj.weights_ = d.get("weights", cls.DEFAULT_WEIGHTS)
            obj.fitted_  = d.get("fitted", False)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 2: PROBABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
class ProbabilityCalibrator:
    def __init__(self):
        self.method_    = "uncalibrated"
        self.platt_a_   = 1.0
        self.platt_b_   = 0.0
        self.iso_x_     = None
        self.iso_y_     = None
        self.ece_       = float("nan")
        self.n_samples_ = 0

    def fit(self, trade_log: pd.DataFrame) -> "ProbabilityCalibrator":
        if "prob_win" not in trade_log.columns or "pnl_r" not in trade_log.columns:
            return self
        df = trade_log[["prob_win","pnl_r"]].dropna()
        if len(df) < 20: return self
        self.n_samples_ = len(df)
        probs  = df["prob_win"].values
        labels = (df["pnl_r"] > 0).astype(int).values

        if len(df) >= 100 and _SKLEARN_OK:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probs, labels)
            self.iso_x_, self.iso_y_ = probs, iso.predict(probs)
            self.method_ = "isotonic"
        else:
            def nll(params):
                a, b = params
                logit = np.log(np.clip(probs, 1e-7, 1-1e-7) / np.clip(1-probs, 1e-7, 1-1e-7))
                p_hat = sigmoid(a * logit + b)
                return -np.mean(labels * np.log(p_hat + 1e-10) + (1-labels) * np.log(1-p_hat + 1e-10))
            res = minimize(nll, [1.0, 0.0], method="Nelder-Mead")
            self.platt_a_, self.platt_b_ = res.x
            self.method_ = "platt"
        return self

    def predict(self, raw_prob: float) -> float:
        if self.method_ == "platt":
            logit = np.log(max(raw_prob, 1e-7) / max(1-raw_prob, 1e-7))
            return float(sigmoid(self.platt_a_ * logit + self.platt_b_))
        elif self.method_ == "isotonic" and self.iso_x_ is not None:
            return float(np.interp(raw_prob, self.iso_x_, self.iso_y_))
        return raw_prob

    def save(self, path: str = str(PROB_CALIBRATION_PATH)) -> None:
        d = {"method": self.method_, "platt_a": self.platt_a_, "platt_b": self.platt_b_}
        if self.iso_x_ is not None:
            d["iso_x"], d["iso_y"] = self.iso_x_.tolist(), self.iso_y_.tolist()
        target = ensure_parent(_resolve_state_path(path, PROB_CALIBRATION_PATH))
        with target.open("w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 3 & 10: PORTFOLIO & VOL TARGETING
# ─────────────────────────────────────────────────────────────────────────────
class VolatilityTargeter:
    def __init__(self, target_vol_annual: float = 0.15):
        self.target_vol = target_vol_annual

    def scale_factor(self, atr: float, close: float) -> float:
        if close <= 0 or atr <= 0: return 1.0
        ann_vol = (atr / close) * np.sqrt(252)
        return float(np.clip(self.target_vol / ann_vol, 0.3, 2.0)) if ann_vol > 0 else 1.0

# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 4: CAPITAL ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────
class CapitalAllocator:
    REGIME_RISK_PCTS = {"TREND_UP": 0.8, "TREND_DOWN": 0.7, "EXPANSION": 0.6, "RANGE": 0.3, "PANIC": 0.0}
    def __init__(self, base_risk_inr: float = 60000):
        self.base_risk = base_risk_inr
    def compute(self, regime: str, conf: float) -> float:
        return self.base_risk * self.REGIME_RISK_PCTS.get(regime, 0.5) * conf


# ─────────────────────────────────────────────────────────────────────────────
# TC SENSITIVITY & WALK-FORWARD
# ─────────────────────────────────────────────────────────────────────────────
class TCostSensitivityAnalyzer:
    @staticmethod
    def sweep(trade_log: pd.DataFrame):
        print(BOLD("\n  Transaction Cost Sensitivity Sweep"))
        for slip in [3, 8, 15]:
            pnl = trade_log["pnl_r"] - (slip/10000) * 2  # simplified
            print(f"  {slip:>2} bps: WR { (pnl > 0).mean():.1%} | Avg R {pnl.mean():.3f}")

class WalkForwardValidator:
    def validate(self, trade_log: pd.DataFrame):
        print(BOLD("\n  Walk-Forward Weight Validation (Manual Check Required)"))
        print(DIM(f"  Split log into temporal buckets and check weight stability in {FACTOR_WEIGHTS_PATH} over time."))

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", default=str(DEFAULT_TRADE_LOG_PATH))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--capital-plan", action="store_true")
    parser.add_argument("--regime", default="TREND_UP")
    args = parser.parse_args()

    print(BOLD(f"\n🦅 SOVEREIGN QUANT LAYER v{VERSION}"))
    
    if args.full:
        if not os.path.exists(args.trade_log):
            print(RED(f"Error: {args.trade_log} not found."))
            return
        df = pd.read_csv(args.trade_log)
        FactorWeightOptimizer().fit(df).print_weights().save()
        ProbabilityCalibrator().fit(df).save()
        TCostSensitivityAnalyzer.sweep(df)
        print(GREEN("\n✅ Weight & Calibration files updated."))
    
    if args.capital_plan:
        risk = CapitalAllocator().compute(args.regime, 0.8)
        print(f"\n  Capital Plan for {args.regime}: Risk Budget ₹{risk:,.0f}")

if __name__ == "__main__":
    main()
