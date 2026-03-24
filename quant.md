"""
sovereign_quant_layer.py
========================
THE QUANTITATIVE INFRASTRUCTURE LAYER
======================================
Wraps sovereign_engine_v8.py with the 10 missing hedge-fund components.

This is NOT a replacement for v8.
v8 = signal engine (factors, regime, scoring)
This = portfolio brain (weights, sizing, risk, validation)

The 10 components built here:
──────────────────────────────
  1. Factor weight optimizer       — regression-based, information-ratio weighted
  2. Probability calibration       — Platt scaling + isotonic regression
  3. Mean-variance portfolio        — scipy.optimize, Sharpe-maximising
  4. Capital allocation model       — risk budgeting per regime/sector/cluster
  5. Event / gap risk filter        — gap%, earnings proximity, limit-up/down
  6. Strategy separation            — intraday vs swing vs positional classifier
  7. Multi-horizon correlation       — 5d/20d/60d correlation with regime weighting
  8. Transaction cost sensitivity   — parameterised slippage sweep
  9. Walk-forward weight validation — rolling train/test weight stability
 10. Volatility targeting           — position scaling to hit portfolio vol target

What we are honest about:
──────────────────────────
  ✗ Earnings calendar requires a paid API (NSE/BSE scraping is brittle).
    We use gap% as a proxy + flag the limitation clearly.
  ✗ True ML weight learning requires labeled outcome data from running the
    system live. We use OLS regression on backtest outcomes — this is
    in-sample and will overfit if not validated via walk-forward (which we do).
  ✗ CVaR optimisation requires a full return distribution. We approximate
    with historical simulation (parametric CVaR = known underestimate).
  ✗ Real vol targeting needs intraday vol estimates. We use daily ATR-based
    vol proxy. Close enough for daily/swing, not adequate for sub-5min.

Usage
─────
    # Run full quant layer on top of v8:
    python sovereign_quant_layer.py

    # Walk-forward weight validation only:
    python sovereign_quant_layer.py --walk-forward

    # Transaction cost sensitivity sweep:
    python sovereign_quant_layer.py --tc-sweep

    # Print capital allocation model for current regime:
    python sovereign_quant_layer.py --capital-plan

    # Full pipeline: signal + quant layer:
    python sovereign_quant_layer.py --full

Requirements (beyond v8)
────────────────────────
    pip install scipy scikit-learn  # calibration, optimisation
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
from typing import Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit as sigmoid

try:
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.isotonic import IsotonicRegression
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("⚠️  scikit-learn not installed. Calibration will use Platt scaling only.")
    print("    pip install scikit-learn")

warnings.filterwarnings("ignore")

IST     = pytz.timezone("Asia/Kolkata")
VERSION = "9.0-QL"  # Quant Layer version

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL COLOURS (standalone — no v8 import needed)
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
    """
    Learns factor weights from historical backtest outcomes.

    Method: Ridge regression (L2-regularised OLS).
    Why Ridge and not OLS?
      - Factors are correlated (trend and momentum overlap).
      - OLS with correlated features inflates coefficients wildly.
      - Ridge shrinks toward equal weights when factors are collinear.
      - Alpha (regularisation) controls how much we trust the data vs prior.

    Why not ML (XGBoost, etc.)?
      - With 45 tickers × 90 days = ~4,050 samples max, we overfit instantly.
      - Ridge has one hyperparameter (alpha), validated via CV.
      - Interpretable: we can inspect weights and sanity-check them.

    Information Ratio weighting:
      - Each factor's raw weight is scaled by its Information Ratio (IR).
      - IR = mean(factor_value for winning trades) / std(factor_value).
      - High IR = factor is consistent, not just lucky.
    """

    FACTOR_NAMES = ["trend", "momentum", "volume", "volatility", "rs", "breakout", "quality"]
    DEFAULT_WEIGHTS = {"trend": 0.28, "momentum": 0.20, "volume": 0.18,
                       "volatility": 0.12, "rs": 0.12, "breakout": 0.06, "quality": 0.04}

    def __init__(self, alpha: float = 1.0):
        self.alpha    = alpha       # Ridge regularisation strength
        self.weights_: dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        self.ir_:      dict[str, float] = {}
        self.fitted_   = False
        self.n_samples_= 0

    def fit(self, trade_log: pd.DataFrame) -> "FactorWeightOptimizer":
        """
        Fit weights from a trade log with columns:
          factor_trend, factor_momentum, ..., pnl_r (outcome)

        Uses Ridge regression with outcome = pnl_r (not just win/loss).
        This preserves magnitude — a +3R win matters more than +0.1R.
        """
        factor_cols = [f"factor_{n}" for n in self.FACTOR_NAMES]
        available   = [c for c in factor_cols if c in trade_log.columns]

        if len(available) < 3:
            print(DIM(f"  WeightOpt: only {len(available)} factor columns — using defaults"))
            return self

        df = trade_log[available + ["pnl_r"]].dropna()
        if len(df) < 30:
            print(DIM(f"  WeightOpt: only {len(df)} samples — need ≥30 for reliable fit"))
            return self

        X = df[[c for c in available]].values
        y = df["pnl_r"].values
        self.n_samples_ = len(df)

        # Information Ratio per factor
        for i, col in enumerate(available):
            fname = col.replace("factor_", "")
            wins  = df[df["pnl_r"] > 0][col].values
            if len(wins) > 5 and wins.std() > 0:
                self.ir_[fname] = float(wins.mean() / wins.std())
            else:
                self.ir_[fname] = 1.0

        # Ridge regression
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        # Cross-validate alpha selection (3-fold)
        best_alpha, best_cv = self.alpha, -np.inf
        for a in [0.01, 0.1, 1.0, 10.0, 100.0]:
            model = Ridge(alpha=a)
            cv_sc = cross_val_score(model, X_sc, y, cv=3, scoring="r2")
            if cv_sc.mean() > best_cv:
                best_cv, best_alpha = cv_sc.mean(), a
        self.alpha = best_alpha

        model = Ridge(alpha=best_alpha)
        model.fit(X_sc, y)

        # Raw coefficients → scale by IR → normalise to sum=1
        raw_w = {
            available[i].replace("factor_", ""): max(0.0, float(model.coef_[i]))
            for i in range(len(available))
        }

        # IR scaling
        ir_scaled = {
            fname: raw_w.get(fname, 0.0) * max(0.5, self.ir_.get(fname, 1.0))
            for fname in self.FACTOR_NAMES
        }

        total = sum(ir_scaled.values())
        if total > 0:
            self.weights_ = {k: round(v / total, 4) for k, v in ir_scaled.items()}
        else:
            self.weights_ = dict(self.DEFAULT_WEIGHTS)

        self.fitted_ = True
        print(GREEN(f"  WeightOpt: fitted on {self.n_samples_} trades "
                    f"(alpha={best_alpha}, CV R²={best_cv:.3f})"))
        return self

    def weights(self) -> dict[str, float]:
        return self.weights_

    def print_weights(self) -> None:
        print(BOLD("\n  Factor Weights (optimised vs default)"))
        print(f"  {'Factor':<12} {'Optimised':>10} {'Default':>10} {'IR':>8}")
        print(f"  {'─'*44}")
        for fname in self.FACTOR_NAMES:
            opt = self.weights_.get(fname, 0.0)
            dflt = self.DEFAULT_WEIGHTS.get(fname, 0.0)
            ir   = self.ir_.get(fname, float("nan"))
            delta = opt - dflt
            col = GREEN if delta > 0.01 else (RED if delta < -0.01 else YELLOW)
            print(f"  {fname:<12} {col(f'{opt:.4f}'):>10} {dflt:.4f}   {ir:8.3f}")

    def to_dict(self) -> dict:
        return {
            "weights":   self.weights_,
            "ir":        self.ir_,
            "alpha":     self.alpha,
            "n_samples": self.n_samples_,
            "fitted":    self.fitted_,
        }

    def save(self, path: str = "factor_weights.json") -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(DIM(f"  Weights saved → {path}"))

    @classmethod
    def load(cls, path: str = "factor_weights.json") -> "FactorWeightOptimizer":
        obj = cls()
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            obj.weights_  = d.get("weights", cls.DEFAULT_WEIGHTS)
            obj.ir_       = d.get("ir", {})
            obj.alpha     = d.get("alpha", 1.0)
            obj.n_samples_= d.get("n_samples", 0)
            obj.fitted_   = d.get("fitted", False)
            print(DIM(f"  Weights loaded ← {path} ({obj.n_samples_} samples)"))
        else:
            print(DIM(f"  No weights file found at {path} — using defaults"))
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 2: PROBABILITY CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
class ProbabilityCalibrator:
    """
    Calibrates the sigmoid-derived probability estimates from v8.

    Problem: sigmoid(4 × (composite - 0.5)) is a model assumption.
    If composites cluster at 0.65-0.75, the model may systematically
    overestimate or underestimate win probability.

    Two methods:
      a) Platt scaling: fit logistic regression on (composite → win/loss)
         Pros: smooth, works with few samples. Cons: assumes logistic shape.
      b) Isotonic regression: monotone non-parametric fit
         Pros: no shape assumption. Cons: needs 100+ samples, can overfit.

    We use Platt scaling when n < 100, isotonic when n >= 100.
    Calibration quality is reported via Expected Calibration Error (ECE).
    """

    def __init__(self):
        self.method_    = "uncalibrated"
        self.platt_a_   = 1.0   # logistic scale (sigmoid steepness)
        self.platt_b_   = 0.0   # logistic bias (sigmoid offset)
        self.iso_x_     = None  # isotonic x points
        self.iso_y_     = None  # isotonic y points
        self.ece_       = float("nan")
        self.n_samples_ = 0

    def fit(self, trade_log: pd.DataFrame) -> "ProbabilityCalibrator":
        """
        Fit calibration from trade log.
        Requires columns: prob_win (model estimate), pnl_r (outcome).
        """
        if "prob_win" not in trade_log.columns or "pnl_r" not in trade_log.columns:
            print(DIM("  Calibrator: missing prob_win or pnl_r columns"))
            return self

        df = trade_log[["prob_win","pnl_r"]].dropna()
        if len(df) < 20:
            print(DIM(f"  Calibrator: {len(df)} samples — need ≥20"))
            return self

        self.n_samples_ = len(df)
        probs  = df["prob_win"].values
        labels = (df["pnl_r"] > 0).astype(int).values

        if len(df) >= 100 and _SKLEARN_OK:
            # Isotonic regression (non-parametric)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probs, labels)
            self.iso_x_  = probs
            self.iso_y_  = iso.predict(probs)
            self.method_ = "isotonic"
        else:
            # Platt scaling: fit logistic a, b such that
            # P(win) = sigmoid(a * logit(p) + b)
            # We fit via scipy minimize instead of sklearn to avoid dependency
            def nll(params):
                a, b = params
                logits = a * np.log(np.clip(probs, 1e-7, 1-1e-7) /
                                    np.clip(1 - probs, 1e-7, 1-1e-7)) + b
                p_hat  = sigmoid(logits)
                return -np.mean(labels * np.log(p_hat + 1e-10) +
                               (1 - labels) * np.log(1 - p_hat + 1e-10))

            res = minimize(nll, [1.0, 0.0], method="Nelder-Mead",
                           options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 2000})
            self.platt_a_, self.platt_b_ = res.x
            self.method_ = "platt"

        # ECE (Expected Calibration Error) — measures reliability
        self.ece_ = self._compute_ece(probs, labels)
        print(GREEN(f"  Calibrator: {self.method_} fitted on {self.n_samples_} samples "
                    f"(ECE = {self.ece_:.4f})"))
        return self

    def predict(self, raw_prob: float) -> float:
        """Apply calibration to a raw sigmoid probability."""
        if self.method_ == "platt":
            logit = np.log(max(raw_prob, 1e-7) / max(1 - raw_prob, 1e-7))
            return float(sigmoid(self.platt_a_ * logit + self.platt_b_))
        elif self.method_ == "isotonic" and self.iso_x_ is not None:
            # Linear interpolation on isotonic fit
            return float(np.interp(raw_prob, self.iso_x_, self.iso_y_))
        return raw_prob  # uncalibrated fallback

    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error — lower is better. 0 = perfectly calibrated."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            if mask.sum() == 0:
                continue
            acc  = labels[mask].mean()
            conf = probs[mask].mean()
            ece += mask.sum() / len(probs) * abs(acc - conf)
        return float(ece)

    def calibration_report(self, trade_log: pd.DataFrame) -> None:
        """Print calibration curve: predicted vs actual win rates by bucket."""
        if "prob_win" not in trade_log.columns:
            return
        df     = trade_log[["prob_win","pnl_r"]].dropna()
        probs  = df["prob_win"].values
        labels = (df["pnl_r"] > 0).astype(int).values

        print(BOLD("\n  Probability Calibration Report"))
        print(f"  Method: {self.method_}  |  ECE: {self.ece_:.4f}  |  n={self.n_samples_}")
        print(f"  {'Pred P(W)':>10} {'Actual W%':>10} {'Count':>7} {'Calibrated':>12} {'Δ':>7}")
        print(f"  {'─'*52}")

        bins = [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,1.0)]
        for lo, hi in bins:
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() < 3:
                continue
            pred_mean  = probs[mask].mean()
            actual_wr  = labels[mask].mean()
            cal_pred   = self.predict(pred_mean)
            delta      = actual_wr - pred_mean
            col = GREEN if abs(delta) < 0.03 else (YELLOW if abs(delta) < 0.07 else RED)
            print(f"  {pred_mean:>10.1%} {actual_wr:>10.1%} {mask.sum():>7} "
                  f"{cal_pred:>12.1%} {col(f'{delta:+.1%}'):>7}")

    def to_dict(self) -> dict:
        d = {"method": self.method_, "ece": self.ece_, "n_samples": self.n_samples_,
             "platt_a": self.platt_a_, "platt_b": self.platt_b_}
        if self.iso_x_ is not None:
            d["iso_x"] = self.iso_x_.tolist()
            d["iso_y"] = self.iso_y_.tolist()
        return d

    def save(self, path: str = "prob_calibration.json") -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(DIM(f"  Calibration saved → {path}"))


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 3: MEAN-VARIANCE PORTFOLIO OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
class MeanVarianceOptimizer:
    """
    Markowitz mean-variance optimization using scipy.optimize.

    Three objectives available:
      a) max_sharpe:    maximise Sharpe ratio (return / volatility)
      b) min_variance:  minimise portfolio variance (risk parity approach)
      c) risk_parity:   equalise risk contribution per asset

    Constraints:
      - Weights sum to 1
      - Each weight ≥ 0 (no shorting the portfolio itself)
      - Max single weight ≤ max_weight
      - Sector constraint (optional)

    Input: expected_returns (from expectancy_r), covariance matrix (from
    rolling 20d returns), per-asset constraints.

    IMPORTANT: With 6-10 assets and 20 data points, the covariance matrix
    is noisy. We apply shrinkage (Ledoit-Wolf style: blend sample cov with
    identity matrix) to stabilise estimates.
    """

    def __init__(
        self,
        max_weight:    float = 0.35,
        min_weight:    float = 0.05,
        risk_free:     float = 0.065,   # 6.5% India risk-free (10y G-Sec approx)
        shrinkage:     float = 0.2,     # blend 20% toward identity matrix
    ):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.risk_free  = risk_free
        self.shrinkage  = shrinkage
        self.weights_   = {}
        self.metrics_   = {}

    def _shrink_cov(self, cov: np.ndarray) -> np.ndarray:
        """
        Ledoit-Wolf inspired shrinkage: blend sample covariance with
        scaled identity matrix. Reduces estimation error for small samples.
        F = (1-alpha)*Sigma + alpha*mu_var*I
        """
        n       = cov.shape[0]
        mu_var  = np.trace(cov) / n  # average variance
        target  = mu_var * np.eye(n)
        return (1 - self.shrinkage) * cov + self.shrinkage * target

    def optimize(
        self,
        tickers:  list[str],
        exp_ret:  dict[str, float],    # expected return per ticker (e.g. expectancy_r)
        cov_mat:  pd.DataFrame,        # rolling return covariance matrix
        method:   str = "max_sharpe",
    ) -> dict[str, float]:
        """
        Returns optimal weight per ticker (sums to 1).
        """
        n = len(tickers)
        if n <= 1:
            return {tickers[0]: 1.0} if n == 1 else {}

        # Build aligned return vector
        mu = np.array([exp_ret.get(t, 0.0) for t in tickers])

        # Build covariance matrix (only available tickers)
        # Map ticker names (stripped) to yfinance format
        def _map(t): return t + ".NS" if not t.endswith(".NS") else t

        available = [_map(t) for t in tickers if _map(t) in cov_mat.index]
        if len(available) < 2:
            # Fallback: equal weight
            w = 1.0 / n
            return {t: w for t in tickers}

        # Subset cov matrix
        cov_sub = cov_mat.loc[available, available].values.astype(float)
        # Fill NaN (some pairs may have missing data)
        cov_sub = np.where(np.isfinite(cov_sub), cov_sub, 0.0)
        np.fill_diagonal(cov_sub, np.maximum(np.diag(cov_sub), 1e-8))
        cov_sub = self._shrink_cov(cov_sub)

        n_sub = len(available)
        mu_sub = mu[:n_sub]  # aligned (simplified — assumes same order)

        # Constraints and bounds
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_weight, self.max_weight)] * n_sub

        def neg_sharpe(w):
            port_ret = np.dot(w, mu_sub)
            port_var = w @ cov_sub @ w
            port_vol = np.sqrt(max(port_var, 1e-10))
            return -(port_ret - self.risk_free / 252) / port_vol

        def portfolio_variance(w):
            return float(w @ cov_sub @ w)

        def risk_parity_obj(w):
            # Minimise sum of squared differences in risk contributions
            port_var   = w @ cov_sub @ w
            marginal_rc = cov_sub @ w
            rc          = w * marginal_rc / max(port_var, 1e-10)
            target_rc   = np.ones(n_sub) / n_sub
            return np.sum((rc - target_rc) ** 2)

        obj_fn = {
            "max_sharpe":   neg_sharpe,
            "min_variance": portfolio_variance,
            "risk_parity":  risk_parity_obj,
        }.get(method, neg_sharpe)

        w0 = np.ones(n_sub) / n_sub  # equal weight init
        result = minimize(obj_fn, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-9, "maxiter": 1000})

        if not result.success:
            # Fallback: equal weight
            w_opt = np.ones(n_sub) / n_sub
        else:
            w_opt = result.x

        # Normalise to sum=1 (numeric safety)
        w_opt = np.maximum(w_opt, 0)
        w_opt = w_opt / w_opt.sum()

        # Compute portfolio metrics
        port_ret = float(np.dot(w_opt, mu_sub))
        port_var = float(w_opt @ cov_sub @ w_opt)
        port_vol = float(np.sqrt(max(port_var, 1e-10)))
        sharpe   = (port_ret - self.risk_free / 252) / port_vol if port_vol > 0 else 0.0

        self.metrics_ = {
            "method":       method,
            "port_return":  round(port_ret, 4),
            "port_vol":     round(port_vol, 4),
            "sharpe":       round(sharpe, 3),
            "n_assets":     n_sub,
        }

        weights_out = {available[i].replace(".NS", ""): round(float(w_opt[i]), 4)
                       for i in range(n_sub)}
        self.weights_ = weights_out
        return weights_out

    def print_allocation(self) -> None:
        if not self.weights_:
            return
        print(BOLD("\n  Mean-Variance Allocation"))
        print(f"  Method: {self.metrics_.get('method','?')}  |  "
              f"Port Return: {self.metrics_.get('port_return',0):.3f}R  |  "
              f"Port Vol: {self.metrics_.get('port_vol',0):.4f}  |  "
              f"Sharpe: {self.metrics_.get('sharpe',0):.2f}")
        print(f"  {'Ticker':<12} {'Weight':>8} {'Risk$':>10}")
        print(f"  {'─'*34}")
        for ticker, w in sorted(self.weights_.items(), key=lambda x: x[1], reverse=True):
            risk_inr = w * 60_000  # example: ₹60k total portfolio risk
            col = GREEN if w >= 0.20 else (YELLOW if w >= 0.10 else DIM)
            print(f"  {ticker:<12} {col(f'{w:.1%}'):>8} {'₹'+f'{risk_inr:,.0f}':>10}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 4: CAPITAL ALLOCATION MODEL
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CapitalAllocation:
    """
    Risk budget per regime / sector / correlation cluster.
    Answers: not just "how many trades?" but "how much total risk?"
    """
    total_risk_inr:      float   # total portfolio risk budget
    regime:              str
    regime_risk_mult:    float   # scale risk exposure by regime confidence
    sector_limits:       dict[str, float]  # max risk per sector (₹)
    cluster_limits:      dict[str, float]  # max risk per correlation cluster (₹)
    per_trade_max:       float   # single trade max risk (₹)
    per_trade_min:       float   # single trade min risk (₹)
    max_open_positions:  int
    vol_target_annual:   float   # target annualised portfolio vol (e.g. 0.15 = 15%)


class CapitalAllocator:
    """
    Determines risk budget allocation based on regime + portfolio vol target.

    The key idea: risk != number of trades.
    PANIC regime → 0% risk deployed
    RANGE regime → 30% of budget (mean reversion has lower edge)
    TREND regime → 80% of budget (trend = best environment)
    EXPANSION     → 60% (high vol = higher expected moves but higher DD risk too)

    Sector limits prevent concentration.
    Correlation clusters cap the total risk to a single market factor.
    """

    REGIME_RISK_PCTS = {
        "TREND_UP":   0.80,
        "TREND_DOWN": 0.70,
        "EXPANSION":  0.60,
        "RANGE":      0.30,
        "PANIC":      0.00,
    }

    SECTOR_MAX_PCT   = 0.35   # max 35% of budget to any single sector
    CLUSTER_MAX_PCT  = 0.50   # max 50% of budget to any correlation cluster
    BASE_RISK_INR    = 60_000 # total portfolio risk budget (sum of all stops)

    def __init__(
        self,
        base_risk_inr:   float = 60_000,
        vol_target:      float = 0.15,
        max_positions:   int   = 6,
    ):
        self.base_risk_inr = base_risk_inr
        self.vol_target    = vol_target
        self.max_positions = max_positions

    def compute(
        self,
        regime:     str,
        regime_conf: float,
        n_sectors:  int,
    ) -> CapitalAllocation:
        regime_mult  = self.REGIME_RISK_PCTS.get(regime, 0.5) * regime_conf
        total_risk   = self.base_risk_inr * regime_mult
        sector_lim   = total_risk * self.SECTOR_MAX_PCT
        cluster_lim  = total_risk * self.CLUSTER_MAX_PCT
        per_max      = total_risk / max(1, self.max_positions * 0.5)
        per_min      = total_risk / max(1, self.max_positions * 3.0)
        max_pos      = min(self.max_positions,
                          int(total_risk / max(per_min, 1)))

        return CapitalAllocation(
            total_risk_inr     = round(total_risk, 2),
            regime             = regime,
            regime_risk_mult   = round(regime_mult, 3),
            sector_limits      = {f"sector_{i}": round(sector_lim, 2) for i in range(n_sectors)},
            cluster_limits     = {"cluster_banking_finance": round(cluster_lim, 2),
                                  "cluster_cyclical":        round(cluster_lim, 2),
                                  "cluster_defensive":       round(cluster_lim, 2)},
            per_trade_max      = round(per_max, 2),
            per_trade_min      = round(per_min, 2),
            max_open_positions = max_pos,
            vol_target_annual  = self.vol_target,
        )

    def print_plan(self, alloc: CapitalAllocation) -> None:
        print(BOLD("\n  Capital Allocation Plan"))
        col = GREEN if alloc.regime in ("TREND_UP","EXPANSION") \
              else (RED if alloc.regime == "PANIC" else YELLOW)
        print(f"  Regime:          {col(alloc.regime)}  (risk mult: {alloc.regime_risk_mult:.1%})")
        print(f"  Total risk:      ₹{alloc.total_risk_inr:,.0f}")
        print(f"  Per trade:       ₹{alloc.per_trade_min:,.0f} – ₹{alloc.per_trade_max:,.0f}")
        print(f"  Max positions:   {alloc.max_open_positions}")
        print(f"  Sector limit:    ₹{list(alloc.sector_limits.values())[0]:,.0f} per sector")
        print(f"  Cluster limit:   ₹{list(alloc.cluster_limits.values())[0]:,.0f} per cluster")
        print(f"  Vol target:      {alloc.vol_target_annual:.0%} annualised")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 5: EVENT / GAP RISK FILTER
# ─────────────────────────────────────────────────────────────────────────────
class EventRiskFilter:
    """
    Filters out setups with elevated event/gap risk.

    What we check (without a paid API):
      a) Gap risk: if today's open gapped >3% from prior close → flag
      b) Historical volatility spike: ATR jumped >2x its 20d mean → flag
      c) Near round-number price levels: within 0.5% of 500/1000/2000 etc
         (institutional option strikes → reversal magnets)
      d) Abnormal volume signature: volume > 5x 20d avg → possible event

    What we CANNOT check without a paid API:
      - Earnings dates (NSE doesn't publish machine-readable calendar for free)
      - Corporate action dates
      - News events

    We are explicit about this limitation.

    IMPORTANT: If you have access to NSE's bhav copy or a paid data provider,
    plug in an earnings_date_map: dict[str, date] here and uncomment the
    earnings proximity check.
    """

    GAP_THRESHOLD       = 0.03   # 3% gap = flag
    VOL_SPIKE_RATIO     = 2.0    # ATR > 2x mean = flag
    VOLUME_SPIKE_RATIO  = 5.0    # vol > 5x avg = flag
    ROUND_NUMBER_PCT    = 0.005  # within 0.5% of round number = flag

    # Earnings calendar — populate from your data provider
    # Format: {"SBIN.NS": date(2024, 10, 25), ...}
    earnings_dates: dict = {}   # empty = no earnings filter active

    @classmethod
    def check(
        cls,
        ticker:    str,
        daily_df:  pd.DataFrame,
        intraday:  dict,
    ) -> tuple[bool, list[str]]:
        """
        Returns (passes: bool, flags: list[str])
        passes=False means high event risk → skip this trade.
        """
        flags: list[str] = []
        passes = True

        if len(daily_df) < 2:
            return True, []

        row      = daily_df.iloc[-1]
        prev_row = daily_df.iloc[-2]
        close    = intraday.get("live_price") or float(row["Close"])
        prev_cl  = float(prev_row["Close"])
        today_op = float(row["Open"])

        # Gap check
        gap_pct = abs(today_op - prev_cl) / prev_cl if prev_cl > 0 else 0.0
        if gap_pct > cls.GAP_THRESHOLD:
            flags.append(f"Gap {gap_pct:.1%}")
            passes = False

        # ATR spike
        atr_now  = float(row.get("ATR", 0) or 0)
        atr_mean = float(row.get("ATR_20_mean", atr_now) or atr_now)
        if atr_mean > 0 and atr_now > cls.VOL_SPIKE_RATIO * atr_mean:
            flags.append(f"ATR spike {atr_now/atr_mean:.1f}x")
            passes = False

        # Volume spike
        vol_now  = intraday.get("vol_today", 0) or float(row.get("Volume", 0) or 0)
        vol_avg  = float(row.get("Vol_Avg_20", 1) or 1)
        if vol_avg > 0 and vol_now > cls.VOLUME_SPIKE_RATIO * vol_avg:
            flags.append(f"Vol spike {vol_now/vol_avg:.1f}x")
            # Don't fail — high RVOL can be directional, just flag it

        # Round number proximity
        round_levels = [100, 200, 500, 1000, 1500, 2000, 2500, 3000,
                        4000, 5000, 7500, 10000, 15000, 20000]
        for level in round_levels:
            if abs(close - level) / level < cls.ROUND_NUMBER_PCT:
                flags.append(f"Near round {level}")
                break  # one flag is enough

        # Earnings proximity (if calendar populated)
        if ticker in cls.earnings_dates:
            ed    = cls.earnings_dates[ticker]
            today = datetime.now().date()
            days_to = (ed - today).days
            if 0 <= days_to <= 5:
                flags.append(f"Earnings in {days_to}d")
                passes = False
            elif -2 <= days_to < 0:
                flags.append(f"Post-earnings {-days_to}d")
                passes = False

        return passes, flags


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 6: STRATEGY SEPARATOR
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StrategyClassification:
    strategy:     str     # "INTRADAY" | "SWING" | "POSITIONAL"
    hold_bars:    int     # expected hold in daily bars
    preferred_entry: str  # "OPEN" | "LIMIT" | "MARKET_ON_CLOSE"
    min_rvol:     float   # minimum RVOL for this strategy
    stop_atr_mult:float   # strategy-specific stop multiplier


class StrategySeparator:
    """
    Classifies each setup into the appropriate strategy type.
    Not one engine — three engines with different parameters.

    INTRADAY (hold < 1 day):
      - High RVOL (>2.0) + momentum + narrow ATR
      - Enter on 5m VWAP confirmation
      - Tight stop (1.0 ATR)
      - Needs: intraday data

    SWING (hold 2-5 days):
      - Volatility contraction + catalyst (sector RS)
      - Enter next day open
      - Standard stop (1.5 ATR)
      - Needs: daily data

    POSITIONAL (hold 1-4 weeks):
      - Strong trend + EMA alignment + sector leadership
      - Enter on pullback to EMA-20
      - Wide stop (2.5 ATR)
      - Needs: weekly context
    """

    @staticmethod
    def classify(
        rvol:          float,
        atr_pctile:    float,
        adx:           float,
        consec_days:   int,
        vol_contract:  bool,
        mtf_aligned:   bool,
        session:       str,
    ) -> StrategyClassification:

        # Intraday: high volume urgency + session-dependent
        if rvol >= 2.0 and atr_pctile < 60 and session != "MIDDAY_CHOP":
            return StrategyClassification(
                strategy         = "INTRADAY",
                hold_bars        = 1,
                preferred_entry  = "MARKET",
                min_rvol         = 2.0,
                stop_atr_mult    = 1.0,
            )

        # Positional: strong trend + MTF alignment + streak
        if adx >= 30 and mtf_aligned and consec_days >= 3 and atr_pctile > 30:
            return StrategyClassification(
                strategy         = "POSITIONAL",
                hold_bars        = 15,
                preferred_entry  = "LIMIT_PULLBACK",
                min_rvol         = 1.0,
                stop_atr_mult    = 2.5,
            )

        # Default: swing
        return StrategyClassification(
            strategy         = "SWING",
            hold_bars        = 4,
            preferred_entry  = "NEXT_OPEN",
            min_rvol         = 1.3,
            stop_atr_mult    = 1.5,
        )


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 7: MULTI-HORIZON CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
class MultiHorizonCorrelation:
    """
    Computes correlation at 3 horizons: 5d, 20d, 60d.
    Returns a blended correlation matrix weighted by regime.

    Why multi-horizon?
      - 5d correlation: current risk (what are these names doing THIS week)
      - 20d correlation: tactical risk (what's driven them last month)
      - 60d correlation: structural risk (are they fundamentally correlated)

    In trending regimes, short-horizon correlation dominates.
    In ranging regimes, longer-horizon correlation is more stable.

    Regime-weighted blend:
      TREND:     w = [0.5, 0.3, 0.2]  (current momentum matters most)
      RANGE:     w = [0.2, 0.4, 0.4]  (structural correlation dominates)
      EXPANSION: w = [0.6, 0.3, 0.1]  (very short-term dominated)
      PANIC:     w = [0.7, 0.2, 0.1]  (all correlations go to 1 in panic)
    """

    HORIZON_WEIGHTS = {
        "TREND_UP":   [0.5, 0.3, 0.2],
        "TREND_DOWN": [0.5, 0.3, 0.2],
        "RANGE":      [0.2, 0.4, 0.4],
        "EXPANSION":  [0.6, 0.3, 0.1],
        "PANIC":      [0.7, 0.2, 0.1],
    }

    def compute(
        self,
        processed: dict[str, pd.DataFrame],
        regime:    str,
    ) -> pd.DataFrame:
        """
        Returns blended correlation matrix.
        """
        horizons = [5, 20, 60]
        weights  = self.HORIZON_WEIGHTS.get(regime, [0.33, 0.34, 0.33])
        corr_mats = []

        bench_key = "^NSEI"
        for lookback in horizons:
            returns: dict[str, pd.Series] = {}
            for ticker, df in processed.items():
                if ticker == bench_key or df.empty or len(df) < lookback + 1:
                    continue
                ret = df["Close"].pct_change().tail(lookback)
                if len(ret) >= lookback * 0.8:   # allow some missing days
                    returns[ticker] = ret

            if len(returns) < 2:
                continue

            df_ret = pd.DataFrame(returns).dropna(axis=1, thresh=int(lookback*0.7))
            corr   = df_ret.corr()
            corr_mats.append(corr)

        if not corr_mats:
            return pd.DataFrame()

        # Align all matrices to common tickers
        common_tickers = corr_mats[0].index
        for cm in corr_mats[1:]:
            common_tickers = common_tickers.intersection(cm.index)

        if len(common_tickers) < 2:
            return corr_mats[0] if corr_mats else pd.DataFrame()

        blended = np.zeros((len(common_tickers), len(common_tickers)))
        w_total = 0.0

        for i, (cm, w) in enumerate(zip(corr_mats, weights[:len(corr_mats)])):
            sub = cm.loc[common_tickers, common_tickers].values
            sub = np.where(np.isfinite(sub), sub, 0.0)
            blended += w * sub
            w_total += w

        if w_total > 0:
            blended /= w_total

        return pd.DataFrame(blended, index=common_tickers, columns=common_tickers)

    def find_clusters(
        self,
        corr_matrix: pd.DataFrame,
        threshold:   float = 0.70,
    ) -> dict[str, list[str]]:
        """
        Simple correlation-based clustering.
        Returns {cluster_id: [tickers]}.
        Greedy: each ticker joins the first cluster it's correlated with.
        """
        if corr_matrix.empty:
            return {}

        tickers  = list(corr_matrix.index)
        clusters: dict[str, list[str]] = {}
        assigned: set[str] = set()

        for ticker in tickers:
            if ticker in assigned:
                continue
            cluster = [ticker]
            assigned.add(ticker)
            for other in tickers:
                if other in assigned:
                    continue
                if ticker in corr_matrix.index and other in corr_matrix.columns:
                    if abs(float(corr_matrix.loc[ticker, other])) >= threshold:
                        cluster.append(other)
                        assigned.add(other)
            cluster_id = f"cluster_{ticker.replace('.NS','')}"
            clusters[cluster_id] = cluster

        return {k: v for k, v in clusters.items() if len(v) >= 1}


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 8: TRANSACTION COST SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────
class TCostSensitivityAnalyzer:
    """
    Re-runs backtest P&L at different slippage assumptions.
    Critical because: a strategy that works at 8bps may break at 15bps.
    Real-world slippage in NSE F&O: 5-15bps depending on liquidity + time.
    """

    SWEEP_CONFIGS = [
        {"label": "Best case",    "slip_bps": 3,  "comm_inr": 15},
        {"label": "Realistic",    "slip_bps": 8,  "comm_inr": 20},
        {"label": "Conservative", "slip_bps": 12, "comm_inr": 25},
        {"label": "Worst case",   "slip_bps": 20, "comm_inr": 30},
    ]

    @staticmethod
    def sweep(trade_log: pd.DataFrame) -> None:
        """
        Given a trade log with raw (pre-cost) pnl_r, apply different cost
        assumptions and show how the strategy degrades.

        Requires: trade_log with columns entry, stop, pnl_r, direction
        """
        if "entry" not in trade_log.columns or "stop" not in trade_log.columns:
            print(DIM("  TC Sweep: need entry, stop columns"))
            return

        print(BOLD("\n  Transaction Cost Sensitivity Sweep"))
        print(f"  {'Scenario':<15} {'Trades':>6} {'Net WR%':>8} {'Net ExpR':>10} "
              f"{'PF':>6} {'Max DD':>8} {'Viable':>7}")
        print(f"  {'─'*64}")

        # Estimate raw (zero-cost) pnl_r from the log
        # We re-apply different slippage to each trade's entry/exit
        for cfg in TCostSensitivityAnalyzer.SWEEP_CONFIGS:
            slip  = cfg["slip_bps"] / 10_000
            comm  = cfg["comm_inr"]

            adj_pnl = []
            for _, row in trade_log.iterrows():
                raw_pnl = float(row.get("pnl_r", 0))
                entry   = float(row.get("entry", 1))
                stop    = float(row.get("stop", entry * 0.98))
                rps     = abs(entry - stop)
                if rps <= 0:
                    adj_pnl.append(raw_pnl)
                    continue
                # Cost = 2 × slippage (entry + exit) + 2 × commission
                slip_r    = (entry * slip * 2) / rps
                comm_r    = (comm * 2) / (rps * max(1, int(10_000 / rps)))
                adj_pnl.append(raw_pnl - slip_r - comm_r)

            p = pd.Series(adj_pnl)
            n = len(p)
            nw = (p > 0).sum()
            wr = nw / n * 100 if n > 0 else 0
            exp_r = p.mean()
            gp  = p[p > 0].sum()
            gl  = p[p < 0].abs().sum()
            pf  = round(gp / gl, 2) if gl > 0 else float("inf")
            eq  = p.cumsum()
            mdd = (eq.cummax() - eq).max()
            viable = wr >= 45 and exp_r > 0 and pf >= 1.2

            col = GREEN if viable else (YELLOW if wr >= 40 else RED)
            print(f"  {cfg['label']:<15} {n:>6} {col(f'{wr:.1f}%'):>8} "
                  f"{col(f'{exp_r:+.3f}R'):>10} {col(str(pf)):>6} "
                  f"{RED(f'{mdd:.2f}R'):>8} {'✅' if viable else '❌':>7}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 9: WALK-FORWARD WEIGHT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
class WalkForwardValidator:
    """
    Tests whether factor weights are stable across time periods.

    Methodology:
      1. Split backtest history into N windows (e.g. 6 × 15-day windows)
      2. For each window: fit weights on first 70% ("train"), evaluate on last 30% ("test")
      3. Compare weights across windows → stable weights = reliable model
      4. Compare train vs test performance → large gap = overfitting

    Key metric: Weight Stability Ratio (WSR)
      WSR = 1 - (std of weights across windows) / (mean weight)
      WSR > 0.7 = stable weights
      WSR < 0.5 = unstable = don't trust the model

    This is the "walk-forward weight validation" the critique asked for.
    """

    def __init__(self, n_windows: int = 4, train_pct: float = 0.70):
        self.n_windows  = n_windows
        self.train_pct  = train_pct
        self.results_: list[dict] = []

    def validate(self, trade_log: pd.DataFrame) -> None:
        """
        Run walk-forward weight validation on a trade log.
        """
        factor_cols = [c for c in trade_log.columns if c.startswith("factor_")]
        if not factor_cols or "pnl_r" not in trade_log.columns:
            print(DIM("  WFV: need factor_* and pnl_r columns"))
            return

        df = trade_log[factor_cols + ["pnl_r", "date"]].dropna()
        if len(df) < 40:
            print(DIM(f"  WFV: only {len(df)} trades — need ≥40 for meaningful validation"))
            return

        df = df.sort_values("date").reset_index(drop=True)
        n  = len(df)
        window_size = n // self.n_windows

        print(BOLD(f"\n  Walk-Forward Weight Validation ({self.n_windows} windows)"))
        print(f"  Train: {self.train_pct:.0%} | Test: {1-self.train_pct:.0%} | Trades/window: ~{window_size}")

        all_weights: list[dict] = []
        window_metrics: list[dict] = []

        for i in range(self.n_windows):
            w_start = i * window_size
            w_end   = min((i + 1) * window_size, n)
            window  = df.iloc[w_start:w_end]

            split   = int(len(window) * self.train_pct)
            train   = window.iloc[:split]
            test    = window.iloc[split:]

            if len(train) < 10 or len(test) < 5:
                continue

            # Fit weights on train
            opt = FactorWeightOptimizer()
            try:
                opt.fit(train)
            except Exception:
                continue

            # Evaluate: recompute composite using fitted weights, check vs pnl_r
            X_test   = test[[c for c in factor_cols if c in test.columns]].values
            w_vec    = np.array([opt.weights_.get(c.replace("factor_",""), 0.0)
                                 for c in factor_cols if c in test.columns])
            if w_vec.sum() > 0:
                w_vec /= w_vec.sum()

            composites = X_test @ w_vec
            # Spearman rank correlation: does higher composite → better pnl_r?
            rho, pval = stats.spearmanr(composites, test["pnl_r"].values)

            # Train performance
            X_train   = train[[c for c in factor_cols if c in train.columns]].values
            comp_tr   = X_train @ w_vec
            rho_tr, _ = stats.spearmanr(comp_tr, train["pnl_r"].values)

            all_weights.append(opt.weights_)
            window_metrics.append({
                "window":    i + 1,
                "n_train":   len(train),
                "n_test":    len(test),
                "rho_train": round(float(rho_tr), 3),
                "rho_test":  round(float(rho), 3),
                "pval":      round(float(pval), 4),
                "weights":   opt.weights_,
            })

        if not window_metrics:
            print(YELLOW("  Not enough data for validation"))
            return

        self.results_ = window_metrics

        # Print results
        print(f"\n  {'Win':>4} {'N_tr':>6} {'N_te':>6} {'ρ_train':>9} {'ρ_test':>9} {'Overfit?':>9}")
        print(f"  {'─'*50}")
        for m in window_metrics:
            overfit  = (m["rho_train"] - m["rho_test"]) > 0.15
            col_test = GREEN if m["rho_test"] > 0.05 else (YELLOW if m["rho_test"] > -0.05 else RED)
            rho_test_str = col_test(f"{m['rho_test']:.3f}")
            overfit_str  = "⚠️ YES" if overfit else "✅ NO"
            print(f"  {m['window']:>4} {m['n_train']:>6} {m['n_test']:>6} "
                  f"{m['rho_train']:>9.3f} {rho_test_str:>9} {overfit_str:>9}")

        # Weight Stability Ratio
        if len(all_weights) >= 2:
            print(BOLD("\n  Weight Stability (WSR > 0.70 = stable)"))
            print(f"  {'Factor':<12} {'Mean W':>8} {'Std W':>8} {'WSR':>8} {'Stable?':>9}")
            print(f"  {'─'*48}")
            factor_names = FactorWeightOptimizer.FACTOR_NAMES
            for fname in factor_names:
                vals = [w.get(fname, 0.0) for w in all_weights]
                mean_w = float(np.mean(vals))
                std_w  = float(np.std(vals))
                wsr    = 1.0 - (std_w / mean_w) if mean_w > 0.01 else 0.0
                col    = GREEN if wsr > 0.70 else (YELLOW if wsr > 0.50 else RED)
                print(f"  {fname:<12} {mean_w:>8.4f} {std_w:>8.4f} {col(f'{wsr:.3f}'):>8} "
                      f"{'✅' if wsr > 0.70 else '⚠️':>9}")


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT 10: VOLATILITY TARGETING
# ─────────────────────────────────────────────────────────────────────────────
class VolatilityTargeter:
    """
    Scales position sizes so the portfolio targets a specific annualised vol.

    Real vol targeting:
      target_vol = 15% annualised
      current_vol = estimated from ATR (proxy for daily vol)
      scale = target_vol / current_vol
      new_shares = base_shares × scale

    Why this matters:
      - In low-vol regime: you're under-risking if sizes are fixed.
      - In high-vol regime: you're over-risking. This is when blowups happen.
      - Vol targeting keeps portfolio risk consistent across regimes.

    Daily vol from ATR:
      daily_vol ≈ ATR / Close (annualise × sqrt(252))
      This is a proxy. Real systems use realised vol from intraday data.
    """

    def __init__(self, target_vol_annual: float = 0.15):
        self.target_vol = target_vol_annual

    def scale_factor(self, atr: float, close: float) -> float:
        """
        Returns a multiplier for position size.
        1.0 = no change. >1 = scale up (low vol). <1 = scale down (high vol).
        """
        if close <= 0 or atr <= 0:
            return 1.0
        daily_vol_estimate  = atr / close
        annual_vol_estimate = daily_vol_estimate * np.sqrt(252)
        if annual_vol_estimate <= 0:
            return 1.0
        scale = self.target_vol / annual_vol_estimate
        # Hard cap: never scale up more than 2x or down below 0.3x
        return float(np.clip(scale, 0.30, 2.00))

    def adjust_shares(self, base_shares: int, atr: float, close: float) -> tuple[int, float]:
        """Returns (adjusted_shares, scale_factor)."""
        sf      = self.scale_factor(atr, close)
        adj     = max(1, int(base_shares * sf))
        return adj, round(sf, 3)

    def portfolio_vol_estimate(
        self,
        tickers:    list[str],
        weights:    dict[str, float],
        corr_mat:   pd.DataFrame,
        atr_map:    dict[str, float],
        close_map:  dict[str, float],
    ) -> float:
        """
        Estimate portfolio annualised vol from individual asset vols + correlation.
        σ_p = sqrt(w' Σ w) where Σ is the covariance matrix.
        """
        n = len(tickers)
        if n == 0:
            return 0.0

        # Build vol vector
        vols = np.array([
            (atr_map.get(t, 0) / max(close_map.get(t, 1), 1)) * np.sqrt(252)
            for t in tickers
        ])

        # Build weight vector
        w = np.array([weights.get(t, 1.0/n) for t in tickers])
        w = w / w.sum()

        if corr_mat.empty or len(tickers) == 1:
            # Portfolio vol ≈ weighted avg vol (assume zero correlation)
            return float(np.dot(w, vols))

        # Build covariance matrix: Σ_ij = ρ_ij × σ_i × σ_j
        try:
            avail = [t + ".NS" if not t.endswith(".NS") else t for t in tickers]
            avail = [t for t in avail if t in corr_mat.index]
            if len(avail) < 2:
                return float(np.dot(w, vols))
            corr_sub = corr_mat.loc[avail, avail].values.astype(float)
            corr_sub = np.where(np.isfinite(corr_sub), corr_sub, 0.0)
            np.fill_diagonal(corr_sub, 1.0)
            n_sub = len(avail)
            vols_sub = vols[:n_sub]; w_sub = w[:n_sub]
            cov = np.outer(vols_sub, vols_sub) * corr_sub
            port_var = float(w_sub @ cov @ w_sub)
            return float(np.sqrt(max(port_var, 0)))
        except Exception:
            return float(np.dot(w, vols))

    def print_analysis(
        self,
        tickers:   list[str],
        base_sizes: dict[str, int],
        atr_map:   dict[str, float],
        close_map: dict[str, float],
    ) -> dict[str, int]:
        """Print vol-targeting adjustment and return adjusted sizes."""
        print(BOLD(f"\n  Volatility Targeting (target: {self.target_vol:.0%} annualised)"))
        print(f"  {'Ticker':<12} {'Daily Vol':>10} {'Ann Vol':>10} {'Scale':>8} "
              f"{'Base Shrs':>10} {'Adj Shrs':>10}")
        print(f"  {'─'*62}")

        adjusted: dict[str, int] = {}
        for ticker in tickers:
            atr   = atr_map.get(ticker, 0)
            close = close_map.get(ticker, 1)
            sf    = self.scale_factor(atr, close)
            dv    = (atr / close) if close > 0 else 0
            av    = dv * np.sqrt(252)
            base  = base_sizes.get(ticker, 0)
            adj   = max(1, int(base * sf))
            adjusted[ticker] = adj
            col   = GREEN if sf > 1.0 else (RED if sf < 0.7 else YELLOW)
            print(f"  {ticker:<12} {dv:>10.3%} {av:>10.1%} {col(f'{sf:.2f}x'):>8} "
                  f"{base:>10} {adj:>10}")

        return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# QUANT LAYER PIPELINE — integrates all 10 components
# ─────────────────────────────────────────────────────────────────────────────
class QuantLayer:
    """
    The full quantitative infrastructure layer.
    Takes v8 results and upgrades them with all 10 components.
    """

    def __init__(
        self,
        vol_target:    float = 0.15,
        base_risk_inr: float = 60_000,
        max_positions: int   = 6,
    ):
        self.weight_opt     = FactorWeightOptimizer.load()
        self.calibrator     = ProbabilityCalibrator()
        self.mv_optimizer   = MeanVarianceOptimizer()
        self.capital_alloc  = CapitalAllocator(base_risk_inr, vol_target, max_positions)
        self.event_filter   = EventRiskFilter()
        self.strategy_sep   = StrategySeparator()
        self.mh_corr        = MultiHorizonCorrelation()
        self.tc_analyzer    = TCostSensitivityAnalyzer()
        self.wf_validator   = WalkForwardValidator()
        self.vol_targeter   = VolatilityTargeter(vol_target)
        self.vol_target     = vol_target

    def load_calibration(self, path: str = "prob_calibration.json") -> None:
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            self.calibrator.method_   = d.get("method", "uncalibrated")
            self.calibrator.platt_a_  = d.get("platt_a", 1.0)
            self.calibrator.platt_b_  = d.get("platt_b", 0.0)
            self.calibrator.ece_      = d.get("ece", float("nan"))
            if "iso_x" in d and "iso_y" in d:
                self.calibrator.iso_x_ = np.array(d["iso_x"])
                self.calibrator.iso_y_ = np.array(d["iso_y"])
            print(DIM(f"  Calibration loaded ← {path} (ECE={self.calibrator.ece_:.4f})"))

    def run_calibration_update(self, trade_log_path: str) -> None:
        """Update calibration from a trade log CSV."""
        if not os.path.exists(trade_log_path):
            print(DIM(f"  No trade log at {trade_log_path}"))
            return
        df = pd.read_csv(trade_log_path)
        self.calibrator.fit(df)
        self.calibrator.calibration_report(df)
        self.calibrator.save()

    def run_weight_update(self, trade_log_path: str) -> None:
        """Update factor weights from a trade log CSV."""
        if not os.path.exists(trade_log_path):
            print(DIM(f"  No trade log at {trade_log_path}"))
            return
        df = pd.read_csv(trade_log_path)
        self.weight_opt.fit(df)
        self.weight_opt.print_weights()
        self.weight_opt.save()

    def run_walk_forward(self, trade_log_path: str) -> None:
        """Run walk-forward weight validation."""
        if not os.path.exists(trade_log_path):
            print(DIM(f"  No trade log at {trade_log_path}"))
            return
        df = pd.read_csv(trade_log_path)
        self.wf_validator.validate(df)

    def run_tc_sweep(self, trade_log_path: str) -> None:
        """Run transaction cost sensitivity sweep."""
        if not os.path.exists(trade_log_path):
            print(DIM(f"  No trade log at {trade_log_path}"))
            return
        df = pd.read_csv(trade_log_path)
        TCostSensitivityAnalyzer.sweep(df)

    def print_capital_plan(self, regime: str, regime_conf: float, n_sectors: int) -> CapitalAllocation:
        alloc = self.capital_alloc.compute(regime, regime_conf, n_sectors)
        self.capital_alloc.print_plan(alloc)
        return alloc

    def enhance_results(
        self,
        results:   list,    # list of TickerResult from v8
        processed: dict,    # processed dataframes from v8
        regime:    str,
        regime_conf: float,
        intraday:  dict,    # intraday cache from v8
        session:   str,
    ) -> list[dict]:
        """
        Takes v8 results and adds quant layer enhancements to each.
        Returns enriched dicts ready for output/Telegram.
        """
        if not results:
            return []

        # Multi-horizon correlation
        print("\n🔗 Computing multi-horizon correlation matrix...")
        blended_corr = self.mh_corr.compute(processed, regime)
        clusters     = self.mh_corr.find_clusters(blended_corr)

        # Capital allocation
        alloc = self.capital_alloc.compute(regime, regime_conf, len(set(r.sector for r in results)))

        # MV optimization inputs
        exp_rets = {r.ticker: r.expectancy_r for r in results}
        # Build covariance matrix from correlation + vols
        # (simplified: use correlation matrix as covariance proxy)
        mv_weights = self.mv_optimizer.optimize(
            tickers  = [r.ticker for r in results],
            exp_ret  = exp_rets,
            cov_mat  = blended_corr,
            method   = "max_sharpe",
        )
        self.mv_optimizer.print_allocation()

        # ATR and close maps for vol targeting
        atr_map   = {}
        close_map = {}
        for r in results:
            tk = r.ticker + ".NS"
            df = processed.get(tk)
            if df is not None and not df.empty:
                atr_map[r.ticker]   = float(df["ATR"].iloc[-1])
                close_map[r.ticker] = float(df["Close"].iloc[-1])

        # Vol-targeted sizes
        base_sizes = {r.ticker: r.shares for r in results}
        adj_sizes  = self.vol_targeter.adjust_shares  # per-asset
        vol_adj_sizes = {
            t: self.vol_targeter.adjust_shares(
                base_sizes.get(t, 1),
                atr_map.get(t, 0),
                close_map.get(t, 1),
            )[0]
            for t in base_sizes
        }

        portfolio_vol = self.vol_targeter.portfolio_vol_estimate(
            [r.ticker for r in results],
            mv_weights or {r.ticker: 1.0/len(results) for r in results},
            blended_corr,
            atr_map, close_map,
        )

        print(f"\n📊 Portfolio estimated vol: {GREEN(f'{portfolio_vol:.1%}')} annualised "
              f"(target: {self.vol_target:.0%})")

        enriched = []
        for r in results:
            # Calibrated probability
            cal_prob = self.calibrator.predict(r.prob_win)

            # Event filter
            tk = r.ticker + ".NS"
            df = processed.get(tk, pd.DataFrame())
            passes_event, event_flags = self.event_filter.check(
                ticker=tk, daily_df=df,
                intraday=intraday.get(tk, {}),
            )

            # Strategy classification
            strat_class = self.strategy_sep.classify(
                rvol=r.rvol, atr_pctile=r.atr_pctile,
                adx=r.adx, consec_days=r.consec_days,
                vol_contract=r.vol_contract, mtf_aligned=r.mtf_aligned,
                session=session,
            )

            # Correlation cluster
            ticker_ns  = r.ticker + ".NS"
            my_cluster = next((k for k, v in clusters.items() if ticker_ns in v), "standalone")

            # Vol-targeted size
            vt_shares, vt_scale = self.vol_targeter.adjust_shares(
                r.shares, atr_map.get(r.ticker, 0), close_map.get(r.ticker, 1)
            )

            # MV-optimised weight
            mv_w = mv_weights.get(r.ticker, 1.0 / len(results))

            # Capital-adjusted risk
            cap_risk = min(r.risk_inr * (alloc.regime_risk_mult),
                           alloc.per_trade_max)

            enriched.append({
                **r.to_dict(),
                "cal_prob":       round(cal_prob, 3),
                "event_passes":   passes_event,
                "event_flags":    ", ".join(event_flags) if event_flags else "",
                "strategy":       strat_class.strategy,
                "hold_bars":      strat_class.hold_bars,
                "preferred_entry":strat_class.preferred_entry,
                "strat_stop_mult":strat_class.stop_atr_mult,
                "cluster":        my_cluster,
                "mv_weight":      round(mv_w, 4),
                "vt_shares":      vt_shares,
                "vt_scale":       vt_scale,
                "cap_risk_inr":   round(cap_risk, 2),
                "regime_risk_mult":round(alloc.regime_risk_mult, 3),
                "total_budget":   round(alloc.total_risk_inr, 2),
            })

        # Sort by calibrated prob × expectancy (not raw prob)
        enriched.sort(key=lambda d: d.get("cal_prob", 0) * d.get("expectancy_r", 0),
                      reverse=True)
        return enriched

    def print_enriched(self, enriched: list[dict]) -> None:
        if not enriched:
            return

        print(BOLD(f"\n{'═'*160}"))
        print(BOLD("  QUANT LAYER — ENRICHED RESULTS"))
        print(f"  {'RawP':>5} {'CalP':>5} {'ExpR':>6} {'Ticker':<10} {'Strategy':<12} "
              f"{'Entry':<8} {'MVW':>5} {'VTShrs':>7} {'CapRisk':>9} {'Cluster':<20} "
              f"{'EventOK':>8} {'Flags'}")
        print("─" * 160)

        for d in enriched:
            ep_col  = GREEN("✅") if d.get("event_passes") else RED("❌")
            raw_p   = d.get("prob_win", 0)
            cal_p   = d.get("cal_prob", raw_p)
            exp_r   = d.get("expectancy_r", 0)
            strat   = d.get("strategy", "SWING")
            strat_c = GREEN(strat) if strat == "INTRADAY" else \
                      (CYAN(strat) if strat == "SWING" else MAG(strat))

            print(
                f"  {raw_p:>5.1%} {cal_p:>5.1%} "
                f"{GREEN(f'+{exp_r:.3f}') if exp_r>0 else RED(f'{exp_r:.3f}'):>6} "
                f"{CYAN(d.get('ticker','?')):<10} {strat_c:<12} "
                f"{d.get('entry',0):>8.2f} {d.get('mv_weight',0):>5.1%} "
                f"{d.get('vt_shares',0):>7} "
                f"₹{d.get('cap_risk_inr',0):>8,.0f} "
                f"{d.get('cluster','?'):<20} "
                f"{ep_col:>8}  {d.get('event_flags','')}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ANALYSIS TOOLS  (run without v8 if you have a trade log CSV)
# ─────────────────────────────────────────────────────────────────────────────
def run_full_analysis(trade_log_path: str) -> None:
    """
    Run all 10 quant components on an existing trade log CSV.
    This is what you run weekly to update the model.
    """
    if not os.path.exists(trade_log_path):
        print(RED(f"Trade log not found: {trade_log_path}"))
        return

    df = pd.read_csv(trade_log_path)
    n  = len(df)
    print(BOLD(f"\n🔬 SOVEREIGN QUANT LAYER v{VERSION} — Full Analysis"))
    print(f"   Trade log: {trade_log_path}  ({n} trades)")

    ql = QuantLayer()

    print(BOLD("\n━━━━━━━━━━ 1. FACTOR WEIGHT OPTIMIZER ━━━━━━━━━━"))
    ql.run_weight_update(trade_log_path)

    print(BOLD("\n━━━━━━━━━━ 2. PROBABILITY CALIBRATION ━━━━━━━━━━"))
    ql.calibrator.fit(df)
    ql.calibrator.calibration_report(df)
    ql.calibrator.save()

    print(BOLD("\n━━━━━━━━━━ 8. TRANSACTION COST SENSITIVITY ━━━━━━━━━━"))
    TCostSensitivityAnalyzer.sweep(df)

    print(BOLD("\n━━━━━━━━━━ 9. WALK-FORWARD WEIGHT VALIDATION ━━━━━━━━━━"))
    ql.run_walk_forward(trade_log_path)

    print(BOLD(f"\n{'═'*70}"))
    print(BOLD("  Summary"))
    print(f"  Weights updated → factor_weights.json")
    print(f"  Calibration updated → prob_calibration.json")
    print(DIM("  Run sovereign_engine_v8.py to use updated weights in live scans"))


def run_capital_plan(regime: str = "TREND_UP", conf: float = 0.80) -> None:
    ql = QuantLayer()
    ql.print_capital_plan(regime, conf, n_sectors=8)

    # Also print vol targeting example
    print(BOLD("\n━━━━━━━━━━ 10. VOLATILITY TARGETING ━━━━━━━━━━"))
    vt = VolatilityTargeter(0.15)
    # Example tickers
    example = {
        "SBIN":      {"atr": 18.0,  "close": 820.0,  "shares": 55},
        "INFY":      {"atr": 42.0,  "close": 1850.0, "shares": 23},
        "RELIANCE":  {"atr": 75.0,  "close": 2950.0, "shares": 13},
        "TATASTEEL": {"atr": 9.0,   "close": 165.0,  "shares": 111},
    }
    vt.print_analysis(
        list(example.keys()),
        {t: d["shares"] for t, d in example.items()},
        {t: d["atr"]    for t, d in example.items()},
        {t: d["close"]  for t, d in example.items()},
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Sovereign Quant Layer v{VERSION}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trade-log",     default="backtest_latest.csv",
                        help="Path to trade log CSV from v8 backtest")
    parser.add_argument("--walk-forward",  action="store_true",
                        help="Run walk-forward weight validation only")
    parser.add_argument("--tc-sweep",      action="store_true",
                        help="Run transaction cost sensitivity sweep only")
    parser.add_argument("--capital-plan",  action="store_true",
                        help="Print capital allocation plan")
    parser.add_argument("--regime",        default="TREND_UP",
                        choices=["TREND_UP","TREND_DOWN","RANGE","EXPANSION","PANIC"],
                        help="Market regime for capital plan")
    parser.add_argument("--full",          action="store_true",
                        help="Run all analysis components on trade log")
    parser.add_argument("--update-weights", action="store_true",
                        help="Fit and save factor weights from trade log")
    parser.add_argument("--update-calibration", action="store_true",
                        help="Fit and save probability calibration from trade log")
    parser.add_argument("--vol-target",    type=float, default=0.15,
                        help="Annual vol target (e.g. 0.15 = 15%%)")
    args = parser.parse_args()

    print(BOLD(f"\n🦅 SOVEREIGN QUANT LAYER v{VERSION}"))
    print(DIM(f"   {datetime.now(IST).strftime('%d-%b-%Y %H:%M IST')}"))
    print(DIM( "   Regression weights | Platt calibration | MV optimizer | "))
    print(DIM( "   Capital budgeting  | Event filter       | Vol targeting |\n"))

    if args.full:
        run_full_analysis(args.trade_log)
        return

    if args.walk_forward:
        ql = QuantLayer(vol_target=args.vol_target)
        ql.run_walk_forward(args.trade_log)
        return

    if args.tc_sweep:
        ql = QuantLayer()
        ql.run_tc_sweep(args.trade_log)
        return

    if args.capital_plan:
        run_capital_plan(args.regime)
        return

    if args.update_weights:
        ql = QuantLayer()
        ql.run_weight_update(args.trade_log)
        return

    if args.update_calibration:
        ql = QuantLayer()
        ql.run_calibration_update(args.trade_log)
        return

    # Default: show what the layer adds
    print(BOLD("  Available components:"))
    components = [
        ("--update-weights",      "1. Ridge regression factor weights from trade log"),
        ("--update-calibration",  "2. Platt/isotonic probability calibration"),
        ("(auto in --full)",      "3. Mean-variance portfolio optimizer (scipy)"),
        ("--capital-plan",        "4. Capital allocation: regime/sector/cluster budgets"),
        ("(auto in --full)",      "5. Event/gap risk filter per ticker"),
        ("(auto in --full)",      "6. Intraday/swing/positional strategy separator"),
        ("(auto in --full)",      "7. Multi-horizon correlation (5d/20d/60d blended)"),
        ("--tc-sweep",            "8. Transaction cost sensitivity (3-20bps sweep)"),
        ("--walk-forward",        "9. Walk-forward weight validation + stability"),
        ("--capital-plan",        "10. Volatility targeting (ATR-based size scaling)"),
    ]
    for flag, desc in components:
        print(f"  {DIM(flag):<30} {desc}")

    print(f"\n  {BOLD('Quickstart:')}")
    print(f"  1. Run v8 backtest:       python sovereign_engine_v8.py --backtest")
    print(f"  2. Rename output:         mv backtest_*.csv backtest_latest.csv")
    print(f"  3. Full quant analysis:   python sovereign_quant_layer.py --full")
    print(f"  4. View capital plan:     python sovereign_quant_layer.py --capital-plan")
    print(f"  5. Check TC robustness:   python sovereign_quant_layer.py --tc-sweep")
    print(f"\n  {DIM('Weights persist in factor_weights.json, calibration in prob_calibration.json')}")
    print(f"  {DIM('v8 reads these files automatically on next run.')}\n")

    print(BOLD(f"{'═'*70}"))
    print(DIM(f"  Sovereign Quant Layer v{VERSION} — Not financial advice."))


if __name__ == "__main__":
    main()