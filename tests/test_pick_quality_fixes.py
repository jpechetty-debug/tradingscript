"""Regression tests for scoring/sizing edge cases."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core.config import CONFIG
from core.portfolio import calculate_kelly_size
from core.scorer import _check_probability_gates


def _candidate():
    regime = SimpleNamespace(allows_mean_reversion=lambda: True)
    return SimpleNamespace(ticker="X.NS", session="CLOSING_TREND", regime=regime, swing_plan=None)


def test_held_position_keeps_positive_but_sub_threshold_expectancy():
    cfg = replace(CONFIG, MIN_EXPECTANCY_R=0.15)
    passed, _ = _check_probability_gates(_candidate(), 0.60, 0.05, cfg, True, False, False)
    assert passed


def test_new_entry_still_requires_full_expectancy():
    cfg = replace(CONFIG, MIN_EXPECTANCY_R=0.15)
    passed, _ = _check_probability_gates(_candidate(), 0.60, 0.05, cfg, False, False, False)
    assert not passed


def test_held_position_with_negative_expectancy_is_dropped():
    cfg = replace(CONFIG, MIN_EXPECTANCY_R=0.15)
    passed, _ = _check_probability_gates(_candidate(), 0.60, -0.10, cfg, True, False, False)
    assert not passed


def test_kelly_nan_probability_does_not_size_with_zero_risk():
    df = pd.DataFrame({"Close": np.linspace(100, 110, 120)})
    shares, risk, _, _ = calculate_kelly_size(100.0, 95.0, float("nan"), 2.0, df, CONFIG)
    assert (shares == 0) == (risk == 0.0)
