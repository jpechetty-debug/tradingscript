"""
tests/test_factors_continuous.py
=================================
Spec-first unit tests defining exact probe point values and continuity
properties for smooth factor curves (RSI, RVOL, ADX, StochRSI) in core/factors.py.
"""

from __future__ import annotations

import sys
import os
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Stub optional heavy dependencies ──────────────────────────────────────────
def _stub_modules() -> None:
    stubs: dict[str, types.ModuleType] = {
        "fyers_apiv3": types.ModuleType("fyers_apiv3"),
        "fyers_apiv3.fyersModel": types.ModuleType("fyers_apiv3.fyersModel"),
    }
    stubs["fyers_apiv3.fyersModel"].FyersModel = MagicMock  # type: ignore[attr-defined]
    stubs["fyers_apiv3"].fyersModel = stubs["fyers_apiv3.fyersModel"]  # type: ignore[attr-defined]
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

_stub_modules()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.factors import (  # noqa: E402
    _rsi_curve_long,
    _rsi_curve_short,
    _rvol_curve,
    _adx_curve,
    _stochrsi_curve_long,
    _stochrsi_curve_short,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spec-First Probe Point Assertions
# ─────────────────────────────────────────────────────────────────────────────

def test_rsi_curve_long_probe_points() -> None:
    """Validate exact probe points for LONG RSI scoring curve."""
    assert _rsi_curve_long(35.0) == pytest.approx(0.10, abs=1e-3)
    assert _rsi_curve_long(38.0) == pytest.approx(0.10, abs=1e-3)
    assert _rsi_curve_long(40.0) == pytest.approx(0.475, abs=1e-3)
    assert _rsi_curve_long(42.0) == pytest.approx(0.85, abs=1e-3)
    assert _rsi_curve_long(45.0) == pytest.approx(0.85, abs=1e-3)
    assert _rsi_curve_long(47.9) == pytest.approx(0.995, abs=1e-3)
    assert _rsi_curve_long(48.0) == pytest.approx(1.00, abs=1e-3)
    assert _rsi_curve_long(60.0) == pytest.approx(1.00, abs=1e-3)
    assert _rsi_curve_long(73.0) == pytest.approx(1.00, abs=1e-3)
    assert _rsi_curve_long(75.5) == pytest.approx(0.60, abs=1e-3)
    assert _rsi_curve_long(78.0) == pytest.approx(0.20, abs=1e-3)
    assert _rsi_curve_long(82.0) == pytest.approx(0.20, abs=1e-3)


def test_rsi_curve_short_probe_points() -> None:
    """Validate exact probe points for SHORT RSI scoring curve."""
    assert _rsi_curve_short(20.0) == pytest.approx(0.20, abs=1e-3)
    assert _rsi_curve_short(22.0) == pytest.approx(0.20, abs=1e-3)
    assert _rsi_curve_short(24.5) == pytest.approx(0.60, abs=1e-3)
    assert _rsi_curve_short(27.0) == pytest.approx(1.00, abs=1e-3)
    assert _rsi_curve_short(35.0) == pytest.approx(1.00, abs=1e-3)
    assert _rsi_curve_short(52.0) == pytest.approx(1.00, abs=1e-3)
    assert _rsi_curve_short(55.0) == pytest.approx(0.50, abs=1e-3)
    assert _rsi_curve_short(58.0) == pytest.approx(0.00, abs=1e-3)
    assert _rsi_curve_short(60.0) == pytest.approx(0.00, abs=1e-3)
    assert _rsi_curve_short(65.0) == pytest.approx(0.00, abs=1e-3)
    assert _rsi_curve_short(67.5) == pytest.approx(0.05, abs=1e-3)
    assert _rsi_curve_short(70.0) == pytest.approx(0.10, abs=1e-3)
    assert _rsi_curve_short(75.0) == pytest.approx(0.10, abs=1e-3)


def test_rvol_curve_probe_points() -> None:
    """Validate exact probe points for non-VCP RVOL scoring curve with smoothstep."""
    # VCP check
    assert _rvol_curve(0.50, is_vcp=True) == pytest.approx(0.40, abs=1e-3)

    # Non-VCP check
    # At RVOL=1.00: t=1/3, S(1/3)=7/27 ≈ 0.25926 -> score = 0.10 * 7/27 ≈ 0.0259
    assert _rvol_curve(0.80, is_vcp=False) == pytest.approx(0.00, abs=1e-3)
    assert _rvol_curve(0.95, is_vcp=False) == pytest.approx(0.00, abs=1e-3)
    assert _rvol_curve(1.00, is_vcp=False) == pytest.approx(0.0259, abs=1e-3)
    assert _rvol_curve(1.05, is_vcp=False) == pytest.approx(0.0741, abs=1e-3)
    assert _rvol_curve(1.10, is_vcp=False) == pytest.approx(0.10, abs=1e-3)
    assert _rvol_curve(1.55, is_vcp=False) == pytest.approx(0.55, abs=1e-3)
    assert _rvol_curve(2.00, is_vcp=False) == pytest.approx(1.00, abs=1e-3)
    assert _rvol_curve(2.50, is_vcp=False) == pytest.approx(1.00, abs=1e-3)


def test_adx_curve_probe_points() -> None:
    """Validate exact probe points for ADX strength curve."""
    assert _adx_curve(15.0) == pytest.approx(0.00, abs=1e-3)
    assert _adx_curve(18.0) == pytest.approx(0.00, abs=1e-3)
    assert _adx_curve(19.0) == pytest.approx(0.25, abs=1e-3)
    assert _adx_curve(20.0) == pytest.approx(0.50, abs=1e-3)
    assert _adx_curve(22.5) == pytest.approx(0.75, abs=1e-3)
    assert _adx_curve(25.0) == pytest.approx(1.00, abs=1e-3)
    assert _adx_curve(30.0) == pytest.approx(1.00, abs=1e-3)


def test_stochrsi_curve_probe_points() -> None:
    """Validate exact probe points for StochRSI curves (LONG and SHORT)."""
    # LONG
    assert _stochrsi_curve_long(10.0) == pytest.approx(0.00, abs=1e-3)
    assert _stochrsi_curve_long(15.0) == pytest.approx(0.00, abs=1e-3)
    assert _stochrsi_curve_long(17.5) == pytest.approx(0.50, abs=1e-3)
    assert _stochrsi_curve_long(20.0) == pytest.approx(1.00, abs=1e-3)
    assert _stochrsi_curve_long(50.0) == pytest.approx(1.00, abs=1e-3)
    assert _stochrsi_curve_long(80.0) == pytest.approx(1.00, abs=1e-3)
    assert _stochrsi_curve_long(85.0) == pytest.approx(0.35, abs=1e-3)
    assert _stochrsi_curve_long(90.0) == pytest.approx(-0.30, abs=1e-3)
    assert _stochrsi_curve_long(95.0) == pytest.approx(-0.30, abs=1e-3)

    # SHORT
    assert _stochrsi_curve_short(5.0) == pytest.approx(-0.30, abs=1e-3)
    assert _stochrsi_curve_short(10.0) == pytest.approx(-0.30, abs=1e-3)
    assert _stochrsi_curve_short(15.0) == pytest.approx(0.35, abs=1e-3)
    assert _stochrsi_curve_short(20.0) == pytest.approx(1.00, abs=1e-3)
    assert _stochrsi_curve_short(50.0) == pytest.approx(1.00, abs=1e-3)
    assert _stochrsi_curve_short(80.0) == pytest.approx(1.00, abs=1e-3)
    assert _stochrsi_curve_short(82.5) == pytest.approx(0.50, abs=1e-3)
    assert _stochrsi_curve_short(85.0) == pytest.approx(0.00, abs=1e-3)
    assert _stochrsi_curve_short(90.0) == pytest.approx(0.00, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Strict C0 Continuity Verification Across Fine Grids
# ─────────────────────────────────────────────────────────────────────────────

def test_c0_continuity_across_grids() -> None:
    """Verify max jump discontinuity across fine 0.05 step grids is <= 0.02."""
    step = 0.05

    # 1. RSI LONG (range 20 to 90)
    rsi_vals = np.arange(20.0, 90.0, step)
    long_scores = [_rsi_curve_long(x) for x in rsi_vals]
    diffs = np.abs(np.diff(long_scores))
    assert np.max(diffs) < 0.02, f"Discontinuity detected in RSI LONG: max jump {np.max(diffs)}"

    # 2. RSI SHORT (range 15 to 85)
    rsi_short_vals = np.arange(15.0, 85.0, step)
    short_scores = [_rsi_curve_short(x) for x in rsi_short_vals]
    diffs_short = np.abs(np.diff(short_scores))
    assert np.max(diffs_short) < 0.02, f"Discontinuity detected in RSI SHORT: max jump {np.max(diffs_short)}"

    # 3. RVOL (range 0.5 to 3.0)
    rvol_vals = np.arange(0.5, 3.0, 0.01)
    rvol_scores = [_rvol_curve(x, is_vcp=False) for x in rvol_vals]
    diffs_rvol = np.abs(np.diff(rvol_scores))
    assert np.max(diffs_rvol) < 0.015, f"Discontinuity detected in RVOL: max jump {np.max(diffs_rvol)}"

    # 4. ADX (range 10 to 40)
    adx_vals = np.arange(10.0, 40.0, step)
    adx_scores = [_adx_curve(x) for x in adx_vals]
    diffs_adx = np.abs(np.diff(adx_scores))
    assert np.max(diffs_adx) < 0.02, f"Discontinuity detected in ADX: max jump {np.max(diffs_adx)}"

    # 5. StochRSI LONG and SHORT (range 0 to 100)
    sk_vals = np.arange(0.0, 100.0, step)
    sk_long_scores = [_stochrsi_curve_long(x) for x in sk_vals]
    diffs_sk_l = np.abs(np.diff(sk_long_scores))
    assert np.max(diffs_sk_l) < 0.02, f"Discontinuity detected in StochRSI LONG: max jump {np.max(diffs_sk_l)}"

    sk_short_scores = [_stochrsi_curve_short(x) for x in sk_vals]
    diffs_sk_s = np.abs(np.diff(sk_short_scores))
    assert np.max(diffs_sk_s) < 0.02, f"Discontinuity detected in StochRSI SHORT: max jump {np.max(diffs_sk_s)}"
