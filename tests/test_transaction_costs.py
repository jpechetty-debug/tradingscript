"""
tests/test_transaction_costs.py
================================
Unit tests for TransactionCostModel and its integration with _realised_r /
walk_forward.

Design contract being tested
-----------------------------
1. ZERO_COST_MODEL produces identical results to the old frictionless backtest.
2. DEFAULT_COST_MODEL reduces net R vs gross R by a positive, deterministic
   amount (friction_r > 0).
3. friction_r is proportional to entry price / sl_dist as documented.
4. _realised_r returns the new 6-tuple and stores gross/net separately.
5. TradeRecord backward-compat: positional construction with 13 args still
   works (gross_r_multiple / friction_r_applied default to 0.0).
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.backtest import (
    DEFAULT_COST_MODEL,
    ZERO_COST_MODEL,
    TransactionCostModel,
    TradeRecord,
    _realised_r,
    walk_forward,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _bars(closes: list[float], *, lows_offset: float = 0.0, highs_offset: float = 0.0) -> pd.DataFrame:
    """Minimal OHLCV DataFrame."""
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    c = pd.Series(closes, index=idx)
    return pd.DataFrame({
        "Open": c, "High": c + highs_offset, "Low": c - lows_offset,
        "Close": c, "Volume": 100_000,
    })


# ── TransactionCostModel unit tests ──────────────────────────────────────────

class TestTransactionCostModel:

    def test_zero_model_has_no_friction(self):
        assert ZERO_COST_MODEL.friction_r(100.0, 2.0) == 0.0

    def test_zero_sl_dist_returns_zero(self):
        assert DEFAULT_COST_MODEL.friction_r(100.0, 0.0) == 0.0
        assert DEFAULT_COST_MODEL.friction_r(100.0, -1.0) == 0.0

    def test_friction_r_is_positive(self):
        fr = DEFAULT_COST_MODEL.friction_r(entry=100.0, sl_dist=1.5)
        assert fr > 0.0

    def test_friction_r_scales_with_entry(self):
        """Doubling the entry price should double friction_r (sl_dist held constant)."""
        fr1 = DEFAULT_COST_MODEL.friction_r(entry=100.0, sl_dist=2.0)
        fr2 = DEFAULT_COST_MODEL.friction_r(entry=200.0, sl_dist=2.0)
        assert abs(fr2 / fr1 - 2.0) < 1e-9

    def test_friction_r_inversely_scales_with_sl_dist(self):
        """Doubling sl_dist should halve friction_r (entry held constant)."""
        fr1 = DEFAULT_COST_MODEL.friction_r(entry=100.0, sl_dist=1.0)
        fr2 = DEFAULT_COST_MODEL.friction_r(entry=100.0, sl_dist=2.0)
        assert abs(fr1 / fr2 - 2.0) < 1e-9

    def test_entry_and_exit_costs_are_positive(self):
        assert DEFAULT_COST_MODEL.entry_cost_pct() > 0
        assert DEFAULT_COST_MODEL.exit_cost_pct() > 0

    def test_custom_model(self):
        """A model with only slippage rounds to expected value."""
        model = TransactionCostModel(
            slippage_pct=0.001, brokerage_pct=0.0, exchange_pct=0.0,
            sebi_pct=0.0, stt_sell_pct=0.0, stamp_buy_pct=0.0, gst_rate=0.0,
        )
        # entry = 100, sl_dist = 1.0, round-trip slippage = 2 × 0.001 × 100 / 1.0 = 0.2 R
        assert abs(model.friction_r(100.0, 1.0) - 0.2) < 1e-9

    def test_nse_defaults_reasonable(self):
        """
        At entry=100, ATR stop = 1.5 (1.5% of price) the round-trip drag
        should be in the range 0.10 – 0.20 R.  This anchors the default
        values to a plausible NSE retail scenario.
        """
        fr = DEFAULT_COST_MODEL.friction_r(entry=100.0, sl_dist=1.5)
        assert 0.10 < fr < 0.25, f"Unexpected friction: {fr}"


# ── _realised_r integration ───────────────────────────────────────────────────

class TestRealisedRWithCosts:

    def test_returns_six_tuple(self):
        bars = _bars([101, 102, 103])
        result = _realised_r("LONG", 100.0, 98.0, 106.0, bars, 10, ZERO_COST_MODEL)
        assert len(result) == 6

    def test_zero_cost_preserves_old_r(self):
        """With ZERO_COST_MODEL the net_r == gross_r."""
        bars = _bars([98.5], lows_offset=0.6)  # low = 97.9 → stop hit
        net_r, hit, _, _, gross_r, friction = _realised_r(
            "LONG", 100.0, 98.0, 106.0, bars, 10, ZERO_COST_MODEL
        )
        assert net_r == gross_r
        assert friction == 0.0

    def test_stop_hit_net_r_worse_than_minus_one(self):
        """After costs a stop-hit trade loses more than 1R."""
        bars = _bars([98.5], lows_offset=1.0)   # low touches stop
        net_r, hit, _, _, gross_r, friction = _realised_r(
            "LONG", 100.0, 98.0, 106.0, bars, 10, DEFAULT_COST_MODEL
        )
        assert gross_r == -1.0
        assert net_r < -1.0
        assert friction > 0.0

    def test_t1_hit_net_r_less_than_gross(self):
        """After costs a target-hit trade returns less than gross RR."""
        bars = _bars([106.5], highs_offset=1.0)   # high exceeds t1
        net_r, hit, _, _, gross_r, friction = _realised_r(
            "LONG", 100.0, 98.0, 106.0, bars, 10, DEFAULT_COST_MODEL
        )
        assert hit is True
        assert net_r < gross_r
        assert net_r == round(gross_r - friction, 4)

    def test_backward_compat_no_cost_model_arg(self):
        """Callers that pass no cost_model get ZERO_COST_MODEL (frictionless)."""
        bars = _bars([101, 102, 103])
        # old 5-arg signature — no cost_model
        net_r, hit, bars_held, exit_date, gross_r, friction = _realised_r(
            "LONG", entry=100.0, stop=98.0, t1=104.0, fwd_bars=bars, time_stop=10
        )
        assert net_r == gross_r
        assert friction == 0.0


# ── TradeRecord backward compat ───────────────────────────────────────────────

def test_trade_record_positional_13_args():
    """Existing positional construction (13 args) must still work."""
    ts = pd.Timestamp("2024-01-01")
    tr = TradeRecord(0, "RELIANCE.NS", "LONG", ts, ts, 2500.0, 2450.0, 2600.0,
                     0.72, 0.58, 1.8, True, 4)
    assert tr.gross_r_multiple == 0.0
    assert tr.friction_r_applied == 0.0


def test_trade_record_full_construction():
    ts = pd.Timestamp("2024-01-01")
    tr = TradeRecord(
        fold=1, ticker="TCS.NS", direction="LONG",
        entry_date=ts, exit_date=ts,
        entry=3800.0, stop=3720.0, t1=3960.0,
        composite=0.75, prob_win=0.61,
        r_multiple=1.85, hit_t1=True, bars_held=3,
        gross_r_multiple=2.0, friction_r_applied=0.15,
    )
    assert tr.r_multiple == 1.85
    assert tr.gross_r_multiple == 2.0
    assert tr.friction_r_applied == 0.15


# ── walk_forward smoke test with cost model ───────────────────────────────────

def _make_ohlcv(n: int = 250) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    c   = pd.Series([100.0 + i for i in range(n)], index=idx, dtype=float)
    return pd.DataFrame({
        "Open": c, "High": c + 1.5, "Low": c - 1.5,
        "Close": c, "Volume": pd.Series(500_000, index=idx),
    })


@pytest.mark.parametrize("model,expect_friction", [
    (ZERO_COST_MODEL, False),
    (DEFAULT_COST_MODEL, True),
])
def test_walk_forward_cost_model_smoke(model, expect_friction):
    from core.config import SystemConfig

    cfg = SystemConfig()
    bench = cfg.BENCHMARK
    raw = {bench: _make_ohlcv(), "RELIANCE.NS": _make_ohlcv()}

    result = walk_forward(raw, cfg, train_days=120, test_days=20,
                          step_days=20, cost_model=model)

    for trade in result.trades:
        if expect_friction:
            # Net R should differ from gross R by exactly friction_r_applied
            assert abs(trade.r_multiple - (trade.gross_r_multiple - trade.friction_r_applied)) < 1e-4
            assert trade.friction_r_applied >= 0.0
        else:
            assert trade.friction_r_applied == 0.0
            assert trade.r_multiple == trade.gross_r_multiple
