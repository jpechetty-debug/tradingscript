import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from core.database import SqliteDatabase
from core.config import SystemConfig, MarketRegimeType
from core.regime import MarketRegime
from core.runtime_paths import RuntimePaths
from core.services import PersistenceService
from core.scorer import score_ticker, score_candidate_pass2, CandidateContext
from core.factors import FactorScores, DEFAULT_WEIGHTS


def _make_dummy_candidate(
    ticker: str = "TEST.NS",
    direction: str = "LONG",
    composite: float = 0.51,
) -> CandidateContext:
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Open": np.linspace(100, 110, n),
        "High": np.linspace(101, 112, n),
        "Low": np.linspace(99, 109, n),
        "Close": np.linspace(100, 110, n),
        "Volume": np.full(n, 1_000_000.0),
        "ATR": np.full(n, 2.0),
        "ATR_Pctile": np.full(n, 50.0),
        "ATR_50_mean": np.full(n, 2.0),
        "EMA_20": np.full(n, 105.0),
        "EMA_50": np.full(n, 103.0),
        "EMA_200": np.full(n, 100.0),
        "Super_Up": np.full(n, True),
        "Supertrend": np.full(n, 98.0),
        "RSI": np.full(n, 55.0),
        "ADX": np.full(n, 25.0),
        "MACD_Hist": np.full(n, 0.5),
        "BB_Width": np.full(n, 0.04),
        "BB_Squeeze": np.full(n, False),
        "StochRSI_K": np.full(n, 60.0),
        "StochRSI_D": np.full(n, 55.0),
        "Vol_Avg_20": np.full(n, 1_000_000.0),
        "Turnover_Avg_20": np.full(n, 100_000_000.0),
        "Up_Day": np.full(n, 1),
        "Dn_Day": np.full(n, 0),
        "RVol_20": np.full(n, 1.2),
        "Change_Pct": np.full(n, 1.0),
    }, index=dates)

    bench = pd.Series(np.linspace(20000, 21000, n), index=dates)
    regime = MarketRegime(
        regime=MarketRegimeType.TREND_UP,
        breadth=0.8,
        adx_median=25.0,
        atr_ratio=1.0,
        confidence=0.8,
        confirmed=True,
    )
    factors = FactorScores(
        trend=composite,
        momentum=composite,
        volume=composite,
        volatility=composite,
        rs=composite,
        breakout=composite,
        quality=composite,
        composite=composite,
        ic_weights=dict(DEFAULT_WEIGHTS),
    )

    return CandidateContext(
        ticker=ticker,
        sector="IT",
        direction=direction,
        close=float(df["Close"].iloc[-1]),
        daily_df=df,
        bench=bench,
        sector_ranks={"IT": 1},
        sector_rs={"IT": 0.05},
        session="OPENING_RANGE",
        regime=regime,
        trade_horizon="SWING",
        factors=factors,
        capital_fraction=1.0,
        intraday={},
        row=df.iloc[-1],
    )


class TestDatabaseOpenPositions:
    def test_crud_open_positions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "state.db"
            db = SqliteDatabase(db_path)

            # Initially empty
            assert db.fetch_open_positions() == {}

            # Insert position
            db.upsert_open_position(
                ticker="RELIANCE.NS",
                direction="LONG",
                entry_price=2500.0,
                shares=10,
                stop_loss=2450.0,
                target=2600.0,
                prob_win=0.55,
                composite=0.60,
            )

            positions = db.fetch_open_positions()
            assert "RELIANCE.NS" in positions
            pos = positions["RELIANCE.NS"]
            assert pos["ticker"] == "RELIANCE.NS"
            assert pos["direction"] == "LONG"
            assert pos["shares"] == 10
            assert pos["entry_price"] == 2500.0

            # Update position
            db.upsert_open_position(
                ticker="RELIANCE.NS",
                direction="LONG",
                entry_price=2520.0,
                shares=15,
                stop_loss=2470.0,
                target=2650.0,
                prob_win=0.58,
                composite=0.62,
            )
            positions = db.fetch_open_positions()
            assert positions["RELIANCE.NS"]["shares"] == 15
            assert positions["RELIANCE.NS"]["entry_price"] == 2520.0

            # Delete position
            deleted = db.delete_open_position("RELIANCE.NS")
            assert deleted is True
            assert db.fetch_open_positions() == {}

    def test_sync_open_positions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "state.db"
            db = SqliteDatabase(db_path)

            initial_positions = [
                {"ticker": "TCS.NS", "direction": "LONG", "entry": 3500.0, "shares": 5, "stop": 3400.0, "t1": 3700.0, "prob_win": 0.54, "composite": 0.58},
                {"ticker": "INFY.NS", "direction": "LONG", "entry": 1500.0, "shares": 20, "stop": 1450.0, "t1": 1600.0, "prob_win": 0.53, "composite": 0.56},
            ]
            db.sync_open_positions(initial_positions)
            stored = db.fetch_open_positions()
            assert set(stored.keys()) == {"TCS.NS", "INFY.NS"}

            # Sync with new list: TCS removed, HDFCBANK added
            updated_positions = [
                {"ticker": "INFY.NS", "direction": "LONG", "entry": 1510.0, "shares": 22, "stop": 1460.0, "t1": 1610.0, "prob_win": 0.55, "composite": 0.59},
                {"ticker": "HDFCBANK.NS", "direction": "LONG", "entry": 1600.0, "shares": 10, "stop": 1550.0, "t1": 1700.0, "prob_win": 0.52, "composite": 0.54},
            ]
            db.sync_open_positions(updated_positions)
            stored2 = db.fetch_open_positions()
            assert set(stored2.keys()) == {"INFY.NS", "HDFCBANK.NS"}
            assert stored2["INFY.NS"]["shares"] == 22

            # Sync with empty list deletes all
            db.sync_open_positions([])
            assert db.fetch_open_positions() == {}


class TestHysteresisGating:
    def test_candidate_with_prob_49_rejected_if_new_accepted_if_open(self):
        config = SystemConfig()
        config.MIN_PROB_WIN = 0.52
        config.PROB_HOLD_FLOOR = 0.47
        config.MIN_EXPECTANCY_R = 0.0

        # With session="OPENING_RANGE" (mult=1.0), PLATT_A=-4.0, PLATT_B=2.0:
        # composite=0.510 -> prob_win = 1/(1+exp(-(-4*0.51+2))) = 1/(1+exp(0.04)) = 0.4900
        cand = _make_dummy_candidate(ticker="INFY.NS", composite=0.510)

        # Case 1: is_open_position=False -> rejected because prob < 0.52
        result_new = score_candidate_pass2(candidate=cand, config=config, is_open_position=False)
        assert result_new is None

        # Case 2: is_open_position=True -> accepted because prob >= 0.47
        result_open = score_candidate_pass2(candidate=cand, config=config, is_open_position=True)
        assert result_open is not None
        assert 0.47 <= result_open.prob_win < 0.52
        assert "HeldPos" in result_open.reasons

    def test_score_ticker_passes_is_open_position(self):
        config = SystemConfig()
        config.PLATT_A = 4.0
        config.PLATT_B = -2.0
        config.MIN_EXPECTANCY_R = 0.0

        cand = _make_dummy_candidate(ticker="INFY.NS", composite=0.510)
        df = cand.daily_df

        # First score with is_open_position=True to retrieve actual prob_win
        baseline = score_ticker(
            ticker="INFY.NS",
            daily_df=df,
            bench=cand.bench,
            sector_ranks=cand.sector_ranks,
            sector_rs=cand.sector_rs,
            session=cand.session,
            regime=cand.regime,
            config=config,
            is_open_position=True,
        )
        assert baseline is not None
        actual_prob = baseline.prob_win

        # Set thresholds so actual_prob is strictly between PROB_HOLD_FLOOR and MIN_PROB_WIN
        config.MIN_PROB_WIN = round(actual_prob + 0.02, 3)
        config.PROB_HOLD_FLOOR = round(actual_prob - 0.02, 3)

        res_new = score_ticker(
            ticker="INFY.NS",
            daily_df=df,
            bench=cand.bench,
            sector_ranks=cand.sector_ranks,
            sector_rs=cand.sector_rs,
            session=cand.session,
            regime=cand.regime,
            config=config,
            is_open_position=False,
        )
        assert res_new is None

        res_open = score_ticker(
            ticker="INFY.NS",
            daily_df=df,
            bench=cand.bench,
            sector_ranks=cand.sector_ranks,
            sector_rs=cand.sector_rs,
            session=cand.session,
            regime=cand.regime,
            config=config,
            is_open_position=True,
        )
        assert res_open is not None
        assert res_open.prob_win == actual_prob
        assert "HeldPos" in res_open.reasons

    def test_persistence_service_loads_open_positions_into_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = RuntimePaths.discover(root=Path(tmp_dir))
            ps = PersistenceService(paths=paths)

            # Insert an open position
            ps.save_open_positions([
                {"ticker": "TCS.NS", "direction": "LONG", "entry": 3500.0, "shares": 5, "stop": 3400.0, "t1": 3700.0, "prob_win": 0.54, "composite": 0.58}
            ])

            config = SystemConfig()
            state = ps.create_scan_state(config)
            assert "TCS.NS" in state.open_positions
