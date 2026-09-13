import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from core.database import SqliteDatabase
from core.config import SystemConfig, MarketRegimeType
from core.regime import MarketRegime
from core.runtime_paths import RuntimePaths
from core.services import PersistenceService
from core.scorer import score_ticker, score_candidate_pass2, CandidateContext, TickerResult
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


def _make_dummy_ticker_result(
    ticker: str,
    sector: str = "IT",
    sharpe_rank: float = 1.0,
    is_held: bool = False,
    reasons: list[str] | None = None,
) -> TickerResult:
    reasons = reasons or (["HeldPos"] if is_held else [])
    factors = FactorScores(
        trend=0.6, momentum=0.6, volume=0.6, volatility=0.6,
        rs=0.6, breakout=0.6, quality=0.6, composite=0.6,
        ic_weights=dict(DEFAULT_WEIGHTS),
    )
    return TickerResult(
        ticker=ticker,
        sector=sector,
        direction="LONG",
        close=100.0,
        change_pct=1.0,
        factors=factors,
        composite=0.6,
        prob_win=0.55 if not is_held else 0.49,
        expectancy_r=0.3,
        sharpe_rank=sharpe_rank,
        entry=100.0,
        stop=95.0,
        t1=110.0,
        t2=115.0,
        breakeven=101.0,
        trail_stop=96.0,
        time_stop_bars=5,
        shares=10,
        risk_inr=50.0,
        rr_t1=2.0,
        kelly_f=0.05,
        kurt_correction=1.0,
        excess_kurtosis=0.0,
        rsi=55.0,
        stochrsi_k=60.0,
        rvol=1.2,
        adx=25.0,
        super_up=True,
        macd_hist=0.5,
        atr_pctile=50.0,
        vol_contract=False,
        rs_vs_nifty=1.0,
        near_52w=True,
        ema200_aligned=True,
        mtf_aligned=True,
        consec_days=3,
        poc=100.0,
        val=98.0,
        vah=102.0,
        regime="TREND_UP",
        session="OPENING_RANGE",
        reasons=reasons,
        is_held=is_held,
    )


class TestPass1Hysteresis:
    def test_intraday_cutoff_blocks_new_candidate_allows_held_position(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        from core.scorer import score_candidate_pass1

        IST = ZoneInfo("Asia/Kolkata")

        cand = _make_dummy_candidate(ticker="INTRADAY.NS")
        df = cand.daily_df.copy()
        df["EMA_20"] = df["Close"] * 1.05
        df["EMA_200"] = df["Close"] * 1.15
        df["Super_Up"] = False
        df["Dn_Day"] = 1
        df["Up_Day"] = 0

        regime = MarketRegime(
            regime=MarketRegimeType.TREND_DOWN,
            breadth=0.30,
            adx_median=25.0,
            atr_ratio=1.0,
            confidence=0.85,
            confirmed=True,
        )

        config = SystemConfig()
        config.SHORT_IS_INTRADAY_ONLY = True
        config.INTRADAY_ENTRY_CUTOFF = "14:30"

        # Evaluation at 14:45 IST during CLOSING_TREND
        eval_dt = datetime(2024, 1, 15, 14, 45, tzinfo=IST)

        # 1. New candidate: blocked by 14:30 entry cutoff
        res_new = score_candidate_pass1(
            ticker="INTRADAY.NS",
            daily_df=df,
            bench=cand.bench,
            sector_ranks=cand.sector_ranks,
            sector_rs=cand.sector_rs,
            session="CLOSING_TREND",
            regime=regime,
            config=config,
            now=eval_dt,
            is_open_position=False,
        )
        assert res_new is None

        # 2. Held position: exempt from entry cutoff
        res_held = score_candidate_pass1(
            ticker="INTRADAY.NS",
            daily_df=df,
            bench=cand.bench,
            sector_ranks=cand.sector_ranks,
            sector_rs=cand.sector_rs,
            session="CLOSING_TREND",
            regime=regime,
            config=config,
            now=eval_dt,
            is_open_position=True,
            held_direction="SHORT",
        )
        assert res_held is not None
        assert res_held.ticker == "INTRADAY.NS"
        assert res_held.trade_horizon == "INTRADAY"

    def test_regime_block_exits_held_position(self):
        from core.scorer import score_candidate_pass1

        cand = _make_dummy_candidate(ticker="HELD_EXIT.NS")
        config = SystemConfig()

        # Regime that forbids LONG
        bear_regime = MarketRegime(
            regime=MarketRegimeType.TREND_DOWN,
            breadth=0.2,
            adx_median=30.0,
            atr_ratio=1.2,
            confidence=0.9,
            confirmed=True,
        )

        res = score_candidate_pass1(
            ticker="HELD_EXIT.NS",
            daily_df=cand.daily_df,
            bench=cand.bench,
            sector_ranks=cand.sector_ranks,
            sector_rs=cand.sector_rs,
            session="OPENING_RANGE",
            regime=bear_regime,
            config=config,
            is_open_position=True,
            held_direction="LONG",
        )
        assert res is None


class TestPortfolioHysteresisCoordination:
    def test_held_position_prioritized_over_higher_sharpe_new_candidate(self):
        from core.portfolio import optimize_portfolio

        config = SystemConfig()
        config.PORTFOLIO_SIZE = 1

        held = _make_dummy_ticker_result(ticker="HELD", sharpe_rank=0.8, is_held=True)
        new_c = _make_dummy_ticker_result(ticker="NEW", sharpe_rank=2.5, is_held=False)

        selected = optimize_portfolio([held, new_c], config)
        assert len(selected) == 1
        assert selected[0].ticker == "HELD"

    def test_held_positions_exempt_from_sector_cap(self):
        from core.portfolio import optimize_portfolio

        config = SystemConfig()
        config.PORTFOLIO_SIZE = 5
        config.MAX_SECTOR_PICKS = 2

        # 3 held positions in Energy (already exceeding cap of 2)
        h1 = _make_dummy_ticker_result(ticker="H1", sector="Energy", sharpe_rank=1.0, is_held=True)
        h2 = _make_dummy_ticker_result(ticker="H2", sector="Energy", sharpe_rank=1.1, is_held=True)
        h3 = _make_dummy_ticker_result(ticker="H3", sector="Energy", sharpe_rank=1.2, is_held=True)

        # 1 new candidate in Energy (should be blocked) and 1 new in IT (should be admitted)
        new_energy = _make_dummy_ticker_result(ticker="NEW_ENG", sector="Energy", sharpe_rank=2.0, is_held=False)
        new_it = _make_dummy_ticker_result(ticker="NEW_IT", sector="IT", sharpe_rank=1.5, is_held=False)

        selected = optimize_portfolio([h1, h2, h3, new_energy, new_it], config)
        tickers = [r.ticker for r in selected]
        assert "H1" in tickers
        assert "H2" in tickers
        assert "H3" in tickers
        assert "NEW_ENG" not in tickers  # blocked by sector cap
        assert "NEW_IT" in tickers       # admitted

    def test_held_positions_exempt_from_mutual_correlation(self):
        from core.portfolio import optimize_portfolio

        config = SystemConfig()
        config.PORTFOLIO_SIZE = 5
        config.MAX_CORR = 0.70

        h1 = _make_dummy_ticker_result(ticker="AAA", is_held=True)
        h2 = _make_dummy_ticker_result(ticker="BBB", is_held=True)
        new_c = _make_dummy_ticker_result(ticker="CCC", is_held=False)

        # AAA and BBB have corr=0.95 (both held). AAA and CCC have corr=0.85 (CCC is new).
        corr = pd.DataFrame(
            [
                [1.00, 0.95, 0.85],
                [0.95, 1.00, 0.20],
                [0.85, 0.20, 1.00],
            ],
            index=["AAA.NS", "BBB.NS", "CCC.NS"],
            columns=["AAA.NS", "BBB.NS", "CCC.NS"],
        )

        selected = optimize_portfolio([h1, h2, new_c], config, corr_matrix=corr)
        tickers = [r.ticker for r in selected]
        assert "AAA" in tickers
        assert "BBB" in tickers          # held positions are exempt from mutual correlation
        assert "CCC" not in tickers      # new candidate dropped due to correlation with AAA

    def test_held_count_exceeds_portfolio_size_retains_all_held(self):
        from core.portfolio import optimize_portfolio

        config = SystemConfig()
        config.PORTFOLIO_SIZE = 2

        h1 = _make_dummy_ticker_result(ticker="H1", sharpe_rank=1.0, is_held=True)
        h2 = _make_dummy_ticker_result(ticker="H2", sharpe_rank=1.1, is_held=True)
        h3 = _make_dummy_ticker_result(ticker="H3", sharpe_rank=1.2, is_held=True)
        new_c = _make_dummy_ticker_result(ticker="NEW", sharpe_rank=2.0, is_held=False)

        selected = optimize_portfolio([h1, h2, h3, new_c], config)
        assert len(selected) == 3        # all 3 active positions retained
        assert "NEW" not in [r.ticker for r in selected]  # 0 new admitted


class TestStopLossAndTargetMonitoring:
    def test_stop_loss_trigger_removes_from_db_and_state(self):
        from core.services import ScanService

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = RuntimePaths.discover(root=Path(tmp_dir))
            ps = PersistenceService(paths=paths)

            # Insert an open position with stop_loss = 2500.0
            ps.save_open_positions([
                {"ticker": "RELIANCE.NS", "direction": "LONG", "entry": 2550.0, "shares": 10, "stop": 2500.0, "t1": 2650.0, "prob_win": 0.55, "composite": 0.60}
            ])

            config = SystemConfig()
            state = ps.create_scan_state(config)
            assert "RELIANCE.NS" in state.open_positions

            # Processed data has Low=2480.0 (breaching 2500 stop)
            df = pd.DataFrame({"Close": [2490.0], "Low": [2480.0], "High": [2510.0]})
            processed = {"RELIANCE.NS": df}

            scan_svc = ScanService(version="14.6.0", persistence=ps)
            scan_svc._monitor_open_position_stops(processed, state)

            # Position must be removed from DB and state
            assert "RELIANCE.NS" not in state.open_positions
            assert ps.load_open_positions() == {}

    def test_target_trigger_removes_from_db_and_state(self):
        from core.services import ScanService

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = RuntimePaths.discover(root=Path(tmp_dir))
            ps = PersistenceService(paths=paths)

            # Insert an open position with target = 2650.0
            ps.save_open_positions([
                {"ticker": "TCS.NS", "direction": "LONG", "entry": 2500.0, "shares": 10, "stop": 2400.0, "t1": 2650.0, "prob_win": 0.55, "composite": 0.60}
            ])

            config = SystemConfig()
            state = ps.create_scan_state(config)
            assert "TCS.NS" in state.open_positions

            # Processed data has High=2680.0 (reaching target)
            df = pd.DataFrame({"Close": [2660.0], "Low": [2490.0], "High": [2680.0]})
            processed = {"TCS.NS": df}

            scan_svc = ScanService(version="14.6.0", persistence=ps)
            scan_svc._monitor_open_position_stops(processed, state)

            assert "TCS.NS" not in state.open_positions
            assert ps.load_open_positions() == {}

    def test_healthy_position_retained(self):
        from core.services import ScanService

        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = RuntimePaths.discover(root=Path(tmp_dir))
            ps = PersistenceService(paths=paths)

            ps.save_open_positions([
                {"ticker": "INFY.NS", "direction": "LONG", "entry": 1500.0, "shares": 10, "stop": 1450.0, "t1": 1600.0, "prob_win": 0.55, "composite": 0.60}
            ])

            config = SystemConfig()
            state = ps.create_scan_state(config)

            df = pd.DataFrame({"Close": [1520.0], "Low": [1480.0], "High": [1550.0]})
            processed = {"INFY.NS": df}

            scan_svc = ScanService(version="14.6.0", persistence=ps)
            scan_svc._monitor_open_position_stops(processed, state)

            assert "INFY.NS" in state.open_positions
            assert "INFY.NS" in ps.load_open_positions()
