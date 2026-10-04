from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.config import CONFIG
from core.factors import FactorScores
from core.regime import MarketRegime, MarketRegimeType
from core.scorer import score_ticker
from core.swing import completed_daily_bars, detect_swing_setup, swing_fill_size


def breakout():
    dates = pd.bdate_range("2023-01-02", periods=200)
    close = np.linspace(80, 100, 200)
    df = pd.DataFrame({"Open": close - .2, "Close": close, "High": close + .4,
                       "Low": close - .8, "ATR": 1., "EMA_20": close - .5,
                       "EMA_50": close - 2, "EMA_200": close - 5, "RVol_20": 1.3}, index=dates)
    df.loc[dates[-1], ["Open", "Close", "High", "Low"]] = [100., 100.6, 100.8, 99.8]
    bench = pd.Series(np.linspace(100, 110, 200), index=dates)
    return df, bench


def test_setup_is_versioned_causal_and_has_bounded_entries():
    df, bench = breakout()
    plan = detect_swing_setup(df, bench, CONFIG)
    assert plan is not None and plan.strategy_id == "SWING_BREAKOUT_V1"
    assert plan.stop < plan.entry_min <= plan.entry <= plan.entry_max < plan.target
    future = pd.Series([1.], index=[df.index[-1] + pd.Timedelta(days=1)])
    assert detect_swing_setup(df, pd.concat([bench, future]), CONFIG) == plan
    assert swing_fill_size(plan, plan.entry_max + .01, CONFIG) == (0, 0.)
    quantity, risk = swing_fill_size(plan, plan.entry, CONFIG)
    assert quantity * plan.entry <= CONFIG.CAPITAL_INR * CONFIG.SWING_MAX_EXPOSURE_FRACTION
    assert risk <= CONFIG.CAPITAL_INR * CONFIG.SWING_RISK_FRACTION
    assert swing_fill_size(plan, plan.entry, replace(CONFIG, CAPITAL_INR=1.)) == (0, 0.)


@pytest.mark.parametrize("issue", ["downtrend", "missing", "bad_bar", "duplicates", "weak_volume"])
def test_rejects_invalid_or_unconfirmed_setups(issue):
    df, bench = breakout()
    if issue == "downtrend":
        df.loc[df.index[-1], "EMA_200"] = 200.
    elif issue == "missing":
        df.loc[df.index[-1], "ATR"] = np.nan
    elif issue == "bad_bar":
        df.loc[df.index[-1], "High"] = 200.
    elif issue == "duplicates":
        df.index = pd.DatetimeIndex([df.index[0]] * len(df))
    else:
        df.loc[df.index[-1], "RVol_20"] = .1
        df.loc[df.index[-1], "Open"] = 101.  # No bullish pullback fallback.
    assert detect_swing_setup(df, bench, CONFIG) is None


def test_pullback_reclaim_uses_structure_and_overhead_resistance():
    df, bench = breakout()
    df.loc[df.index[-10], "High"] = 106.
    df.loc[df.index[-1], ["Open", "Close", "High", "Low"]] = [100., 100.6, 100.8, 99.8]
    plan = detect_swing_setup(df, bench, CONFIG)
    assert plan is not None and plan.strategy_id == "SWING_PULLBACK_V1"
    assert plan.target <= 106.


def test_scorer_uses_completed_daily_setup_and_fixed_risk(monkeypatch):
    df, bench = breakout()
    for column, value in {"ATR_50_mean": 1., "ATR_Pctile": 50., "RSI": 55.,
                          "ADX": 25., "MACD_Hist": .5, "Super_Up": True,
                          "Vol_Avg_20": 10_000_000., "Turnover_Avg_20": 1e10,
                          "Up_Day": 1, "Dn_Day": 0}.items():
        df[column] = value
    plan = detect_swing_setup(df, bench, CONFIG)
    assert plan is not None
    # Add a partial next-session candle which must not enter factors or targets.
    partial_date = df.index[-1] + pd.offsets.BDay()
    partial = df.iloc[[-1]].copy()
    partial.index = pd.DatetimeIndex([partial_date])
    partial["Close"] = 150.
    df = pd.concat([df, partial])
    bench = pd.concat([bench, pd.Series([1.], index=partial.index)])
    captured = []

    def factors(**kwargs):
        captured.append(kwargs)
        return FactorScores(*([.8] * 8), ic_weights={}, volume_profile=(100., 99., 101.))

    monkeypatch.setattr("core.scorer.compute_factors", factors)
    monkeypatch.setattr("core.scorer.calculate_kelly_size", lambda **kw: pytest.fail("Swing must use fixed risk"))
    regime = MarketRegime(MarketRegimeType.TREND_UP, .8, 25., 1., .9, True)
    config = replace(CONFIG, SWING_SETUP_ENABLED=True, INTRADAY_ENABLED=True, PLATT_A=100., PLATT_B=100.)
    now = partial_date.to_pydatetime().replace(hour=10)
    results = [score_ticker("RELIANCE.NS", df, bench, {}, {}, session, regime, config,
                            intraday={"live_price": 150., "above_vwap": False},
                            factor_weights={"trend": 1.}, now=now)
               for session in ("OPENING_RANGE", "MIDDAY_CHOP", "CLOSING_TREND")]
    for result in results:
        assert result is not None
        assert result.strategy_id == plan.strategy_id and result.trade_horizon == "SWING"
        assert result.entry == plan.entry and result.stop == plan.stop and result.t1 == plan.target
        assert result.signal_time == plan.signal_time
        assert result.probability_status == "HEURISTIC_UNVALIDATED" and result.prob_win > .7
        assert result.shares > 0 and result.risk_inr <= config.CAPITAL_INR * config.SWING_RISK_FRACTION
    assert len({r.prob_win for r in results if r is not None}) == 1
    for kwargs in captured:
        assert kwargs["daily_df"].index[-1] < partial_date
        assert kwargs["bench"].index[-1] < partial_date
        assert kwargs["close"] == plan.entry and kwargs["intraday"] == {}
        assert kwargs["weights"] != {"trend": 1.}


def test_market_preparation_excludes_partial_and_stale_bars(monkeypatch):
    from core.services import MarketDataService
    df, _ = breakout()
    now = df.index[-1].to_pydatetime().replace(hour=10)
    captured = []
    monkeypatch.setattr("core.services.completed_daily_bars", lambda frame, config: completed_daily_bars(frame, config, now))
    monkeypatch.setattr("core.services.add_indicators", lambda frame, config: captured.append(frame.index[-1]) or frame)
    monkeypatch.setattr("core.services.compute_sector_rs", lambda frames, bench, config: {})
    config = replace(CONFIG, SWING_SETUP_ENABLED=True)
    service = MarketDataService(fetcher=lambda tickers, cfg: {
        config.BENCHMARK: df, "RELIANCE.NS": df, "STALE.NS": df.iloc[:-2]})
    prepared = service.prepare_scan_data([], config)
    assert all(date < df.index[-1] for date in captured)
    assert prepared.bench_series.index[-1] == df.index[-2]
    assert set(prepared.processed) == {config.BENCHMARK, "RELIANCE.NS"}
    after_close = now.replace(hour=16)
    assert completed_daily_bars(df, config, after_close).index[-1] == df.index[-1]
