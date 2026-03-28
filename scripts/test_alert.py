import logging
from _bootstrap import ensure_repo_root

ensure_repo_root()

from core.config import CONFIG
from core.regime import MarketRegime, MarketRegimeType
from core.scorer import TickerResult
from core.factors import FactorScores
from screener_v14_modular import _send_alert

logging.basicConfig(level=logging.INFO)

mock_factors = FactorScores(
    trend=0.8, momentum=0.9, volume=0.7, volatility=0.6,
    rs=0.8, breakout=0.5, quality=0.7, composite=0.75,
    ic_weights={"trend": 1/7, "momentum": 1/7, "volume": 1/7, "volatility": 1/7, "rs": 1/7, "breakout": 1/7, "quality": 1/7}
)

mock_r = TickerResult(
    ticker="MOCK.NS",
    sector="MockSector",
    direction="LONG",
    close=100.0,
    change_pct=2.0,
    factors=mock_factors,
    composite=0.75,
    prob_win=0.85,
    expectancy_r=1.5,
    sharpe_rank=0.5,
    entry=100.0,
    stop=95.0,
    t1=110.0,
    t2=120.0,
    breakeven=105.0,
    trail_stop=98.0,
    time_stop_bars=5,
    shares=100,
    risk_inr=500.0,
    rr_t1=2.0,
    kelly_f=0.1,
    kurt_correction=1.0,
    excess_kurtosis=0.0,
    rsi=60.0, stochrsi_k=50.0, rvol=1.5, adx=30.0,
    super_up=True, macd_hist=0.5, atr_pctile=40.0,
    vol_contract=False, rs_vs_nifty=5.0, near_52w=True,
    ema200_aligned=True, mtf_aligned=True, consec_days=3,
    poc=100.0, val=98.0, vah=102.0,
    regime="TREND_UP", session="MIDDAY",
    reasons=["Mock Signal"]
)

mock_reg = MarketRegime(
    regime=MarketRegimeType.TREND_UP,
    breadth=0.65,
    adx_median=30.0,
    atr_ratio=1.1,
    confidence=0.75,
    confirmed=True
)

print("Sending mock alert...")
_send_alert([mock_r], mock_reg, CONFIG)
print("Done.")
