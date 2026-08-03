
import os
import logging
from dotenv import load_dotenv
from core.runtime_components import RegimeAwareTelegramAlerter, RegimeProbabilityGate, TieredCapitalScaler
from core.regime import MarketRegime, MarketRegimeType
from core.scorer import TickerResult, FactorScores

# Replicate logging to see errors
logging.basicConfig(level=logging.INFO)

def test_live_alert():
    load_dotenv()

    print(f"Token: {os.getenv('TELEGRAM_BOT_TOKEN')}")
    print(f"Chat ID: {os.getenv('TELEGRAM_CHAT_ID')}")

    # Mock data for a fake alert
    fake_ticker = TickerResult(
        ticker="TEST_TICKER",
        sector="TEST_SECTOR",
        direction="LONG",
        close=100.0,
        change_pct=1.5,
        factors=FactorScores(
            trend=0.8, momentum=0.8, volume=0.8, volatility=0.8, rs=0.8, breakout=0.8, quality=0.8,
            composite=0.7, ic_weights={"trend": 1/7, "momentum": 1/7, "volume": 1/7, "volatility": 1/7, "rs": 1/7, "breakout": 1/7, "quality": 1/7}
        ),
        composite=0.7,
        prob_win=0.65,
        expectancy_r=1.2,
        sharpe_rank=2.0,
        entry=100.0,
        stop=95.0,
        t1=110.0,
        t2=120.0,
        breakeven=105.0,
        trail_stop=98.0,
        time_stop_bars=5,
        shares=10,
        risk_inr=50.0,
        rr_t1=2.0,
        kelly_f=0.05,
        kurt_correction=1.0,
        excess_kurtosis=0.0,
        rsi=60.0,
        stochrsi_k=70.0,
        rvol=1.5,
        adx=30.0,
        super_up=True,
        macd_hist=0.5,
        atr_pctile=50.0,
        vol_contract=False,
        rs_vs_nifty=1.2,
        near_52w=True,
        ema200_aligned=True,
        mtf_aligned=True,
        consec_days=3,
        poc=100.0,
        val=98.0,
        vah=102.0,
        regime="TREND_UP",
        session="OPENING_RANGE",
        reasons=["Test Reason"]
    )

    regime = MarketRegime(
        regime=MarketRegimeType.TREND_UP,
        breadth=0.65,
        adx_median=25.0,
        atr_ratio=1.1,
        confidence=0.8,
        confirmed=True,
        breadth_delta=0.02,
        sector_concentration=0.3
    )

    gate = RegimeProbabilityGate()
    scaler = TieredCapitalScaler(portfolio_peak=1000000.0)
    alerter = RegimeAwareTelegramAlerter(gate=gate, scaler=scaler)

    print("Attempting to send alert...")
    success = alerter.send_daily_summary(
        regime=regime.regime.value,
        top_picks=[fake_ticker.__dict__],
        current_nav=875000.0
    )
    print(f"Alert sent: {success}")

if __name__ == "__main__":
    test_live_alert()
