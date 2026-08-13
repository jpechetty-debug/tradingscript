import os
import sys
from dotenv import load_dotenv

# Add current dir to path to import screener
sys.path.append(os.getcwd())
import screener

def test_telegram():
    load_dotenv()

    # Create a mock TickerResult
    mock_factors = screener.FactorScores(
        trend=0.9, momentum=0.8, volume=0.7,
        volatility=0.6, rs=0.8, breakout=0.5,
        quality=0.9, composite=0.85
    )

    res = screener.TickerResult(
        ticker="TEST",
        sector="TEST_SECTOR",
        direction="LONG",
        close=100.0,
        change_pct=2.5,
        factors=mock_factors,
        prob_win=0.75,
        expectancy_r=0.5,
        composite=0.85,
        display_score=85,
        regime="TREND_UP",
        rsi=65.0,
        stochrsi_k=70.0,
        rvol=2.0,
        adx=30.0,
        super_up=True,
        macd_hist=0.5,
        vol_contract=False,
        rs_vs_nifty=2.0,
        near_52w=True,
        ema200_aligned=True,
        mtf_aligned=True,
        consec_days=3,
        poc=98.0,
        val=95.0,
        vah=102.0,
        atr_pctile=45,
        entry=100.0,
        stop=95.0,
        t1=110.0,
        t2=120.0,
        breakeven=97.5,
        trail_stop=98.0,
        time_stop_bars=5,
        shares=100,
        risk_inr=500.0,
        rr_t1=2.0,
        reasons=["TEST ALERT", "BOT CONNECTED"],
        sector_rs_rank=1,
        sector_rs_pct=2.5
    )

    regime = screener.MarketRegime("TREND_UP", 0.6, 25.0, 1.1, 0.8)

    print("Testing Telegram alert with mock data...")
    success = screener.send_telegram_alert([res], "TEST_SESSION", regime)

    if success:
        print("✅ Test alert sent successfully!")
    else:
        print("❌ Failed to send test alert. Check .env and console for details.")

if __name__ == "__main__":
    test_telegram()
