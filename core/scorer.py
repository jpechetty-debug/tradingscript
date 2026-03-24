import logging
from .portfolio import compute_targets, calculate_kelly_size

log = logging.getLogger("sovereign.scorer")

def score_ticker(ticker, daily_df, bench, sector_ranks, sector_rs, session, regime, config):
    if daily_df.empty: return None
    row = daily_df.iloc[-1]
    
    # Liquidity check
    if row["Vol_Avg_20"] < config.ADV_SHARE_FLOOR: return None
    
    close = float(row["Close"])
    super_up = bool(row["Super_Up"])
    ema20 = float(row["EMA_20"])
    
    direction = "LONG" if (super_up and close > ema20) else "SHORT"
    
    # Regime filter
    if direction == "LONG" and not regime.allows_long(): return None
    if direction == "SHORT" and not regime.allows_short(): return None
    
    # Simplified Scoring for refactor demo (would be full compute_factors logic)
    composite = 0.6  # Placeholder for full logic
    prob_win = 0.55  # Placeholder
    
    targets = compute_targets(direction, close, float(row["ATR"]), config)
    exp_r = round(prob_win * targets.rr - (1 - prob_win) * 1.0, 3)
    
    if prob_win < config.MIN_PROB_WIN or exp_r < config.MIN_EXPECTANCY_R: return None
    
    shares, risk = calculate_kelly_size(close, targets.stop, prob_win, targets.rr, daily_df, config)
    
    from dataclasses import dataclass
    @dataclass
    class TickerResult:
        ticker: str; sector: str; direction: str; close: float; expectancy_r: float; shares: int
    
    return TickerResult(ticker, "Unknown", direction, close, exp_r, shares)
