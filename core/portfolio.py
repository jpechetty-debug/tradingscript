import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import kurtosis

@dataclass
class TradeTargets:
    stop: float
    t1: float
    t2: float
    rr: float

def compute_targets(direction, close, atr, config) -> TradeTargets:
    sl_dist = config.STOP_ATR_MULT * atr
    if direction == "LONG":
        stop = round(close - sl_dist, 2)
        t1 = round(close + config.TARGET1_ATR_MULT * atr, 2)
        t2 = round(close + config.TARGET2_ATR_MULT * atr, 2)
    else:
        stop = round(close + sl_dist, 2)
        t1 = round(close - config.TARGET1_ATR_MULT * atr, 2)
        t2 = round(close - config.TARGET2_ATR_MULT * atr, 2)
    rr = round(abs(t1 - close) / sl_dist, 2) if sl_dist > 0 else 0.0
    return TradeTargets(stop, t1, t2, rr)

def calculate_kelly_size(entry, stop, prob_win, rr, daily_df, config):
    rps = abs(entry - stop)
    if rps <= 0: return 0, 0.0
    f_star = max(0.0, (prob_win * (rr + 1) - 1) / rr) if rr > 0 else 0.0
    
    rets = daily_df["Close"].pct_change().dropna().tail(config.KELLY_KURTOSIS_WINDOW).values
    ek = float(kurtosis(rets, fisher=True)) if len(rets) >= config.KELLY_KURTOSIS_MIN_OBS else config.KELLY_KURTOSIS_FALLBACK
    k_corr = 3.0 / (3.0 + np.clip(ek, 0, 20))
    
    f = f_star * config.KELLY_FRACTION * k_corr
    risk = np.clip(config.RISK_PER_TRADE_INR * f * 100, config.RISK_PER_TRADE_INR * 0.25, config.RISK_PER_TRADE_INR * config.KELLY_MAX_MULT)
    shares = max(config.KELLY_MIN_SHARES, int(risk / rps))
    return shares, round(shares * rps, 2)

def optimize_portfolio(candidates, config):
    candidates.sort(key=lambda x: x.expectancy_r, reverse=True)
    selected = []
    sector_counts = {}
    for c in candidates:
        if len(selected) >= config.PORTFOLIO_SIZE: break
        if sector_counts.get(c.sector, 0) >= config.MAX_SECTOR_PICKS: continue
        selected.append(c)
        sector_counts[c.sector] = sector_counts.get(c.sector, 0) + 1
    return selected
