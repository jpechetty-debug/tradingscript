"""Causal research observations, without selecting or filtering trading signals."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import SystemConfig
from .universe import TICKER_TO_SECTOR


def percentile_bucket(value: float | None, width: int = 10) -> str:
    if value is None or not np.isfinite(value):
        return "UNKNOWN"
    lower = min(100 - width, int(max(0., value) // width) * width)
    return f"{lower:02d}-{lower + width:03d}"


def leadership_snapshot(processed: dict[str, pd.DataFrame], config: SystemConfig,
                        date: pd.Timestamp) -> dict[str, dict[str, Any]]:
    """Rank liquid available stocks BEFORE setup selection; no future bars enter ranks.

    Sector RS is a leave-one-out median of peer RS, not an investable sector index.
    Current sector mappings and snapshot constituents remain retrospective proxies.
    """
    benchmark = processed.get(config.BENCHMARK)
    if benchmark is None:
        return {}
    benchmark = benchmark.loc[:date]
    if benchmark.empty or benchmark.index[-1] != date:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for ticker, full in processed.items():
        if ticker == config.BENCHMARK:
            continue
        frame = full.loc[:date]
        if frame.empty or frame.index[-1] != date or len(frame) < 50 or "Volume" not in frame:
            continue
        turnover = float((frame.Close * frame.Volume).tail(20).mean())
        volume = float(frame.Volume.tail(20).mean())
        if not np.isfinite([turnover, volume]).all() or turnover < config.ADV_TURNOVER_FLOOR or volume < config.ADV_SHARE_FLOOR:
            continue
        close = frame.Close.reindex(benchmark.index)
        rs: dict[str, float | None] = {}
        for lookback in (63, 252):
            values = close.iloc[-lookback - 1:]
            market = benchmark.Close.iloc[-lookback - 1:]
            valid = len(values) == lookback + 1 and np.isfinite(values).all() and (values > 0).all() and (market > 0).all()
            rs[f"rs_{lookback}"] = float((values.iloc[-1] / values.iloc[0]) / (market.iloc[-1] / market.iloc[0]) - 1) if valid else None
        row = frame.iloc[-1]
        ema200 = float(row.get("EMA_200", np.nan))
        ema50 = float(row.get("EMA_50", np.nan))
        true_range = pd.concat([frame.High - frame.Low, (frame.High - frame.Close.shift()).abs(),
                                (frame.Low - frame.Close.shift()).abs()], axis=1).max(axis=1)
        atr20 = true_range.ewm(alpha=1 / 20, adjust=False, min_periods=20).mean()
        previous_atr = atr20.iloc[-253:-1].dropna()
        atr_rank = float((previous_atr < atr20.iloc[-1]).mean() * 100) if len(previous_atr) >= 60 else None
        expansion_base = atr20.iloc[-21:-1].mean()
        bandwidth = 4 * frame.Close.rolling(20).std(ddof=0) / frame.Close.rolling(20).mean()
        rows[ticker] = {**rs, "sector": TICKER_TO_SECTOR.get(ticker, "UNKNOWN"),
                        "turnover_inr": turnover, "above_ema50": bool(row.Close > ema50) if np.isfinite(ema50) else None,
                        "above_ema200": bool(row.Close > ema200) if np.isfinite(ema200) else None,
                        "distance_ema200": float(row.Close / ema200 - 1) if np.isfinite(ema200) and ema200 > 0 else None,
                        "atr20_percentile": atr_rank,
                        "atr20_expansion": float(atr20.iloc[-1] / expansion_base) if expansion_base > 0 else None,
                        "bandwidth_expanding": bool(bandwidth.iloc[-1] > bandwidth.iloc[-2]) if np.isfinite(bandwidth.iloc[-2:]).all() else None}
    if not rows:
        return {}
    for lookback in (63, 252):
        ranks = pd.Series({t: r[f"rs_{lookback}"] for t, r in rows.items()}, dtype=float).rank(method="average", pct=True) * 100
        for ticker, row in rows.items():
            row[f"rs_{lookback}_percentile"] = float(ranks[ticker]) if np.isfinite(ranks[ticker]) else None
            row[f"rs_{lookback}_rank_count"] = int(ranks.notna().sum())
    for ticker, row in rows.items():
        peers = [r["rs_63"] for t, r in rows.items() if t != ticker and r["sector"] == row["sector"]
                 and r["rs_63"] is not None and row["sector"] != "UNKNOWN"]
        row["sector_peer_count"] = len(peers)
        row["sector_rs_63"] = float(np.median(peers)) if len(peers) >= 2 else None
        for span in (50, 200):
            valid = [r[f"above_ema{span}"] for r in rows.values() if r[f"above_ema{span}"] is not None]
            row[f"breadth_ema{span}"] = float(np.mean(valid)) if valid else None
            row[f"breadth_ema{span}_denominator"] = len(valid)
        row["rank_universe_count"] = len(rows)
        row["benchmark_above_ema200"] = bool(benchmark.Close.iloc[-1] > benchmark.EMA_200.iloc[-1]) if "EMA_200" in benchmark and np.isfinite(benchmark.EMA_200.iloc[-1]) else None
        row["market_date"] = date.isoformat()
        row["schema"] = "SWING_LEADERSHIP_V1"
        row["protocol_version"] = "SWING_STRUCTURAL_V1"
        row["universe_status"] = "AVAILABLE_LIQUID_SNAPSHOT_NOT_POINT_IN_TIME"
        row["earnings_status"] = "UNKNOWN_NO_ASOF_EVENT_FEED"
    return rows


def trade_excursions(trade: Any, frame: pd.DataFrame) -> dict[str, Any]:
    """Bounds in initial-risk units. Never call exit-day extremes pre-exit facts.

    Prior-session OHLC extremes are observed while held. Exit-session extremes
    are only upper bounds, except opening exits where exposure ends at the open.
    """
    distance = float(trade.entry - trade.stop)
    if distance <= 0 or trade.direction != "LONG":
        raise ValueError("Excursions currently require a positive-risk long trade")
    held = frame.loc[pd.Timestamp(trade.entry_date):pd.Timestamp(trade.exit_date)]
    if held.empty or held.index[0] != pd.Timestamp(trade.entry_date) or held.index[-1] != pd.Timestamp(trade.exit_date):
        raise ValueError("Trade holding-period bars are missing")
    exit_bar = held.iloc[-1]
    at_open = trade.exit_reason in ("STOP", "TARGET", "TRAIL_STOP") and abs(float(trade.exit_price) - float(exit_bar.Open)) < 1e-8
    known = held.iloc[:-1]
    favorable = [0., (float(trade.exit_price) - trade.entry) / distance]
    adverse = [0., (trade.entry - float(trade.exit_price)) / distance]
    if len(known):
        favorable.append(float((known.High.max() - trade.entry) / distance))
        adverse.append(float((trade.entry - known.Low.min()) / distance))
    lower_mfe, lower_mae = max(favorable), max(adverse)
    # TIME/DATA_END exits occur at the close, so that entire session is held.
    whole_day = trade.exit_reason in ("TIME", "DATA_END")
    upper_mfe = max(lower_mfe, float((exit_bar.High - trade.entry) / distance)) if not at_open else lower_mfe
    upper_mae = max(lower_mae, float((trade.entry - exit_bar.Low) / distance)) if not at_open else lower_mae
    return {"mfe_r_lower": upper_mfe if whole_day else lower_mfe, "mfe_r_upper": upper_mfe,
            "mae_r_lower": upper_mae if whole_day else lower_mae, "mae_r_upper": upper_mae,
            "exit_day_order_unknown": not (at_open or whole_day), "initial_risk_per_share": distance}
