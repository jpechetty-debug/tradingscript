"""Chronological cash book for the standalone LP/PEAD scheduled opportunities.

This does not generate signals or infer intraday paths. The caller supplies an
explicit OPEN, INTRADAY or CLOSE exit phase. Sizing uses the previous close NAV;
only actual opening exits release cash before opening entries. Quantities remain
fractional, matching the standalone research experiment rather than broker lots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScheduledReplay:
    equity: pd.DataFrame
    fills: pd.DataFrame
    decisions: pd.DataFrame


def replay_scheduled_trades(
    trades: pd.DataFrame,
    history: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    setups: Sequence[str],
    *,
    cost: float = 0.004,
    risk: float = 0.005,
    cap: float = 0.10,
    slots: int = 20,
) -> ScheduledReplay:
    """Replay one opening entry batch, then later exit phases, each session.

    Required columns: setup, tk, sig, ent, ext, e, x, st, prio, exit_phase.
    Prices/exit dates in an opportunity are scheduled execution inputs, never
    inputs to earlier sizing. A rejected opportunity is not retried later.
    Missing closes carry the last observed mark; future closes are never used.
    """
    if not np.isfinite([cost, risk, cap]).all() or not (0 <= cost < 2 and 0 < risk <= 1 and 0 < cap <= 1):
        raise ValueError("Invalid cost, risk or position cap")
    if slots < 1 or not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError("Replay requires positive slots and ordered unique sessions")
    required = {"setup", "tk", "sig", "ent", "ext", "e", "x", "st", "prio", "exit_phase"}
    if not required.issubset(trades.columns):
        raise ValueError(f"Missing scheduled trade columns: {sorted(required - set(trades.columns))}")
    selected = trades.loc[trades.setup.isin(setups)].copy()
    selected = selected.sort_values(["ent", "prio", "tk", "setup"], ascending=[True, False, True, True], kind="stable")
    for row in selected.itertuples():
        if not np.isfinite([row.e, row.x, row.st, row.prio]).all() or not (row.e > row.st > 0 and row.x > 0):
            raise ValueError("Scheduled trade requires finite prices and positive initial risk")
        if row.sig >= row.ent or row.ext < row.ent or row.exit_phase not in {"OPEN", "INTRADAY", "CLOSE"}:
            raise ValueError("Invalid signal/entry/exit chronology or exit phase")
        if row.ext == row.ent and row.exit_phase == "OPEN":
            raise ValueError("A new position cannot already have an opening exit")
        if row.tk not in history:
            raise ValueError(f"Missing history for {row.tk}")
        if len(dates) and dates[0] <= row.ent <= dates[-1] and row.ent not in dates:
            raise ValueError("Entry is not a replay session")
        if len(dates) and dates[0] <= row.ext <= dates[-1] and row.ext not in dates:
            raise ValueError("Exit is not a replay session")
    closes = {ticker: history[ticker].Close.dropna() for ticker in selected.tk.unique()}
    if any(not s.index.is_monotonic_increasing or not s.index.is_unique for s in closes.values()):
        raise ValueError("Price history must have ordered unique sessions")
    by_entry = {date: group for date, group in selected.groupby("ent")}
    cash, previous_nav = 1.0, 1.0
    book: dict[str, dict[str, Any]] = {}
    equity: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    def exit_due(date: pd.Timestamp, phase: str) -> None:
        nonlocal cash
        for ticker, position in list(book.items()):
            if position["ext"] != date or position["exit_phase"] != phase:
                continue
            quantity, price = position["quantity"], position["x"]
            fee = quantity * price * cost / 2
            cash += quantity * price - fee
            fills.append({"date": date, "phase": phase, "action": "SELL", "tk": ticker,
                          "setup": position["setup"], "quantity": quantity, "price": price,
                          "fees": fee, "cash": cash, "sizing_nav": None})
            del book[ticker]

    for date in dates:
        # Locked before today's prices or proceeds can influence opening sizes.
        sizing_nav = previous_nav
        exit_due(date, "OPEN")
        batch = by_entry.get(date)
        if batch is not None:
            for row in batch.itertuples():
                reason = "already_held" if row.tk in book else "position_cap" if len(book) >= slots else ""
                if reason:
                    decisions.append({"date": date, "tk": row.tk, "setup": row.setup, "reason": reason})
                    continue
                quantity = min(risk * sizing_nav / (row.e - row.st), cap * sizing_nav / row.e,
                               cash / (row.e * (1 + cost / 2)))
                if quantity <= 0:
                    decisions.append({"date": date, "tk": row.tk, "setup": row.setup, "reason": "cash_or_nav"})
                    continue
                fee = quantity * row.e * cost / 2
                cash = max(0., cash - quantity * row.e - fee)
                book[row.tk] = {"ext": row.ext, "exit_phase": row.exit_phase, "x": row.x,
                                "quantity": quantity, "mark": row.e, "mark_date": date, "setup": row.setup}
                fills.append({"date": date, "phase": "OPEN", "action": "BUY", "tk": row.tk,
                              "setup": row.setup, "quantity": quantity, "price": row.e,
                              "fees": fee, "cash": cash, "sizing_nav": sizing_nav})
        # Includes trades opened today: no same-day stop/close can be missed.
        exit_due(date, "INTRADAY")
        exit_due(date, "CLOSE")
        for ticker, position in book.items():
            series = closes[ticker]
            index = series.index.searchsorted(date, side="right") - 1
            if index >= 0 and series.index[index] >= position["mark_date"]:
                mark = float(series.iloc[index])
                if not np.isfinite(mark) or mark <= 0:
                    raise ValueError(f"Invalid observed close for {ticker} at {date}")
                position["mark"] = mark
                position["mark_date"] = series.index[index]
        previous_nav = cash + sum(p["quantity"] * p["mark"] for p in book.values())
        equity.append({"date": date, "nav": previous_nav, "cash": cash, "positions": len(book)})
    return ScheduledReplay(
        pd.DataFrame(equity, columns=["date", "nav", "cash", "positions"]).set_index("date"),
        pd.DataFrame(fills, columns=["date", "phase", "action", "tk", "setup", "quantity", "price", "fees", "cash", "sizing_nav"]),
        pd.DataFrame(decisions, columns=["date", "tk", "setup", "reason"]),
    )
