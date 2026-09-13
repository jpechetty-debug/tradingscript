"""
core/database.py
================
High-performance, raw SQL SQLite persistence layer for the Sovereign Engine.

Features:
- Standard library sqlite3 (zero external dependencies).
- ACID compliant with Write-Ahead Logging (WAL) for high concurrency.
- Tables: portfolio_state, platt_calibration, trade_log, factor_weights.
- Auto-migration support from existing JSON files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

log = logging.getLogger("sovereign.database")


class SqliteDatabase:
    """Raw SQL SQLite interface optimized for low-latency engine state and trade logging."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a SQLite connection configured with WAL and busy timeout."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create state tables and indexes if they do not exist, handling schema migrations."""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    current_nav REAL NOT NULL,
                    peak_nav REAL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS platt_calibration (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    a REAL NOT NULL,
                    b REAL NOT NULL,
                    fitted_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    ticker TEXT,
                    direction TEXT,
                    pnl REAL,
                    factors TEXT,
                    composite REAL,
                    raw_payload TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trade_ticker ON trade_log(ticker);
                CREATE INDEX IF NOT EXISTS idx_trade_timestamp ON trade_log(timestamp);

                CREATE TABLE IF NOT EXISTS factor_weights (
                    factor_name TEXT PRIMARY KEY,
                    weight REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS open_positions (
                    ticker TEXT PRIMARY KEY,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    stop_loss REAL NOT NULL,
                    target REAL NOT NULL,
                    prob_win REAL NOT NULL,
                    composite REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)

            # Migration: ensure trade_id column exists and populate empty legacy records
            cols = [col[1] for col in conn.execute("PRAGMA table_info(trade_log);").fetchall()]
            if "trade_id" not in cols:
                conn.execute("ALTER TABLE trade_log ADD COLUMN trade_id TEXT;")
            conn.execute("UPDATE trade_log SET trade_id = 'legacy_' || id WHERE trade_id IS NULL OR trade_id = '';")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_unique_id ON trade_log(trade_id);")

    # ── Portfolio State ─────────────────────────────────────────────────────────

    def upsert_portfolio_state(
        self,
        current_nav: float,
        peak_nav: Optional[float] = None,
        updated_at: Optional[str] = None,
    ) -> None:
        ts = updated_at or datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_state (id, current_nav, peak_nav, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    current_nav = excluded.current_nav,
                    peak_nav = excluded.peak_nav,
                    updated_at = excluded.updated_at;
                """,
                (float(current_nav), float(peak_nav) if peak_nav is not None else None, ts),
            )

    def fetch_latest_portfolio_state(self) -> Optional[dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT current_nav, peak_nav, updated_at FROM portfolio_state WHERE id = 1;"
            ).fetchone()
            if row:
                return {
                    "current_nav": float(row["current_nav"]),
                    "peak_nav": float(row["peak_nav"]) if row["peak_nav"] is not None else None,
                    "updated_at": row["updated_at"],
                }
            return None

    # ── Platt Calibration ───────────────────────────────────────────────────────

    def upsert_platt(
        self,
        a: float,
        b: float,
        fitted_at: Optional[str] = None,
    ) -> None:
        ts = fitted_at or datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO platt_calibration (id, a, b, fitted_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    a = excluded.a,
                    b = excluded.b,
                    fitted_at = excluded.fitted_at;
                """,
                (float(a), float(b), ts),
            )

    def fetch_latest_platt(self) -> Optional[tuple[float, float, str]]:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT a, b, fitted_at FROM platt_calibration WHERE id = 1;"
            ).fetchone()
            if row:
                return float(row["a"]), float(row["b"]), str(row["fitted_at"])
            return None

    # ── Trade Log ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_trade_id(trade: dict[str, Any], ticker: Any, direction: Any, ts: str, pnl: Any) -> str:
        trade_id = str(trade.get("trade_id") or trade.get("id") or "").strip()
        if trade_id:
            return trade_id
        raw = f"{ticker}_{direction}_{ts}_{pnl}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def insert_trade(self, trade: dict[str, Any]) -> int:
        ticker = trade.get("ticker")
        direction = trade.get("direction")
        pnl = float(trade["pnl"]) if trade.get("pnl") is not None else None
        factors_raw = trade.get("factors")
        factors_str = json.dumps(factors_raw) if factors_raw is not None else None
        composite = float(trade["composite"]) if trade.get("composite") is not None else None
        ts = str(trade.get("timestamp") or trade.get("ts") or datetime.now(timezone.utc).isoformat())
        trade_id = self._resolve_trade_id(trade, ticker, direction, ts, pnl)
        raw_json = json.dumps(trade, default=str)

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trade_log (trade_id, ticker, direction, pnl, factors, composite, raw_payload, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    ticker = excluded.ticker,
                    direction = excluded.direction,
                    pnl = excluded.pnl,
                    factors = excluded.factors,
                    composite = excluded.composite,
                    raw_payload = excluded.raw_payload,
                    timestamp = excluded.timestamp;
                """,
                (trade_id, ticker, direction, pnl, factors_str, composite, raw_json, ts),
            )
            return cursor.lastrowid or 0

    def insert_trades(self, trades: list[dict[str, Any]]) -> None:
        with self.get_connection() as conn:
            for trade in trades:
                ticker = trade.get("ticker")
                direction = trade.get("direction")
                pnl = float(trade["pnl"]) if trade.get("pnl") is not None else None
                factors_raw = trade.get("factors")
                factors_str = json.dumps(factors_raw) if factors_raw is not None else None
                composite = float(trade["composite"]) if trade.get("composite") is not None else None
                ts = str(trade.get("timestamp") or trade.get("ts") or datetime.now(timezone.utc).isoformat())
                trade_id = self._resolve_trade_id(trade, ticker, direction, ts, pnl)
                raw_json = json.dumps(trade, default=str)

                conn.execute(
                    """
                    INSERT INTO trade_log (trade_id, ticker, direction, pnl, factors, composite, raw_payload, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trade_id) DO UPDATE SET
                        ticker = excluded.ticker,
                        direction = excluded.direction,
                        pnl = excluded.pnl,
                        factors = excluded.factors,
                        composite = excluded.composite,
                        raw_payload = excluded.raw_payload,
                        timestamp = excluded.timestamp;
                    """,
                    (trade_id, ticker, direction, pnl, factors_str, composite, raw_json, ts),
                )

    def fetch_trades(
        self,
        ticker: Optional[str] = None,
        limit: int = 2000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = "SELECT raw_payload FROM trade_log"
        params: list[Any] = []
        if ticker:
            query += " WHERE ticker = ?"
            params.append(ticker)
        query += " ORDER BY id ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            trades: list[dict[str, Any]] = []
            for r in rows:
                try:
                    payload = json.loads(r["raw_payload"])
                    if isinstance(payload, dict):
                        trades.append(payload)
                except Exception:
                    continue
            return trades

    def count_trades(self) -> int:
        with self.get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM trade_log;").fetchone()
            return int(row["cnt"]) if row else 0

    # ── Factor Weights ─────────────────────────────────────────────────────────

    def upsert_factor_weights(
        self,
        weights: dict[str, float],
        updated_at: Optional[str] = None,
    ) -> None:
        ts = updated_at or datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            for name, weight in weights.items():
                conn.execute(
                    """
                    INSERT INTO factor_weights (factor_name, weight, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(factor_name) DO UPDATE SET
                        weight = excluded.weight,
                        updated_at = excluded.updated_at;
                    """,
                    (str(name), float(weight), ts),
                )

    def fetch_factor_weights(self) -> Optional[dict[str, float]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT factor_name, weight FROM factor_weights;").fetchall()
            if not rows:
                return None
            return {str(r["factor_name"]): float(r["weight"]) for r in rows}

    # ── Open Positions ─────────────────────────────────────────────────────────

    def upsert_open_position(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        shares: int,
        stop_loss: float,
        target: float,
        prob_win: float,
        composite: float,
        opened_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ) -> None:
        now_ts = datetime.now(timezone.utc).isoformat()
        opened = opened_at or now_ts
        updated = updated_at or now_ts
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO open_positions (
                    ticker, direction, entry_price, shares, stop_loss, target,
                    prob_win, composite, opened_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    direction = excluded.direction,
                    entry_price = excluded.entry_price,
                    shares = excluded.shares,
                    stop_loss = excluded.stop_loss,
                    target = excluded.target,
                    prob_win = excluded.prob_win,
                    composite = excluded.composite,
                    updated_at = excluded.updated_at;
                """,
                (
                    str(ticker), str(direction), float(entry_price), int(shares),
                    float(stop_loss), float(target), float(prob_win), float(composite),
                    opened, updated,
                ),
            )

    def delete_open_position(self, ticker: str) -> bool:
        with self.get_connection() as conn:
            cur = conn.execute("DELETE FROM open_positions WHERE ticker = ?;", (str(ticker),))
            return cur.rowcount > 0

    def fetch_open_positions(self) -> dict[str, dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM open_positions ORDER BY ticker ASC;").fetchall()
            return {
                str(r["ticker"]): {
                    "ticker": str(r["ticker"]),
                    "direction": str(r["direction"]),
                    "entry_price": float(r["entry_price"]),
                    "shares": int(r["shares"]),
                    "stop_loss": float(r["stop_loss"]),
                    "target": float(r["target"]),
                    "prob_win": float(r["prob_win"]),
                    "composite": float(r["composite"]),
                    "opened_at": str(r["opened_at"]),
                    "updated_at": str(r["updated_at"]),
                }
                for r in rows
            }

    def sync_open_positions(self, positions: list[dict[str, Any]]) -> None:
        """
        Synchronize the open_positions table to exactly match the given active positions list.
        Positions no longer in the list are removed; new/existing positions are upserted.
        """
        active_tickers = {str(p["ticker"]) for p in positions if "ticker" in p}
        now_ts = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            if active_tickers:
                placeholders = ",".join("?" * len(active_tickers))
                conn.execute(
                    f"DELETE FROM open_positions WHERE ticker NOT IN ({placeholders});",
                    list(active_tickers),
                )
            else:
                conn.execute("DELETE FROM open_positions;")

            for p in positions:
                ticker = str(p["ticker"])
                direction = str(p.get("direction", "LONG"))
                entry_price = float(p.get("entry", p.get("entry_price", p.get("close", 0.0))))
                shares = int(p.get("shares", 0))
                stop_loss = float(p.get("stop", p.get("stop_loss", 0.0)))
                target = float(p.get("t1", p.get("target", 0.0)))
                prob_win = float(p.get("prob_win", 0.50))
                composite = float(p.get("composite", 0.50))
                opened_at = str(p.get("opened_at") or now_ts)
                updated_at = str(p.get("updated_at") or now_ts)
                conn.execute(
                    """
                    INSERT INTO open_positions (
                        ticker, direction, entry_price, shares, stop_loss, target,
                        prob_win, composite, opened_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker) DO UPDATE SET
                        direction = excluded.direction,
                        entry_price = excluded.entry_price,
                        shares = excluded.shares,
                        stop_loss = excluded.stop_loss,
                        target = excluded.target,
                        prob_win = excluded.prob_win,
                        composite = excluded.composite,
                        updated_at = excluded.updated_at;
                    """,
                    (
                        ticker, direction, entry_price, shares, stop_loss, target,
                        prob_win, composite, opened_at, updated_at,
                    ),
                )
