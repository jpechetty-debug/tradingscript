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
