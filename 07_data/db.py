"""SQLite persistence layer for momentum scan results.

Schema
------
Table ``momentum_scans``:

    id          INTEGER PRIMARY KEY AUTOINCREMENT
    symbol      TEXT NOT NULL
    scanned_at  TEXT NOT NULL        -- ISO-8601 UTC timestamp
    state       TEXT NOT NULL        -- 'entry' | 'exit' | 'hold'
    confidence  REAL NOT NULL        -- [0.0, 1.0]
    return_pct  REAL                 -- rolling return used for signal
    price_close REAL                 -- latest close price
    lookback    INTEGER              -- bars used
    source      TEXT                 -- 'YahooFinance' | 'Synthetic' | …

Usage
-----
    from 07_data.db import ScanDB
    db = ScanDB("data/scans.db")
    db.save(result)
    df = db.query(state="entry", limit=50)
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------


@dataclass
class ScanRecord:
    """Single row in the momentum_scans table.

    Attributes:
        symbol: Ticker symbol.
        scanned_at: UTC timestamp (ISO-8601 string).
        state: Signal state — ``'entry'``, ``'exit'``, or ``'hold'``.
        confidence: Signal confidence in ``[0, 1]``.
        return_pct: Rolling return (%) used to produce the signal.
        price_close: Latest close price at scan time.
        lookback: Number of bars used by the strategy.
        source: Data source identifier.
    """

    symbol: str
    scanned_at: str
    state: str
    confidence: float
    return_pct: float
    price_close: float
    lookback: int
    source: str

    @classmethod
    def now(
        cls,
        symbol: str,
        state: str,
        confidence: float,
        return_pct: float,
        price_close: float,
        lookback: int,
        source: str,
    ) -> "ScanRecord":
        """Create a record stamped with the current UTC time."""
        ts = datetime.now(tz=timezone.utc).isoformat()
        return cls(
            symbol=symbol,
            scanned_at=ts,
            state=state,
            confidence=confidence,
            return_pct=return_pct,
            price_close=price_close,
            lookback=lookback,
            source=source,
        )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS momentum_scans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    scanned_at  TEXT    NOT NULL,
    state       TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    return_pct  REAL,
    price_close REAL,
    lookback    INTEGER,
    source      TEXT
);

CREATE INDEX IF NOT EXISTS idx_symbol      ON momentum_scans (symbol);
CREATE INDEX IF NOT EXISTS idx_scanned_at  ON momentum_scans (scanned_at);
CREATE INDEX IF NOT EXISTS idx_state       ON momentum_scans (state);
"""


class ScanDB:
    """Thin wrapper around a SQLite database for momentum scan results.

    Args:
        db_path: Path to the ``.db`` file. Created automatically if missing.
    """

    def __init__(self, db_path: str | Path = "data/scans.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, record: ScanRecord) -> None:
        """Persist a single :class:`ScanRecord` to the database.

        Args:
            record: Scan result to store.
        """
        row = asdict(record)
        columns = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        sql = f"INSERT INTO momentum_scans ({columns}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.execute(sql, row)

    def save_many(self, records: list[ScanRecord]) -> None:
        """Bulk-insert a list of :class:`ScanRecord` objects.

        Args:
            records: Scan results to store.
        """
        if not records:
            return
        rows = [asdict(r) for r in records]
        columns = ", ".join(rows[0].keys())
        placeholders = ", ".join(f":{k}" for k in rows[0])
        sql = f"INSERT INTO momentum_scans ({columns}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.executemany(sql, rows)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        state: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Query scan history with optional filters.

        Args:
            state: Filter by signal state (``'entry'``, ``'exit'``, ``'hold'``).
            symbol: Filter by ticker symbol.
            limit: Maximum number of rows (most recent first).

        Returns:
            DataFrame with all columns from ``momentum_scans``, ordered by
            ``scanned_at`` descending.
        """
        clauses: list[str] = []
        params: dict[str, object] = {}

        if state:
            clauses.append("state = :state")
            params["state"] = state
        if symbol:
            clauses.append("symbol = :symbol")
            params["symbol"] = symbol.upper()

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT * FROM momentum_scans {where} "
            f"ORDER BY scanned_at DESC LIMIT :limit"
        )
        params["limit"] = limit

        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def latest_per_symbol(self) -> pd.DataFrame:
        """Return the most recent scan result for every symbol.

        Returns:
            DataFrame with one row per symbol, the most recently scanned entry.
        """
        sql = """
            SELECT *
            FROM momentum_scans
            WHERE id IN (
                SELECT MAX(id)
                FROM momentum_scans
                GROUP BY symbol
            )
            ORDER BY scanned_at DESC
        """
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn)

    def entry_signals(self, limit: int = 100) -> pd.DataFrame:
        """Shortcut: return only ``state='entry'`` rows.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            DataFrame filtered to entry signals.
        """
        return self.query(state="entry", limit=limit)

    def row_count(self) -> int:
        """Return total number of rows in the scan history table."""
        with self._connect() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM momentum_scans")
            return int(cur.fetchone()[0])

    def clear(self) -> None:
        """Delete ALL rows from the scan table (destructive!)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM momentum_scans")
