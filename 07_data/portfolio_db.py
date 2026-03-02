"""SQLite persistence for portfolio positions and transaction history.

Schema
------
``positions``   — current holdings (one row per symbol, upserted on change).
``transactions``— immutable ledger of every buy/sell/rebalance event.
``reallocation``— symbols flagged for reallocation (from News Filter).
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class Position:
    """A portfolio holding.

    Attributes:
        symbol: Ticker symbol.
        qty: Number of shares / units held.
        avg_cost: Average cost basis per unit.
        current_price: Latest market price.
        source: How the position was created ('manual', 'news_filter', 'scanner').
        added_at: ISO-8601 UTC timestamp.
        notes: Free-text notes.
    """

    symbol: str
    qty: float
    avg_cost: float
    current_price: float = 0.0
    source: str = "manual"
    added_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.added_at:
            self.added_at = datetime.now(tz=timezone.utc).isoformat()

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def unrealised_pnl(self) -> float:
        return self.qty * (self.current_price - self.avg_cost)

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price / self.avg_cost - 1.0) * 100.0


@dataclass
class Transaction:
    """Immutable ledger entry.

    Attributes:
        symbol: Ticker symbol.
        action: ``'buy'``, ``'sell'``, or ``'rebalance'``.
        qty: Units traded.
        price: Execution price per unit.
        executed_at: ISO-8601 UTC timestamp.
        notes: Free-text notes.
    """

    symbol: str
    action: str
    qty: float
    price: float
    executed_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.executed_at:
            self.executed_at = datetime.now(tz=timezone.utc).isoformat()


_DDL = """
CREATE TABLE IF NOT EXISTS positions (
    symbol        TEXT PRIMARY KEY,
    qty           REAL NOT NULL,
    avg_cost      REAL NOT NULL,
    current_price REAL DEFAULT 0,
    source        TEXT DEFAULT 'manual',
    added_at      TEXT,
    notes         TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    action      TEXT NOT NULL,
    qty         REAL NOT NULL,
    price       REAL NOT NULL,
    executed_at TEXT,
    notes       TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS reallocation (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT NOT NULL,
    reason     TEXT,
    added_at   TEXT,
    resolved   INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tx_symbol ON transactions (symbol);
CREATE INDEX IF NOT EXISTS idx_realloc_resolved ON reallocation (resolved);
"""


class PortfolioDB:
    """Portfolio persistence layer backed by SQLite.

    Args:
        db_path: Path to the ``.db`` file. Created automatically if missing.
    """

    def __init__(self, db_path: str | Path = "data/portfolio.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def upsert_position(self, pos: Position) -> None:
        """Insert or update a position (keyed by symbol).

        Args:
            pos: :class:`Position` to persist.
        """
        sql = """
            INSERT INTO positions (symbol, qty, avg_cost, current_price, source, added_at, notes)
            VALUES (:symbol, :qty, :avg_cost, :current_price, :source, :added_at, :notes)
            ON CONFLICT(symbol) DO UPDATE SET
                qty           = excluded.qty,
                avg_cost      = excluded.avg_cost,
                current_price = excluded.current_price,
                source        = excluded.source,
                notes         = excluded.notes
        """
        with self._connect() as conn:
            conn.execute(sql, asdict(pos))

    def get_positions(self) -> pd.DataFrame:
        """Return all portfolio positions as a DataFrame."""
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM positions ORDER BY symbol", conn)

    def delete_position(self, symbol: str) -> None:
        """Remove a position by symbol.

        Args:
            symbol: Ticker to remove.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol.upper(),))

    def update_price(self, symbol: str, price: float) -> None:
        """Update the current_price for a position.

        Args:
            symbol: Ticker symbol.
            price: Latest market price.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE positions SET current_price = ? WHERE symbol = ?",
                (price, symbol.upper()),
            )

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def add_transaction(self, tx: Transaction) -> None:
        """Append a transaction to the ledger.

        Args:
            tx: :class:`Transaction` to record.
        """
        row = asdict(tx)
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO transactions (symbol, action, qty, price, executed_at, notes)
                   VALUES (:symbol, :action, :qty, :price, :executed_at, :notes)""",
                row,
            )

    def get_transactions(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Return transaction history, optionally filtered by symbol.

        Args:
            symbol: Optional ticker filter.

        Returns:
            DataFrame ordered by ``executed_at`` descending.
        """
        if symbol:
            sql = "SELECT * FROM transactions WHERE symbol = ? ORDER BY executed_at DESC"
            with self._connect() as conn:
                return pd.read_sql_query(sql, conn, params=(symbol.upper(),))
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM transactions ORDER BY executed_at DESC", conn
            )

    # ------------------------------------------------------------------
    # Reallocation queue
    # ------------------------------------------------------------------

    def add_reallocation(self, symbol: str, reason: str = "") -> None:
        """Flag a symbol for reallocation.

        Args:
            symbol: Ticker to reallocate.
            reason: Explanation (e.g. from news filter).
        """
        ts = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO reallocation (symbol, reason, added_at) VALUES (?, ?, ?)",
                (symbol.upper(), reason, ts),
            )

    def get_reallocation(self, include_resolved: bool = False) -> pd.DataFrame:
        """Return pending reallocation items.

        Args:
            include_resolved: If True, also return resolved items.

        Returns:
            DataFrame ordered by ``added_at`` descending.
        """
        where = "" if include_resolved else "WHERE resolved = 0"
        with self._connect() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM reallocation {where} ORDER BY added_at DESC", conn
            )

    def resolve_reallocation(self, row_id: int) -> None:
        """Mark a reallocation item as resolved.

        Args:
            row_id: Primary key of the reallocation row.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE reallocation SET resolved = 1 WHERE id = ?", (row_id,)
            )
