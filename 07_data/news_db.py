"""SQLite persistence for financial news sources and AI-processed feed items.

Schema
------
``news_sources``  — registered sources (YouTube channel, Twitter, Email, TradingView, RSS).
``news_items``    — processed news items with AI-structured summary.
                    Each item carries: ticker, ROI near/mid/long, benchmark,
                    strategy_fit flag, and auto-routing decision.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class NewsSource:
    """A registered financial information source.

    Attributes:
        name: Display name (e.g. "Finanzfluss YouTube").
        source_type: One of ``youtube``, ``twitter``, ``email``,
            ``tradingview``, ``rss``, ``other``.
        url: Source URL or identifier.
        active: Whether the source is active for scanning.
        added_at: ISO-8601 UTC timestamp.
    """

    name: str
    source_type: str
    url: str = ""
    active: int = 1
    added_at: str = ""

    def __post_init__(self) -> None:
        if not self.added_at:
            self.added_at = datetime.now(tz=timezone.utc).isoformat()


@dataclass
class NewsItem:
    """AI-structured news/feed item.

    Attributes:
        source_name: Which source produced this item.
        raw_text: Original pasted or fetched text.
        ticker: Extracted ticker symbol (e.g. ``"NVDA"``).
        headline: One-sentence summary.
        roi_near: Near-term ROI expectation (narrative or %, e.g. ``"+5% (1M)"``).
        roi_mid: Mid-term ROI expectation.
        roi_long: Long-term ROI expectation.
        benchmark: Comparison benchmark (e.g. ``"S&P500"``, ``"BTC"``).
        strategy_fit: ``"yes"``, ``"no"``, or ``"partial"``.
        routing: ``"portfolio"``, ``"reallocation"``, or ``"watch"``.
        processed_at: ISO-8601 UTC timestamp.
        notes: AI reasoning or manual notes.
    """

    source_name: str
    raw_text: str
    ticker: str = ""
    headline: str = ""
    roi_near: str = ""
    roi_mid: str = ""
    roi_long: str = ""
    benchmark: str = ""
    strategy_fit: str = "no"
    routing: str = "watch"
    processed_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.processed_at:
            self.processed_at = datetime.now(tz=timezone.utc).isoformat()


_DDL = """
CREATE TABLE IF NOT EXISTS news_sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    source_type TEXT NOT NULL,
    url         TEXT DEFAULT '',
    active      INTEGER DEFAULT 1,
    added_at    TEXT
);

CREATE TABLE IF NOT EXISTS news_items (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name  TEXT,
    raw_text     TEXT,
    ticker       TEXT DEFAULT '',
    headline     TEXT DEFAULT '',
    roi_near     TEXT DEFAULT '',
    roi_mid      TEXT DEFAULT '',
    roi_long     TEXT DEFAULT '',
    benchmark    TEXT DEFAULT '',
    strategy_fit TEXT DEFAULT 'no',
    routing      TEXT DEFAULT 'watch',
    processed_at TEXT,
    notes        TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_ni_ticker   ON news_items (ticker);
CREATE INDEX IF NOT EXISTS idx_ni_routing  ON news_items (routing);
CREATE INDEX IF NOT EXISTS idx_ni_fit      ON news_items (strategy_fit);
"""


class NewsDB:
    """News source and feed-item persistence backed by SQLite.

    Args:
        db_path: Path to the ``.db`` file. Created automatically if missing.
    """

    def __init__(self, db_path: str | Path = "data/news.db") -> None:
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
    # Sources
    # ------------------------------------------------------------------

    def add_source(self, source: NewsSource) -> int:
        """Persist a new news source.

        Args:
            source: :class:`NewsSource` to add.

        Returns:
            Newly inserted row id.
        """
        row = asdict(source)
        row.pop("added_at", None)
        row["added_at"] = source.added_at
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO news_sources (name, source_type, url, active, added_at)
                   VALUES (:name, :source_type, :url, :active, :added_at)""",
                row,
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_sources(self, active_only: bool = False) -> pd.DataFrame:
        """Return all (or only active) news sources.

        Args:
            active_only: If ``True``, filter to active sources only.

        Returns:
            DataFrame of news sources.
        """
        where = "WHERE active = 1" if active_only else ""
        with self._connect() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM news_sources {where} ORDER BY name", conn
            )

    def toggle_source(self, source_id: int) -> None:
        """Toggle a source between active and inactive.

        Args:
            source_id: Primary key of the source row.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE news_sources SET active = 1 - active WHERE id = ?",
                (source_id,),
            )

    def delete_source(self, source_id: int) -> None:
        """Delete a source by id.

        Args:
            source_id: Primary key of the source row to delete.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM news_sources WHERE id = ?", (source_id,))

    # ------------------------------------------------------------------
    # Items
    # ------------------------------------------------------------------

    def save_item(self, item: NewsItem) -> int:
        """Persist a processed news item.

        Args:
            item: :class:`NewsItem` to save.

        Returns:
            Newly inserted row id.
        """
        row = asdict(item)
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO news_items
                   (source_name, raw_text, ticker, headline,
                    roi_near, roi_mid, roi_long, benchmark,
                    strategy_fit, routing, processed_at, notes)
                   VALUES
                   (:source_name, :raw_text, :ticker, :headline,
                    :roi_near, :roi_mid, :roi_long, :benchmark,
                    :strategy_fit, :routing, :processed_at, :notes)""",
                row,
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_items(
        self,
        routing: Optional[str] = None,
        ticker: Optional[str] = None,
        strategy_fit: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Query news items with optional filters.

        Args:
            routing: Filter by routing (``'portfolio'``, ``'reallocation'``,
                ``'watch'``).
            ticker: Filter by ticker symbol.
            strategy_fit: Filter by fit (``'yes'``, ``'no'``, ``'partial'``).
            limit: Maximum rows to return (most recent first).

        Returns:
            DataFrame ordered by ``processed_at`` descending.
        """
        clauses: list[str] = []
        params: dict[str, object] = {}
        if routing:
            clauses.append("routing = :routing")
            params["routing"] = routing
        if ticker:
            clauses.append("ticker = :ticker")
            params["ticker"] = ticker.upper()
        if strategy_fit:
            clauses.append("strategy_fit = :strategy_fit")
            params["strategy_fit"] = strategy_fit
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params["limit"] = limit
        with self._connect() as conn:
            return pd.read_sql_query(
                f"SELECT * FROM news_items {where} "
                f"ORDER BY processed_at DESC LIMIT :limit",
                conn,
                params=params,
            )

    def item_count(self) -> int:
        """Return total number of processed news items."""
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0])
