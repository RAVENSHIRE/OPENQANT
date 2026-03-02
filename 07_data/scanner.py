"""Sequential momentum scanner with SQLite persistence.

Architecture
------------
::

    WATCHLIST  ──►  MomentumScanner.run_full_scan()
                         │
                         ▼  (sequential, one symbol at a time)
                    yfinance / Synthetic
                         │
                         ▼
                    SeasonalMomentumStrategy.generate_signal()
                         │
                         ▼
                    ScanRecord  ──►  ScanDB.save()
                         │
                         ▼
                    pd.DataFrame  ──►  Dashboard / CLI

Usage
-----
    from 07_data.scanner import MomentumScanner

    scanner = MomentumScanner()
    df = scanner.run_full_scan()        # scan everything in WATCHLIST
    entries = scanner.db.entry_signals()  # query results from DB

    # scan only a subset
    df = scanner.run_full_scan(["AAPL", "BTC-USD", "^GSPC"])
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent


def _load_db_module():
    """Dynamically load db module to avoid relative-import errors."""
    spec = importlib.util.spec_from_file_location("scan_db_module", _DATA_DIR / "db.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["scan_db_module"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_db_mod = _load_db_module()
ScanDB = _db_mod.ScanDB
ScanRecord = _db_mod.ScanRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Watchlist — categories with default symbols
# ---------------------------------------------------------------------------

WATCHLIST: dict[str, list[str]] = {
    "Indizes": [
        "^GSPC",   # S&P 500
        "^NDX",    # NASDAQ 100
        "^DJI",    # Dow Jones
        "^RUT",    # Russell 2000
        "^STOXX50E",  # Euro Stoxx 50
    ],
    "ETFs": [
        "SPY",   # S&P 500 ETF
        "QQQ",   # NASDAQ 100 ETF
        "IWM",   # Russell 2000 ETF
        "EFA",   # MSCI EAFE (Europa/Asien)
        "GLD",   # Gold ETF
        "TLT",   # US Long Bonds
    ],
    "Aktien": [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "NVDA",  # NVIDIA
        "AMZN",  # Amazon
        "META",  # Meta
        "TSLA",  # Tesla
    ],
    "Krypto": [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "BNB-USD",
        "XRP-USD",
    ],
}

ALL_SYMBOLS: list[str] = [s for symbols in WATCHLIST.values() for s in symbols]


# ---------------------------------------------------------------------------
# Dynamic loader for SeasonalMomentumStrategy (avoids package import)
# ---------------------------------------------------------------------------


def _load_strategy() -> type:
    """Load ``SeasonalMomentumStrategy`` from the strategies folder.

    Returns:
        The ``SeasonalMomentumStrategy`` class.

    Raises:
        ImportError: If the module cannot be found or loaded.
    """
    strat_path = Path(__file__).resolve().parents[1] / "02_strategies" / "momentum.py"
    spec = importlib.util.spec_from_file_location("momentum_strategy", strat_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load momentum strategy from {strat_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["momentum_strategy"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.SeasonalMomentumStrategy


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    """Intermediate result before DB persistence.

    Attributes:
        symbol: Ticker symbol.
        state: Signal state (``'entry'``, ``'exit'``, ``'hold'``).
        confidence: Signal confidence in ``[0, 1]``.
        return_pct: Rolling return (%) over the lookback window.
        price_close: Latest close price.
        lookback: Bars used for the momentum calculation.
        source: Data source used.
        error: Non-empty string if the scan failed.
    """

    symbol: str
    state: str = "hold"
    confidence: float = 0.0
    return_pct: float = 0.0
    price_close: float = 0.0
    lookback: int = 20
    source: str = "YahooFinance"
    error: str = ""

    @property
    def ok(self) -> bool:
        """Return ``True`` if the scan completed without errors."""
        return self.error == ""


class MomentumScanner:
    """Sequential momentum scanner for a configurable watchlist.

    Fetches OHLCV data for each symbol (via yfinance or synthetic fallback),
    passes close prices through :class:`SeasonalMomentumStrategy`, and
    persists every result to a SQLite database.

    Args:
        db_path: Path to the SQLite file. Defaults to ``data/scans.db``
            relative to the project root.
        lookback: Momentum lookback window in trading days.
        entry_threshold: Minimum return to emit an ``entry`` signal.
        exit_threshold: Return below which an ``exit`` signal is emitted.
        delay_seconds: Pause between sequential requests (rate-limit safety).
        source: Data source for price fetching.
        period_days: Calendar days of history to download per symbol.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        lookback: int = 20,
        entry_threshold: float = 0.02,
        exit_threshold: float = -0.01,
        delay_seconds: float = 0.3,
        source: str = "YahooFinance",
        period_days: int = 90,
    ) -> None:
        if db_path is None:
            db_path = Path(__file__).resolve().parents[1] / "data" / "scans.db"
        self.db = ScanDB(db_path)
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.delay_seconds = delay_seconds
        self.source = source
        self.period_days = period_days

        # lazy-load strategy class
        self._strategy_cls = _load_strategy()
        self._strategy = self._strategy_cls(
            lookback=lookback,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_prices(self, symbol: str) -> np.ndarray:
        """Download close prices for *symbol* using the configured source.

        Falls back to synthetic data when yfinance returns nothing.

        Args:
            symbol: Ticker to download.

        Returns:
            1-D numpy array of close prices, at least ``lookback + 1`` long.

        Raises:
            RuntimeError: If neither real nor synthetic data can be fetched.
        """
        from datetime import date, timedelta

        # Dynamic load to avoid relative-import errors
        _fspec = importlib.util.spec_from_file_location("fetcher_mod", _DATA_DIR / "fetcher.py")
        _fmod = importlib.util.module_from_spec(_fspec)  # type: ignore[arg-type]
        sys.modules["fetcher_mod"] = _fmod
        _fspec.loader.exec_module(_fmod)  # type: ignore[union-attr]
        fetch_ohlcv = _fmod.fetch_ohlcv

        end = date.today().isoformat()
        start = (date.today() - timedelta(days=self.period_days)).isoformat()

        try:
            df = fetch_ohlcv(symbol, start_date=start, end_date=end, source=self.source)
            prices = df["close"].dropna().to_numpy(dtype=float)
            if len(prices) < self.lookback + 1:
                raise ValueError(
                    f"Only {len(prices)} bars — need {self.lookback + 1}."
                )
            return prices
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Real data fetch failed for '%s' (%s). Falling back to synthetic.",
                symbol,
                exc,
            )

        # Synthetic fallback
        df_syn = fetch_ohlcv(symbol, start_date=start, end_date=end, source="Synthetic")
        return df_syn["close"].dropna().to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_symbol(self, symbol: str) -> ScanResult:
        """Scan a single symbol and return a :class:`ScanResult`.

        The result is **not** automatically saved to the DB; use
        :meth:`run_full_scan` or call :meth:`~ScanDB.save` manually.

        Args:
            symbol: Ticker symbol to scan.

        Returns:
            :class:`ScanResult` with signal state, confidence and price info.
        """
        sym = symbol.upper()
        try:
            prices = self._fetch_prices(sym)
            signal = self._strategy.generate_signal(prices)
            roll_ret = float(prices[-1] / prices[-self.lookback - 1] - 1.0)
            return ScanResult(
                symbol=sym,
                state=signal.state,
                confidence=round(float(signal.confidence), 4),
                return_pct=round(roll_ret * 100, 4),
                price_close=round(float(prices[-1]), 4),
                lookback=self.lookback,
                source=self.source,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("scan_symbol('%s') failed: %s", sym, exc)
            return ScanResult(symbol=sym, error=str(exc))

    def run_full_scan(
        self,
        symbols: Optional[list[str]] = None,
        save_to_db: bool = True,
    ) -> pd.DataFrame:
        """Run a sequential scan over the provided (or default) watchlist.

        Each symbol is fetched, signalled, and (optionally) persisted to the
        SQLite DB.  A configurable delay between requests prevents rate-limiting
        by Yahoo Finance.

        Args:
            symbols: List of tickers to scan. Defaults to
                :data:`ALL_SYMBOLS` when ``None``.
            save_to_db: If ``True`` (default), persist each successful
                :class:`ScanRecord` to :attr:`db`.

        Returns:
            DataFrame with columns ``symbol``, ``state``, ``confidence``,
            ``return_pct``, ``price_close``, ``lookback``, ``source``,
            ``error`` — one row per scanned symbol.
        """
        targets = symbols or ALL_SYMBOLS
        results: list[ScanResult] = []
        records: list[ScanRecord] = []

        for i, sym in enumerate(targets, start=1):
            logger.info("[%d/%d] Scanning %s …", i, len(targets), sym)
            result = self.scan_symbol(sym)
            results.append(result)

            if result.ok and save_to_db:
                records.append(
                    ScanRecord.now(
                        symbol=result.symbol,
                        state=result.state,
                        confidence=result.confidence,
                        return_pct=result.return_pct,
                        price_close=result.price_close,
                        lookback=result.lookback,
                        source=result.source,
                    )
                )

            if i < len(targets):
                time.sleep(self.delay_seconds)

        if records:
            self.db.save_many(records)
            logger.info("Saved %d scan records to DB.", len(records))

        return pd.DataFrame(
            [
                {
                    "symbol": r.symbol,
                    "state": r.state,
                    "confidence": r.confidence,
                    "return_pct": r.return_pct,
                    "price_close": r.price_close,
                    "lookback": r.lookback,
                    "source": r.source,
                    "error": r.error,
                }
                for r in results
            ]
        )

    def run_category_scan(
        self,
        category: str,
        save_to_db: bool = True,
    ) -> pd.DataFrame:
        """Scan a single watchlist category by name.

        Args:
            category: One of ``"Indizes"``, ``"ETFs"``, ``"Aktien"``,
                ``"Krypto"``.
            save_to_db: Persist results to the DB.

        Returns:
            DataFrame as returned by :meth:`run_full_scan`.

        Raises:
            KeyError: If the category is not in :data:`WATCHLIST`.
        """
        if category not in WATCHLIST:
            raise KeyError(
                f"Unknown category '{category}'. "
                f"Choose from: {list(WATCHLIST.keys())}"
            )
        return self.run_full_scan(symbols=WATCHLIST[category], save_to_db=save_to_db)
