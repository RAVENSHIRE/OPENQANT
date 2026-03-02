"""Market data fetcher — yfinance (real) + deterministic synthetic fallback.

Supported sources
-----------------
* ``YahooFinance``  — live data via ``yfinance``.
* ``Synthetic``     — fast, deterministic GBM series (no network required).
* ``AlphaVantage``  — reserved; raises :exc:`NotImplementedError`.
* ``Quandl``        — reserved; raises :exc:`NotImplementedError`.

Usage
-----
    df = fetch_ohlcv("AAPL", "2024-01-01", "2024-12-31")
    df = fetch_ohlcv("SPY", "2023-01-01", "2023-12-31", source="Synthetic")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_SOURCES = {"YahooFinance", "AlphaVantage", "Quandl", "Synthetic"}
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _yfinance_download(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance via ``yfinance``.

    Args:
        symbol: Ticker symbol (e.g. ``"AAPL"``, ``"^GSPC"``, ``"BTC-USD"``).
        start: Inclusive start date ``YYYY-MM-DD``.
        end: Inclusive end date ``YYYY-MM-DD``.

    Returns:
        DataFrame with lowercase OHLCV columns and a tz-naive DatetimeIndex.

    Raises:
        ImportError: If ``yfinance`` is not installed.
        ValueError: If no data is returned (unknown symbol or empty range).
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for source='YahooFinance'. "
            "Install it with:  pip install yfinance"
        ) from exc

    ticker = yf.Ticker(symbol)
    raw = ticker.history(start=start, end=end, auto_adjust=True)

    if raw.empty:
        raise ValueError(
            f"No data returned for '{symbol}' between {start} and {end}. "
            "Check the ticker symbol and date range."
        )

    raw = raw.rename(columns=str.lower)
    available = [c for c in _OHLCV_COLUMNS if c in raw.columns]
    frame = raw[available].copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "date"
    logger.info("YahooFinance: fetched %d rows for '%s'.", len(frame), symbol)
    return frame


def _synthetic_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV via Geometric Brownian Motion."""
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    seed = abs(hash((symbol, start_date, end_date, "Synthetic"))) % (2**32)
    rng = np.random.default_rng(seed)
    n = len(dates)
    innovations = rng.normal(loc=0.0002, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(innovations))
    high = close * (1.0 + rng.uniform(0.0001, 0.01, size=n))
    low = close * (1.0 - rng.uniform(0.0001, 0.01, size=n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(100_000, 2_000_000, size=n).astype(float)
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(high, np.maximum(open_, close)),
            "low": np.minimum(low, np.minimum(open_, close)),
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    frame.index.name = "date"
    return frame


@dataclass
class MarketDataFetcher:
    """Deterministic synthetic data generator with in-memory caching."""

    _cache: Dict[Tuple[str, int, int], np.ndarray] = field(default_factory=dict)

    def get_synthetic(self, symbol: str, points: int, seed: int = 42) -> np.ndarray:
        """Generate or retrieve cached synthetic close prices.

        Args:
            symbol: Instrument symbol (used as part of cache key).
            points: Number of data points to generate.
            seed: Random seed for reproducibility.

        Returns:
            1-D array of synthetic close prices.

        Raises:
            ValueError: If ``points`` is not a positive integer.
        """
        if points <= 0:
            raise ValueError("points must be positive.")
        key = (symbol, points, seed)
        if key in self._cache:
            return self._cache[key].copy()
        rng = np.random.default_rng(seed)
        innovations = rng.normal(loc=0.0003, scale=0.012, size=points)
        series = 100.0 * np.exp(np.cumsum(innovations))
        self._cache[key] = series
        return series.copy()

    def is_cached(self, symbol: str, points: int, seed: int = 42) -> bool:
        """Return whether synthetic data for the key exists in memory.

        Args:
            symbol: Instrument symbol.
            points: Number of generated points.
            seed: Random seed used for generation.

        Returns:
            ``True`` if the data is already cached.
        """
        return (symbol, points, seed) in self._cache


def fetch_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    source: str = "YahooFinance",
) -> pd.DataFrame:
    """Fetch OHLCV market data as a tidy DataFrame.

    Args:
        symbol: Ticker or instrument identifier (e.g. ``"AAPL"``,
            ``"BTC-USD"``, ``"^GSPC"``).
        start_date: Inclusive start date in ``YYYY-MM-DD`` format.
        end_date: Inclusive end date in ``YYYY-MM-DD`` format.
        source: Data source — one of ``"YahooFinance"``, ``"Synthetic"``,
            ``"AlphaVantage"`` (reserved), or ``"Quandl"`` (reserved).

    Returns:
        DataFrame indexed by date with columns
        ``open``, ``high``, ``low``, ``close``, ``volume``.

    Raises:
        ValueError: Unsupported source, invalid dates, or empty result.
        NotImplementedError: Reserved sources (AlphaVantage, Quandl).

    Examples:
        >>> df = fetch_ohlcv("AAPL", "2024-01-01", "2024-03-01")
        >>> list(df.columns)
        ['open', 'high', 'low', 'close', 'volume']

        >>> df = fetch_ohlcv("SPY", "2023-01-01", "2023-12-31", source="Synthetic")
        >>> len(df) > 200
        True
    """
    if source not in VALID_SOURCES:
        raise ValueError(
            f"Unsupported source '{source}'. Choose from: {sorted(VALID_SOURCES)}"
        )
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    if start_dt >= end_dt:
        raise ValueError(
            f"start_date '{start_date}' must be before end_date '{end_date}'."
        )
    if source == "YahooFinance":
        return _yfinance_download(symbol, start_date, end_date)
    if source == "Synthetic":
        frame = _synthetic_ohlcv(symbol, start_date, end_date)
        if frame.empty:
            raise ValueError("No business days in the requested synthetic range.")
        return frame
    raise NotImplementedError(
        f"Source '{source}' is reserved and not yet implemented. "
        "Use 'YahooFinance' or 'Synthetic'."
    )


# ===========================================================================
# New DataFetcher class (2026 API)
# ===========================================================================

_OHLCV_REQUIRED = {"open", "high", "low", "close", "volume"}


class DataFetcher:
    """High-level data fetcher with config-dict API and optional disk cache.

    Args:
        cache_dir: Directory for optional persistent caching (unused for
            synthetic data, reserved for future live sources).

    Usage::

        fetcher = DataFetcher(cache_dir="/tmp/cache")
        df = fetcher.fetch({"ticker": "SPY", "source": "synthetic",
                            "start": "2020-01-01", "end": "2023-12-31"})
        universe = fetcher.fetch_universe(["SPY", "QQQ"], {...})
    """

    def __init__(self, cache_dir: str = "/tmp/openqant_cache") -> None:
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, config: dict) -> pd.DataFrame:
        """Fetch and clean OHLCV data according to *config*.

        Args:
            config: Dict with keys:

                * ``ticker``  \u2013 instrument symbol
                * ``source``  \u2013 ``"synthetic"`` or ``"YahooFinance"`` (default ``"synthetic"``)
                * ``start``   \u2013 start date ``YYYY-MM-DD``
                * ``end``     \u2013 end date ``YYYY-MM-DD``

        Returns:
            Cleaned OHLCV DataFrame.

        Raises:
            ValueError: If required columns are missing after fetching.
        """
        source = str(config.get("source", "synthetic")).lower()
        ticker = str(config["ticker"])
        start = str(config["start"])
        end = str(config["end"])

        if source in ("synthetic", "Synthetic"):
            raw = self._fetch_synthetic(ticker, start, end)
        else:
            raw = fetch_ohlcv(ticker, start, end, source=source)

        return self._clean(raw)

    def fetch_universe(self, tickers: list[str], config: dict) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers using the same config template.

        Args:
            tickers: List of instrument symbols.
            config: Base config dict (``ticker`` key is overridden per symbol).

        Returns:
            Dict mapping each ticker to its cleaned OHLCV DataFrame.
        """
        result: dict[str, pd.DataFrame] = {}
        for t in tickers:
            cfg = {**config, "ticker": t}
            result[t] = self.fetch(cfg)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_synthetic(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Generate deterministic synthetic OHLCV through the module-level helper.

        Args:
            ticker: Instrument symbol (used as seed component).
            start: Start date ``YYYY-MM-DD``.
            end: End date ``YYYY-MM-DD``.

        Returns:
            Raw OHLCV DataFrame (not yet cleaned).
        """
        return _synthetic_ohlcv(ticker, start, end)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate schema and remove degenerate rows.

        Args:
            df: Raw OHLCV DataFrame.

        Returns:
            Cleaned DataFrame with positive volume and no missing columns.

        Raises:
            ValueError: If any required OHLCV column is absent.
        """
        missing = _OHLCV_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"Missing column: {sorted(missing)}")
        # Drop zero-volume rows (illiquid / bad ticks)
        return df[df["volume"] > 0].copy()
