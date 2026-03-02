"""Base strategy abstract class and TradeSignal domain model.

All strategy modules in ``02_strategies`` must subclass :class:`BaseStrategy`
and implement :meth:`generate_signals`.

Usage
-----
    from base_strategy import BaseStrategy, TradeSignal

    class MyStrategy(BaseStrategy):
        def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


@dataclass
class TradeSignal:
    """Unified signal model returned by every strategy.

    Args:
        ticker: Instrument identifier (e.g. ``"AAPL"``, ``"BTC-USD"``).
        direction: One of ``"LONG"``, ``"SHORT"``, or ``"FLAT"``.
        confidence: Signal strength in ``[0.0, 1.0]``.
        entry_price: Suggested entry price.
        stop_loss: Stop-loss price level.
        take_profit: Take-profit price level.
        strategy_name: Human-readable strategy identifier.
        metadata: Arbitrary extra data (Kelly fraction, hold-days, …).
    """

    ticker: str
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {"LONG", "SHORT", "FLAT"}:
            raise ValueError(
                f"direction must be 'LONG', 'SHORT', or 'FLAT', got {self.direction!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")


class BaseStrategy(ABC):
    """Abstract base class for all OPENQANT strategy modules.

    Subclasses must implement :meth:`generate_signals`.  The :meth:`fit`
    method stores validated OHLCV data and sets ``self._fitted = True``.

    Args:
        config: Strategy configuration dict.  Expected keys:

            * ``name``   – human-readable label (default: class name)
            * ``ticker`` – instrument symbol (default: ``"UNKNOWN"``)
    """

    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.name: str = config.get("name", self.__class__.__name__)
        self.ticker: str = config.get("ticker", "UNKNOWN")
        self._fitted: bool = False
        self._data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> None:
        """Validate and store training data.

        Args:
            data: OHLCV DataFrame with a DatetimeIndex.

        Raises:
            ValueError: If required columns are missing.
        """
        self.validate_data(data)
        self._data = data.copy()
        self._fitted = True

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Return a list of :class:`TradeSignal` objects for *data*.

        Args:
            data: OHLCV DataFrame (same schema as in :meth:`fit`).

        Returns:
            Non-empty list of :class:`TradeSignal` instances.
        """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate OHLCV schema and forward-fill NaN values in-place.

        Args:
            data: Input DataFrame.

        Returns:
            ``True`` if validation passes.

        Raises:
            ValueError: If any required column is absent.
        """
        missing = _REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        return True

    # ------------------------------------------------------------------
    # Shared technical indicators
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Wilder RSI.

        Args:
            close: Close price series.
            period: Smoothing period.

        Returns:
            RSI series in ``[0, 100]`` (NaN for insufficient history).
        """
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range (Wilder smoothing).

        Args:
            data: OHLCV DataFrame.
            period: Smoothing period.

        Returns:
            ATR series (non-negative, NaN for first ``period`` bars).
        """
        high = data["high"]
        low = data["low"]
        prev_close = data["close"].shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()
