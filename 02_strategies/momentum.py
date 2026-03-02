"""Seasonal equity momentum strategy — DataFrame API.

Backward-compatible: the legacy ``generate_signal(prices)`` scalar API is
preserved alongside the new ``fit()`` / ``generate_signals(data)`` interface.
"""

from __future__ import annotations

import sys
from importlib import util
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: load BaseStrategy / TradeSignal from 01_core
# ---------------------------------------------------------------------------

def _load_module(rel: str, key: str):
    if key in sys.modules:
        return sys.modules[key]
    p = Path(__file__).resolve().parents[1] / rel
    spec = util.spec_from_file_location(key, p)
    mod = util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[key] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_base_mod = _load_module("01_core/base_strategy.py", "base_strategy")
_signal_mod = _load_module("04_signals/signal.py", "signal_model")

BaseStrategy = _base_mod.BaseStrategy
TradeSignal = _base_mod.TradeSignal
Signal = _signal_mod.Signal  # legacy compat


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class SeasonalMomentumStrategy(BaseStrategy):  # type: ignore[misc]
    """Generate seasonal momentum signals from rolling returns.

    Accepts both a ``config`` dict (new API) and plain dataclass-style kwargs
    (legacy API) so existing callers are not broken.

    Args:
        config: Strategy config dict with optional keys:

            * ``lookback``        – momentum window (default 20)
            * ``entry_threshold`` – min return for LONG (default 0.02)
            * ``exit_threshold``  – return below which FLAT (default -0.01)
    """

    # legacy frozen-dataclass defaults kept as class attributes
    lookback: int = 20
    entry_threshold: float = 0.02
    exit_threshold: float = -0.01

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.lookback = int(cfg.get("lookback", self.__class__.lookback))
        self.entry_threshold = float(cfg.get("entry_threshold", self.__class__.entry_threshold))
        self.exit_threshold = float(cfg.get("exit_threshold", self.__class__.exit_threshold))

    # ------------------------------------------------------------------
    # New DataFrame API
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Return signals for the latest bar of *data*.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.

        Returns:
            Single-element list with a :class:`TradeSignal`.

        Raises:
            ValueError: If fewer bars than ``lookback`` are available.
        """
        close = data["close"]
        if len(close) <= self.lookback:
            raise ValueError("Not enough prices for momentum calculation.")

        atr_s = self._atr(data)
        current_atr = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else float(close.iloc[-1]) * 0.02
        entry_price = float(close.iloc[-1])
        momentum = (entry_price / float(close.iloc[-(self.lookback + 1)])) - 1.0

        # March seasonality boost
        is_march = hasattr(data.index, "month") and int(data.index[-1].month) == 3
        seasonal_boost = 0.05 if is_march else 0.0

        if momentum >= self.entry_threshold or (momentum > 0 and seasonal_boost > 0):
            confidence = min(1.0, abs(momentum) * 5.0 + 0.5 + seasonal_boost)
            stop_loss = entry_price - max(current_atr * 1.5, entry_price * 0.02)
            take_profit = entry_price + max(current_atr * 4.5, entry_price * 0.06)
            return [TradeSignal(
                ticker=self.ticker, direction="LONG", confidence=confidence,
                entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                strategy_name=self.name,
            )]

        confidence = min(1.0, abs(momentum) * 5.0 + 0.2)
        stop_loss = entry_price * 0.98
        take_profit = entry_price * 1.04
        return [TradeSignal(
            ticker=self.ticker, direction="FLAT", confidence=confidence,
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            strategy_name=self.name,
        )]

    # ------------------------------------------------------------------
    # Legacy scalar API (backward compat for old tests / integration)
    # ------------------------------------------------------------------

    def generate_signal(self, prices: Sequence[float]) -> Signal:
        """Legacy scalar API — accepts raw price sequence.

        Args:
            prices: Ordered close prices.

        Returns:
            :class:`Signal` with ``state`` and ``confidence``.
        """
        values = np.asarray(prices, dtype=float)
        if values.size <= self.lookback:
            raise ValueError("Not enough prices for momentum calculation.")
        start = float(values[-self.lookback - 1])
        end = float(values[-1])
        momentum = (end / start) - 1.0
        if momentum >= self.entry_threshold:
            return Signal(state="entry", confidence=min(1.0, momentum * 5.0 + 0.5))
        if momentum <= self.exit_threshold:
            return Signal(state="exit", confidence=min(1.0, abs(momentum) * 5.0 + 0.5))
        return Signal(state="hold", confidence=0.5)
