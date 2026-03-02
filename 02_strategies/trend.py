"""Trend-following breakout strategy — DataFrame API.

Legacy ``generate_signal(prices)`` is preserved for backward compatibility.
"""

from __future__ import annotations

import sys
from importlib import util
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


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
Signal = _signal_mod.Signal


class TrendBreakoutStrategy(BaseStrategy):  # type: ignore[misc]
    """Detect breakouts beyond historical channel bounds.

    Args:
        config: Strategy config dict with optional keys:

            * ``channel_window``      – lookback for high/low channel (default 30)
            * ``atr_multiplier_stop`` – stop ATR multiple (default 1.5)
            * ``risk_reward_ratio``   – minimum R:R (default 3.0)
    """

    channel_window: int = 30

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.channel_window = int(cfg.get("channel_window", self.__class__.channel_window))
        self.atr_multiplier_stop = float(cfg.get("atr_multiplier_stop", 1.5))
        self.risk_reward_ratio = float(cfg.get("risk_reward_ratio", 3.0))

    # ------------------------------------------------------------------
    # New DataFrame API
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Return breakout signal for the latest bar.

        Args:
            data: OHLCV DataFrame.

        Returns:
            Single-element list with a :class:`TradeSignal`.
        """
        close = data["close"]
        if len(close) <= self.channel_window:
            raise ValueError("Not enough prices for breakout calculation.")

        atr_s = self._atr(data)
        current_atr = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else float(close.iloc[-1]) * 0.02
        entry_price = float(close.iloc[-1])
        reference = close.iloc[-(self.channel_window + 1):-1]
        chan_high = float(reference.max())
        chan_low = float(reference.min())

        if entry_price > chan_high:
            stop_loss = entry_price - self.atr_multiplier_stop * current_atr
            risk = max(entry_price - stop_loss, entry_price * 0.005)
            take_profit = entry_price + self.risk_reward_ratio * risk
            return [TradeSignal(
                ticker=self.ticker, direction="LONG", confidence=0.8,
                entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                strategy_name=self.name,
            )]

        stop_loss = entry_price * 0.98
        take_profit = entry_price * 1.04
        return [TradeSignal(
            ticker=self.ticker, direction="FLAT", confidence=0.4,
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            strategy_name=self.name,
        )]

    # ------------------------------------------------------------------
    # Legacy scalar API
    # ------------------------------------------------------------------

    def generate_signal(self, prices: Sequence[float]) -> Signal:
        """Legacy scalar API.

        Args:
            prices: Ordered close prices.

        Returns:
            :class:`Signal` with ``state`` and ``confidence``.
        """
        values = np.asarray(prices, dtype=float)
        if values.size <= self.channel_window:
            raise ValueError("Not enough prices for breakout calculation.")
        reference = values[-self.channel_window - 1:-1]
        current = float(values[-1])
        if current > float(np.max(reference)):
            return Signal(state="entry", confidence=0.8)
        if current < float(np.min(reference)):
            return Signal(state="exit", confidence=0.8)
        return Signal(state="hold", confidence=0.5)

