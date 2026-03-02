"""Crypto momentum strategy for high-volatility assets — DataFrame API.

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

_MAX_CRYPTO_POSITION = 0.20  # Kelly cap for crypto


class CryptoMomentumStrategy(BaseStrategy):  # type: ignore[misc]
    """Momentum signals adapted for high crypto volatility.

    Args:
        config: Strategy config dict with optional keys:

            * ``lookback``              – momentum window (default 14)
            * ``entry_threshold``       – min return for LONG (default 0.04)
            * ``exit_threshold``        – return for FLAT (default -0.03)
            * ``rsi_exit_threshold``    – RSI level above which no LONG (default 75)
            * ``trailing_stop_atr_mult``– ATR multiplier for stop (default 3.0)
    """

    lookback: int = 14
    entry_threshold: float = 0.04
    exit_threshold: float = -0.03

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.lookback = int(cfg.get("lookback", self.__class__.lookback))
        self.entry_threshold = float(cfg.get("entry_threshold", self.__class__.entry_threshold))
        self.exit_threshold = float(cfg.get("exit_threshold", self.__class__.exit_threshold))
        self.rsi_exit_threshold = float(cfg.get("rsi_exit_threshold", 75.0))
        self.trailing_stop_atr_mult = float(cfg.get("trailing_stop_atr_mult", 3.0))

    # ------------------------------------------------------------------
    # New DataFrame API
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Return crypto momentum signal for the latest bar.

        Includes ``suggested_position_pct`` (Kelly fraction, capped at 20%)
        in :attr:`TradeSignal.metadata` for LONG signals.

        Args:
            data: OHLCV DataFrame.

        Returns:
            Single-element list with a :class:`TradeSignal`.
        """
        close = data["close"]
        if len(close) <= self.lookback:
            raise ValueError("Not enough prices for crypto momentum calculation.")

        atr_s = self._atr(data)
        rsi_s = self._rsi(close)
        current_atr = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else float(close.iloc[-1]) * 0.03
        current_rsi = float(rsi_s.iloc[-1]) if not np.isnan(rsi_s.iloc[-1]) else 50.0
        entry_price = float(close.iloc[-1])
        momentum = (entry_price / float(close.iloc[-(self.lookback + 1)])) - 1.0
        confidence = min(1.0, abs(momentum) * 6.0)

        if momentum >= self.entry_threshold and current_rsi < self.rsi_exit_threshold:
            stop_loss = entry_price - self.trailing_stop_atr_mult * current_atr
            take_profit = entry_price + self.trailing_stop_atr_mult * current_atr * 2.0
            # Kelly position sizing
            win_prob = min(0.9, 0.5 + momentum * 2.0)
            payout = 2.0
            raw_kelly = win_prob - (1.0 - win_prob) / payout
            kelly_pct = min(_MAX_CRYPTO_POSITION, max(0.0, raw_kelly))
            return [TradeSignal(
                ticker=self.ticker, direction="LONG", confidence=confidence,
                entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                strategy_name=self.name,
                metadata={"suggested_position_pct": round(kelly_pct, 4)},
            )]

        stop_loss = entry_price * 0.97
        take_profit = entry_price * 1.06
        return [TradeSignal(
            ticker=self.ticker, direction="FLAT", confidence=confidence,
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            strategy_name=self.name,
            metadata={"suggested_position_pct": 0.0},
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
        if values.size <= self.lookback:
            raise ValueError("Not enough prices for crypto momentum calculation.")
        start = float(values[-self.lookback - 1])
        end = float(values[-1])
        momentum = (end / start) - 1.0
        confidence = min(1.0, abs(momentum) * 6.0)
        if momentum >= self.entry_threshold:
            return Signal(state="entry", confidence=confidence)
        if momentum <= self.exit_threshold:
            return Signal(state="exit", confidence=confidence)
        return Signal(state="hold", confidence=0.5)
