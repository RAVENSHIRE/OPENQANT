"""Small-cap mean-reversion strategy — DataFrame API.

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


class SmallCapReversalStrategy(BaseStrategy):  # type: ignore[misc]
    """Detect short-term oversold conditions in small-cap series.

    Args:
        config: Strategy config dict with optional keys:

            * ``lookback``         – z-score window (default 10)
            * ``zscore_entry``     – entry threshold (default -1.2)
            * ``zscore_exit``      – exit threshold (default 1.0)
            * ``earnings_surprise``– positive earnings surprise [0, 1] (default 0.0)
            * ``hold_max_days``    – maximum holding period in days (default 5)
    """

    lookback: int = 10
    zscore_entry: float = -1.2
    zscore_exit: float = 1.0

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.lookback = int(cfg.get("lookback", self.__class__.lookback))
        self.zscore_entry = float(cfg.get("zscore_entry", self.__class__.zscore_entry))
        self.zscore_exit = float(cfg.get("zscore_exit", self.__class__.zscore_exit))
        self.earnings_surprise = float(cfg.get("earnings_surprise", 0.0))
        self.hold_max_days = int(cfg.get("hold_max_days", 5))

    # ------------------------------------------------------------------
    # New DataFrame API
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> list[TradeSignal]:
        """Return reversal signal for the latest bar.

        ``metadata`` always contains ``max_hold_days``.
        A positive ``earnings_surprise`` boosts ``confidence``.

        Args:
            data: OHLCV DataFrame.

        Returns:
            Single-element list with a :class:`TradeSignal`.
        """
        close = data["close"]
        if len(close) <= self.lookback:
            raise ValueError("Not enough prices for reversal calculation.")

        atr_s = self._atr(data)
        current_atr = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else float(close.iloc[-1]) * 0.02
        entry_price = float(close.iloc[-1])
        window = close.iloc[-self.lookback:].to_numpy()
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0

        meta = {"max_hold_days": self.hold_max_days}

        if std == 0.0:
            return [TradeSignal(
                ticker=self.ticker, direction="FLAT", confidence=0.0,
                entry_price=entry_price, stop_loss=entry_price * 0.98,
                take_profit=entry_price * 1.04, strategy_name=self.name, metadata=meta,
            )]

        zscore = (entry_price - mean) / std
        confidence = min(1.0, abs(zscore) / 2.0 + self.earnings_surprise * 0.2)
        confidence = min(1.0, confidence)

        if zscore <= self.zscore_entry:
            stop_loss = entry_price - max(current_atr * 1.5, entry_price * 0.02)
            take_profit = entry_price + max(current_atr * 3.0, entry_price * 0.04)
            return [TradeSignal(
                ticker=self.ticker, direction="LONG", confidence=confidence,
                entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                strategy_name=self.name, metadata=meta,
            )]

        stop_loss = entry_price * 0.98
        take_profit = entry_price * 1.04
        return [TradeSignal(
            ticker=self.ticker, direction="FLAT", confidence=confidence,
            entry_price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
            strategy_name=self.name, metadata=meta,
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
            raise ValueError("Not enough prices for reversal calculation.")
        window = values[-self.lookback:]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1)) if window.size > 1 else 0.0
        if std == 0.0:
            return Signal(state="hold", confidence=0.5)
        zscore = (float(values[-1]) - mean) / std
        confidence = min(1.0, abs(zscore) / 2.0)
        if zscore <= self.zscore_entry:
            return Signal(state="entry", confidence=confidence)
        if zscore >= self.zscore_exit:
            return Signal(state="exit", confidence=confidence)
        return Signal(state="hold", confidence=0.5)
