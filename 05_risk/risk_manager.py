"""Risk management \u2014 position sizing, drawdown halt, regime filter.

Two public APIs co-exist:

* **New API** \u2014 :class:`RiskConfig` + mutable :class:`RiskManager`
  (used by the current test suite and recommended for new code).
* **Legacy API** \u2014 :class:`KellyCriterion` function + frozen
  :class:`_LegacyRiskManager` dataclass (kept for backward compat).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ===========================================================================
# Shared helpers
# ===========================================================================

def KellyCriterion(win_probability: float, payout_ratio: float) -> float:
    """Calculate optimal risk fraction using the Kelly criterion.

    Args:
        win_probability: Probability of a winning trade in ``[0, 1]``.
        payout_ratio: Mean win divided by mean loss (must be > 0).

    Returns:
        Kelly fraction clipped to ``[0, 1]``.

    Raises:
        ValueError: If inputs are outside valid ranges.
    """
    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("win_probability must be between 0 and 1.")
    if payout_ratio <= 0.0:
        raise ValueError("payout_ratio must be positive.")
    raw = win_probability - ((1.0 - win_probability) / payout_ratio)
    return min(max(raw, 0.0), 1.0)


# ===========================================================================
# New API
# ===========================================================================

@dataclass
class RiskConfig:
    """Immutable risk parameters used by :class:`RiskManager`.

    Args:
        total_capital: Starting equity in account currency.
        max_risk_per_trade_pct: Maximum capital at risk per trade (e.g. 0.02 = 2 %).
        max_position_pct: Hard cap on single-position market value (e.g. 0.20 = 20 %).
        max_drawdown_threshold: Halt trading when drawdown exceeds this level.
        use_regime_filter: Whether to apply the SMA-based regime filter.
        regime_sma_period: SMA lookback for regime detection.
    """

    total_capital: float = 100_000.0
    max_risk_per_trade_pct: float = 0.02
    max_position_pct: float = 0.20
    max_drawdown_threshold: float = 0.20
    use_regime_filter: bool = True
    regime_sma_period: int = 200


class RiskManager:
    """Stateful risk gateway: position sizing, drawdown halt, regime filter.

    Maintains running ``equity`` and ``peak_equity`` across calls.

    Args:
        config: :class:`RiskConfig` instance (uses defaults if omitted).

    Legacy usage (backward compat)::

        rm = RiskManager(max_drawdown=0.10)   # old kwargs still work
        rm.breach_drawdown(peak_equity=100, current_equity=85)
    """

    def __init__(self, config: RiskConfig | None = None, **legacy_kwargs: Any) -> None:
        # Legacy path: RiskManager(max_drawdown=0.1)
        if config is None and legacy_kwargs:
            config = RiskConfig(
                max_drawdown_threshold=float(legacy_kwargs.get("max_drawdown", 0.20)),
            )
        self.config: RiskConfig = config or RiskConfig()
        self.equity: float = self.config.total_capital
        self.peak_equity: float = self.config.total_capital
        self._positions: dict[str, float] = {}  # ticker -> invested value

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _calc_position_size(self, signal: Any, market_data: dict) -> float:
        """Calculate position size in units respecting all risk limits.

        Args:
            signal: :class:`TradeSignal`-compatible object with
                ``entry_price``, ``stop_loss``, ``confidence``.
            market_data: Dict mapping ticker to OHLCV DataFrame (unused
                for sizing, inspected for regime filter).

        Returns:
            Number of units to buy (0.0 if no valid position possible).
        """
        entry = float(signal.entry_price)
        stop = float(signal.stop_loss)
        risk_per_unit = abs(entry - stop)

        if risk_per_unit < 1e-9 or entry <= 0:
            return 0.0

        # Capital at risk = capital × max_risk_pct × confidence
        confidence = min(max(float(signal.confidence), 0.0), 1.0)
        capital_at_risk = self.equity * self.config.max_risk_per_trade_pct * confidence
        size = capital_at_risk / risk_per_unit

        # Hard cap: position market value ≤ max_position_pct × confidence × equity
        max_value = self.equity * self.config.max_position_pct * confidence
        size = min(size, max_value / entry)

        return max(0.0, size)

    # ------------------------------------------------------------------
    # Drawdown halt
    # ------------------------------------------------------------------

    def _in_drawdown_halt(self) -> bool:
        """Return ``True`` when current drawdown exceeds the configured threshold.

        Returns:
            ``True`` if trading should be halted.
        """
        if self.peak_equity <= 0:
            return False
        drawdown = 1.0 - (self.equity / self.peak_equity)
        return drawdown >= self.config.max_drawdown_threshold

    # backward compat alias
    def breach_drawdown(self, peak_equity: float, current_equity: float) -> bool:
        """Legacy API \u2014 check drawdown against configured threshold.

        Args:
            peak_equity: Historical equity peak.
            current_equity: Current equity value.

        Returns:
            ``True`` if drawdown >= :attr:`RiskConfig.max_drawdown_threshold`.
        """
        if peak_equity <= 0 or current_equity <= 0:
            raise ValueError("equity values must be positive.")
        dd = 1.0 - (current_equity / peak_equity)
        return dd >= self.config.max_drawdown_threshold

    # ------------------------------------------------------------------
    # Regime filter
    # ------------------------------------------------------------------

    def _regime_is_bearish(self, ticker: str, market_data: dict) -> bool:
        """Return ``True`` when price is below its long-term SMA (bearish regime).

        Always returns ``False`` when :attr:`RiskConfig.use_regime_filter` is
        disabled or when ``ticker`` is not found in ``market_data``.

        Args:
            ticker: Instrument symbol.
            market_data: Dict mapping ticker to OHLCV DataFrame.

        Returns:
            ``True`` if regime is bearish, ``False`` otherwise.
        """
        if not self.config.use_regime_filter:
            return False
        df = market_data.get(ticker)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return False
        close = df["close"] if "close" in df.columns else None
        if close is None or len(close) < self.config.regime_sma_period:
            return False
        sma = float(close.rolling(self.config.regime_sma_period).mean().iloc[-1])
        return float(close.iloc[-1]) < sma

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def process_signals(self, signals: list, market_data: dict) -> list[dict]:
        """Filter and size a list of signals, respecting halt and regime rules.

        Args:
            signals: List of :class:`TradeSignal`-compatible objects.
            market_data: Dict mapping ticker to OHLCV DataFrame.

        Returns:
            List of order dicts ``{ticker, direction, size}`` \u2014 empty when in halt.
        """
        if self._in_drawdown_halt():
            return []
        orders = []
        for sig in signals:
            if self._regime_is_bearish(getattr(sig, "ticker", ""), market_data):
                continue
            size = self._calc_position_size(sig, market_data)
            if size > 0:
                orders.append({
                    "ticker": getattr(sig, "ticker", ""),
                    "direction": getattr(sig, "direction", "FLAT"),
                    "size": round(size, 6),
                })
        return orders

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a snapshot of the current risk state.

        Returns:
            Dict with keys: ``equity``, ``peak_equity``, ``drawdown``,
            ``n_positions``, ``invested_pct``.
        """
        drawdown = (
            max(0.0, 1.0 - self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0
        )
        invested = sum(self._positions.values())
        invested_pct = (invested / self.equity) if self.equity > 0 else 0.0
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "drawdown": round(drawdown, 6),
            "n_positions": len(self._positions),
            "invested_pct": round(invested_pct, 6),
        }


# ===========================================================================
# Legacy frozen dataclass (kept for backward compat)
# ===========================================================================

@dataclass(frozen=True)
class _LegacyRiskManager:
    """Frozen risk manager \u2014 kept for backward compat only."""

    max_position_fraction: float = 0.10
    max_drawdown: float = 0.20

    def position_size(self, equity: float, price: float, confidence: float) -> float:
        if equity <= 0 or price <= 0:
            raise ValueError("equity and price must be positive.")
        allocation = equity * self.max_position_fraction * min(max(confidence, 0.0), 1.0)
        return allocation / price

    def breach_drawdown(self, peak_equity: float, current_equity: float) -> bool:
        if peak_equity <= 0 or current_equity <= 0:
            raise ValueError("equity values must be positive.")
        return (1.0 - current_equity / peak_equity) >= self.max_drawdown
