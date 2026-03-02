"""Backtest engine coordinating strategy, risk and return calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BacktestEngine:
    """Run a minimal vectorized backtest over price data."""

    transaction_cost_bps: float = 5.0

    def run(self, prices: Sequence[float], signals: Sequence[str]) -> dict[str, float]:
        """Run simulation and return key metrics.

        Args:
            prices: Ordered close prices.
            signals: Signal states aligned to `prices` (`entry`, `exit`, `hold`).

        Returns:
            Summary dictionary with cumulative return and Sharpe ratio.

        Raises:
            ValueError: If inputs have incompatible shapes.
        """
        price_values = np.asarray(prices, dtype=float)
        signal_values = np.asarray([_state_to_position(state) for state in signals], dtype=float)
        if price_values.size != signal_values.size:
            raise ValueError("prices and signals must have same length.")
        if price_values.size < 3:
            raise ValueError("At least three datapoints are required.")

        returns = np.diff(price_values) / price_values[:-1]
        shifted_signals = signal_values[:-1]
        gross = shifted_signals * returns

        turnover = np.abs(np.diff(signal_values, prepend=signal_values[0]))
        costs = (turnover[:-1] * self.transaction_cost_bps) / 10_000.0
        net = gross - costs

        cumulative = float(np.prod(1.0 + net) - 1.0)
        volatility = float(np.std(net, ddof=1)) if net.size > 1 else 0.0
        sharpe = float(np.mean(net) / volatility * np.sqrt(252)) if volatility > 0 else 0.0
        return {"cumulative_return": cumulative, "sharpe": sharpe}


def _state_to_position(state: str) -> int:
    """Map signal state to numeric portfolio position.

    Args:
        state: One of `entry`, `exit`, `hold`.

    Returns:
        Numeric position in `{-1, 0, 1}`.

    Raises:
        ValueError: If state is unknown.
    """
    if state == "entry":
        return 1
    if state == "exit":
        return -1
    if state == "hold":
        return 0
    raise ValueError(f"Unknown state '{state}'.")
