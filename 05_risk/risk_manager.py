"""Risk manager for position limits and safety checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskManager:
    """Enforce core risk constraints for orders.

    Args:
        max_position_fraction: Maximum fraction of equity for a single position.
        max_drawdown: Maximum tolerated drawdown.
    """

    max_position_fraction: float = 0.10
    max_drawdown: float = 0.20

    def position_size(self, equity: float, price: float, confidence: float) -> float:
        """Calculate capped position quantity.

        Args:
            equity: Available account equity.
            price: Current instrument price.
            confidence: Signal confidence in range [0, 1].

        Returns:
            Position size in units.

        Raises:
            ValueError: If equity or price are non-positive.
        """
        if equity <= 0 or price <= 0:
            raise ValueError("equity and price must be positive.")

        bounded_confidence = min(max(confidence, 0.0), 1.0)
        allocation = equity * self.max_position_fraction * bounded_confidence
        return allocation / price

    def breach_drawdown(self, peak_equity: float, current_equity: float) -> bool:
        """Check whether current drawdown breaches maximum threshold.

        Args:
            peak_equity: Highest observed equity.
            current_equity: Current account equity.

        Returns:
            True if drawdown is at or above configured limit.

        Raises:
            ValueError: If equity inputs are non-positive.
        """
        if peak_equity <= 0 or current_equity <= 0:
            raise ValueError("equity values must be positive.")

        drawdown = 1.0 - (current_equity / peak_equity)
        return drawdown >= self.max_drawdown


def KellyCriterion(win_probability: float, payout_ratio: float) -> float:
    """Calculate optimal risk fraction using Kelly criterion.

    Args:
        win_probability: Probability of a winning trade in range [0, 1].
        payout_ratio: Mean win divided by mean loss (> 0).

    Returns:
        Kelly fraction clipped to [0, 1].

    Raises:
        ValueError: If inputs are outside valid ranges.
    """
    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("win_probability must be between 0 and 1.")
    if payout_ratio <= 0.0:
        raise ValueError("payout_ratio must be positive.")

    raw_fraction = win_probability - ((1.0 - win_probability) / payout_ratio)
    return min(max(raw_fraction, 0.0), 1.0)
