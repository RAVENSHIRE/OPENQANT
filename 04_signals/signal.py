"""Domain signal model shared across strategies and orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    """Represent strategy output in a consistent API format.

    Args:
        state: Signal state, one of `entry`, `exit`, or `hold`.
        confidence: Confidence score in the range [0, 1].
    """

    state: str
    confidence: float

    def __post_init__(self) -> None:
        """Validate signal invariants.

        Raises:
            ValueError: If state or confidence are invalid.
        """
        if self.state not in {"entry", "exit", "hold"}:
            raise ValueError("state must be one of: entry, exit, hold")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
