"""Consensus signal aggregation functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AggregationResult:
    """Summary of signal aggregation.

    Args:
        state: Aggregated state (`entry`, `exit`, `hold`).
        confidence: Confidence score in range [0, 1].
    """

    state: str
    confidence: float


def aggregate_consensus(signals: Sequence[str], min_votes: int = 2) -> AggregationResult:
    """Aggregate strategy votes into a consensus directional signal.

    Args:
        signals: Iterable of strategy states (`entry`, `exit`, `hold`).
        min_votes: Minimum absolute score required for non-neutral output.

    Returns:
        Aggregated result with consensus state and confidence.

    Raises:
        ValueError: If `min_votes` is not positive.
    """
    if min_votes <= 0:
        raise ValueError("min_votes must be positive.")

    entry_votes = sum(1 for signal in signals if signal == "entry")
    exit_votes = sum(1 for signal in signals if signal == "exit")
    total = max(len(signals), 1)

    if entry_votes >= min_votes and entry_votes > exit_votes:
        return AggregationResult(state="entry", confidence=entry_votes / total)
    if exit_votes >= min_votes and exit_votes > entry_votes:
        return AggregationResult(state="exit", confidence=exit_votes / total)
    return AggregationResult(state="hold", confidence=0.5)
