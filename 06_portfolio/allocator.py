"""Portfolio allocator using confidence-weighted scores."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def confidence_weighted_allocation(confidences: Sequence[float]) -> np.ndarray:
    """Convert confidence scores into normalized allocation weights.

    Args:
        confidences: Non-negative confidence scores per asset.

    Returns:
        Normalized weight vector summing to one; zeros if all scores are zero.

    Raises:
        ValueError: If any confidence is negative.
    """
    values = np.asarray(confidences, dtype=float)
    if np.any(values < 0.0):
        raise ValueError("Confidence values must be non-negative.")

    total = float(np.sum(values))
    if total == 0.0:
        return np.zeros_like(values)
    return values / total
