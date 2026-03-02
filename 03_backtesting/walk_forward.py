"""Walk-forward utilities for rolling out-of-sample evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def walk_forward_scores(prices: Sequence[float], folds: int = 4) -> list[float]:
    """Compute mean fold returns across walk-forward partitions.

    Args:
        prices: Ordered close prices.
        folds: Number of walk-forward folds.

    Returns:
        List with one score per fold.

    Raises:
        ValueError: If folds are invalid or data is insufficient.
    """
    if folds <= 0:
        raise ValueError("folds must be positive.")

    values = np.asarray(prices, dtype=float)
    fold_size = values.size // folds
    if fold_size < 2:
        raise ValueError("Insufficient data for requested fold count.")

    scores: list[float] = []
    for fold in range(folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold < folds - 1 else values.size
        fold_values = values[start:end]
        fold_returns = np.diff(fold_values) / fold_values[:-1]
        scores.append(float(np.mean(fold_returns)))
    return scores
