"""Monte-Carlo helpers for distribution-based scenario testing."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def monte_carlo_terminal_values(
    prices: Sequence[float], simulations: int = 500, horizon: int = 20, seed: int = 11
) -> np.ndarray:
    """Sample bootstrap return paths and compute terminal multipliers.

    Args:
        prices: Ordered close prices.
        simulations: Number of simulated paths.
        horizon: Number of sampled steps per path.
        seed: Random seed for deterministic runs.

    Returns:
        Array of terminal multipliers for each simulation.

    Raises:
        ValueError: If simulations/horizon are non-positive.
    """
    if simulations <= 0 or horizon <= 0:
        raise ValueError("simulations and horizon must be positive.")

    values = np.asarray(prices, dtype=float)
    returns = np.diff(values) / values[:-1]
    if returns.size == 0:
        raise ValueError("At least two prices are required.")

    rng = np.random.default_rng(seed)
    sampled = rng.choice(returns, size=(simulations, horizon), replace=True)
    return np.prod(1.0 + sampled, axis=1)
