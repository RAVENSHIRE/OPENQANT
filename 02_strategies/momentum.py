"""Seasonal equity momentum strategy implementation."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Sequence

import numpy as np


from importlib import util
from pathlib import Path


def _load_signal_model() -> type:
    """Load `Signal` dataclass from the signal module.

    Returns:
        Signal class object.
    """
    module_path = Path(__file__).resolve().parents[1] / "04_signals" / "signal.py"
    spec = util.spec_from_file_location("signal_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load signal model module.")
    module = util.module_from_spec(spec)
    sys.modules["signal_model"] = module
    spec.loader.exec_module(module)
    return module.Signal


Signal = _load_signal_model()


@dataclass(frozen=True)
class SeasonalMomentumStrategy:
    """Generate seasonal momentum signals from rolling returns.

    Args:
        lookback: Number of bars used for momentum estimation.
        entry_threshold: Minimum return threshold to emit an `entry` signal.
        exit_threshold: Return threshold below which an `exit` signal is emitted.
    """

    lookback: int = 20
    entry_threshold: float = 0.02
    exit_threshold: float = -0.01

    def generate_signal(self, prices: Sequence[float]) -> Signal:
        """Compute momentum signal as documented API object.

        Args:
            prices: Ordered close prices.

        Returns:
            Signal with one of `entry`, `exit`, `hold`.

        Raises:
            ValueError: If not enough prices are provided.
        """
        values = np.asarray(prices, dtype=float)
        if values.size <= self.lookback:
            raise ValueError("Not enough prices for momentum calculation.")

        start_price = float(values[-self.lookback - 1])
        end_price = float(values[-1])
        momentum = (end_price / start_price) - 1.0
        if momentum >= self.entry_threshold:
            return Signal(state="entry", confidence=min(1.0, momentum * 5.0 + 0.5))
        if momentum <= self.exit_threshold:
            return Signal(state="exit", confidence=min(1.0, abs(momentum) * 5.0 + 0.5))
        return Signal(state="hold", confidence=0.5)
