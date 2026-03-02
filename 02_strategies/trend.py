"""Trend-following breakout strategy implementation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


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
class TrendBreakoutStrategy:
    """Detect breakouts beyond historical channel bounds.

    Args:
        channel_window: Lookback for channel high/low.
    """

    channel_window: int = 30

    def generate_signal(self, prices: Sequence[float]) -> Signal:
        """Create a breakout signal.

        Args:
            prices: Ordered close prices.

        Returns:
            Signal with `entry`, `exit` or `hold`.

        Raises:
            ValueError: If `prices` length is smaller than the channel window.
        """
        values = np.asarray(prices, dtype=float)
        if values.size <= self.channel_window:
            raise ValueError("Not enough prices for breakout calculation.")

        reference = values[-self.channel_window - 1 : -1]
        current = float(values[-1])
        if current > float(np.max(reference)):
            return Signal(state="entry", confidence=0.8)
        if current < float(np.min(reference)):
            return Signal(state="exit", confidence=0.8)
        return Signal(state="hold", confidence=0.5)

