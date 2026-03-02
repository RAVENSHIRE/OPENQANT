"""Small-cap mean-reversion strategy implementation."""

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
class SmallCapReversalStrategy:
    """Detect short-term oversold conditions in small-cap series.

    Args:
        lookback: Rolling window for z-score estimate.
        zscore_entry: Entry threshold for oversold reversal.
        zscore_exit: Exit threshold for overbought condition.
    """

    lookback: int = 10
    zscore_entry: float = -1.2
    zscore_exit: float = 1.0

    def generate_signal(self, prices: Sequence[float]) -> Signal:
        """Generate reversal signal from rolling z-score.

        Args:
            prices: Ordered close prices.

        Returns:
            Signal with `entry`, `exit`, or `hold` state.

        Raises:
            ValueError: If insufficient data is provided.
        """
        values = np.asarray(prices, dtype=float)
        if values.size <= self.lookback:
            raise ValueError("Not enough prices for reversal calculation.")

        window = values[-self.lookback :]
        mean = float(np.mean(window))
        std = float(np.std(window, ddof=1)) if window.size > 1 else 0.0
        if std == 0.0:
            return Signal(state="hold", confidence=0.5)

        zscore = (float(values[-1]) - mean) / std
        confidence = min(1.0, abs(zscore) / 2.0)

        if zscore <= self.zscore_entry:
            return Signal(state="entry", confidence=confidence)
        if zscore >= self.zscore_exit:
            return Signal(state="exit", confidence=confidence)
        return Signal(state="hold", confidence=0.5)
