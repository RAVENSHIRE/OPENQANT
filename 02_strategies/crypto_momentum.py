"""Crypto momentum strategy for high-volatility assets."""

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
class CryptoMomentumStrategy:
    """Compute momentum signals adapted for crypto volatility.

    Args:
        lookback: Window length for return estimation.
        entry_threshold: Threshold above which an entry is emitted.
        exit_threshold: Threshold below which an exit is emitted.
    """

    lookback: int = 14
    entry_threshold: float = 0.04
    exit_threshold: float = -0.03

    def generate_signal(self, prices: Sequence[float]) -> Signal:
        """Generate `entry`, `exit`, or `hold` from crypto momentum.

        Args:
            prices: Ordered close prices.

        Returns:
            Signal object compliant with signal engine interfaces.

        Raises:
            ValueError: If not enough prices are provided.
        """
        values = np.asarray(prices, dtype=float)
        if values.size <= self.lookback:
            raise ValueError("Not enough prices for crypto momentum calculation.")

        start_price = float(values[-self.lookback - 1])
        end_price = float(values[-1])
        momentum = (end_price / start_price) - 1.0
        confidence = min(1.0, abs(momentum) * 6.0)

        if momentum >= self.entry_threshold:
            return Signal(state="entry", confidence=confidence)
        if momentum <= self.exit_threshold:
            return Signal(state="exit", confidence=confidence)
        return Signal(state="hold", confidence=0.5)
