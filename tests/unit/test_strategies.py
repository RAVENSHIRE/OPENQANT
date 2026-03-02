"""Unit tests for strategy modules in `02_strategies`."""

from __future__ import annotations

from importlib import util
from pathlib import Path
import sys

import numpy as np


def load_module(module_path: Path, module_name: str):
    """Load module from file path for numeric folder structure."""
    spec = util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_all_four_strategies_return_valid_states(project_root) -> None:
    """Ensure all four strategy modules emit valid signal states."""
    momentum_module = load_module(project_root / "02_strategies" / "momentum.py", "seasonal_momentum")
    trend_module = load_module(project_root / "02_strategies" / "trend.py", "trend_breakout")
    crypto_module = load_module(project_root / "02_strategies" / "crypto_momentum.py", "crypto_momentum")
    reversal_module = load_module(
        project_root / "02_strategies" / "small_cap_reversal.py", "small_cap_reversal"
    )

    prices = np.linspace(100.0, 120.0, 64)
    signals = [
        momentum_module.SeasonalMomentumStrategy().generate_signal(prices),
        trend_module.TrendBreakoutStrategy().generate_signal(prices),
        crypto_module.CryptoMomentumStrategy().generate_signal(prices),
        reversal_module.SmallCapReversalStrategy().generate_signal(prices),
    ]

    for signal in signals:
        assert signal.state in {"entry", "exit", "hold"}
        assert 0.0 <= signal.confidence <= 1.0


def test_strategy_determinism_for_equal_inputs(project_root) -> None:
    """Verify deterministic output for repeated calls with identical input."""
    momentum_module = load_module(project_root / "02_strategies" / "momentum.py", "seasonal_momentum_determinism")
    prices = np.linspace(50.0, 90.0, 40)

    strategy = momentum_module.SeasonalMomentumStrategy()
    first = strategy.generate_signal(prices)
    second = strategy.generate_signal(prices)
    assert first == second
