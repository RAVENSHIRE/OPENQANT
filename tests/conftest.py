"""Global pytest configuration and shared fixtures.

Automatically loaded by pytest before any test is collected.

Path layout
-----------
All sub-packages (01_core \u2026 07_data) are added to ``sys.path`` so tests can
import modules by their bare filename without package prefixes.
"""

from __future__ import annotations

import sys
from importlib import util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sys.path bootstrap \u2014 runs at collection time
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

for _subdir in [
    "01_core",
    "02_strategies",
    "03_backtesting",
    "04_signals",
    "05_risk",
    "06_portfolio",
    "07_data",
]:
    _p = str(ROOT / _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Legacy helper (used by integration tests)
# ---------------------------------------------------------------------------

def load_module(module_path: Path, module_name: str) -> ModuleType:
    """Load a Python module from a filesystem path.

    Args:
        module_path: Absolute path to ``.py`` file.
        module_name: Synthetic module name used for ``sys.modules``.

    Returns:
        Loaded module object.

    Raises:
        ImportError: If the module cannot be loaded.
    """
    spec = util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# Session-scoped OHLCV fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return repository root path."""
    return ROOT


@pytest.fixture(scope="session")
def sample_ohlcv_300() -> pd.DataFrame:
    """300 business-day uptrending OHLCV frame. Session-scoped for speed."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = np.abs(100 + np.cumsum(np.random.randn(n) * 0.5 + 0.2)) + 1.0
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.ones(n) * 1e6,
        },
        index=dates,
    )


@pytest.fixture(scope="session")
def sample_ohlcv_1000() -> pd.DataFrame:
    """1 000 business-day OHLCV frame for longer backtests. Session-scoped."""
    np.random.seed(7)
    n = 1000
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    close = np.abs(100 + np.cumsum(np.random.randn(n) * 0.5 + 0.15)) + 1.0
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Signal fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_valid_signal():
    """A minimal, valid LONG TradeSignal."""
    from base_strategy import TradeSignal  # noqa: PLC0415

    return TradeSignal(
        ticker="SPY",
        direction="LONG",
        confidence=0.75,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=115.0,
        strategy_name="TestStrategy",
    )


@pytest.fixture
def flat_signal():
    """A FLAT signal (no action)."""
    from base_strategy import TradeSignal  # noqa: PLC0415

    return TradeSignal(
        ticker="SPY",
        direction="FLAT",
        confidence=0.0,
        entry_price=100.0,
        stop_loss=0.0,
        take_profit=0.0,
        strategy_name="TestStrategy",
    )


# ---------------------------------------------------------------------------
# pytest marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:  # type: ignore[override]
    config.addinivalue_line(
        "markers", "slow: tests that take > 5 s (Walk-Forward, Monte Carlo)"
    )
    config.addinivalue_line(
        "markers", "integration: end-to-end integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: NFR-01 performance benchmarks"
    )
