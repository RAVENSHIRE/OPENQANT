"""Unit tests for risk management and data fetching modules."""

from __future__ import annotations

from importlib import util
from pathlib import Path
import sys


def load_module(module_path: Path, module_name: str):
    """Load module from file path for numeric folder structure."""
    spec = util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_kelly_criterion_bounds(project_root) -> None:
    """Kelly output must be clipped to [0, 1]."""
    risk_module = load_module(project_root / "05_risk" / "risk_manager.py", "risk_manager")
    fraction = risk_module.KellyCriterion(win_probability=0.6, payout_ratio=1.5)
    assert 0.0 <= fraction <= 1.0


def test_drawdown_halt_trigger(project_root) -> None:
    """Drawdown limit should trigger halt for severe losses."""
    risk_module = load_module(project_root / "05_risk" / "risk_manager.py", "risk_manager_drawdown")
    manager = risk_module.RiskManager(max_drawdown=0.1)
    assert manager.breach_drawdown(peak_equity=100.0, current_equity=85.0)


def test_fetch_ohlcv_schema(project_root) -> None:
    """Fetched OHLCV frame should contain standard columns."""
    data_module = load_module(project_root / "07_data" / "fetcher.py", "fetcher_module")
    frame = data_module.fetch_ohlcv("SPY", "2024-01-01", "2024-03-01", source="Synthetic")
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) > 0


def test_data_fetcher_cache(project_root) -> None:
    """Synthetic data generator must cache by key."""
    data_module = load_module(project_root / "07_data" / "fetcher.py", "fetcher_cache")
    fetcher = data_module.MarketDataFetcher()
    _ = fetcher.get_synthetic("BTC-USD", points=128, seed=5)
    assert fetcher.is_cached("BTC-USD", points=128, seed=5)
