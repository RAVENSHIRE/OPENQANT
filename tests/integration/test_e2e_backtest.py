"""Integration tests for the documented 02-07 system architecture."""

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


def test_full_pipeline_integration(project_root) -> None:
    """Run Fetch -> Strategies -> Aggregation -> Backtest pipeline."""
    fetcher_module = load_module(project_root / "07_data" / "fetcher.py", "fetcher_integration")
    momentum_module = load_module(project_root / "02_strategies" / "momentum.py", "momentum_integration")
    trend_module = load_module(project_root / "02_strategies" / "trend.py", "trend_integration")
    aggregator_module = load_module(project_root / "04_signals" / "aggregator.py", "aggregator_integration")
    engine_module = load_module(project_root / "03_backtesting" / "engine.py", "engine_integration")

    frame = fetcher_module.fetch_ohlcv("SPY", "2024-01-01", "2024-05-01", source="Synthetic")
    prices = frame["close"].to_numpy()

    strategy_states = [
        momentum_module.SeasonalMomentumStrategy().generate_signal(prices).state,
        trend_module.TrendBreakoutStrategy().generate_signal(prices).state,
        "hold",
        "hold",
    ]
    aggregated = aggregator_module.aggregate_consensus(strategy_states, min_votes=2)

    signals = ["hold"] * (len(prices) - 1) + [aggregated.state]
    result = engine_module.BacktestEngine().run(prices, signals)
    assert "cumulative_return" in result
    assert "sharpe" in result


def test_walk_forward_optimization(project_root) -> None:
    """Walk-forward helper should produce one score per fold."""
    walk_module = load_module(project_root / "03_backtesting" / "walk_forward.py", "walk_forward_integration")
    prices = np.linspace(100.0, 140.0, 80)
    scores = walk_module.walk_forward_scores(prices, folds=4)
    assert len(scores) == 4


def test_monte_carlo_simulation(project_root) -> None:
    """Monte-Carlo helper should return terminal values of requested shape."""
    mc_module = load_module(project_root / "03_backtesting" / "monte_carlo.py", "mc_integration")
    prices = np.linspace(100.0, 130.0, 90)
    terminal = mc_module.monte_carlo_terminal_values(prices, simulations=200, horizon=12)
    assert terminal.shape == (200,)


def test_signal_aggregation(project_root) -> None:
    """Aggregation should respect minimum vote threshold."""
    aggregator_module = load_module(project_root / "04_signals" / "aggregator.py", "aggregator_votes")
    result = aggregator_module.aggregate_consensus(["entry", "entry", "hold", "exit"], min_votes=2)
    assert result.state == "entry"
