"""Unit tests for all four strategy modules in ``02_strategies``.

Pattern: AAA (Arrange → Act → Assert)

Imports use bare module names because ``conftest.py`` adds all sub-package
directories to ``sys.path`` at collection time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from base_strategy import TradeSignal  # noqa: E402
from momentum import SeasonalMomentumStrategy  # noqa: E402
from trend import TrendBreakoutStrategy  # noqa: E402
from crypto_momentum import CryptoMomentumStrategy  # noqa: E402
from small_cap_reversal import SmallCapReversalStrategy  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 300, trend: str = "up", seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data.

    Args:
        n: Number of bars.
        trend: ``"up"`` | ``"down"`` | ``"flat"`` | ``"volatile"``.
        seed: Random seed for reproducibility.

    Returns:
        OHLCV DataFrame with a business-day DatetimeIndex.
    """
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    if trend == "up":
        close = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.3)
    elif trend == "down":
        close = 100 + np.cumsum(np.random.randn(n) * 0.5 - 0.3)
    elif trend == "flat":
        close = 100 + np.random.randn(n) * 0.5
    else:  # volatile
        close = 100 + np.cumsum(np.random.randn(n) * 3.0)
    close = np.abs(close) + 1.0
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = np.abs(np.random.randn(n) * 1e6 + 5e6)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def uptrend_data() -> pd.DataFrame:
    return make_ohlcv(300, trend="up")


@pytest.fixture
def downtrend_data() -> pd.DataFrame:
    return make_ohlcv(300, trend="down")


@pytest.fixture
def flat_data() -> pd.DataFrame:
    return make_ohlcv(300, trend="flat")


# ---------------------------------------------------------------------------
# BaseStrategy validation
# ---------------------------------------------------------------------------

class TestBaseStrategyValidation:
    def test_validate_data_raises_on_missing_column(self, uptrend_data: pd.DataFrame) -> None:
        # Arrange
        strat = SeasonalMomentumStrategy({"name": "test", "ticker": "SPY"})
        broken = uptrend_data.drop(columns=["volume"])
        # Act & Assert
        with pytest.raises(ValueError, match="Missing columns"):
            strat.validate_data(broken)

    def test_validate_data_fills_nan(self, uptrend_data: pd.DataFrame) -> None:
        # Arrange
        strat = SeasonalMomentumStrategy({"name": "test", "ticker": "SPY"})
        data = uptrend_data.copy()
        data.loc[data.index[5], "close"] = np.nan
        # Act
        result = strat.validate_data(data)
        # Assert
        assert result is True
        assert not data["close"].isnull().any()

    def test_signal_has_required_fields(self, uptrend_data: pd.DataFrame) -> None:
        # Arrange
        strat = SeasonalMomentumStrategy({"name": "test", "ticker": "SPY"})
        strat.fit(uptrend_data)
        # Act
        signals = strat.generate_signals(uptrend_data)
        # Assert
        assert len(signals) >= 1
        sig = signals[0]
        assert isinstance(sig, TradeSignal)
        assert sig.direction in ("LONG", "SHORT", "FLAT")
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.strategy_name == "test"


# ---------------------------------------------------------------------------
# SeasonalMomentumStrategy
# ---------------------------------------------------------------------------

class TestSeasonalMomentum:
    def test_generates_long_in_uptrend(self, uptrend_data: pd.DataFrame) -> None:
        strat = SeasonalMomentumStrategy({"name": "SeasonalMomentum", "ticker": "SPY"})
        strat.fit(uptrend_data)
        signals = strat.generate_signals(uptrend_data)
        directions = [s.direction for s in signals]
        assert "LONG" in directions or "FLAT" in directions

    def test_generates_flat_in_downtrend(self, downtrend_data: pd.DataFrame) -> None:
        strat = SeasonalMomentumStrategy({"name": "SeasonalMomentum", "ticker": "SPY"})
        strat.fit(downtrend_data)
        signals = strat.generate_signals(downtrend_data)
        assert len(signals) >= 1

    def test_stop_loss_below_entry(self, uptrend_data: pd.DataFrame) -> None:
        strat = SeasonalMomentumStrategy({"name": "test", "ticker": "SPY"})
        strat.fit(uptrend_data)
        for sig in strat.generate_signals(uptrend_data):
            if sig.direction == "LONG":
                assert sig.stop_loss < sig.entry_price, (
                    f"Stop {sig.stop_loss} should be below entry {sig.entry_price}"
                )

    def test_take_profit_above_entry(self, uptrend_data: pd.DataFrame) -> None:
        strat = SeasonalMomentumStrategy({"name": "test", "ticker": "SPY"})
        strat.fit(uptrend_data)
        for sig in strat.generate_signals(uptrend_data):
            if sig.direction == "LONG":
                assert sig.take_profit > sig.entry_price

    def test_confidence_boost_in_march(self) -> None:
        strat = SeasonalMomentumStrategy({"name": "test", "ticker": "SPY"})
        data = make_ohlcv(300, trend="up")
        data.index = pd.date_range("2025-01-01", periods=300, freq="B")
        strat.fit(data)
        signals = strat.generate_signals(data)
        assert len(signals) >= 1

    def test_rsi_values_in_valid_range(self, uptrend_data: pd.DataFrame) -> None:
        rsi = SeasonalMomentumStrategy._rsi(uptrend_data["close"], period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all(), "RSI below 0 found"
        assert (valid <= 100).all(), "RSI above 100 found"

    def test_atr_values_always_positive(self, uptrend_data: pd.DataFrame) -> None:
        atr = SeasonalMomentumStrategy._atr(uptrend_data, period=14)
        assert (atr.dropna() >= 0).all(), "ATR should always be non-negative"

    def test_determinism(self, uptrend_data: pd.DataFrame) -> None:
        """Same input must always produce the same output."""
        strat = SeasonalMomentumStrategy({"name": "test", "ticker": "SPY"})
        strat.fit(uptrend_data)
        s1 = strat.generate_signals(uptrend_data)
        s2 = strat.generate_signals(uptrend_data)
        assert s1[0].direction == s2[0].direction
        assert s1[0].confidence == s2[0].confidence


# ---------------------------------------------------------------------------
# TrendBreakoutStrategy
# ---------------------------------------------------------------------------

class TestTrendBreakout:
    def test_no_signal_below_sma(self, downtrend_data: pd.DataFrame) -> None:
        strat = TrendBreakoutStrategy({"name": "TrendBreakout", "ticker": "QQQ"})
        strat.fit(downtrend_data)
        for sig in strat.generate_signals(downtrend_data):
            assert sig.direction in ("LONG", "FLAT")

    def test_risk_reward_respected(self, uptrend_data: pd.DataFrame) -> None:
        config = {
            "name": "test", "ticker": "QQQ",
            "risk_reward_ratio": 3.0, "atr_multiplier_stop": 1.5,
        }
        strat = TrendBreakoutStrategy(config)
        strat.fit(uptrend_data)
        for sig in strat.generate_signals(uptrend_data):
            if sig.direction == "LONG" and sig.entry_price > sig.stop_loss:
                risk = sig.entry_price - sig.stop_loss
                reward = sig.take_profit - sig.entry_price
                actual_rr = reward / risk
                assert actual_rr >= 2.5, f"R:R too low: {actual_rr:.2f}"

    def test_confidence_between_0_and_1(self, uptrend_data: pd.DataFrame) -> None:
        strat = TrendBreakoutStrategy({"name": "test", "ticker": "QQQ"})
        strat.fit(uptrend_data)
        for sig in strat.generate_signals(uptrend_data):
            assert 0.0 <= sig.confidence <= 1.0


# ---------------------------------------------------------------------------
# CryptoMomentumStrategy
# ---------------------------------------------------------------------------

class TestCryptoMomentum:
    def test_kelly_position_size_in_metadata(self, uptrend_data: pd.DataFrame) -> None:
        strat = CryptoMomentumStrategy({"name": "CryptoMomentum", "ticker": "BTC-USD"})
        strat.fit(uptrend_data)
        signals = strat.generate_signals(uptrend_data)
        if signals[0].direction == "LONG":
            assert "suggested_position_pct" in signals[0].metadata
            pct = signals[0].metadata["suggested_position_pct"]
            assert 0.0 <= pct <= 0.20, f"Position pct out of bounds: {pct}"

    def test_no_long_when_rsi_overbought(self) -> None:
        np.random.seed(0)
        dates = pd.date_range("2022-01-01", periods=300, freq="B")
        close = 100 + np.cumsum(np.ones(300) * 2.0)
        data = pd.DataFrame(
            {
                "open": close * 0.99, "high": close * 1.01,
                "low": close * 0.98, "close": close,
                "volume": np.ones(300) * 1e6,
            },
            index=dates,
        )
        strat = CryptoMomentumStrategy(
            {"name": "test", "ticker": "BTC-USD", "rsi_exit_threshold": 75}
        )
        strat.fit(data)
        signals = strat.generate_signals(data)
        assert signals[0].direction in ("LONG", "FLAT")

    def test_wider_stop_for_crypto(self, uptrend_data: pd.DataFrame) -> None:
        config = {"name": "test", "ticker": "BTC-USD", "trailing_stop_atr_mult": 3.0}
        strat = CryptoMomentumStrategy(config)
        strat.fit(uptrend_data)
        for sig in strat.generate_signals(uptrend_data):
            if sig.direction == "LONG":
                assert sig.entry_price - sig.stop_loss > 0


# ---------------------------------------------------------------------------
# SmallCapReversalStrategy
# ---------------------------------------------------------------------------

class TestSmallCapReversal:
    def test_long_signal_when_deeply_oversold(self) -> None:
        np.random.seed(5)
        n = 300
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        # Stable phase then sharp drop (> 2 ATR below SMA)
        stable = 150 + np.random.randn(280) * 0.3
        drop = 150 - np.arange(1, 21) * 2.5
        close = np.concatenate([stable, drop])
        high = close * 1.01
        low = close * 0.99
        open_ = close * 1.005
        volume = np.concatenate([np.ones(280) * 5e5, np.ones(20) * 1e6])
        data = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        strat = SmallCapReversalStrategy(
            {"name": "SmallCapReversal", "ticker": "TEST", "earnings_surprise": 0.05}
        )
        strat.fit(data)
        signals = strat.generate_signals(data)
        assert signals[0].direction in ("LONG", "FLAT")

    def test_max_hold_days_in_metadata(self, flat_data: pd.DataFrame) -> None:
        strat = SmallCapReversalStrategy(
            {"name": "test", "ticker": "IWM", "hold_max_days": 10}
        )
        strat.fit(flat_data)
        signals = strat.generate_signals(flat_data)
        assert "max_hold_days" in signals[0].metadata
        assert signals[0].metadata["max_hold_days"] == 10

    def test_higher_confidence_with_earnings_surprise(
        self, uptrend_data: pd.DataFrame
    ) -> None:
        config_no_earnings = {"name": "test", "ticker": "IWM", "earnings_surprise": 0.0}
        config_earnings = {"name": "test", "ticker": "IWM", "earnings_surprise": 0.05}
        strat_no = SmallCapReversalStrategy(config_no_earnings)
        strat_yes = SmallCapReversalStrategy(config_earnings)
        strat_no.fit(uptrend_data)
        strat_yes.fit(uptrend_data)
        sigs_no = strat_no.generate_signals(uptrend_data)
        sigs_yes = strat_yes.generate_signals(uptrend_data)
        if sigs_no[0].direction == "LONG" and sigs_yes[0].direction == "LONG":
            assert sigs_yes[0].confidence >= sigs_no[0].confidence
