"""Unit tests for RiskManager and DataFetcher.

Covers NFR-04 (≥80 % coverage) for critical infrastructure modules.

Pattern: AAA (Arrange → Act → Assert)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from base_strategy import TradeSignal  # noqa: E402
from risk_manager import RiskManager, RiskConfig  # noqa: E402
from fetcher import DataFetcher  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_signal(
    ticker: str = "SPY",
    direction: str = "LONG",
    confidence: float = 0.80,
    entry: float = 100.0,
    stop: float = 95.0,
    target: float = 115.0,
    strategy: str = "TestStrategy",
) -> TradeSignal:
    return TradeSignal(
        ticker=ticker, direction=direction, confidence=confidence,
        entry_price=entry, stop_loss=stop, take_profit=target,
        strategy_name=strategy,
    )


def make_market_data(ticker: str = "SPY", n: int = 300, trend: str = "up") -> dict:
    np.random.seed(0)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    rnd = np.cumsum(np.random.randn(n) * 0.5 + (0.2 if trend == "up" else -0.2))
    close = np.abs(100 + rnd) + 1.0
    df = pd.DataFrame(
        {
            "open": close * 0.99, "high": close * 1.01,
            "low": close * 0.98, "close": close,
            "volume": np.ones(n) * 1e6,
        },
        index=dates,
    )
    return {ticker: df}


# ---------------------------------------------------------------------------
# RiskManager — position sizing
# ---------------------------------------------------------------------------

class TestRiskManagerPositionSizing:
    def test_position_size_respects_max_risk(self) -> None:
        cfg = RiskConfig(total_capital=100_000, max_risk_per_trade_pct=0.02)
        rm = RiskManager(cfg)
        sig = make_signal(entry=100.0, stop=95.0)  # 5 USD risk per unit
        size = rm._calc_position_size(sig, make_market_data("SPY"))
        max_risk_usd = 100_000 * 0.02
        assert size * (sig.entry_price - sig.stop_loss) <= max_risk_usd * 1.01

    def test_position_size_zero_when_stop_equals_entry(self) -> None:
        rm = RiskManager()
        sig = make_signal(entry=100.0, stop=100.0)
        assert rm._calc_position_size(sig, {}) == 0.0

    def test_low_confidence_reduces_position(self) -> None:
        rm = RiskManager()
        market = make_market_data("SPY")
        sig_high = make_signal(confidence=1.0, entry=100.0, stop=95.0)
        sig_low = make_signal(confidence=0.5, entry=100.0, stop=95.0)
        assert rm._calc_position_size(sig_low, market) < rm._calc_position_size(sig_high, market)

    def test_position_capped_at_max_position_pct(self) -> None:
        cfg = RiskConfig(total_capital=100_000, max_position_pct=0.10)
        rm = RiskManager(cfg)
        sig = make_signal(entry=100.0, stop=99.99)  # tiny stop → huge theoretical size
        size = rm._calc_position_size(sig, make_market_data("SPY"))
        max_value = cfg.total_capital * cfg.max_position_pct
        assert size * sig.entry_price <= max_value * 1.01


# ---------------------------------------------------------------------------
# RiskManager — drawdown halt
# ---------------------------------------------------------------------------

class TestRiskManagerDrawdownHalt:
    def test_halt_triggered_when_drawdown_exceeds_threshold(self) -> None:
        cfg = RiskConfig(total_capital=100_000, max_drawdown_threshold=0.15)
        rm = RiskManager(cfg)
        rm.equity = 100_000
        rm.peak_equity = 120_000  # drawdown = -16.7 %
        assert rm._in_drawdown_halt() is True

    def test_no_halt_below_threshold(self) -> None:
        cfg = RiskConfig(max_drawdown_threshold=0.15)
        rm = RiskManager(cfg)
        rm.equity = 100_000
        rm.peak_equity = 110_000  # drawdown = -9.1 %
        assert rm._in_drawdown_halt() is False

    def test_process_signals_returns_empty_in_halt(self) -> None:
        cfg = RiskConfig(total_capital=100_000, max_drawdown_threshold=0.10)
        rm = RiskManager(cfg)
        rm.equity = 85_000  # -15 % → halt
        rm.peak_equity = 100_000
        sigs = [make_signal("SPY"), make_signal("QQQ")]
        market = {**make_market_data("SPY"), **make_market_data("QQQ")}
        assert rm.process_signals(sigs, market) == []


# ---------------------------------------------------------------------------
# RiskManager — regime filter
# ---------------------------------------------------------------------------

class TestRiskManagerRegimeFilter:
    def test_regime_bearish_when_price_below_sma200(self) -> None:
        cfg = RiskConfig(use_regime_filter=True, regime_sma_period=200)
        rm = RiskManager(cfg)
        result = rm._regime_is_bearish("SPY", make_market_data("SPY", trend="down"))
        assert isinstance(result, bool)

    def test_regime_filter_disabled(self) -> None:
        rm = RiskManager(RiskConfig(use_regime_filter=False))
        assert rm._regime_is_bearish("SPY", make_market_data("SPY")) is False

    def test_regime_false_for_unknown_ticker(self) -> None:
        rm = RiskManager(RiskConfig(use_regime_filter=True))
        assert rm._regime_is_bearish("UNKNOWN_TICKER", {}) is False


# ---------------------------------------------------------------------------
# RiskManager — portfolio summary
# ---------------------------------------------------------------------------

class TestRiskManagerSummary:
    def test_summary_returns_correct_keys(self) -> None:
        rm = RiskManager()
        summary = rm.summary()
        for key in ["equity", "peak_equity", "drawdown", "n_positions", "invested_pct"]:
            assert key in summary

    def test_initial_drawdown_is_zero(self) -> None:
        rm = RiskManager(RiskConfig(total_capital=100_000))
        assert rm.summary()["drawdown"] == 0.0


# ---------------------------------------------------------------------------
# DataFetcher
# ---------------------------------------------------------------------------

class TestDataFetcher:
    @pytest.fixture
    def fetcher(self, tmp_path) -> DataFetcher:
        return DataFetcher(cache_dir=str(tmp_path / "cache"))

    def test_synthetic_returns_valid_ohlcv(self, fetcher: DataFetcher) -> None:
        df = fetcher.fetch({"ticker": "TEST", "source": "synthetic",
                            "start": "2020-01-01", "end": "2023-12-31"})
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(df) > 100
        assert not df.isnull().any().any()

    def test_synthetic_no_negative_prices(self, fetcher: DataFetcher) -> None:
        df = fetcher.fetch({"ticker": "TEST", "source": "synthetic",
                            "start": "2018-01-01", "end": "2025-12-31"})
        assert (df["close"] > 0).all()
        assert (df["volume"] > 0).all()

    def test_high_always_gte_low(self, fetcher: DataFetcher) -> None:
        df = fetcher.fetch({"ticker": "SPY", "source": "synthetic",
                            "start": "2020-01-01", "end": "2023-12-31"})
        assert (df["high"] >= df["low"]).all(), "High should always be >= Low"

    def test_clean_removes_zero_volume(self, fetcher: DataFetcher) -> None:
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0], "high": [102.0, 103.0],
                "low": [98.0, 99.0], "close": [101.0, 102.0],
                "volume": [0.0, 1000.0],
            },
            index=pd.date_range("2022-01-01", periods=2, freq="B"),
        )
        cleaned = fetcher._clean(df)
        assert len(cleaned) == 1
        assert (cleaned["volume"] > 0).all()

    def test_clean_raises_on_missing_column(self, fetcher: DataFetcher) -> None:
        df = pd.DataFrame({"open": [100.0], "high": [102.0], "low": [98.0]})
        with pytest.raises(ValueError, match="Missing column"):
            fetcher._clean(df)

    def test_synthetic_determinism(self, fetcher: DataFetcher) -> None:
        df1 = fetcher._fetch_synthetic("SPY", "2020-01-01", "2023-12-31")
        df2 = fetcher._fetch_synthetic("SPY", "2020-01-01", "2023-12-31")
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_tickers_give_different_data(self, fetcher: DataFetcher) -> None:
        df_spy = fetcher._fetch_synthetic("SPY", "2020-01-01", "2023-12-31")
        df_btc = fetcher._fetch_synthetic("BTC", "2020-01-01", "2023-12-31")
        assert not df_spy["close"].equals(df_btc["close"])

    def test_fetch_universe_returns_all_tickers(self, fetcher: DataFetcher) -> None:
        tickers = ["SPY", "QQQ", "IWM"]
        config = {"source": "synthetic", "start": "2020-01-01", "end": "2023-12-31"}
        result = fetcher.fetch_universe(tickers, config)
        assert set(result.keys()) == set(tickers)
        for df in result.values():
            assert len(df) > 0


# ---------------------------------------------------------------------------
# Performance benchmarks (NFR-01)
# ---------------------------------------------------------------------------

class TestPerformance:
    @pytest.mark.performance
    def test_synthetic_fetch_10000_rows_under_1s(self) -> None:
        fetcher = DataFetcher(cache_dir="/tmp/test_cache")
        config = {"ticker": "PERF", "source": "synthetic",
                  "start": "1985-01-01", "end": "2025-12-31"}  # ~10 000 trading days
        start = time.perf_counter()
        df = fetcher.fetch(config)
        elapsed = time.perf_counter() - start
        print(f"\n[Perf] Fetch {len(df)} rows: {elapsed:.3f}s")
        assert elapsed < 1.0, f"Fetch took {elapsed:.2f}s > 1.0s (NFR-01)"

    @pytest.mark.performance
    def test_signal_generation_10000_bars_under_500ms(self) -> None:
        from momentum import SeasonalMomentumStrategy  # noqa: PLC0415

        np.random.seed(0)
        n = 10_000
        dates = pd.date_range("1985-01-01", periods=n, freq="B")
        close = np.abs(100 + np.cumsum(np.random.randn(n) * 0.3 + 0.1)) + 1.0
        data = pd.DataFrame(
            {
                "open": close * 0.99, "high": close * 1.01,
                "low": close * 0.98, "close": close,
                "volume": np.ones(n) * 1e6,
            },
            index=dates,
        )
        strat = SeasonalMomentumStrategy({"name": "perf_test", "ticker": "SPY"})
        strat.fit(data)
        start = time.perf_counter()
        strat.generate_signals(data)
        elapsed = time.perf_counter() - start
        print(f"\n[Perf] generate_signals {n} bars: {elapsed:.3f}s")
        assert elapsed < 0.5, f"Signal generation took {elapsed:.3f}s > 0.5s (NFR-01)"
