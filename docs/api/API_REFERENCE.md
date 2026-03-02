# API Referenz

## 1. Modul: `04_signals/signal.py`

### `Signal(state: str, confidence: float)`

**Beschreibung:** Einheitliches Signalobjekt für alle Strategien.

**Parameter:**

| Parameter | Typ | Beschreibung |
|---|---|---|
| `state` | `str` | Einer von `entry`, `exit`, `hold`. |
| `confidence` | `float` | Konfidenz im Bereich `[0, 1]`. |

## 2. Modul: `02_strategies`

### `SeasonalMomentumStrategy.generate_signal(prices: Sequence[float]) -> Signal`
### `TrendBreakoutStrategy.generate_signal(prices: Sequence[float]) -> Signal`
### `CryptoMomentumStrategy.generate_signal(prices: Sequence[float]) -> Signal`
### `SmallCapReversalStrategy.generate_signal(prices: Sequence[float]) -> Signal`

**Beschreibung:** Vier dokumentierte Kernstrategien gemäß Konfiguration in `09_config/strategies.yaml`.

## 3. Modul: `04_signals/aggregator.py`

### `aggregate_consensus(signals: Sequence[str], min_votes: int = 2) -> AggregationResult`

**Beschreibung:** Aggregiert mehrere Strategie-States zu einem Konsens-Signal.

## 4. Modul: `05_risk/risk_manager.py`

### `KellyCriterion(win_probability: float, payout_ratio: float) -> float`

**Beschreibung:** Berechnet den optimalen Kapitalanteil gemäß Kelly-Kriterium.

## 5. Modul: `07_data/fetcher.py`

### `fetch_ohlcv(symbol: str, start_date: str, end_date: str, source: str = 'YahooFinance') -> pd.DataFrame`

**Beschreibung:** Liefert OHLCV-Daten als DataFrame mit Spalten `open`, `high`, `low`, `close`, `volume`.

**Unterstützte Quellen:** `YahooFinance`, `AlphaVantage`, `Quandl`, `Synthetic`.
