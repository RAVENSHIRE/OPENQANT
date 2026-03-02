# API Referenz

## 1. Modul: `strategies`

### `RSIStrategy(period: int, entry_threshold: float, exit_threshold: float) -> Signal`

**Beschreibung:** Implementiert eine Relative Strength Index (RSI) basierte Handelsstrategie.

**Parameter:**

| Parameter | Typ | Beschreibung |
|---|---|---|
| `period` | `int` | Die Anzahl der Perioden für die RSI-Berechnung. |
| `entry_threshold` | `float` | Der RSI-Schwellenwert für den Einstieg in eine Position. |
| `exit_threshold` | `float` | Der RSI-Schwellenwert für den Ausstieg aus einer Position. |

**Rückgabe:**

| Typ | Beschreibung |
|---|---|
| `Signal` | Ein Signalobjekt, das `entry`, `exit` oder `hold` angibt. |

## 2. Modul: `risk_management`

### `KellyCriterion(win_probability: float, payout_ratio: float) -> float`

**Beschreibung:** Berechnet die optimale Positionsgröße basierend auf dem Kelly-Kriterium.

**Parameter:**

| Parameter | Typ | Beschreibung |
|---|---|---|
| `win_probability` | `float` | Die Wahrscheinlichkeit eines Gewinns. |
| `payout_ratio` | `float` | Das Verhältnis von durchschnittlichem Gewinn zu durchschnittlichem Verlust. |

**Rückgabe:**

| Typ | Beschreibung |
|---|---|
| `float` | Der optimale Anteil des Kapitals, der riskiert werden sollte. |

## 3. Modul: `data_fetcher`

### `fetch_ohlcv(symbol: str, start_date: str, end_date: str, source: str = 'YahooFinance') -> pd.DataFrame`

**Beschreibung:** Ruft OHLCV-Daten (Open, High, Low, Close, Volume) für ein bestimmtes Symbol ab.

**Parameter:**

| Parameter | Typ | Beschreibung |
|---|---|---|
| `symbol` | `str` | Das Tickersymbol des Finanzinstruments. |
| `start_date` | `str` | Das Startdatum im Format 'YYYY-MM-DD'. |
| `end_date` | `str` | Das Enddatum im Format 'YYYY-MM-DD'. |
| `source` | `str` | Die Datenquelle (Standard: 'YahooFinance'). |

**Rückgabe:**

| Typ | Beschreibung |
|---|---|
| `pd.DataFrame` | Ein Pandas DataFrame mit den OHLCV-Daten. |
