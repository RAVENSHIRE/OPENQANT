# OPENQANT — Quantitative Trading System

**Version: 1.1 | Status: Active Development | Stand: März 2026**

Regelbasiertes, modulares Multi-Strategie-Trading-System. Die vollständige Pipeline umfasst Datenbeschaffung, Signalgenerierung, Risikomanagement, Portfolio-Konstruktion und ein interaktives Streamlit-Dashboard. Vier historisch robuste Strategien werden durch einen Konsens-Filter kombiniert und gegen Overfitting durch Walk-Forward-Validierung und Monte-Carlo-Simulation abgesichert.

Vollständiges PRD → [docs/prd/PRD.md](docs/prd/PRD.md)

---

## Implementierungsstand (März 2026)

| Modul | Dateien | Status |
|---|---|---|
| Dashboard (6 Tabs) | `10_dashboard/app.py` | ✅ Vollständig |
| 4 Strategien | `02_strategies/*.py` | ✅ Vollständig |
| Backtesting Engine | `03_backtesting/engine.py` | ✅ Vollständig |
| Walk-Forward + Monte Carlo | `03_backtesting/walk_forward.py`, `monte_carlo.py` | ✅ Vollständig |
| Signal-Aggregator | `04_signals/aggregator.py`, `signal.py` | ✅ Vollständig |
| RiskManager | `05_risk/risk_manager.py` | ✅ Vollständig |
| Portfolio-Konstruktion | `06_portfolio/allocator.py` | ✅ Vollständig |
| DataFetcher + Scanner | `07_data/fetcher.py`, `scanner.py` | ✅ Vollständig |
| Datenbanken (SQLite) | `07_data/db.py`, `portfolio_db.py`, `news_db.py` | ✅ Vollständig |
| AI Agent (OpenAI) | `07_data/ai_agent.py` | ✅ Vollständig |
| Orchestrator | `01_core/orchestrator.py` | ⚠️ Stub – Ausbau geplant |
| Test-Suite | `tests/unit/`, `tests/integration/` | ✅ Vorhanden |

---

## Architektur

```text
OPENQUANT/
├── docs/
│   ├── prd/PRD.md                 ← User Personas, FR, NFR, Risiko, DoD
│   ├── roadmap/ROADMAP.md         ← Meilensteine, Sprint-Status, Tech-Roadmap
│   ├── api/API_REFERENCE.md       ← Vollständige Modul- & Interface-Referenz
│   └── standards/CODE_STANDARDS.md← PEP8, Type Hints, Git, CI/CD, Metriken
├── tests/
│   ├── conftest.py                ← Session-Fixtures, load_module()-Helper
│   ├── unit/
│   │   ├── test_strategies.py     ← Alle 4 Strategien: Signal-State, Determinismus
│   │   └── test_risk_and_data.py  ← Kelly, Drawdown, OHLCV-Schema, Cache
│   └── integration/
│       └── test_e2e_backtest.py   ← End-to-End Backtest-Pipeline
├── 01_core/
│   └── orchestrator.py            ← Pipeline-Steuerung (Stub, Ausbau geplant)
├── 02_strategies/
│   ├── momentum.py                ← SeasonalMomentumStrategy
│   ├── trend.py                   ← TrendBreakoutStrategy
│   ├── crypto_momentum.py         ← CryptoMomentumStrategy
│   └── small_cap_reversal.py      ← SmallCapReversalStrategy
├── 03_backtesting/
│   ├── engine.py                  ← Backtesting-Engine (OHLCV → Equity-Kurve)
│   ├── walk_forward.py            ← Walk-Forward-Validierung (5 Splits)
│   └── monte_carlo.py             ← Monte-Carlo-Simulation (1.000 Runs)
├── 04_signals/
│   ├── signal.py                  ← TradeSignal(state, confidence)
│   └── aggregator.py              ← Konsens-Aggregator (min. 2/4 Stimmen)
├── 05_risk/
│   └── risk_manager.py            ← KellyCriterion, RiskManager, Drawdown-Halt
├── 06_portfolio/
│   └── allocator.py               ← Risk Parity + Confidence-Weights
├── 07_data/
│   ├── fetcher.py                 ← fetch_ohlcv(), MarketDataFetcher (Cache)
│   ├── scanner.py                 ← MomentumScanner (Watchlist → SQLite)
│   ├── db.py                      ← ScanDB (scan_results)
│   ├── portfolio_db.py            ← PortfolioDB, Position, Transaction
│   ├── news_db.py                 ← NewsDB, NewsSource, NewsItem
│   └── ai_agent.py                ← QuantAgent (OpenAI, Portfolio-aware Chat)
├── 09_config/
│   └── strategies.yaml            ← Alle Strategie-Parameter zentral
├── 10_dashboard/
│   └── app.py                     ← Streamlit Dashboard (6 Tabs)
├── data/                          ← SQLite-Datenbanken (auto-erstellt)
├── pytest.ini                     ← Testpfade, Marker-Definitionen
├── setup.cfg                      ← flake8 (max-line-length=100), isort, mypy
└── requirements.txt               ← Alle Python-Abhängigkeiten
```

---

## Dashboard — 6 Tabs

| Tab | Funktion |
|---|---|
| 💼 **Portfolio** | Positionen, P&L, Allocation-Pie, Reallocation-Queue, CRUD |
| 🔍 **Scanner** | Momentum-Scan (Watchlist → SQLite), Entry-Schwelle konfigurierbar, Auto-Add ins Portfolio |
| 📰 **News & Feeds** | Quellen-Verwaltung (NewsDB), KI-Filter, Auto-Routing von Entry-Signalen |
| 💬 **Chat** | Portfolio-bewusster KI-Agent QUANT (OpenAI GPT, Offline-Fallback) |
| 📊 **Chart** | OHLCV Candlestick (yfinance live oder Synthetic), technische Indikatoren |
| 📋 **Scan History** | DB-Abfrage historischer Scan-Ergebnisse, Export |

---

## Strategien

| # | Klasse | Asset-Universum | Risiko |
|---|---|---|---|
| 1 | `SeasonalMomentumStrategy` | S&P500, Europa, Schweiz | Niedrig–Mittel |
| 2 | `TrendBreakoutStrategy` | Aktien, ETFs | Mittel |
| 3 | `CryptoMomentumStrategy` | BTC, ETH, SOL | Hoch |
| 4 | `SmallCapReversalStrategy` | Russell 2000 | Sehr hoch |

Alle Strategien implementieren die gleiche Schnittstelle:
```python
signal = Strategy().generate_signal(prices)  # → TradeSignal(state, confidence)
assert signal.state in {"entry", "exit", "hold"}
assert 0.0 <= signal.confidence <= 1.0
```

---

## Schnellstart

```bash
# 1. Abhängigkeiten
pip install -r requirements.txt

# 2. Optionaler KI-Agent (Chat-Tab)
cp .env.example .env
# OPENAI_API_KEY=sk-... in .env eintragen

# 3. Dashboard starten
streamlit run 10_dashboard/app.py

# 4. Tests ausführen
pytest tests/unit/ -v                          # schnell (~5 s, kein Netz)
pytest tests/ -v                               # inkl. Integration
pytest tests/ --cov=. --cov-report=html        # Coverage-Report → htmlcov/

# 5. Linting
flake8 . --max-line-length=100
black . --check
isort . --check
```

---

## Dokumentation

| Dokument | Inhalt | Zielgruppe |
|---|---|---|
| [docs/prd/PRD.md](docs/prd/PRD.md) | Personas, FR, NFR, Risiken, DoD | PM, Stakeholder |
| [docs/roadmap/ROADMAP.md](docs/roadmap/ROADMAP.md) | Meilensteine, Sprint-Status, Tech-Roadmap | Team, PM |
| [docs/api/API_REFERENCE.md](docs/api/API_REFERENCE.md) | Alle Module, Klassen, Funktionen | Entwickler |
| [docs/standards/CODE_STANDARDS.md](docs/standards/CODE_STANDARDS.md) | PEP8, Git-Workflow, CI/CD, Metriken | Entwickler |

---

*Disclaimer: Dieses System dient Bildungs- und Forschungszwecken. Backtests garantieren keine zukünftigen Renditen.*
