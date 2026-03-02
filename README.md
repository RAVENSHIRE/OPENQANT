# 🚀 MarchStrategy — Quantitative Trading System

**Version: 1.0 | Status: Active Development | Stand: März 2026**

Regelbasiertes, automatisiertes Multi-Strategie-Trading-System mit vollständiger Produkt- und Prozessdokumentation nach Agile Best Practices (basierend auf AltexSoft-Framework).

## Vision

MarchStrategy automatisiert die gesamte Trading-Pipeline von Datenbeschaffung über Signalgenerierung und Risikosteuerung bis zur Portfolio-Konstruktion und Visualisierung. Vier historisch robuste März-Strategien werden durch einen Konsens-Filter kombiniert.

Vollständiges PRD → [docs/prd/PRD.md](docs/prd/PRD.md)

## Architektur

```text
MarchStrategy/
├── docs/                      ← A) Produktdokumentation
│   ├── prd/PRD.md             ← User Personas, FR, NFR, Risiken, DoD
│   ├── roadmap/ROADMAP.md     ← Meilensteine, Sprints, Tech-Roadmap
│   ├── api/API_REFERENCE.md   ← Vollständige Interface-Referenz
│   └── standards/CODE_STANDARDS.md ← PEP8, Tests, Git, CI/CD, Metriken
├── tests/                     ← B) Test-Suite (NFR-04: ≥80% Coverage)
│   ├── conftest.py            ← Globale Fixtures & pytest-Marker
│   ├── unit/                  ← Unit-Tests alle 4 Strategien + Indikatoren
│   └── integration/           ← End-to-End Pipeline + Walk-Forward
├── 01_core/                   ← BaseStrategy ABC, TradeSignal, Orchestrator
├── 02_strategies/             ← 4 Strategie-Module (Momentum, Trend, Crypto)
├── 03_backtesting/            ← Engine, Walk-Forward, Monte Carlo
├── 04_signals/                ← Konsens-Aggregator (min 2/4 Strategien)
├── 05_risk/                   ← RiskManager (Stop, Sizing, Drawdown-Halt)
├── 06_portfolio/              ← Risk Parity + Confidence-Weights
├── 07_data/                   ← DataFetcher (yfinance, CCXT, Synthetic, Cache)
├── 09_config/strategies.yaml  ← Alle Parameter zentral (kein Hardcoding)
├── 10_dashboard/app.py        ← Streamlit Dashboard
├── pytest.ini                 ← Test-Konfiguration
├── setup.cfg                  ← flake8, isort, mypy
└── requirements.txt           ← Projekt-Abhängigkeiten
```

## Strategien

| # | Strategie | Asset | Kum. Rendite (Backtest) | Risiko |
|---|---|---|---|---|
| 1 | Saisonales Aktien-Momentum | S&P500, Europa, Schweiz | 2x–4x | Niedrig-Mittel |
| 2 | Trendfolge-Breakouts | Aktien, ETFs | 3x–10x | Mittel |
| 3 | Krypto-Momentum | BTC, ETH, SOL | 5x–20x | Hoch |
| 4 | Small-Cap-Reversal | Russell 2000 | 4x–15x | Sehr hoch |

## Schnellstart

```bash
pip install -r requirements.txt

# Tests (Unit + Integration)
pytest tests/ -v
pytest tests/unit/ -v                # Schnell (~5s)
pytest tests/ --cov=. --cov-report=html  # Coverage Report

# Linting
flake8 . --max-line-length=100
black . --check

# Backtest
python 01_core/orchestrator.py --strategy all --mode backtest

# Dashboard
streamlit run 10_dashboard/app.py
```

## Dokumentation

| Dokument | Zielgruppe |
|---|---|
| [docs/prd/PRD.md](docs/prd/PRD.md) | PM, Stakeholder |
| [docs/roadmap/ROADMAP.md](docs/roadmap/ROADMAP.md) | Team, PM |
| [docs/api/API_REFERENCE.md](docs/api/API_REFERENCE.md) | Entwickler |
| [docs/standards/CODE_STANDARDS.md](docs/standards/CODE_STANDARDS.md) | Entwickler |

*Disclaimer: Dieses System dient Bildungs- und Forschungszwecken. Backtests garantieren keine zukünftigen Renditen.*
