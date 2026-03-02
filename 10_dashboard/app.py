"""MarchStrategy — Streamlit Dashboard.

Tabs
----
1. 💼 Portfolio   — Positionen, Allocation-Pie, P&L, Reallocation-Queue
2. 🔍 Scanner     — Sequenzieller Momentum-Scan → SQLite-DB
3. 📰 News & Feeds — Quellen-Verwaltung, KI-Filter, Auto-Routing
4. 💬 Chat        — Portfolio-bewusster KI-Agent (QUANT)
5. 📊 Chart        — OHLCV Candlestick (yfinance oder Synthetic)
6. 📋 Scan History — DB-Abfrage historischer Scans

Start
-----
    streamlit run 10_dashboard/app.py
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Project root & dynamic loader
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


def _load(rel_path: str, module_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@st.cache_resource(show_spinner=False)
def _scanner_mod():
    m = _load("07_data/scanner.py", "scanner")
    return m.MomentumScanner, m.WATCHLIST, m.ALL_SYMBOLS


@st.cache_resource(show_spinner=False)
def _scan_db_cls():
    return _load("07_data/db.py", "scan_db").ScanDB


@st.cache_resource(show_spinner=False)
def _portfolio_db_cls():
    m = _load("07_data/portfolio_db.py", "portfolio_db")
    return m.PortfolioDB, m.Position, m.Transaction


@st.cache_resource(show_spinner=False)
def _news_mod():
    m = _load("07_data/news_db.py", "news_db")
    return m.NewsDB, m.NewsSource, m.NewsItem


@st.cache_resource(show_spinner=False)
def _ai_mod():
    return _load("07_data/ai_agent.py", "ai_agent")


@st.cache_resource(show_spinner=False)
def _fetch_fn():
    return _load("07_data/fetcher.py", "fetcher").fetch_ohlcv


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCAN_DB_PATH = ROOT / "data" / "scans.db"
PORTFOLIO_DB_PATH = ROOT / "data" / "portfolio.db"
NEWS_DB_PATH = ROOT / "data" / "news.db"

# ---------------------------------------------------------------------------
# Page config & global search
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MarchStrategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=56)
    st.title("MarchStrategy")
    st.caption("Quant Trading System v1.0")
    st.divider()

    # Global search bar
    search_q = st.text_input("🔎 Symbol / News suchen", placeholder="AAPL, BTC, …")
    if search_q:
        st.info(f"Suche nach: **{search_q.upper()}** → wähle Tab für Details")

    st.divider()
    st.markdown("**Datenbanken**")
    PortfolioDB, Position, Transaction = _portfolio_db_cls()
    pdb = PortfolioDB(PORTFOLIO_DB_PATH)
    ScanDB = _scan_db_cls()
    sdb = ScanDB(SCAN_DB_PATH)
    NewsDB, NewsSource, NewsItem = _news_mod()
    ndb = NewsDB(NEWS_DB_PATH)

    n_pos = len(pdb.get_positions())
    n_scans = sdb.row_count()
    n_news = ndb.item_count()
    st.metric("Positionen", n_pos)
    st.metric("Scan-Einträge", n_scans)
    st.metric("News-Items", n_news)

    st.divider()
    ai_key_set = bool(__import__("os").environ.get("OPENAI_API_KEY") or "")
    if ai_key_set:
        st.success("🤖 KI: OpenAI aktiv")
    else:
        st.warning("🤖 KI: Offline-Modus\nOpenAI_API_KEY in .env setzen")

# ---- Tabs ------------------------------------------------------------------
(
    tab_portfolio,
    tab_scanner,
    tab_news,
    tab_chat,
    tab_chart,
    tab_history,
) = st.tabs([
    "💼 Portfolio",
    "🔍 Scanner",
    "📰 News & Feeds",
    "💬 Chat",
    "📊 Chart",
    "📋 Scan History",
])

# ===========================================================================
# TAB 1 — PORTFOLIO
# ===========================================================================

with tab_portfolio:
    st.subheader("💼 Portfolio Overview")

    df_pos = pdb.get_positions()

    if df_pos.empty:
        st.info("Noch keine Positionen. Füge unten eine hinzu oder lass den News-Filter routen.")
    else:
        # Metrics row
        total_cost = (df_pos["qty"] * df_pos["avg_cost"]).sum()
        total_mv = (df_pos["qty"] * df_pos["current_price"]).sum()
        total_pnl = total_mv - total_cost
        pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Investiert", f"${total_cost:,.2f}")
        c2.metric("📈 Marktwert", f"${total_mv:,.2f}")
        c3.metric("P&L", f"${total_pnl:,.2f}", delta=f"{pnl_pct:.1f}%")
        c4.metric("Positionen", len(df_pos))

        st.divider()

        col_tbl, col_pie = st.columns([3, 2])

        with col_tbl:
            st.markdown("**Positionen**")
            display = df_pos.copy()
            display["MktWert $"] = (display["qty"] * display["current_price"]).round(2)
            display["P&L $"] = (display["qty"] * (display["current_price"] - display["avg_cost"])).round(2)
            display["P&L %"] = ((display["current_price"] / display["avg_cost"].replace(0, 1) - 1) * 100).round(2)

            def _pnl_color(val):
                return "color: #00cc00" if val > 0 else ("color: #cc0000" if val < 0 else "")

            st.dataframe(
                display[["symbol", "qty", "avg_cost", "current_price", "MktWert $", "P&L $", "P&L %", "source", "notes"]]
                .style.applymap(_pnl_color, subset=["P&L $", "P&L %"]),
                use_container_width=True,
                hide_index=True,
            )

        with col_pie:
            st.markdown("**Allocation**")
            if total_mv > 0:
                display["weight"] = (display["qty"] * display["current_price"]) / total_mv * 100
                fig_pie = px.pie(
                    display,
                    names="symbol",
                    values="weight",
                    hole=0.4,
                    template="plotly_dark",
                )
                fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=280)
                st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    st.markdown("**➕ Position hinzufügen / aktualisieren**")
    padd1, padd2, padd3, padd4, padd5 = st.columns(5)
    with padd1:
        p_sym = st.text_input("Symbol", key="p_sym", placeholder="AAPL").upper()
    with padd2:
        p_qty = st.number_input("Stückzahl", min_value=0.0001, value=1.0, step=0.1, key="p_qty")
    with padd3:
        p_cost = st.number_input("Avg. Einstand $", min_value=0.01, value=100.0, step=0.01, key="p_cost")
    with padd4:
        p_cur = st.number_input("Akt. Kurs $", min_value=0.0, value=0.0, step=0.01, key="p_cur",
                                help="0 = Kurs manuell nachtragen")
    with padd5:
        p_src = st.selectbox("Quelle", ["manual", "scanner", "news_filter"], key="p_src")
    p_notes = st.text_input("Notiz (optional)", key="p_notes")

    col_add, col_del = st.columns(2)
    with col_add:
        if st.button("💾 Speichern", use_container_width=True, key="btn_save_pos"):
            if p_sym:
                pos = Position(
                    symbol=p_sym,
                    qty=p_qty,
                    avg_cost=p_cost,
                    current_price=p_cur,
                    source=p_src,
                    notes=p_notes,
                )
                pdb.upsert_position(pos)
                pdb.add_transaction(Transaction(symbol=p_sym, action="buy", qty=p_qty, price=p_cost))
                st.success(f"✅ {p_sym} gespeichert.")
                st.rerun()
    with col_del:
        del_sym = st.text_input("Symbol löschen", key="del_sym", placeholder="AAPL").upper()
        if st.button("🗑 Position löschen", use_container_width=True, key="btn_del_pos"):
            if del_sym:
                pdb.delete_position(del_sym)
                st.warning(f"❌ {del_sym} entfernt.")
                st.rerun()

    # Reallocation queue
    df_realloc = pdb.get_reallocation()
    if not df_realloc.empty:
        st.divider()
        st.markdown("**🔄 Reallocation Queue**")
        st.dataframe(df_realloc, use_container_width=True, hide_index=True)
        resolve_id = st.number_input("ID auflösen", min_value=1, step=1, key="resolve_id")
        if st.button("✅ Aufgelöst markieren", key="btn_resolve"):
            pdb.resolve_reallocation(int(resolve_id))
            st.rerun()

# ===========================================================================
# TAB 2 — SCANNER
# ===========================================================================

with tab_scanner:
    st.subheader("🔍 Momentum Scanner")
    st.markdown(
        "Scannt die Watchlist **sequenziell** mit yfinance, "
        "speichert Signale in der DB und routet Entry-Signale optional ins Portfolio."
    )

    MomentumScanner, WATCHLIST, ALL_SYMBOLS = _scanner_mod()

    scfg1, scfg2, scfg3, scfg4 = st.columns(4)
    with scfg1:
        sc_cat = st.selectbox("Kategorie", ["Alle"] + list(WATCHLIST.keys()), key="sc_cat")
    with scfg2:
        sc_lb = st.slider("Lookback (Tage)", 5, 60, 20, key="sc_lb")
    with scfg3:
        sc_thr = st.slider("Entry-Schwelle (%)", 0.5, 10.0, 2.0, 0.5, key="sc_thr")
    with scfg4:
        sc_auto_add = st.checkbox("Entry → Portfolio auto-hinzufügen", value=False, key="sc_auto")

    if st.button("▶ Scan starten", type="primary", use_container_width=True, key="btn_scan"):
        targets = None if sc_cat == "Alle" else WATCHLIST[sc_cat]
        scanner = MomentumScanner(
            db_path=SCAN_DB_PATH,
            lookback=sc_lb,
            entry_threshold=sc_thr / 100,
            exit_threshold=-0.01,
            delay_seconds=0.25,
        )
        with st.spinner("Scanne …"):
            df_scan = scanner.run_full_scan(symbols=targets, save_to_db=True)

        st.success(f"✅ {len(df_scan)} Symbole gescannt.")

        if sc_auto_add:
            entries = df_scan[df_scan["state"] == "entry"]
            for _, row in entries.iterrows():
                p = Position(
                    symbol=row["symbol"],
                    qty=1.0,
                    avg_cost=row["price_close"],
                    current_price=row["price_close"],
                    source="scanner",
                    notes=f"Auto-added by scanner (conf={row['confidence']:.2f})",
                )
                pdb.upsert_position(p)
            if len(entries):
                st.info(f"📥 {len(entries)} Entry-Signale ins Portfolio übernommen.")

        def _sc_color(val):
            m = {"entry": "background-color:#1a7a1a;color:white",
                 "exit": "background-color:#8b0000;color:white",
                 "hold": "background-color:#444;color:white"}
            return m.get(val, "")

        st.dataframe(
            df_scan.style.applymap(_sc_color, subset=["state"]),
            use_container_width=True, hide_index=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 Entry", int((df_scan["state"] == "entry").sum()))
        m2.metric("🔴 Exit", int((df_scan["state"] == "exit").sum()))
        m3.metric("⚪ Hold", int((df_scan["state"] == "hold").sum()))

# ===========================================================================
# TAB 3 — NEWS & FEEDS
# ===========================================================================

with tab_news:
    st.subheader("📰 News & Feeds — KI-Filter")

    ai = _ai_mod()

    ntab_sources, ntab_add, ntab_feed = st.tabs(
        ["📡 Quellen verwalten", "➕ News verarbeiten", "📋 Feed anzeigen"]
    )

    with ntab_sources:
        st.markdown("**Registrierte Quellen**")
        df_src = ndb.get_sources()
        if df_src.empty:
            st.info("Noch keine Quellen. Füge unten eine hinzu.")
        else:
            st.dataframe(df_src, use_container_width=True, hide_index=True)
            del_src_id = st.number_input("Quellen-ID löschen", min_value=1, step=1, key="del_src_id")
            if st.button("🗑 Quelle löschen", key="btn_del_src"):
                ndb.delete_source(int(del_src_id))
                st.rerun()

        st.divider()
        st.markdown("**➕ Neue Quelle hinzufügen**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            s_name = st.text_input("Name", placeholder="Finanzfluss YouTube", key="s_name")
        with sc2:
            s_type = st.selectbox("Typ", ["youtube", "twitter", "email", "tradingview", "rss", "other"], key="s_type")
        with sc3:
            s_url = st.text_input("URL / Handle", placeholder="https://youtube.com/...", key="s_url")
        if st.button("💾 Quelle speichern", key="btn_save_src"):
            if s_name:
                ndb.add_source(NewsSource(name=s_name, source_type=s_type, url=s_url))
                st.success(f"✅ Quelle '{s_name}' gespeichert.")
                st.rerun()

    with ntab_add:
        st.markdown(
            "Füge rohen Text ein (Tweet, YouTube-Beschreibung, TradingView-Idee, "
            "E-Mail-Snippet). Der KI-Filter extrahiert Ticker, ROI-Erwartungen und "
            "routet automatisch ins **Portfolio**, in die **Reallocation Queue** oder "
            "auf die **Watchlist**."
        )

        src_list = ndb.get_sources()
        src_names = src_list["name"].tolist() if not src_list.empty else []
        src_names = ["(keine Quelle)"] + src_names

        fc1, fc2 = st.columns([2, 1])
        with fc1:
            raw_input = st.text_area(
                "📋 Roh-Text einfügen",
                height=160,
                placeholder="NVDA just broke out above the 52-week high with strong volume. "
                            "Analysts target +25% in 6M. Momentum clearly bullish. BUY signal.",
                key="raw_input",
            )
        with fc2:
            filter_source = st.selectbox("Quelle zuordnen", src_names, key="filter_source")
            st.markdown("**KI-Routing-Logik:**")
            st.markdown("- ✅ `strategy_fit=yes` → **Portfolio**")
            st.markdown("- 🔄 `strategy_fit=no` → **Reallocation**")
            st.markdown("- 👁 `strategy_fit=partial` → **Watch**")
            auto_portfolio = st.checkbox("Portfolio-Eintrag autom. anlegen", value=True, key="auto_port")

        if st.button("🤖 KI-Filter anwenden", type="primary", use_container_width=True, key="btn_filter"):
            if raw_input.strip():
                src_label = filter_source if filter_source != "(keine Quelle)" else "manual"
                with st.spinner("Analysiere …"):
                    result = ai.filter_news(raw_text=raw_input.strip(), source_name=src_label)

                # Save to news DB
                item = NewsItem(**result)
                ndb.save_item(item)

                st.success("✅ News verarbeitet und gespeichert.")

                # Show structured result
                r1, r2, r3 = st.columns(3)
                r1.metric("Ticker", result["ticker"] or "—")
                r2.metric("Strategy Fit", result["strategy_fit"].upper())
                r3.metric("Routing", result["routing"].upper())

                st.markdown(f"**📝 Headline:** {result['headline']}")
                rcol1, rcol2, rcol3 = st.columns(3)
                rcol1.info(f"**ROI kurz:** {result['roi_near']}")
                rcol2.info(f"**ROI mittel:** {result['roi_mid']}")
                rcol3.info(f"**ROI lang:** {result['roi_long']}")
                st.caption(f"Benchmark: {result['benchmark']} | Notiz: {result['notes']}")

                # Auto-route
                ticker = result["ticker"]
                if ticker:
                    if result["routing"] == "portfolio" and auto_portfolio:
                        p = Position(
                            symbol=ticker,
                            qty=1.0,
                            avg_cost=0.0,
                            source="news_filter",
                            notes=result["headline"][:100],
                        )
                        pdb.upsert_position(p)
                        st.success(f"📥 **{ticker}** ins Portfolio übernommen (Kurs bitte nachtragen).")
                    elif result["routing"] == "reallocation":
                        pdb.add_reallocation(symbol=ticker, reason=result["headline"][:200])
                        st.warning(f"🔄 **{ticker}** in die Reallocation Queue eingetragen.")
            else:
                st.warning("Bitte Text einfügen.")

    with ntab_feed:
        st.markdown("**Gefilterter News-Feed**")

        nf1, nf2, nf3 = st.columns(3)
        with nf1:
            feed_routing = st.selectbox("Routing", ["alle", "portfolio", "reallocation", "watch"], key="feed_routing")
        with nf2:
            feed_fit = st.selectbox("Strategy Fit", ["alle", "yes", "no", "partial"], key="feed_fit")
        with nf3:
            feed_limit = st.number_input("Max. Zeilen", 10, 500, 100, key="feed_limit")

        df_feed = ndb.get_items(
            routing=None if feed_routing == "alle" else feed_routing,
            strategy_fit=None if feed_fit == "alle" else feed_fit,
            limit=int(feed_limit),
        )
        if df_feed.empty:
            st.info("Keine Einträge mit diesen Filtern.")
        else:
            def _fit_color(val):
                m = {"yes": "background-color:#1a7a1a;color:white",
                     "no": "background-color:#8b0000;color:white",
                     "partial": "background-color:#8b6000;color:white"}
                return m.get(str(val), "")

            st.dataframe(
                df_feed[["processed_at", "source_name", "ticker", "headline",
                          "roi_near", "roi_mid", "roi_long", "benchmark",
                          "strategy_fit", "routing", "notes"]]
                .style.applymap(_fit_color, subset=["strategy_fit"]),
                use_container_width=True, hide_index=True,
            )

# ===========================================================================
# TAB 4 — CHAT
# ===========================================================================

with tab_chat:
    st.subheader("💬 QUANT — KI Trading Assistant")

    ai = _ai_mod()

    # Build portfolio context string for the agent
    df_p = pdb.get_positions()
    if not df_p.empty:
        port_ctx = "Positions: " + ", ".join(
            f"{r['symbol']} ({r['qty']} units @ ${r['avg_cost']:.2f})"
            for _, r in df_p.iterrows()
        )
    else:
        port_ctx = "Portfolio is empty."

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict[str, str]] = []

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Frage stellen … (z.B. 'Zeige mein Portfolio' oder 'Welche Signale heute?')")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("QUANT denkt …"):
                reply = ai.chat(
                    messages=st.session_state.chat_history,
                    portfolio_context=port_ctx,
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.button("🗑 Chat löschen", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

# ===========================================================================
# TAB 5 — CHART
# ===========================================================================

with tab_chart:
    st.subheader("📊 OHLCV Chart")
    fetch_ohlcv = _fetch_fn()

    # Pre-fill from sidebar search if set
    default_sym = search_q.upper() if search_q else "AAPL"

    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1:
        c_sym = st.text_input("Symbol", value=default_sym, key="c_sym").upper()
    with cc2:
        c_start = st.date_input("Von", value=date.today() - timedelta(days=180), key="c_start")
    with cc3:
        c_end = st.date_input("Bis", value=date.today(), key="c_end")
    with cc4:
        c_src = st.selectbox("Quelle", ["YahooFinance", "Synthetic"], key="c_src")

    if st.button("📊 Chart laden", use_container_width=True, key="btn_chart"):
        try:
            df_chart = fetch_ohlcv(
                c_sym,
                start_date=c_start.isoformat(),
                end_date=c_end.isoformat(),
                source=c_src,
            )
            fig = go.Figure(
                go.Candlestick(
                    x=df_chart.index,
                    open=df_chart["open"],
                    high=df_chart["high"],
                    low=df_chart["low"],
                    close=df_chart["close"],
                    name=c_sym,
                )
            )
            # Volume bars
            fig.add_trace(
                go.Bar(
                    x=df_chart.index,
                    y=df_chart["volume"],
                    name="Volume",
                    marker_color="rgba(100,149,237,0.35)",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                title=f"{c_sym} — {c_src}",
                yaxis=dict(title="Kurs", domain=[0.25, 1.0]),
                yaxis2=dict(title="Volumen", domain=[0.0, 0.20], showgrid=False),
                xaxis_rangeslider_visible=False,
                height=600,
                template="plotly_dark",
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{len(df_chart)} Handelstage · {c_src}")
        except Exception as exc:
            st.error(f"Fehler: {exc}")

# ===========================================================================
# TAB 6 — SCAN HISTORY
# ===========================================================================

with tab_history:
    st.subheader("📋 Scan-Historie")
    total = sdb.row_count()

    if total == 0:
        st.info("Keine Scan-Daten. Bitte zuerst einen Scan im Scanner-Tab starten.")
    else:
        st.caption(f"Gesamt: **{total}** Einträge")

        hf1, hf2, hf3 = st.columns(3)
        with hf1:
            h_state = st.selectbox("State", ["Alle", "entry", "hold", "exit"], key="h_state")
        with hf2:
            h_sym = st.text_input("Symbol", key="h_sym", placeholder="AAPL").strip().upper()
        with hf3:
            h_limit = st.number_input("Max. Zeilen", 10, 1000, 200, key="h_limit")

        df_hist = sdb.query(
            state=None if h_state == "Alle" else h_state,
            symbol=h_sym or None,
            limit=int(h_limit),
        )
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        st.markdown("### Letzter Stand pro Symbol")
        df_lat = sdb.latest_per_symbol()
        if not df_lat.empty:
            st.dataframe(df_lat, use_container_width=True, hide_index=True)

