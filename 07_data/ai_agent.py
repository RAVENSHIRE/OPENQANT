"""AI-powered news filter and chat agent.

Uses OpenAI (GPT-4o-mini by default) when ``OPENAI_API_KEY`` is set in the
environment or ``.env`` file.  Falls back to a deterministic rule-based
extractor when no key is available, so the system works fully offline.

Public API
----------
``filter_news(raw_text, source_name, strategies) -> NewsItem``
    Parse raw financial text into a structured :class:`~news_db.NewsItem`.

``chat(messages, portfolio_context) -> str``
    Single-turn chat with the portfolio-aware agent.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional: load .env for OPENAI_API_KEY
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Ticker extraction helper (rule-based, used by both paths)
# ---------------------------------------------------------------------------

# Known symbols in our watchlist — extend as needed
_KNOWN_TICKERS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "GOOG",
    "SPY", "QQQ", "IWM", "EFA", "GLD", "TLT",
    "BTC", "BTC-USD", "ETH", "ETH-USD", "SOL", "SOL-USD", "BNB", "XRP",
    "^GSPC", "^NDX", "^DJI", "^RUT", "^STOXX50E",
    "INTC", "AMD", "BABA", "NFLX", "DIS", "JPM", "GS", "BAC",
}

_TICKER_RE = re.compile(r"\b([A-Z]{2,5}(?:-USD)?)\b")


def _extract_ticker(text: str) -> str:
    """Heuristic: find first known ticker in text, else first ALL-CAPS word."""
    upper = text.upper()
    for match in _TICKER_RE.finditer(upper):
        candidate = match.group(1)
        if candidate in _KNOWN_TICKERS:
            return candidate
    # fallback: first 2-5 uppercase sequence
    m = _TICKER_RE.search(upper)
    return m.group(1) if m else ""


def _simple_roi(text: str, horizon: str) -> str:
    """Extract a rough ROI mention for a given horizon keyword."""
    patterns = [
        r"(\+?-?\d+(?:\.\d+)?%)",  # percentage
        r"(\d+x)",  # multiplier
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return f"{m.group(1)} ({horizon})"
    return "n/a"


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _rule_based_filter(raw_text: str, source_name: str) -> dict:
    """Parse a news snippet without AI — deterministic heuristics.

    Args:
        raw_text: Raw pasted or fetched financial text.
        source_name: Name of the originating source.

    Returns:
        Dictionary matching :class:`~news_db.NewsItem` fields.
    """
    text_up = raw_text.upper()
    ticker = _extract_ticker(raw_text)

    # Headline = first sentence (≤120 chars)
    first_sent = raw_text.split(".")[0].strip()
    headline = first_sent[:120] if first_sent else raw_text[:120]

    # ROI hints
    roi_near = _simple_roi(raw_text, "1M")
    roi_mid = _simple_roi(raw_text, "6M")
    roi_long = _simple_roi(raw_text, "1Y+")

    # Benchmark
    benchmark = "S&P500"
    if any(k in text_up for k in ("BTC", "ETH", "CRYPTO", "SOL")):
        benchmark = "BTC"
    elif any(k in text_up for k in ("NASDAQ", "NDX", "QQQ")):
        benchmark = "NASDAQ"

    # Strategy fit
    positive_keywords = {
        "MOMENTUM", "BREAKOUT", "BULLISH", "UPTREND", "BUY", "ACCUMULATE",
        "GROWTH", "STRONG", "OUTPERFORM", "SURGE", "RALLY",
    }
    negative_keywords = {
        "BEARISH", "DOWNTREND", "SELL", "WEAK", "UNDERPERFORM",
        "CRASH", "DUMP", "CAUTION", "RISK",
    }
    pos_hits = sum(1 for k in positive_keywords if k in text_up)
    neg_hits = sum(1 for k in negative_keywords if k in text_up)

    if pos_hits > neg_hits:
        strategy_fit = "yes"
        routing = "portfolio"
    elif neg_hits > pos_hits:
        strategy_fit = "no"
        routing = "reallocation"
    else:
        strategy_fit = "partial"
        routing = "watch"

    return {
        "source_name": source_name,
        "raw_text": raw_text,
        "ticker": ticker,
        "headline": headline,
        "roi_near": roi_near,
        "roi_mid": roi_mid,
        "roi_long": roi_long,
        "benchmark": benchmark,
        "strategy_fit": strategy_fit,
        "routing": routing,
        "notes": "Rule-based extraction (no OpenAI key configured).",
    }


# ---------------------------------------------------------------------------
# OpenAI-powered filter
# ---------------------------------------------------------------------------

_FILTER_SYSTEM = """You are a quantitative financial analyst. Given a raw news snippet or
social-media post, extract structured information and respond ONLY with
valid JSON — no markdown, no explanation.

JSON schema:
{
  "ticker":       "<string — main ticker or empty string>",
  "headline":     "<string — one sentence summary, max 120 chars>",
  "roi_near":     "<string — near-term ROI estimate (1 month), e.g. '+3% (1M)' or 'n/a'>",
  "roi_mid":      "<string — mid-term ROI estimate (6 months), e.g. '+12% (6M)' or 'n/a'>",
  "roi_long":     "<string — long-term ROI estimate (1 year+), e.g. '+35% (1Y+)' or 'n/a'>",
  "benchmark":    "<string — relevant benchmark, e.g. 'S&P500', 'BTC', 'NASDAQ'>",
  "strategy_fit": "<string — exactly one of: yes / no / partial>",
  "routing":      "<string — exactly one of: portfolio / reallocation / watch>",
  "notes":        "<string — brief reasoning, max 200 chars>"
}

Routing rules:
- strategy_fit=yes  → routing=portfolio
- strategy_fit=no   → routing=reallocation
- strategy_fit=partial → routing=watch

Strategy context: momentum, trend-following, mean-reversion for equities and crypto.
"""

_CHAT_SYSTEM = """You are QUANT, the AI trading assistant for MarchStrategy — a
quantitative multi-strategy system. You have access to the user's portfolio context
(positions, scan signals, news feed).

Be concise, data-driven, and factual. For buy/sell decisions always ask for
confirmation before acting. Format numbers clearly. Respond in the same language
as the user."""


def _openai_filter(raw_text: str, source_name: str) -> dict:
    """Use OpenAI to structure a news snippet.

    Args:
        raw_text: Raw text to analyse.
        source_name: Source label.

    Returns:
        Dictionary matching :class:`~news_db.NewsItem` fields.

    Raises:
        RuntimeError: If the OpenAI call fails or returns malformed JSON.
    """
    import openai  # type: ignore

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _FILTER_SYSTEM},
            {"role": "user", "content": raw_text[:4000]},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI returned non-JSON: {content}") from exc

    return {
        "source_name": source_name,
        "raw_text": raw_text,
        "ticker": data.get("ticker", ""),
        "headline": data.get("headline", ""),
        "roi_near": data.get("roi_near", "n/a"),
        "roi_mid": data.get("roi_mid", "n/a"),
        "roi_long": data.get("roi_long", "n/a"),
        "benchmark": data.get("benchmark", "S&P500"),
        "strategy_fit": data.get("strategy_fit", "partial"),
        "routing": data.get("routing", "watch"),
        "notes": data.get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Public: filter_news
# ---------------------------------------------------------------------------


def filter_news(
    raw_text: str,
    source_name: str = "manual",
) -> dict:
    """Parse raw financial text into a structured news item dictionary.

    Tries OpenAI (GPT-4o-mini) when ``OPENAI_API_KEY`` is available;
    falls back to rule-based extraction otherwise.

    Args:
        raw_text: Pasted tweet, YouTube description, article snippet, etc.
        source_name: Label of the originating source.

    Returns:
        Dictionary with all fields needed to construct a
        :class:`~news_db.NewsItem`.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            return _openai_filter(raw_text, source_name)
        except Exception as exc:  # noqa: BLE001 — fall through to rule-based
            return {
                **_rule_based_filter(raw_text, source_name),
                "notes": f"OpenAI error ({exc}); rule-based fallback used.",
            }
    return _rule_based_filter(raw_text, source_name)


# ---------------------------------------------------------------------------
# Public: chat
# ---------------------------------------------------------------------------


def chat(
    messages: list[dict[str, str]],
    portfolio_context: Optional[str] = None,
) -> str:
    """Single-turn portfolio-aware chat with the AI agent.

    Args:
        messages: List of ``{"role": "user"|"assistant", "content": "..."}``
            dicts representing the conversation history.
        portfolio_context: Optional serialised portfolio summary injected into
            the system prompt.

    Returns:
        Assistant reply as a plain string.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")

    system_content = _CHAT_SYSTEM
    if portfolio_context:
        system_content += f"\n\n## Current Portfolio Context\n{portfolio_context}"

    if not api_key:
        # ------- offline fallback -------
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        return _offline_chat(last_user)

    try:
        import openai  # type: ignore

        client = openai.OpenAI()
        full_messages = [{"role": "system", "content": system_content}] + messages
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_messages,
            temperature=0.4,
            max_tokens=600,
        )
        return response.choices[0].message.content or "(no response)"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ OpenAI error: {exc}. Set OPENAI_API_KEY in .env to enable AI chat."


def _offline_chat(user_message: str) -> str:
    """Simple keyword-based offline chat fallback."""
    msg = user_message.lower()
    if any(k in msg for k in ("portfolio", "positions", "holdings")):
        return (
            "📊 Your portfolio positions are shown in the **Portfolio** tab. "
            "Add positions manually or let the News Filter route 'entry' signals automatically."
        )
    if any(k in msg for k in ("scan", "momentum", "signal")):
        return (
            "🔍 Use the **Scanner** tab to run a momentum scan across all watchlist "
            "categories. Results are saved to the SQLite DB and can be queried in "
            "the **Scan History** tab."
        )
    if any(k in msg for k in ("news", "feed", "filter")):
        return (
            "📰 In the **News & Feeds** tab you can add sources (YouTube, Twitter, "
            "Email, TradingView) and paste raw content. The AI extracts ticker, "
            "ROI estimates, and routes items to Portfolio or Reallocation automatically."
        )
    if any(k in msg for k in ("help", "what", "how")):
        return (
            "👋 I am **QUANT**, your MarchStrategy assistant. I can help you with:\n"
            "- 📊 Portfolio overview & P&L\n"
            "- 🔍 Momentum scans & signals\n"
            "- 📰 News filtering & routing\n"
            "- 📈 OHLCV chart lookups\n\n"
            "Set **OPENAI_API_KEY** in `.env` to enable full AI capabilities."
        )
    return (
        "🤖 **QUANT (offline mode)** — I can answer basic questions about your "
        "portfolio, scanner, and news feed. For full AI reasoning, set "
        "`OPENAI_API_KEY` in your `.env` file."
    )
