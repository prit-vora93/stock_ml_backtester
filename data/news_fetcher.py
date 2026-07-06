"""
data/news_fetcher.py
--------------------
Production-quality financial news sentiment pipeline for ML stock prediction.

Architecture (preserved from original):
    score_headline()
        ↓
    fetch_yfinance_news()
        ↓
    fetch_rss_news()
        ↓
    build_daily_sentiment()
        ↓
    fetch_news_sentiment()          ← main public API (unchanged signature)
        ↓
    get_sentiment_feature_columns()

Key improvements over v1:
    - Duplicate detection via normalized headline similarity
    - Source reliability weighting (Reuters > ET > Moneycontrol > Yahoo)
    - News importance scoring (earnings/merger/RBI get higher weight)
    - Event classification (earnings, RBI, merger, lawsuit etc.)
    - Extensible company matching (supports aliases, avoids false positives)
    - 40+ ML features (rolling stats, z-scores, event diversity, etc.)
    - FinBERT-ready: swap scorer in ONE place (_score_text)
    - Detailed logging (articles, duplicates, events, timing)

Output:
    One daily feature DataFrame merged into feature_engineer.py.
    All existing column names preserved — no breaking changes.
"""

import re
import time
import unicodedata
from datetime import datetime, timedelta, date
from functools import lru_cache
from typing import Optional

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONFIGURATION
# All tunable parameters in one place.
# ─────────────────────────────────────────────────────────────────────────────

# ── Sentiment scorer (swap here to replace VADER with FinBERT) ───────────────
# Design: _score_text() is the ONLY function that calls the scorer.
# To use FinBERT later: replace SentimentIntensityAnalyzer with your model
# and update _score_text(). Nothing else in this file needs to change.
_vader = SentimentIntensityAnalyzer()


# ── Source reliability weights ────────────────────────────────────────────────
# Higher weight = more reliable source = stronger influence on daily sentiment.
# Based on editorial standards, fact-checking quality, and financial focus.
SOURCE_WEIGHTS: dict[str, float] = {
    "reuters":          1.00,   # Gold standard for financial news
    "bloomberg":        1.00,   # Premium financial data provider
    "economic_times":   0.95,   # India's largest financial newspaper
    "business_standard":0.93,   # Strong Indian financial coverage
    "moneycontrol":     0.90,   # Popular Indian financial portal
    "livemint":         0.90,   # Quality Indian financial journalism
    "yfinance":         0.85,   # Yahoo Finance aggregated news
    "hindu_business":   0.85,   # The Hindu BusinessLine
    "unknown":          0.70,   # Fallback for unrecognized sources
}

# ── News importance keywords ──────────────────────────────────────────────────
# These keywords indicate HIGH-IMPACT news that deserves extra weight.
# Why: An earnings beat affects a stock far more than a routine update.
# Organized by category for clarity and easy extension.
IMPORTANCE_KEYWORDS: dict[str, float] = {
    # Corporate events — direct stock price impact
    "earnings":         1.0,
    "profit":           0.9,
    "revenue":          0.9,
    "results":          0.8,
    "guidance":         0.9,
    "outlook":          0.7,
    "forecast":         0.7,

    # Corporate actions — structural changes
    "merger":           1.0,
    "acquisition":      1.0,
    "takeover":         1.0,
    "buyout":           0.9,
    "ipo":              0.8,
    "buyback":          0.8,
    "dividend":         0.7,
    "split":            0.6,
    "demerger":         0.8,

    # Negative events — high urgency
    "bankruptcy":       1.0,
    "default":          1.0,
    "fraud":            1.0,
    "lawsuit":          0.9,
    "penalty":          0.8,
    "investigation":    0.9,
    "scam":             1.0,
    "downgrade":        0.8,
    "recall":           0.7,

    # Macro/policy — market-wide impact
    "rbi":              1.0,   # Reserve Bank of India
    "fed":              0.9,   # US Federal Reserve
    "rate":             0.8,
    "inflation":        0.8,
    "gdp":              0.7,
    "budget":           1.0,   # Union Budget (massive Indian market event)
    "policy":           0.7,
    "regulation":       0.7,
    "sanction":         0.8,
    "tariff":           0.7,

    # Contracts/growth — positive catalyst
    "contract":         0.8,
    "deal":             0.7,
    "partnership":      0.6,
    "expansion":        0.6,
    "launch":           0.6,

    # Credit/ratings — investor confidence
    "credit rating":    0.9,
    "upgrade":          0.8,
    "outlook":          0.7,
}

# ── Event classification taxonomy ────────────────────────────────────────────
# Maps event category names → trigger keywords.
# A headline can belong to multiple categories.
# These become binary ML features (0/1 per category per day).
EVENT_CATEGORIES: dict[str, list[str]] = {
    "earnings":     ["earnings", "profit", "revenue", "results", "quarterly", "annual", "eps", "pat", "ebitda"],
    "merger":       ["merger", "acquisition", "takeover", "buyout", "acquires", "merge", "demerger"],
    "dividend":     ["dividend", "buyback", "bonus share", "rights issue"],
    "guidance":     ["guidance", "outlook", "forecast", "raises guidance", "cuts guidance"],
    "lawsuit":      ["lawsuit", "legal", "court", "penalty", "investigation", "fraud", "scam", "sebi"],
    "government":   ["government", "ministry", "policy", "regulation", "budget", "tax", "gst", "plr"],
    "rbi":          ["rbi", "reserve bank", "monetary policy", "repo rate", "crr", "slr"],
    "fed":          ["federal reserve", "fed", "powell", "fomc", "interest rate"],
    "inflation":    ["inflation", "cpi", "wpi", "iip", "wholesale price", "consumer price"],
    "oil":          ["crude", "oil", "petroleum", "brent", "opec", "fuel"],
    "war":          ["war", "conflict", "geopolitical", "sanctions", "invasion", "tension"],
    "ceo":          ["ceo", "cfo", "coo", "md ", "managing director", "chief executive", "leadership", "appoints"],
    "rating":       ["rating", "upgrade", "downgrade", "credit", "moody", "fitch", "s&p", "crisil"],
    "contract":     ["contract", "deal", "wins", "order", "partnership", "agreement"],
    "product":      ["launch", "product", "new product", "service", "platform", "unveiled"],
    "ipo":          ["ipo", "listing", "public offer", "issue price"],
}

# ── RSS feed sources ──────────────────────────────────────────────────────────
# (symbol, url, source_name_for_weight_lookup)
RSS_FEEDS: list[tuple[str, str, str]] = [
    ("economic_times",   "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",  "economic_times"),
    ("moneycontrol",     "https://www.moneycontrol.com/rss/MCtopnews.xml",                        "moneycontrol"),
    ("livemint",         "https://www.livemint.com/rss/markets",                                  "livemint"),
    ("business_standard","https://www.business-standard.com/rss/markets-106.rss",                 "business_standard"),
]

# ── Company symbol → aliases mapping ─────────────────────────────────────────
# Designed for easy extension to hundreds of NSE stocks.
# Each entry: symbol → list of name variants to match in headlines.
# Ordered from most specific to least specific to reduce false positives.
SYMBOL_ALIASES: dict[str, list[str]] = {
    "RELIANCE.NS":   ["Reliance Industries", "RIL", "Reliance Jio", "Jio", "Mukesh Ambani"],
    "INFY.NS":       ["Infosys", "Infy", "Infosys Technologies"],
    "TCS.NS":        ["Tata Consultancy", "TCS"],
    "WIPRO.NS":      ["Wipro"],
    "HDFCBANK.NS":   ["HDFC Bank", "HDFCBank"],
    "ICICIBANK.NS":  ["ICICI Bank", "ICICI"],
    "SBIN.NS":       ["State Bank", "SBI"],
    "TATAMOTORS.NS": ["Tata Motors", "TaMo"],
    "BAJFINANCE.NS": ["Bajaj Finance", "BAF"],
    "ADANIENT.NS":   ["Adani Enterprises", "Adani Group", "Gautam Adani"],
}

# Minimum similarity ratio to consider two headlines as duplicates (0–1).
# 0.85 = headlines must be 85% similar to be considered duplicates.
DUPLICATE_SIMILARITY_THRESHOLD = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CORE SCORER
# Single function that wraps the underlying NLP model.
# Swap VADER for FinBERT here — nothing else changes.
# ─────────────────────────────────────────────────────────────────────────────

def _score_text(text: str) -> float:
    """
    Scores text sentiment using the configured NLP model.

    Returns:
        Float -1.0 (very negative) to +1.0 (very positive)

    EXTENSIBILITY:
        This is the ONLY function that calls the NLP scorer.
        To replace VADER with FinBERT:

            from transformers import pipeline
            _finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

            def _score_text(text: str) -> float:
                result = _finbert(text[:512])[0]
                score  = result["score"]
                return score if result["label"] == "positive" else -score

        Everything else in this file remains unchanged.
    """
    if not text or not isinstance(text, str):
        return 0.0
    scores = _vader.polarity_scores(text)
    return float(scores["compound"])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DUPLICATE DETECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_headline(headline: str) -> str:
    """
    Normalizes a headline for duplicate comparison.

    Steps:
        1. Lowercase
        2. Remove unicode accents (é → e)
        3. Remove punctuation and extra spaces
        4. Strip common filler words

    Why: "Reliance beats Q3 estimates!" and
         "Reliance beats Q3 estimates" are the same story.
         Normalization makes them comparable.

    Args:
        headline: Raw headline string

    Returns:
        Normalized string for comparison
    """
    if not headline:
        return ""

    # Lowercase
    text = headline.lower()

    # Remove unicode accents
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove punctuation except spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _headline_similarity(a: str, b: str) -> float:
    """
    Computes similarity between two normalized headlines.
    Uses token overlap (Jaccard similarity) — fast and good enough.

    Jaccard similarity = |intersection| / |union| of word sets.

    Example:
        a = "reliance beats q3 estimates"
        b = "reliance beats earnings estimates"
        intersection = {reliance, beats, estimates} → 3 words
        union        = {reliance, beats, q3, estimates, earnings} → 5 words
        similarity   = 3/5 = 0.60

    Args:
        a, b: Normalized headline strings

    Returns:
        Float 0.0 (completely different) to 1.0 (identical)
    """
    if not a or not b:
        return 0.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union        = tokens_a | tokens_b

    return len(intersection) / len(union)


def _remove_duplicates(articles: list[dict]) -> tuple[list[dict], int]:
    """
    Removes duplicate news articles using headline similarity.

    Algorithm:
        For each article, compare its normalized headline to all
        already-accepted articles. If similarity > threshold, skip it.
        Keep the article from the highest-weight source.

    Args:
        articles: List of article dicts (must have "title", "source_weight")

    Returns:
        (deduplicated_list, number_of_duplicates_removed)

    Why this matters:
        Economic Times and Moneycontrol often publish the same Reuters story.
        Without deduplication, that story counts twice in daily sentiment,
        artificially amplifying its effect.
    """
    if not articles:
        return [], 0

    # Sort by source weight descending so we keep the most reliable version
    sorted_articles  = sorted(articles, key=lambda x: x.get("source_weight", 0.7), reverse=True)
    accepted         = []
    accepted_norms   = []
    duplicates_found = 0

    for article in sorted_articles:
        norm = _normalize_headline(article.get("title", ""))

        is_duplicate = any(
            _headline_similarity(norm, existing) >= DUPLICATE_SIMILARITY_THRESHOLD
            for existing in accepted_norms
        )

        if is_duplicate:
            duplicates_found += 1
        else:
            accepted.append(article)
            accepted_norms.append(norm)

    return accepted, duplicates_found


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: COMPANY MATCHING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _get_aliases(symbol: str) -> list[str]:
    """
    Returns list of name aliases for a stock symbol.
    Cached so repeated calls for same symbol are free.

    Designed for easy extension:
        Add new symbols to SYMBOL_ALIASES dict above.
        No code changes needed anywhere else.

    Args:
        symbol: NSE symbol e.g. "RELIANCE.NS"

    Returns:
        List of alias strings to match in headlines
    """
    return SYMBOL_ALIASES.get(
        symbol,
        [symbol.replace(".NS", "").replace(".BO", "")]
    )


def _is_relevant(headline: str, symbol: str) -> bool:
    """
    Checks whether a headline is relevant to a given stock symbol.

    Matching rules:
        1. Check all aliases for the symbol
        2. Match is case-insensitive
        3. Require word-boundary match to avoid false positives
           ("TCS" should not match "TACTICS")

    Why word-boundary matters:
        Without boundary check: "TACTICS report" matches TCS → wrong.
        With boundary check: only "TCS reports" matches TCS → correct.

    Args:
        headline: Raw headline string
        symbol:   NSE symbol e.g. "TCS.NS"

    Returns:
        True if headline is about this stock, False otherwise
    """
    aliases = _get_aliases(symbol)
    hl_lower = headline.lower()

    for alias in aliases:
        # Word boundary pattern: alias must appear as a whole word/phrase
        pattern = r'\b' + re.escape(alias.lower()) + r'\b'
        if re.search(pattern, hl_lower):
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: IMPORTANCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _compute_importance(headline: str) -> float:
    """
    Computes a rule-based importance score for a headline.

    Logic:
        Scan headline for high-impact keywords.
        Take the MAXIMUM keyword score found (not sum).
        Normalize to 0.5–1.0 range so even ordinary news has some weight.

    Why max and not sum?
        A headline like "RBI raises rates, inflation concern" shouldn't
        score double just for mentioning two important topics.
        The most important topic determines the score.

    Why 0.5 minimum?
        Even routine news contributes some signal.
        A score of 0 would mean the article is completely ignored.

    Args:
        headline: Raw headline string

    Returns:
        Float 0.5 (ordinary news) to 1.0 (critical event)

    Example:
        _compute_importance("Infosys beats Q3 earnings, raises guidance")
        # "earnings" → 1.0, "guidance" → 0.9 → max = 1.0
        # Returns: 1.0

        _compute_importance("Infosys participates in industry event")
        # No keywords match → Returns: 0.5
    """
    hl_lower    = headline.lower()
    max_score   = 0.0

    for keyword, weight in IMPORTANCE_KEYWORDS.items():
        if keyword in hl_lower:
            max_score = max(max_score, weight)

    # Normalize: map [0, 1] keyword range to [0.5, 1.0] output range
    # importance = 0.5 + (max_score * 0.5)
    return 0.5 + (max_score * 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: EVENT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _classify_events(headline: str) -> list[str]:
    """
    Classifies a headline into one or more event categories.

    A headline can belong to multiple categories.
    Example: "RBI raises repo rate amid inflation concerns"
        → ["rbi", "inflation"]

    Why this matters for ML:
        Instead of just "negative news day", the model can learn that
        "RBI rate hike days" specifically predict IT sector decline
        but banking sector gains — much more granular signal.

    Args:
        headline: Raw headline string

    Returns:
        List of matched category names (empty list if none match)

    Example:
        _classify_events("Infosys misses Q3 earnings, CEO resigns")
        # Returns: ["earnings", "ceo"]
    """
    hl_lower = headline.lower()
    matched  = []

    for category, keywords in EVENT_CATEGORIES.items():
        if any(kw in hl_lower for kw in keywords):
            matched.append(category)

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: HEADLINE SCORER (public function — unchanged signature)
# ─────────────────────────────────────────────────────────────────────────────

def score_headline(headline: str) -> float:
    """
    Scores a single news headline sentiment.
    Public function — signature preserved from v1.

    Internally delegates to _score_text() so the NLP backend
    can be swapped without changing this function's interface.

    Args:
        headline: Raw headline string

    Returns:
        Float -1.0 (very negative) to +1.0 (very positive)

    Example:
        score_headline("Infosys beats Q3 estimates, raises guidance")
        # Returns: ~0.72

        score_headline("Wipro misses revenue, shares fall 8%")
        # Returns: ~-0.65
    """
    return _score_text(headline)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: NEWS FETCHERS (signatures preserved from v1)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yfinance_news(symbol: str) -> list[dict]:
    """
    Fetches recent news for a stock from yfinance.
    Signature preserved from v1.

    Now returns enriched articles with:
        title, date, sentiment, source, source_weight,
        importance_score, events

    Note:
        yfinance returns only the last ~10 news items.
        Best for real-time predictions, not historical training.

    Args:
        symbol: NSE symbol e.g. "RELIANCE.NS"

    Returns:
        List of enriched article dicts
    """
    try:
        ticker     = yf.Ticker(symbol)
        news_items = ticker.news or []

        results = []
        for item in news_items:
            title = item.get("title", "")
            if not title:
                continue

            try:
                pub_date = date.fromtimestamp(item.get("providerPublishTime", 0))
            except Exception:
                pub_date = date.today()

            results.append({
                "title":            title,
                "date":             pub_date,
                "sentiment":        score_headline(title),
                "source":           "yfinance",
                "source_weight":    SOURCE_WEIGHTS["yfinance"],
                "importance_score": _compute_importance(title),
                "events":           _classify_events(title),
            })

        logger.info(f"yfinance → {len(results)} articles for {symbol}")
        return results

    except Exception as e:
        logger.error(f"fetch_yfinance_news failed for {symbol}: {e}")
        return []


def fetch_rss_news(symbol: str) -> list[dict]:
    """
    Fetches news from RSS feeds, filtered for a specific stock symbol.
    Signature preserved from v1.

    Now uses:
        - Improved word-boundary company matching
        - Source weight lookup
        - Importance scoring
        - Event classification

    Args:
        symbol: NSE symbol e.g. "RELIANCE.NS"

    Returns:
        List of enriched article dicts relevant to this stock
    """
    results = []

    for feed_key, feed_url, source_name in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                logger.warning(f"RSS parse issue: {feed_key}")
                continue

            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title:
                    continue

                # Use improved word-boundary matching
                if not _is_relevant(title, symbol):
                    continue

                try:
                    pub_date = date(*entry.published_parsed[:3])
                except Exception:
                    pub_date = date.today()

                results.append({
                    "title":            title,
                    "date":             pub_date,
                    "sentiment":        score_headline(title),
                    "source":           source_name,
                    "source_weight":    SOURCE_WEIGHTS.get(source_name, SOURCE_WEIGHTS["unknown"]),
                    "importance_score": _compute_importance(title),
                    "events":           _classify_events(title),
                })

        except Exception as e:
            logger.warning(f"RSS feed {feed_key} failed: {e}")
            continue

    logger.info(f"RSS feeds → {len(results)} relevant articles for {symbol}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: DAILY SENTIMENT BUILDER (signature preserved from v1)
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_sentiment(
    news_items: list[dict],
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    """
    Aggregates article-level data into daily ML features.
    Signature preserved from v1.

    Improvements over v1:
        - Weighted sentiment (source reliability × importance)
        - Per-category event flags
        - News volume z-score
        - Positive/negative/neutral ratios
        - Rolling features (3d, 5d, 10d averages, std, z-score)
        - Sentiment acceleration, volatility, trend
        - Days since last negative/positive news

    Args:
        news_items: List of enriched article dicts
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"

    Returns:
        DataFrame indexed by date with 50+ ML feature columns
    """

    t_start = time.time()

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    # Build date-keyed bucket for all articles
    all_dates   = pd.date_range(start=start, end=end, freq="D")
    daily_bucket: dict[date, list[dict]] = {d.date(): [] for d in all_dates}

    for item in news_items:
        d = item.get("date")
        if d and start <= d <= end:
            daily_bucket[d].append(item)

    # ── Aggregate per day ─────────────────────────────────────────────────────
    rows = []

    for day in sorted(daily_bucket.keys()):
        articles = daily_bucket[day]
        row      = {"date": day}

        if not articles:
            # No news = neutral baseline
            row.update({
                "sentiment":          0.0,
                "weighted_sentiment": 0.0,
                "news_count":         0,
                "positive_ratio":     0.0,
                "negative_ratio":     0.0,
                "neutral_ratio":      1.0,
                "average_importance": 0.5,
                "weighted_positive":  0.0,
                "weighted_negative":  0.0,
                "news_intensity":     0.0,
                "event_diversity":    0.0,
            })
            # Zero all event category flags
            for cat in EVENT_CATEGORIES:
                row[f"event_{cat}"] = 0
        else:
            sentiments   = [a["sentiment"]        for a in articles]
            weights      = [a["source_weight"] * a["importance_score"] for a in articles]
            importances  = [a["importance_score"] for a in articles]

            total_weight = sum(weights) or 1.0

            # Weighted average sentiment
            weighted_sent = sum(s * w for s, w in zip(sentiments, weights)) / total_weight

            # Simple average sentiment (kept for backward compatibility)
            avg_sentiment = sum(sentiments) / len(sentiments)

            # Sentiment distribution ratios
            pos_count  = sum(1 for s in sentiments if s >  0.05)
            neg_count  = sum(1 for s in sentiments if s < -0.05)
            neu_count  = len(sentiments) - pos_count - neg_count
            n          = len(sentiments)

            # Weighted positive/negative (importance-adjusted)
            w_positive = sum(
                s * w for s, w in zip(sentiments, weights) if s > 0.05
            ) / total_weight

            w_negative = sum(
                abs(s) * w for s, w in zip(sentiments, weights) if s < -0.05
            ) / total_weight

            # Event diversity: how many DIFFERENT event types today?
            all_events     = [e for a in articles for e in a.get("events", [])]
            unique_events  = set(all_events)
            event_diversity= len(unique_events)

            # News intensity: weighted count (importance-adjusted volume)
            news_intensity = sum(importances)

            row.update({
                "sentiment":          avg_sentiment,
                "weighted_sentiment": weighted_sent,
                "news_count":         n,
                "positive_ratio":     pos_count / n,
                "negative_ratio":     neg_count / n,
                "neutral_ratio":      neu_count / n,
                "average_importance": sum(importances) / n,
                "weighted_positive":  w_positive,
                "weighted_negative":  w_negative,
                "news_intensity":     news_intensity,
                "event_diversity":    float(event_diversity),
            })

            # Binary event category flags (1 if any article today had this event)
            for cat in EVENT_CATEGORIES:
                row[f"event_{cat}"] = int(cat in unique_events)

        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # ── Rolling features ──────────────────────────────────────────────────────
    df = _add_rolling_features(df)

    # ── Advanced feature engineering ─────────────────────────────────────────
    df = _add_advanced_features(df)

    # ── Backward-compatible columns (v1 names preserved) ─────────────────────
    df["sentiment_ma3"]        = df["sentiment_rolling_mean_3"]
    df["sentiment_ma5"]        = df["sentiment_rolling_mean_5"]
    df["negative_flag"]        = (df["sentiment"] < -0.3).astype(int)
    df["positive_flag"]        = (df["sentiment"] >  0.3).astype(int)
    df["sentiment_momentum"]   = df["sentiment"].diff(3)

    # ── Logging summary ───────────────────────────────────────────────────────
    elapsed      = time.time() - t_start
    news_days    = (df["news_count"] > 0).sum()
    avg_sent     = df["sentiment"].mean()
    avg_imp      = df["average_importance"].mean()

    event_totals = {
        cat: int(df[f"event_{cat}"].sum())
        for cat in EVENT_CATEGORIES
        if f"event_{cat}" in df.columns and df[f"event_{cat}"].sum() > 0
    }

    logger.info(
        f"Daily sentiment built: {len(df)} days | "
        f"{news_days} days with news | "
        f"avg sentiment: {avg_sent:+.3f} | "
        f"avg importance: {avg_imp:.3f} | "
        f"elapsed: {elapsed:.1f}s"
    )
    if event_totals:
        logger.info(f"Event counts: {event_totals}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: ROLLING + ADVANCED FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling window statistics for sentiment.

    Why rolling features matter for ML:
        Today's sentiment alone is noisy — one misleading headline
        can flip the sign. A 5-day rolling average captures whether
        the NEWS TREND is improving or deteriorating, which is a
        much stronger predictor than any single day's value.

    Windows: 3-day (short), 5-day (medium), 10-day (longer term)
    """

    s = df["sentiment"]

    for window in [3, 5, 10]:
        w = str(window)
        r = s.rolling(window, min_periods=1)

        df[f"sentiment_rolling_mean_{w}"]  = r.mean()
        df[f"sentiment_rolling_std_{w}"]   = r.std().fillna(0)

        # Z-score: how many standard deviations from recent average?
        # A large positive z-score = unusually positive news burst
        mean = r.mean()
        std  = r.std().replace(0, np.nan)
        df[f"sentiment_zscore_{w}"] = ((s - mean) / std).fillna(0)

        # Rolling momentum: is sentiment improving or worsening?
        df[f"sentiment_momentum_{w}"] = s.diff(window)

    # News volume z-score: is today's article count unusually high?
    # High volume + negative sentiment = stronger sell signal
    vol_mean = df["news_count"].rolling(20, min_periods=1).mean()
    vol_std  = df["news_count"].rolling(20, min_periods=1).std().replace(0, np.nan)
    df["news_volume_zscore"] = ((df["news_count"] - vol_mean) / vol_std).fillna(0)

    return df


def _add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds higher-order sentiment features.

    sentiment_volatility:
        How erratic has sentiment been recently?
        High volatility = uncertain news environment = higher risk.

    sentiment_trend:
        Is news getting better or worse over last 5 days?
        Positive = improving narrative, Negative = deteriorating.

    sentiment_acceleration:
        Is the trend itself speeding up or slowing down?
        Like second derivative of sentiment.

    days_since_negative/positive:
        How long since last major news event?
        Recent bad news still affects investor psychology.
    """

    s = df["sentiment"]

    # Sentiment volatility: 10-day rolling std of sentiment
    df["sentiment_volatility"] = s.rolling(10, min_periods=1).std().fillna(0)

    # Sentiment trend: 5-day linear slope (positive = improving)
    df["sentiment_trend"] = s.rolling(5, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
        raw=True
    ).fillna(0)

    # Sentiment acceleration: change in trend (second derivative)
    df["sentiment_acceleration"] = df["sentiment_trend"].diff(1).fillna(0)

    # Rolling max and min (recent sentiment range)
    df["sentiment_rolling_max_5"] = s.rolling(5, min_periods=1).max()
    df["sentiment_rolling_min_5"] = s.rolling(5, min_periods=1).min()

    # Days since last strongly negative news
    # Why: A major negative event (fraud, earnings miss) keeps affecting
    # stock for days as investors digest the news.
    neg_days = (s < -0.3).astype(int)
    df["days_since_negative"] = _days_since_event(neg_days)

    # Days since last strongly positive news
    pos_days = (s >  0.3).astype(int)
    df["days_since_positive"] = _days_since_event(pos_days)

    return df


def _days_since_event(event_series: pd.Series) -> pd.Series:
    """
    Computes number of days since the last occurrence of an event.

    Args:
        event_series: Binary Series (1 = event occurred, 0 = no event)

    Returns:
        Series with count of days since last event.
        Capped at 30 (beyond 30 days, event is no longer relevant).
        Filled with 30 where no prior event exists.

    Example:
        events: [0, 0, 1, 0, 0, 0, 1, 0]
        result: [30,30, 0, 1, 2, 3, 0, 1]
    """
    result  = []
    counter = 30   # Start at max (no prior event)

    for val in event_series:
        if val == 1:
            counter = 0
        else:
            counter = min(counter + 1, 30)
        result.append(counter)

    return pd.Series(result, index=event_series.index)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: MAIN PUBLIC API (signature preserved from v1)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news_sentiment(
    symbol:     str,
    start_date: str,
    end_date:   str,
) -> Optional[pd.DataFrame]:
    """
    Fetches news from all sources, deduplicates, scores sentiment,
    and returns a daily feature DataFrame ready for feature_engineer.py.

    Signature IDENTICAL to v1 — no changes needed in feature_engineer.py.

    Pipeline:
        1. Fetch from yfinance + all RSS feeds
        2. Deduplicate overlapping stories
        3. Score sentiment + importance + events per article
        4. Aggregate to daily features (weighted by source + importance)
        5. Compute rolling + advanced features
        6. Return DataFrame indexed by date

    Args:
        symbol:     NSE symbol e.g. "RELIANCE.NS"
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"

    Returns:
        DataFrame indexed by date with 50+ sentiment feature columns.
        None if all sources fail (feature_engineer.py handles gracefully).

    Example:
        df = fetch_news_sentiment("RELIANCE.NS", "2022-01-01", "2024-01-01")
        print(df.shape)           # (730, 52)
        print(df["sentiment"].mean())        # e.g. 0.12
        print(df["weighted_sentiment"].mean()) # e.g. 0.15
    """

    t0 = time.time()
    logger.info(f"{'─'*50}")
    logger.info(f"News sentiment pipeline: {symbol} | {start_date} → {end_date}")

    all_articles: list[dict] = []

    # ── Fetch from all sources ────────────────────────────────────────────────
    yf_articles  = fetch_yfinance_news(symbol)
    rss_articles = fetch_rss_news(symbol)

    all_articles.extend(yf_articles)
    all_articles.extend(rss_articles)

    logger.info(
        f"Raw articles: {len(all_articles)} total "
        f"(yfinance: {len(yf_articles)}, RSS: {len(rss_articles)})"
    )

    # ── Deduplicate ───────────────────────────────────────────────────────────
    clean_articles, n_dupes = _remove_duplicates(all_articles)

    logger.info(
        f"After deduplication: {len(clean_articles)} articles "
        f"({n_dupes} duplicates removed)"
    )

    if not clean_articles:
        logger.warning(
            f"No articles for {symbol}. "
            f"All days will use neutral sentiment (0.0)."
        )

    # ── Build daily sentiment ─────────────────────────────────────────────────
    sentiment_df = build_daily_sentiment(clean_articles, start_date, end_date)

    elapsed = time.time() - t0
    logger.success(
        f"Sentiment pipeline complete: {symbol} | "
        f"{len(sentiment_df)} days | "
        f"{len(sentiment_df.columns)} features | "
        f"{elapsed:.1f}s total"
    )
    logger.info(f"{'─'*50}")

    return sentiment_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: FEATURE COLUMN REGISTRY (extended from v1)
# ─────────────────────────────────────────────────────────────────────────────

def get_sentiment_feature_columns() -> list[str]:
    """
    Returns complete list of sentiment feature columns.
    Used by feature_engineer.py to know which columns to merge.

    Backward-compatible: all v1 columns still present.
    New columns added at the end.
    """
    # ── v1 columns (preserved) ────────────────────────────────────────────────
    v1_cols = [
        "sentiment",
        "sentiment_ma3",
        "sentiment_ma5",
        "news_count",
        "negative_flag",
        "positive_flag",
        "sentiment_momentum",
    ]

    # ── v2 new columns ────────────────────────────────────────────────────────
    v2_core = [
        "weighted_sentiment",
        "positive_ratio",
        "negative_ratio",
        "neutral_ratio",
        "average_importance",
        "weighted_positive",
        "weighted_negative",
        "news_intensity",
        "event_diversity",
        "news_volume_zscore",
    ]

    v2_rolling = [
        f"sentiment_rolling_mean_{w}" for w in [3, 5, 10]
    ] + [
        f"sentiment_rolling_std_{w}"  for w in [3, 5, 10]
    ] + [
        f"sentiment_zscore_{w}"       for w in [3, 5, 10]
    ] + [
        f"sentiment_momentum_{w}"     for w in [3, 5, 10]
    ]

    v2_advanced = [
        "sentiment_volatility",
        "sentiment_trend",
        "sentiment_acceleration",
        "sentiment_rolling_max_5",
        "sentiment_rolling_min_5",
        "days_since_negative",
        "days_since_positive",
    ]

    v2_events = [f"event_{cat}" for cat in EVENT_CATEGORIES]

    return v1_cols + v2_core + v2_rolling + v2_advanced + v2_events