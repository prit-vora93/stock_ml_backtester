"""
data/news_storage.py
---------------------
Saves and retrieves fetched news articles from PostgreSQL.

Mirrors the save/read pattern used in data/storage.py for OHLCV data.

Why this exists:
    fetch_yfinance_news() and fetch_rss_news() (in news_fetcher.py) only
    ever see the last ~10 / currently-live headlines — there is no
    historical news archive behind a free API. Without persistence, every
    multi-year preprocess() run silently fabricated a neutral (0.0)
    sentiment for almost every historical day.

    This module lets news_fetcher.py save every article it ever fetches,
    so repeated runs accumulate a real historical archive over time
    instead of discarding what was fetched. It does not retroactively
    backfill news from before this table existed — that data simply
    doesn't exist anywhere for free — but it stops throwing away what
    IS available and makes future backtests over recent history
    increasingly accurate the more often the fetcher runs.

Usage:
    from data.news_storage import save_news_articles, get_news_articles

    saved = save_news_articles("RELIANCE.NS", articles)
    rows  = get_news_articles("RELIANCE.NS", "2020-01-01", "2024-01-01")
"""

import hashlib
import re
from datetime import date as date_cls, datetime

from api.database import SessionLocal, NewsArticle
from utils.logger import logger


def _title_hash(title: str) -> str:
    """
    Normalizes a headline (lowercase, strip punctuation/whitespace) and
    returns its MD5 hash — used as the de-duplication key alongside
    (symbol, date).
    """
    normalized = re.sub(r"[^\w\s]", " ", title.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def save_news_articles(symbol: str, articles: list[dict]) -> int:
    """
    Saves enriched article dicts (as produced by news_fetcher.py) into
    the news_articles table. Duplicates (same symbol + date + normalized
    title) are silently skipped.

    Args:
        symbol:   Stock symbol e.g. "RELIANCE.NS"
        articles: List of dicts with keys:
                  title, date, sentiment, source, source_weight,
                  importance_score, events (list[str])

    Returns:
        Number of NEW rows actually inserted.
    """
    if not articles:
        return 0

    db = SessionLocal()
    rows_saved = 0
    rows_skip  = 0

    try:
        existing_hashes = {
            (row_date, title_hash) for (row_date, title_hash) in
            db.query(NewsArticle.date, NewsArticle.title_hash)
              .filter(NewsArticle.symbol == symbol)
              .all()
        }

        new_rows = []
        for article in articles:
            title = article.get("title", "")
            if not title:
                continue

            article_date = article.get("date")
            if isinstance(article_date, datetime):
                article_date = article_date.date()
            if not isinstance(article_date, date_cls):
                continue   # unparseable/missing date — skip this article

            key = (article_date, _title_hash(title))
            if key in existing_hashes:
                rows_skip += 1
                continue

            existing_hashes.add(key)   # dedupe within this same batch too
            new_rows.append(NewsArticle(
                symbol           = symbol,
                date             = article_date,
                title            = title[:500],
                title_hash       = key[1],
                source           = article.get("source", "unknown")[:30],
                source_weight    = float(article.get("source_weight", 0.7)),
                sentiment        = float(article.get("sentiment", 0.0)),
                importance_score = float(article.get("importance_score", 0.5)),
                events           = ",".join(article.get("events", []))[:300],
            ))
            rows_saved += 1

        if new_rows:
            db.bulk_save_objects(new_rows)
        db.commit()

        logger.info(
            f"{symbol}: {rows_saved} new articles archived | "
            f"{rows_skip} duplicates skipped"
        )
        return rows_saved

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save news articles for {symbol}: {e}")
        return 0

    finally:
        db.close()


def get_news_articles(symbol: str, start_date: str, end_date: str) -> list[dict]:
    """
    Reads back all archived articles for a symbol within a date range,
    in the same dict shape news_fetcher.py's fetchers produce (so it can
    be passed straight into build_daily_sentiment()).

    Args:
        symbol:     Stock symbol e.g. "RELIANCE.NS"
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"

    Returns:
        List of article dicts: title, date, sentiment, source,
        source_weight, importance_score, events (list[str])
    """
    db = SessionLocal()
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end   = datetime.strptime(end_date,   "%Y-%m-%d").date()

        rows = (
            db.query(NewsArticle)
              .filter(
                  NewsArticle.symbol == symbol,
                  NewsArticle.date >= start,
                  NewsArticle.date <= end,
              )
              .order_by(NewsArticle.date.asc())
              .all()
        )

        return [
            {
                "title":            row.title,
                "date":             row.date,
                "sentiment":        row.sentiment,
                "source":           row.source,
                "source_weight":    row.source_weight,
                "importance_score": row.importance_score,
                "events":           row.events.split(",") if row.events else [],
            }
            for row in rows
        ]

    finally:
        db.close()


def get_news_coverage(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Returns coverage stats for a symbol/date range — how many distinct
    days in the range have at least one real archived article.

    Used to warn callers when sentiment features for most of a requested
    range will be fabricated neutral placeholders rather than real signal.

    Returns:
        {"total_days": int, "days_with_news": int, "coverage_pct": float}
    """
    db = SessionLocal()
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end   = datetime.strptime(end_date,   "%Y-%m-%d").date()
        total_days = (end - start).days + 1

        days_with_news = (
            db.query(NewsArticle.date)
              .filter(
                  NewsArticle.symbol == symbol,
                  NewsArticle.date >= start,
                  NewsArticle.date <= end,
              )
              .distinct()
              .count()
        )

        coverage_pct = (days_with_news / total_days) if total_days > 0 else 0.0
        return {
            "total_days":     total_days,
            "days_with_news": days_with_news,
            "coverage_pct":   coverage_pct,
        }
    finally:
        db.close()
