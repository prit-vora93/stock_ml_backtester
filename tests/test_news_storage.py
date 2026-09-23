"""
tests/test_news_storage.py
---------------------------
Tests for the news sentiment persistence fix.

Bug fixed:
    fetch_yfinance_news() only returns yfinance's last ~10 items and
    fetch_rss_news() only returns whatever's currently in the RSS feeds —
    neither has a historical archive. Without persistence, a multi-year
    fetch_news_sentiment() call silently filled almost every requested day
    with a fabricated neutral (0.0) sentiment baseline.

    data/news_storage.py + the updated fetch_news_sentiment() now persist
    every fetched article to PostgreSQL and read the full archived range
    back on each call, so repeated runs accumulate real historical
    coverage instead of discarding what was fetched.

These tests need a live Postgres connection (available in this sandbox)
but NOT network access — all news fetching is mocked, since Yahoo
Finance / RSS feeds are blocked from this environment.

Run:
    pytest tests/test_news_storage.py -v
"""

from datetime import date, timedelta
from unittest.mock import patch

import pytest

from api.database import SessionLocal, NewsArticle, create_tables
import data.news_fetcher as news_fetcher
from data.news_storage import (
    save_news_articles,
    get_news_articles,
    get_news_coverage,
)


TEST_SYMBOL = "PYTEST_NEWS.NS"


@pytest.fixture(autouse=True)
def _clean_test_articles():
    """Removes any PYTEST_NEWS.NS rows before and after each test so tests
    are independent and repeatable."""
    create_tables()
    db = SessionLocal()
    try:
        db.query(NewsArticle).filter(NewsArticle.symbol == TEST_SYMBOL).delete()
        db.commit()
    finally:
        db.close()

    yield

    db = SessionLocal()
    try:
        db.query(NewsArticle).filter(NewsArticle.symbol == TEST_SYMBOL).delete()
        db.commit()
    finally:
        db.close()


def _article(title, day, sentiment=0.5, source="reuters", weight=1.0,
             importance=0.8, events=None):
    return {
        "title": title,
        "date": day,
        "sentiment": sentiment,
        "source": source,
        "source_weight": weight,
        "importance_score": importance,
        "events": events or [],
    }


class TestNewsStorage:

    def test_save_new_articles(self):
        day = date(2024, 1, 15)
        articles = [
            _article("Company beats Q3 estimates", day, events=["earnings"]),
            _article("Company launches new product", day, events=["product"]),
        ]
        saved = save_news_articles(TEST_SYMBOL, articles)
        assert saved == 2

    def test_duplicate_articles_are_skipped(self):
        day = date(2024, 1, 15)
        articles = [_article("Company beats Q3 estimates", day)]

        first  = save_news_articles(TEST_SYMBOL, articles)
        second = save_news_articles(TEST_SYMBOL, articles)

        assert first == 1
        assert second == 0, "Re-saving the same article must not duplicate it"

    def test_articles_accumulate_across_separate_calls(self):
        """
        This is the core behavior the fix depends on: each save_news_articles
        call only ever has a handful of CURRENT articles, but successive
        calls on different days must accumulate into a real archive.
        """
        day1 = date(2024, 1, 15)
        day2 = date(2024, 1, 16)

        save_news_articles(TEST_SYMBOL, [_article("Day one story", day1)])
        save_news_articles(TEST_SYMBOL, [_article("Day two story", day2)])

        archive = get_news_articles(TEST_SYMBOL, "2024-01-01", "2024-01-31")
        assert len(archive) == 2
        assert {a["date"] for a in archive} == {day1, day2}

    def test_get_news_articles_respects_date_range(self):
        in_range     = date(2024, 1, 15)
        out_of_range = date(2024, 6, 1)

        save_news_articles(TEST_SYMBOL, [
            _article("In range story", in_range),
            _article("Out of range story", out_of_range),
        ])

        archive = get_news_articles(TEST_SYMBOL, "2024-01-01", "2024-01-31")
        assert len(archive) == 1
        assert archive[0]["title"] == "In range story"

    def test_events_round_trip_as_list(self):
        day = date(2024, 1, 15)
        save_news_articles(TEST_SYMBOL, [
            _article("RBI raises rates", day, events=["rbi", "inflation"])
        ])
        archive = get_news_articles(TEST_SYMBOL, "2024-01-01", "2024-01-31")
        assert archive[0]["events"] == ["rbi", "inflation"]

    def test_coverage_reflects_real_archived_days(self):
        day = date(2024, 1, 15)
        save_news_articles(TEST_SYMBOL, [_article("Only story", day)])

        coverage = get_news_coverage(TEST_SYMBOL, "2024-01-01", "2024-01-31")
        assert coverage["total_days"] == 31
        assert coverage["days_with_news"] == 1
        assert coverage["coverage_pct"] == pytest.approx(1 / 31)

    def test_coverage_zero_when_no_articles(self):
        coverage = get_news_coverage(TEST_SYMBOL, "2024-01-01", "2024-01-31")
        assert coverage["days_with_news"] == 0
        assert coverage["coverage_pct"] == 0.0


class TestFetchNewsSentimentPersistence:
    """
    Integration tests for fetch_news_sentiment() with the fetch sources
    mocked out (no network in this sandbox) — verifies the end-to-end
    persist-then-read-back behavior.
    """

    def test_second_run_sees_first_runs_articles(self):
        day1 = date(2024, 3, 1)
        day2 = date(2024, 3, 5)

        with patch.object(news_fetcher, "fetch_yfinance_news",
                           return_value=[_article("First run story", day1, events=["earnings"])]), \
             patch.object(news_fetcher, "fetch_rss_news", return_value=[]):
            df1 = news_fetcher.fetch_news_sentiment(TEST_SYMBOL, "2024-01-01", "2024-03-31")

        assert df1.loc["2024-03-01", "news_count"] == 1

        with patch.object(news_fetcher, "fetch_yfinance_news",
                           return_value=[_article("Second run story", day2, events=["contract"])]), \
             patch.object(news_fetcher, "fetch_rss_news", return_value=[]):
            df2 = news_fetcher.fetch_news_sentiment(TEST_SYMBOL, "2024-01-01", "2024-03-31")

        # Second run's dataframe must still show the FIRST run's article —
        # this is the actual bug fix: coverage accumulates instead of
        # being limited to whatever's "currently" fetchable.
        assert df2.loc["2024-03-01", "news_count"] == 1, (
            "First run's archived article must still be visible on the second run"
        )
        assert df2.loc["2024-03-05", "news_count"] == 1

    def test_no_articles_still_returns_full_date_range(self):
        with patch.object(news_fetcher, "fetch_yfinance_news", return_value=[]), \
             patch.object(news_fetcher, "fetch_rss_news", return_value=[]):
            df = news_fetcher.fetch_news_sentiment(TEST_SYMBOL, "2024-01-01", "2024-01-31")

        assert len(df) == 31
        assert (df["news_count"] == 0).all()
        assert (df["sentiment"] == 0.0).all()
