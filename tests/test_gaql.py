import datetime as dt

from app.google_ads_client import (
    build_campaign_search_term_query,
    build_landing_page_query,
    build_landing_page_view_query,
    build_search_term_fallback_query,
)


def test_build_campaign_search_term_query_uses_campaign_view():
    start = dt.date(2024, 1, 1)
    end = dt.date(2024, 1, 31)
    query = build_campaign_search_term_query(["123", "456"], start, end)
    assert "FROM campaign_search_term_view" in query
    normalized = query.replace(" ", "").replace("\n", "")
    assert "campaign.idIN(123,456)" in normalized
    assert "segments.date BETWEEN '2024-01-01' AND '2024-01-31'" in query


def test_build_search_term_fallback_query():
    start = dt.date(2024, 2, 1)
    end = dt.date(2024, 2, 15)
    query = build_search_term_fallback_query(["789"], start, end)
    assert "FROM search_term_view" in query
    assert "campaign.id IN (789)" in query


def test_build_landing_page_query_includes_campaign_id():
    query = build_landing_page_query([111, 222])
    assert "FROM ad_group_ad" in query
    assert "campaign.id" in query


def test_build_landing_page_view_query_includes_campaign():
    query = build_landing_page_view_query([111])
    assert "FROM landing_page_view" in query
    assert "campaign.id" in query
