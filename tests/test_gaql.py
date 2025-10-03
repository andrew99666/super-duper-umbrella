import datetime as dt

from app.google_ads_client import (
    SearchTermRow,
    aggregate_search_term_rows,
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


def test_aggregate_search_term_rows_merges_duplicates():
    rows = [
        SearchTermRow(
            campaign_id="123",
            campaign_name="Alpha",
            search_term="Shoes",
            date=dt.date(2024, 1, 1),
            match_type="BROAD",
            match_source="SEARCH",
            impressions=10,
            clicks=2,
            cost_micros=1500,
            conversions=1.0,
        ),
        SearchTermRow(
            campaign_id="123",
            campaign_name="Alpha",
            search_term="Shoes",
            date=dt.date(2024, 1, 2),
            match_type="PHRASE",
            match_source="SEARCH",
            impressions=5,
            clicks=1,
            cost_micros=500,
            conversions=0.0,
        ),
        SearchTermRow(
            campaign_id="999",
            campaign_name="Beta",
            search_term="Socks",
            date=dt.date(2024, 1, 1),
            match_type="BROAD",
            match_source="SEARCH",
            impressions=2,
            clicks=0,
            cost_micros=200,
            conversions=0.0,
        ),
    ]

    aggregated = aggregate_search_term_rows(rows)

    assert len(aggregated) == 2
    primary = next(row for row in aggregated if row.campaign_id == "123")
    assert primary.impressions == 15
    assert primary.clicks == 3
    assert primary.cost_micros == 2000
    assert primary.conversions == 1.0
    assert primary.match_type == "mixed"
    assert primary.match_source == "SEARCH"

    secondary = next(row for row in aggregated if row.campaign_id == "999")
    assert secondary.search_term == "Socks"
