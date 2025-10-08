"""Tests for helper utilities in ``app.main``."""

from app.google_ads_client import SearchTermRow
import app.main as main_module


def _row(term: str, impressions: int = 10, clicks: int = 1, cost: int = 100, conversions: float = 0.0) -> SearchTermRow:
    return SearchTermRow(
        campaign_id="1",
        campaign_name="Test",
        search_term=term,
        date=None,
        match_type="EXACT",
        match_source="SEARCH",
        impressions=impressions,
        clicks=clicks,
        cost_micros=cost,
        conversions=conversions,
    )


def test_prioritise_search_terms_caps_and_orders():
    main_module.DEFAULT_MAX_TERMS = 3
    main_module.MIN_METRIC_THRESHOLD = 0
    rows = [
        _row("brand", cost=10, clicks=1),
        _row("generic high cost", cost=500, clicks=3),
        _row("converter", cost=50, conversions=2.0),
        _row("medium", cost=200, clicks=2),
    ]

    selected = main_module.prioritise_search_terms(rows)

    assert len(selected) == 3
    # Converting terms are always retained
    assert any(term.search_term == "converter" for term in selected)
    # Highest cost non-converters should take the remaining slots in descending order
    assert [term.search_term for term in selected[:2]] == ["converter", "generic high cost"]


def test_prioritise_respects_impression_threshold():
    main_module.DEFAULT_MAX_TERMS = 10
    main_module.MIN_METRIC_THRESHOLD = 5
    rows = [_row("low impressions", impressions=2), _row("kept", impressions=10)]

    selected = main_module.prioritise_search_terms(rows)

    assert len(selected) == 1
    assert selected[0].search_term == "kept"
