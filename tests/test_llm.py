import json

from app import llm


def _build_response_from_prompt(prompt: str) -> str:
    payload = json.loads(prompt)
    terms = []
    for term in payload["terms"]:
        query = term["query"]
        terms.append(
            {
                "query": query,
                "relevancy_label": "irrelevant",
                "reason": f"Test result for {query}",
                "confidence": 0.9,
                "suggest_negative": True,
                "suggested_match_type": "exact",
                "match_type_rationale": "Unit test rationale",
            }
        )
    return json.dumps({"terms": terms})


def test_analyze_search_terms_splits_on_repeated_failures(monkeypatch):
    call_count = {"value": 0}

    def fake_call(prompt: str) -> str:
        call_count["value"] += 1
        if call_count["value"] <= 3:
            return "{"  # invalid JSON to trigger the fallback
        return _build_response_from_prompt(prompt)

    monkeypatch.setattr(llm, "_call_relevancy_model", fake_call)
    monkeypatch.setattr(llm, "_max_parallel_requests", lambda: 1)

    terms = [
        {"query": "alpha", "impressions": 10, "clicks": 0, "cost_micros": 0, "conversions": 0},
        {"query": "beta", "impressions": 5, "clicks": 0, "cost_micros": 0, "conversions": 0},
    ]

    results = llm.analyze_search_terms("summary", "context", terms)

    assert [result.query for result in results] == ["alpha", "beta"]
    assert call_count["value"] >= 5  # three failures + at least two successful splits


def test_analyze_search_terms_drops_single_term_after_failures(monkeypatch):
    def fake_call(prompt: str) -> str:  # pragma: no cover - behaviour under test
        return "{"  # always invalid JSON

    monkeypatch.setattr(llm, "_call_relevancy_model", fake_call)
    monkeypatch.setattr(llm, "_max_parallel_requests", lambda: 1)

    terms = [
        {"query": "gamma", "impressions": 2, "clicks": 0, "cost_micros": 0, "conversions": 0},
    ]

    results = llm.analyze_search_terms("summary", "context", terms)

    assert results == []
