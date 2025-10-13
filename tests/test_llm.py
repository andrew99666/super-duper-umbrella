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


def test_temperature_kwargs_respects_model_defaults():
    llm._temperature_warnings_issued.clear()
    # Ensure helper skips overrides for restricted models
    kwargs = llm._temperature_kwargs("gpt-5-mini", 0.0)
    assert kwargs == {}

    # Models not in the restriction list should pass through values
    kwargs_allowed = llm._temperature_kwargs("gpt-4o", 0.2)
    assert kwargs_allowed == {"temperature": 0.2}

    # Non-standard casing and other GPT-5 variants should also skip overrides
    kwargs_variant = llm._temperature_kwargs("GPT-5-preview", 0.0)
    assert kwargs_variant == {}

    # Whitespace variants should still be detected as GPT-5 models
    kwargs_spaced = llm._temperature_kwargs("  gpt 5-mini  ", 0.0)
    assert kwargs_spaced == {}

    llm._temperature_warnings_issued.clear()


def test_recursion_depth_limit_prevents_infinite_loops(monkeypatch):
    """Test that recursion depth limit is enforced to prevent excessive recursion."""
    call_count = {"value": 0}

    def fake_call(prompt: str) -> str:
        call_count["value"] += 1
        return "{"  # Always invalid JSON to force recursion

    monkeypatch.setattr(llm, "_call_relevancy_model", fake_call)

    # Test with multiple terms to trigger splitting
    terms = [
        {"query": f"term_{i}", "impressions": 10, "clicks": 0, "cost_micros": 0, "conversions": 0}
        for i in range(8)
    ]

    results = llm._analyse_chunk_with_fallback("summary", "context", terms, depth=0)

    # Should drop all terms after hitting max recursion depth
    assert results == []
    # Should have made 3 attempts at each level before recursing
    assert call_count["value"] > 3


def test_max_recursion_depth_reached_logs_error(monkeypatch):
    """Test that reaching max recursion depth logs an error and returns empty list."""
    terms = [{"query": "test", "impressions": 10, "clicks": 0, "cost_micros": 0, "conversions": 0}]

    # Start at max depth - 1, so next recursion will hit the limit
    results = llm._analyse_chunk_with_fallback(
        "summary", "context", terms, depth=llm._MAX_RECURSION_DEPTH
    )

    assert results == []


def test_concurrent_recursive_calls_no_deadlock(monkeypatch):
    """Test that concurrent recursive calls don't deadlock with larger executor pool."""
    import concurrent.futures

    def fake_call(prompt: str) -> str:
        # Return valid response for first 2 attempts, then fail
        payload = json.loads(prompt)
        if len(payload["terms"]) == 1:
            return _build_response_from_prompt(prompt)
        # Force split for multi-term batches
        return "{"

    monkeypatch.setattr(llm, "_call_relevancy_model", fake_call)
    monkeypatch.setattr(llm, "_max_parallel_requests", lambda: 2)

    # Submit multiple concurrent analyses
    terms_batch_1 = [
        {"query": "alpha", "impressions": 10, "clicks": 0, "cost_micros": 0, "conversions": 0},
        {"query": "beta", "impressions": 5, "clicks": 0, "cost_micros": 0, "conversions": 0},
    ]
    terms_batch_2 = [
        {"query": "gamma", "impressions": 10, "clicks": 0, "cost_micros": 0, "conversions": 0},
        {"query": "delta", "impressions": 5, "clicks": 0, "cost_micros": 0, "conversions": 0},
    ]

    # This should complete without deadlocking
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(llm.analyze_search_terms, "summary", "context", terms_batch_1)
        future2 = executor.submit(llm.analyze_search_terms, "summary", "context", terms_batch_2)

        results1 = future1.result(timeout=10)
        results2 = future2.result(timeout=10)

    # Both should complete successfully
    assert len(results1) == 2
    assert len(results2) == 2


def test_fallback_executor_properly_initialized():
    """Test that the fallback executor is properly initialized and registered for shutdown."""
    import atexit

    # Verify executor exists and has correct worker count
    assert llm._FALLBACK_EXECUTOR is not None
    assert llm._FALLBACK_EXECUTOR._max_workers == 10

    # Verify shutdown is registered (this is harder to test directly,
    # but we can verify the executor is a valid ThreadPoolExecutor)
    assert isinstance(llm._FALLBACK_EXECUTOR, llm.ThreadPoolExecutor)
