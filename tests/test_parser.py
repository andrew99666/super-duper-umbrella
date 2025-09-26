import json

from app.llm import parse_relevancy_response
from app.scrape import extract_page_content


def test_extract_page_content_parses_headings():
    html = """
    <html>
      <head>
        <title>Test Landing Page</title>
        <meta name="description" content="A simple test page" />
        <link rel="canonical" href="https://example.com/test" />
      </head>
      <body>
        <h1>Primary Heading</h1>
        <h2>Subheading</h2>
        <p>Some interesting content about widgets and services.</p>
      </body>
    </html>
    """
    page = extract_page_content("https://example.com", html)
    assert page.title == "Test Landing Page"
    assert page.meta_description == "A simple test page"
    assert page.canonical_url == "https://example.com/test"
    assert "Primary Heading" in page.h1s[0]
    assert "Subheading" in page.h2s[0]
    assert "widgets" in page.visible_text_excerpt


def test_parse_relevancy_response_validates_schema():
    payload = {
        "terms": [
            {
                "query": "test query",
                "relevancy_label": "irrelevant",
                "reason": "Does not match the landing page offering.",
                "confidence": 0.92,
                "suggest_negative": True,
                "suggested_match_type": "exact",
                "match_type_rationale": "Narrow bad query",
            }
        ]
    }
    json_str = json.dumps(payload)
    results = parse_relevancy_response(json_str)
    assert len(results) == 1
    assert results[0].query == "test query"
    assert results[0].relevancy_label == "irrelevant"
