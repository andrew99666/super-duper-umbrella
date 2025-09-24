"""OpenAI client helpers for landing page summarisation and relevancy scoring."""
from __future__ import annotations

import json
import logging
import os
from typing import Iterable, List, Sequence

from openai import APIError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, ValidationError, field_validator

from .schemas import PageContent

logger = logging.getLogger(__name__)

PAGE_SUMMARY_SYSTEM_PROMPT = (
    "You analyze a landing page to infer its product/service, audience, and exclusions. "
    "Return a concise, factual summary (bulleted), avoid marketing fluff."
)

RELEVANCY_SYSTEM_PROMPT = (
    "You classify paid search queries as relevant vs irrelevant for a given landing page. "
    "Be conservative. Do not block brand or near-brand terms. Prefer EXACT negatives for single clear bad "
    "queries; PHRASE if many bad variants share a phrase. Output only JSON matching the provided schema."
)


class RelevancyResult(BaseModel):
    query: str
    relevancy_label: str = Field(pattern=r"^(relevant|possibly_related|irrelevant)$")
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    suggest_negative: bool
    suggested_match_type: str = Field(pattern=r"^(exact|phrase)$")
    match_type_rationale: str

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("query cannot be empty")
        return value


class RelevancyBatch(BaseModel):
    terms: List[RelevancyResult]


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=api_key)


def _openai_retry():
    return retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=1, max=16),
        retry=retry_if_exception_type((RateLimitError, APIError)),
    )


def _model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@_openai_retry()
def generate_page_summary(page: PageContent) -> str:
    """Create a compact landing-page summary using the LLM."""
    client = _get_openai_client()
    prompt = {
        "url": page.url,
        "title": page.title or "",
        "meta_description": page.meta_description or "",
        "h1s": page.h1s,
        "h2s": page.h2s,
        "visible_text_excerpt": page.visible_text_excerpt[:1500],
        "canonical_url": page.canonical_url or "",
    }
    response = client.chat.completions.create(
        model=_model_name(),
        temperature=0.2,
        messages=[
            {"role": "system", "content": PAGE_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content or ""
    logger.debug("LLM summary response: %s", content)
    return content.strip()


def chunk_terms(terms: Sequence[dict], size: int = 200) -> Iterable[Sequence[dict]]:
    for idx in range(0, len(terms), size):
        yield terms[idx : idx + size]


def _relevancy_user_prompt(
    page_summary: str, campaign_context: str | None, terms: Sequence[dict]
) -> str:
    payload = {
        "page_summary": page_summary,
        "campaign_context": campaign_context or "",
        "terms": terms,
        "response_schema": {
            "type": "object",
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "query",
                            "relevancy_label",
                            "reason",
                            "confidence",
                            "suggest_negative",
                            "suggested_match_type",
                            "match_type_rationale",
                        ],
                        "properties": {
                            "query": {"type": "string"},
                            "relevancy_label": {
                                "type": "string",
                                "enum": ["relevant", "possibly_related", "irrelevant"],
                            },
                            "reason": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "suggest_negative": {"type": "boolean"},
                            "suggested_match_type": {
                                "type": "string",
                                "enum": ["exact", "phrase"],
                            },
                            "match_type_rationale": {"type": "string"},
                        },
                    },
                }
            },
            "required": ["terms"],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_relevancy_response(content: str) -> list[RelevancyResult]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from LLM: {exc}") from exc
    try:
        batch = RelevancyBatch.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"LLM response did not match schema: {exc}") from exc
    return batch.terms


@_openai_retry()
def _call_relevancy_model(prompt: str) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=_model_name(),
        temperature=0.0,
        messages=[
            {"role": "system", "content": RELEVANCY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    logger.debug("LLM relevancy response: %s", content)
    return content


def analyze_search_terms(
    page_summary: str,
    campaign_context: str | None,
    terms: Sequence[dict],
) -> list[RelevancyResult]:
    """Analyse search terms in batches and validate JSON output."""
    if not terms:
        return []

    results: list[RelevancyResult] = []
    for chunk in chunk_terms(list(terms)):
        prompt = _relevancy_user_prompt(page_summary, campaign_context, chunk)
        for attempt in range(3):
            try:
                response_text = _call_relevancy_model(prompt)
                parsed = parse_relevancy_response(response_text)
            except ValueError as exc:
                logger.warning("LLM JSON parse failure (attempt %s): %s", attempt + 1, exc)
                if attempt == 2:
                    raise
                continue
            results.extend(parsed)
            break
    return results
