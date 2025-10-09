"""OpenAI client helpers for landing page summarisation and relevancy scoring."""
from __future__ import annotations

import atexit
import json
import logging
import threading
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Sequence

try:  # pragma: no cover - optional dependency shim for tests
    from openai import APIError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - fallback when openai is missing
    class APIError(Exception):
        """Placeholder API error when openai package is unavailable."""

        pass

    class RateLimitError(Exception):
        """Placeholder rate-limit error when openai package is unavailable."""

        pass

    class OpenAI:  # type: ignore[override]
        """Minimal stub that signals the OpenAI SDK is missing."""

        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise ImportError("openai package is not installed")
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, ValidationError, field_validator

from .schemas import PageContent

logger = logging.getLogger(__name__)

# Module-level shared thread pool for recursive fallback processing
# Use a larger pool to prevent deadlocks during recursive analysis
_FALLBACK_EXECUTOR = ThreadPoolExecutor(max_workers=10)
_MAX_RECURSION_DEPTH = 10

# Register shutdown hook to ensure proper cleanup
atexit.register(_FALLBACK_EXECUTOR.shutdown, wait=False)

DEFAULT_PAGE_SUMMARY_SYSTEM_PROMPT = (
    "You analyze a landing page to infer its product/service, audience, and exclusions. "
    "Return a concise, factual summary (bulleted), avoid marketing fluff."
)

DEFAULT_RELEVANCY_SYSTEM_PROMPT = (
    "You classify paid search queries as relevant vs irrelevant for a given landing page. "
    "Be conservative. Do not block brand or near-brand terms. Prefer EXACT negatives for single clear bad "
    "queries; PHRASE if many bad variants share a phrase. Output only JSON matching the provided schema."
)


def _page_summary_prompt() -> str:
    return os.getenv("OPENAI_PAGE_SUMMARY_SYSTEM_PROMPT", DEFAULT_PAGE_SUMMARY_SYSTEM_PROMPT)


def _relevancy_prompt() -> str:
    return os.getenv("OPENAI_RELEVANCY_SYSTEM_PROMPT", DEFAULT_RELEVANCY_SYSTEM_PROMPT)


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

    @field_validator("suggested_match_type", mode="before")
    @classmethod
    def normalise_match_type(cls, value: str | None) -> str:
        """Coerce unexpected match types into a supported value."""

        if value is None:
            logger.debug("Missing match type from LLM; defaulting to exact")
            return "exact"

        normalised = str(value).strip().lower()
        if normalised in {"exact", "phrase"}:
            return normalised

        if normalised in {"none", "unknown", ""}:
            logger.debug(
                "Unsupported match type '%s' from LLM; coercing to exact", normalised
            )
            return "exact"

        logger.debug(
            "Unexpected match type '%s' from LLM; coercing to exact", normalised
        )
        return "exact"


class RelevancyBatch(BaseModel):
    terms: List[RelevancyResult]


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    try:
        return OpenAI(api_key=api_key)
    except ImportError as exc:  # pragma: no cover - missing dependency
        raise RuntimeError("The openai Python package is required to call the API.") from exc


def _openai_retry():
    return retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=1, max=16),
        retry=retry_if_exception_type((RateLimitError, APIError)),
    )


def _model_name() -> str:
    """Return the configured OpenAI model name."""

    value = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    return value.strip()


# Models in the GPT-5 family currently ignore custom temperature values and
# return 400 errors when one is supplied. We therefore suppress overrides for
# those models entirely while still allowing explicit values for others.
_temperature_warnings_issued: set[str] = set()
_temperature_warning_lock = threading.Lock()


def _temperature_kwargs(model: str, desired: float | None) -> dict[str, float]:
    """Return kwargs for temperature respecting model limitations."""

    if desired is None:
        return {}

    model_key = model.strip().lower()
    normalised = model_key.replace("_", "-")
    compact = normalised.replace(" ", "")
    if compact.startswith("gpt-5") or compact.startswith("gpt5"):
        if desired != 1:
            with _temperature_warning_lock:
                if compact not in _temperature_warnings_issued:
                    logger.info(
                        "Model %s ignores custom temperature; skipping override %.2f",
                        model,
                        desired,
                    )
                    _temperature_warnings_issued.add(compact)
        return {}

    return {"temperature": desired}


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
    start = time.perf_counter()
    model = _model_name()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _page_summary_prompt()},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        **_temperature_kwargs(model, 0.2),
    )
    content = response.choices[0].message.content or ""
    logger.info(
        "Generated LLM summary for %s in %.2fs",
        page.url,
        time.perf_counter() - start,
    )
    logger.debug("LLM summary response: %s", content)
    return content.strip()


def chunk_terms(terms: Sequence[dict], size: int) -> Iterable[Sequence[dict]]:
    for idx in range(0, len(terms), size):
        yield terms[idx : idx + size]


def _max_parallel_requests() -> int:
    """Return the number of concurrent OpenAI calls to issue.

    The limit is derived from the optional ``OPENAI_MAX_CONCURRENT_REQUESTS``
    environment variable. To honour the user's request to stay 15%% below the
    advertised cap we multiply the configured maximum by 0.85. When the value
    is unset or invalid we assume a generous limit of 120 concurrent calls,
    yielding 102 workers after the reduction.
    """

    default_limit = 120
    raw = os.getenv("OPENAI_MAX_CONCURRENT_REQUESTS")
    if not raw:
        configured = default_limit
    else:
        try:
            configured = int(raw)
        except ValueError:
            logger.warning(
                "Invalid OPENAI_MAX_CONCURRENT_REQUESTS value %s; falling back to %s",
                raw,
                default_limit,
            )
            configured = default_limit
    if configured <= 0:
        return 1
    calculated = max(1, int(configured * 0.85))
    return calculated or 1


def _relevancy_chunk_size() -> int:
    default_size = 50
    raw = os.getenv("OPENAI_RELEVANCY_CHUNK_SIZE")
    if not raw:
        return default_size
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid OPENAI_RELEVANCY_CHUNK_SIZE value %s; falling back to %s",
            raw,
            default_size,
        )
        return default_size
    return max(10, min(value, 200))


def _analyse_chunk(
    page_summary: str,
    campaign_context: str | None,
    chunk: Sequence[dict],
) -> list[RelevancyResult]:
    return _analyse_chunk_with_fallback(page_summary, campaign_context, list(chunk), depth=0)


def _analyse_chunk_with_fallback(
    page_summary: str,
    campaign_context: str | None,
    chunk: list[dict],
    depth: int = 0,
) -> list[RelevancyResult]:
    if not chunk:
        return []

    # Prevent excessive recursion
    if depth >= _MAX_RECURSION_DEPTH:
        logger.error(
            "Max recursion depth reached for chunk of size %d; dropping terms",
            len(chunk),
        )
        return []

    prompt = _relevancy_user_prompt(page_summary, campaign_context, chunk)
    last_error: ValueError | None = None
    for attempt in range(3):
        logger.debug(
            "Calling OpenAI relevancy model for %d terms (attempt %d, depth %d)",
            len(chunk),
            attempt + 1,
            depth,
        )
        try:
            response_text = _call_relevancy_model(prompt)
            return parse_relevancy_response(response_text)
        except ValueError as exc:
            last_error = exc
            logger.warning("LLM JSON parse failure (attempt %s): %s", attempt + 1, exc)

    logger.info(
        "Falling back to smaller batches after repeated LLM parse failures (chunk_size=%s, depth=%d, last_error=%s)",
        len(chunk),
        depth,
        last_error,
    )

    if len(chunk) == 1:
        logger.error(
            "Dropping search term %s after repeated LLM parse failures",
            chunk[0].get("query"),
        )
        return []

    mid = max(1, len(chunk) // 2)
    # Use module-level shared thread pool for recursive fallback
    left_future = _FALLBACK_EXECUTOR.submit(_analyse_chunk_with_fallback, page_summary, campaign_context, chunk[:mid], depth + 1)
    right_future = _FALLBACK_EXECUTOR.submit(_analyse_chunk_with_fallback, page_summary, campaign_context, chunk[mid:], depth + 1)
    left = left_future.result()
    right = right_future.result()
    return left + right


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
    model = _model_name()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _relevancy_prompt()},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        **_temperature_kwargs(model, 0.0),
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

    chunk_size = _relevancy_chunk_size()
    start = time.perf_counter()
    chunk_list = list(chunk_terms(list(terms), size=chunk_size))
    if not chunk_list:
        return []

    worker_count = min(len(chunk_list), _max_parallel_requests())
    logger.info(
        "Analysing %d search terms in %d chunks (chunk_size=%d, workers=%d)",
        len(terms),
        len(chunk_list),
        chunk_size,
        worker_count,
    )
    results: Dict[int, list[RelevancyResult]] = {}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map: Dict[Future[list[RelevancyResult]], int] = {}
        for index, chunk in enumerate(chunk_list):
            future = executor.submit(_analyse_chunk, page_summary, campaign_context, chunk)
            future_map[future] = index

        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()

    ordered: list[RelevancyResult] = []
    for index in sorted(results):
        ordered.extend(results[index])
    logger.info(
        "OpenAI relevancy analysis completed with %d results in %.2fs",
        len(ordered),
        time.perf_counter() - start,
    )
    return ordered
