"""FastAPI application entrypoint."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import logging
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List
from uuid import uuid4

from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from .db import get_session, init_db
from .google_ads_client import (
    GoogleAdsOAuthError,
    GoogleAdsService,
    LandingPageRow,
    SearchTermRow,
    CustomerClientSummary,
    aggregate_search_term_rows,
    build_google_ads_client,
    build_oauth_flow,
    campaigns_to_dict,
    store_user_credentials,
)
from .logging_setup import configure_logging
from .llm import analyze_search_terms, generate_page_summary
from .models import (
    AnalysisRun,
    Campaign,
    GoogleAdsCustomer,
    LandingPage,
    SearchTerm,
    SearchTermAnalysis,
    Suggestion,
    User,
)
from .scrape import extract_page_content, fetch_page

LOG_FILE_PATH = configure_logging(os.getenv("LOG_LEVEL"))
logger = logging.getLogger(__name__)
logger.info("Logging configured. File output: %s", LOG_FILE_PATH)

app = FastAPI(title="Search Term Relevancy Assistant")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("APP_SECRET_KEY", "change-me"))

TEMPLATES = Jinja2Templates(directory="app/templates")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
SUMMARY_TTL = dt.timedelta(days=3)

DEFAULT_MAX_TERMS = int(os.getenv("OPENAI_MAX_TERMS", "1500") or 0) or 1500
MIN_METRIC_THRESHOLD = int(os.getenv("OPENAI_MIN_IMPRESSIONS", "0") or 0)


def _format_duration(seconds: float) -> str:
    """Return a human friendly duration string."""

    if seconds <= 0:
        return "<1 ms"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 120:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def prioritise_search_terms(rows: list[SearchTermRow]) -> list[SearchTermRow]:
    """Down-select rows for LLM analysis based on performance metrics.

    The OpenAI calls are the slowest portion of the workflow. Large accounts
    can return thousands of aggregated queries which would otherwise result in
    many long-running API calls. We keep all terms that have recorded
    conversions, then prioritise the remainder by cost, clicks, and
    impressions. Low-activity rows (below ``MIN_METRIC_THRESHOLD`` impressions)
    are dropped entirely. The final list is capped at ``DEFAULT_MAX_TERMS`` to
    ensure the analysis completes in minutes rather than hours.
    """

    if not rows:
        return []

    filtered: list[SearchTermRow] = []
    for row in rows:
        if row.impressions and row.impressions < MIN_METRIC_THRESHOLD:
            continue
        filtered.append(row)

    if len(filtered) <= DEFAULT_MAX_TERMS:
        return filtered

    converters = [r for r in filtered if (r.conversions or 0) > 0]
    non_converters = [r for r in filtered if (r.conversions or 0) <= 0]
    non_converters.sort(
        key=lambda r: (
            r.cost_micros or 0,
            r.clicks or 0,
            r.impressions or 0,
            r.search_term,
        ),
        reverse=True,
    )

    prioritised = converters + non_converters
    if len(prioritised) <= DEFAULT_MAX_TERMS:
        return prioritised

    return prioritised[:DEFAULT_MAX_TERMS]


@app.on_event("startup")
def on_startup() -> None:
    start = time.perf_counter()
    logger.info("Starting application initialisation")
    init_db()
    duration = time.perf_counter() - start
    logger.info("Database initialised in %.2fs", duration)


def get_current_user(request: Request, session: Session = Depends(get_session)) -> User:
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_302_FOUND, headers={"Location": "/"})
    user = session.get(User, user_id)
    if not user:
        request.session.pop("user_id", None)
        raise HTTPException(status_code=status.HTTP_302_FOUND, headers={"Location": "/"})
    return user


@app.get("/", response_class=HTMLResponse)
def index(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
    user = None
    user_id = request.session.get("user_id")
    if user_id:
        user = session.get(User, user_id)
    return TEMPLATES.TemplateResponse("index.html", {"request": request, "user": user})


@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


@app.get("/auth/google")
def auth_google(request: Request) -> RedirectResponse:
    try:
        state = str(uuid4())
        flow = build_oauth_flow(BASE_URL, state=state)
        authorization_url, state = flow.authorization_url(
            access_type="offline",
            prompt="consent",
            include_granted_scopes="true",
        )
        logger.info("Generated Google OAuth URL for state %s", state)
    except GoogleAdsOAuthError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    request.session["oauth_state"] = state
    request.session["code_verifier"] = flow.code_verifier
    return RedirectResponse(url=authorization_url, status_code=status.HTTP_302_FOUND)


@app.get("/oauth/callback")
def oauth_callback(request: Request, session: Session = Depends(get_session)) -> RedirectResponse:
    state = request.query_params.get("state")
    stored_state = request.session.get("oauth_state")
    if not state or stored_state != state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing OAuth code")

    logger.info("Received OAuth callback for state %s", state)
    flow = build_oauth_flow(BASE_URL, state=state)
    code_verifier = request.session.get("code_verifier")
    if code_verifier:
        flow.code_verifier = code_verifier
    try:
        flow.fetch_token(code=code)
        logger.info("OAuth token exchange succeeded for state %s", state)
    except Exception as exc:  # pragma: no cover - network
        logger.exception("OAuth token exchange failed: %s", exc)
        raise HTTPException(status_code=400, detail="OAuth token exchange failed") from exc

    credentials = flow.credentials
    refresh_token = credentials.refresh_token
    if not refresh_token:
        raise HTTPException(status_code=400, detail="No refresh token returned")

    token_hash = hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()
    user = session.execute(select(User).where(User.token_hash == token_hash)).scalar_one_or_none()
    if not user:
        user = User(token_hash=token_hash, google_refresh_token=b"")
        session.add(user)
        session.flush()

    store_user_credentials(user, credentials)
    session.add(user)

    try:
        logger.info("Building Google Ads client for user %s", user.id)
        client = build_google_ads_client(user)
        service = GoogleAdsService(client)
        sync_customers(session, user, service)
        session.commit()
    except GoogleAdsOAuthError as exc:
        session.rollback()
        logger.exception("Failed to sync Google Ads accounts: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    request.session["user_id"] = user.id
    request.session.pop("oauth_state", None)
    request.session.pop("code_verifier", None)
    return RedirectResponse(url="/campaigns", status_code=status.HTTP_302_FOUND)


def sync_customers(session: Session, user: User, service: GoogleAdsService) -> None:
    sync_start = time.perf_counter()
    logger.info("Syncing accessible customers for user %s", user.id)
    accessible = service.list_accessible_customers()
    logger.info("Retrieved %d accessible customer resource names", len(accessible))
    default_login_id = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")

    # Ensure we prefer the configured login customer ID if provided.
    if not accessible and default_login_id and not user.login_customer_id:
        user.login_customer_id = default_login_id
        session.add(user)
        return

    metadata: dict[str, CustomerClientSummary] = {}
    parent_logins: dict[str, str] = {}
    queue: deque[str] = deque()
    preferred_login = user.login_customer_id or default_login_id
    for entry in accessible:
        resource_name = entry.get("resource_name")
        if resource_name:
            queue.append(resource_name)
            customer_id = resource_name.split("/")[-1]
            parent_logins.setdefault(resource_name, preferred_login or customer_id)

    if not queue:
        logger.warning("No accessible customers returned for user %s", user.id)
        return

    seen_ids: set[str] = set()
    manager_candidates: list[str] = []

    while queue:
        resource_name = queue.popleft()
        customer_id = resource_name.split("/")[-1]
        if customer_id in seen_ids:
            continue
        seen_ids.add(customer_id)

        summary = metadata.get(resource_name)
        login_customer_id = parent_logins.get(resource_name)
        descriptive_name = summary.descriptive_name if summary else None
        currency_code = summary.currency_code if summary else None
        time_zone = summary.time_zone if summary else None
        is_manager = summary.manager if summary else False

        try:
            fetch_start = time.perf_counter()
            customer_data = service.get_customer(
                resource_name, login_customer_id=login_customer_id
            )
            logger.debug(
                "Fetched customer %s details in %.2fs",
                resource_name,
                time.perf_counter() - fetch_start,
            )
            descriptive_name = getattr(customer_data, "descriptive_name", descriptive_name)
            currency_code = getattr(customer_data, "currency_code", currency_code)
            time_zone = getattr(customer_data, "time_zone", time_zone)
            is_manager = bool(getattr(customer_data, "manager", is_manager))
        except Exception as exc:  # pragma: no cover - network error fallback
            logger.debug("Failed to hydrate customer %s: %s", resource_name, exc)

        existing = session.execute(
            select(GoogleAdsCustomer).where(GoogleAdsCustomer.resource_name == resource_name)
        ).scalar_one_or_none()
        if existing:
            customer = existing
        else:
            customer = GoogleAdsCustomer(
                user_id=user.id,
                resource_name=resource_name,
                customer_id=customer_id,
            )

        customer.descriptive_name = descriptive_name
        customer.currency_code = currency_code
        customer.time_zone = time_zone
        session.add(customer)

        if is_manager and customer_id not in manager_candidates:
            manager_candidates.append(customer_id)

        # Only list customer clients for manager accounts
        # Non-manager accounts don't have child accounts and will fail with permission errors
        if is_manager:
            try:
                child_start = time.perf_counter()
                child_customers = service.list_customer_clients(
                    customer_id, login_customer_id=login_customer_id or customer_id
                )
                logger.info(
                    "Customer %s returned %d child accounts in %.2fs",
                    customer_id,
                    len(child_customers),
                    time.perf_counter() - child_start,
                )
            except Exception as exc:  # pragma: no cover - network error fallback
                logger.debug("Failed to list child customers for %s: %s", customer_id, exc)
                child_customers = []
        else:
            child_customers = []
            logger.debug("Skipping list_customer_clients for non-manager account %s", customer_id)

        for child in child_customers:
            if child.id in seen_ids:
                continue
            metadata.setdefault(child.resource_name, child)
            queue.append(child.resource_name)
            # For child accounts, use the parent manager's ID as login-customer-id
            # Manager accounts can use their own ID, non-managers must use their parent manager
            if child.manager:
                parent_logins.setdefault(child.resource_name, child.id)
            else:
                # Non-manager child accounts need the manager account ID
                parent_logins.setdefault(child.resource_name, login_customer_id or customer_id)
            if child.manager and child.id not in manager_candidates:
                manager_candidates.append(child.id)

    if not user.login_customer_id:
        if default_login_id:
            user.login_customer_id = default_login_id
        elif manager_candidates:
            user.login_customer_id = manager_candidates[0]
        elif accessible:
            first = accessible[0].get("resource_name")
            if first:
                user.login_customer_id = first.split("/")[-1]
        session.add(user)
        logger.info("Selected login customer ID %s for user %s", user.login_customer_id, user.id)

    logger.info("Customer sync completed in %.2fs", time.perf_counter() - sync_start)


@app.get("/campaigns", response_class=HTMLResponse)
async def list_campaigns(
    request: Request,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> HTMLResponse:
    logger.info("Listing campaigns for user %s", user.id)
    client = build_google_ads_client(user)
    service = GoogleAdsService(client)

    customers = (
        session.execute(
            select(GoogleAdsCustomer).where(GoogleAdsCustomer.user_id == user.id)
        )
        .scalars()
        .all()
    )

    requested_customer = request.query_params.get("customer_id")
    session_selected = request.session.get("selected_customer_id")
    customer_id = requested_customer or session_selected or user.login_customer_id
    if not customer_id and customers:
        customer_id = customers[0].customer_id

    if not customer_id:
        raise HTTPException(status_code=400, detail="No customer ID selected")

    valid_customer_ids = {customer.customer_id for customer in customers}
    if customer_id not in valid_customer_ids:
        raise HTTPException(status_code=404, detail="Unknown customer ID")

    request.session["selected_customer_id"] = customer_id

    list_start = time.perf_counter()
    campaigns = service.list_campaigns(
        customer_id, login_customer_id=user.login_customer_id or os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
    )
    logger.info(
        "Fetched %d campaigns for customer %s in %.2fs",
        len(campaigns),
        customer_id,
        time.perf_counter() - list_start,
    )
    persist_campaigns(session, user, customer_id, campaigns)

    customers = (
        session.execute(
            select(GoogleAdsCustomer).where(GoogleAdsCustomer.user_id == user.id)
        )
        .scalars()
        .all()
    )

    campaign_payload = campaigns_to_dict(campaigns)
    for campaign in campaign_payload:
        status = (campaign.get("status") or "").replace("_", " ").title()
        campaign["status_label"] = status if status else "Unknown"

    active_campaigns: list[dict[str, str | None]] = []
    paused_campaigns: list[dict[str, str | None]] = []
    archived_campaigns: list[dict[str, str | None]] = []

    for campaign in campaign_payload:
        status_value = (campaign.get("status") or "").upper()
        if status_value == "ENABLED":
            active_campaigns.append(campaign)
        elif status_value in {"PAUSED", "PENDING"}:
            paused_campaigns.append(campaign)
        else:
            archived_campaigns.append(campaign)

    sort_key = lambda c: (c.get("name") or "").lower()
    active_campaigns.sort(key=sort_key)
    paused_campaigns.sort(key=sort_key)
    archived_campaigns.sort(key=sort_key)

    context = {
        "request": request,
        "active_campaigns": active_campaigns,
        "paused_campaigns": paused_campaigns,
        "archived_campaigns": archived_campaigns,
        "customers": customers,
        "selected_customer": customer_id,
        "default_start": (dt.date.today() - dt.timedelta(days=7)).isoformat(),
        "default_end": dt.date.today().isoformat(),
        "user": user,
    }
    return TEMPLATES.TemplateResponse("campaigns.html", context)


def persist_campaigns(
    session: Session, user: User, customer_id: str, campaigns: Iterable
) -> None:
    start = time.perf_counter()
    campaign_list = list(campaigns)
    customer = session.execute(
        select(GoogleAdsCustomer).where(GoogleAdsCustomer.customer_id == customer_id)
    ).scalar_one_or_none()
    if not customer:
        customer = GoogleAdsCustomer(
            user_id=user.id,
            resource_name=f"customers/{customer_id}",
            customer_id=customer_id,
        )
        session.add(customer)
        session.flush()

    existing_campaigns = {
        c.campaign_id: c
        for c in session.execute(
            select(Campaign).where(Campaign.customer_id == customer.id)
        ).scalars()
    }
    for summary in campaign_list:
        record = existing_campaigns.get(summary.campaign_id)
        if not record:
            record = Campaign(
                customer=customer,
                campaign_id=summary.campaign_id,
                name=summary.name,
                status=summary.status,
                advertising_channel_type=summary.advertising_channel_type,
            )
        else:
            record.name = summary.name
            record.status = summary.status
            record.advertising_channel_type = summary.advertising_channel_type
        session.add(record)
    logger.info(
        "Persisted %d campaigns for customer %s in %.2fs",
        len(campaign_list),
        customer_id,
        time.perf_counter() - start,
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    campaign_ids: List[str] = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    auto_select: str | None = Form(None),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> HTMLResponse:
    if not campaign_ids:
        raise HTTPException(status_code=400, detail="No campaigns selected")

    analysis_timer = time.perf_counter()
    logger.info(
        "Starting analysis for campaigns %s with range %s to %s",
        campaign_ids,
        start_date,
        end_date,
    )
    try:
        start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date format") from exc

    if start > end:
        raise HTTPException(status_code=400, detail="Start date must be before end date")

    analysis_run = AnalysisRun(
        user_id=user.id,
        campaign_ids=",".join(campaign_ids),
        start_date=start,
        end_date=end,
    )
    session.add(analysis_run)
    session.flush()

    client = build_google_ads_client(user)
    service = GoogleAdsService(client)

    auto_select_irrelevant = auto_select == "irrelevant"

    landing_duration_total = 0.0
    search_duration_total = 0.0
    persist_duration_total = 0.0
    llm_duration_total = 0.0
    total_landing_urls = 0
    total_raw_rows = 0
    total_aggregated_rows = 0
    total_prioritised_rows = 0
    total_skipped_rows = 0
    total_llm_results = 0
    total_llm_suggestions = 0
    analysis_steps: list[dict[str, str]] = []

    campaign_records = session.execute(
        select(Campaign).where(Campaign.campaign_id.in_(campaign_ids))
    ).scalars().all()
    campaigns_by_customer: Dict[str, list[Campaign]] = defaultdict(list)
    for campaign in campaign_records:
        customer = session.get(GoogleAdsCustomer, campaign.customer_id)
        if not customer:
            continue
        campaigns_by_customer[customer.customer_id].append(campaign)

    landing_page_map: Dict[str, LandingPage] = {}

    error_message = None
    try:
        login_header = user.login_customer_id or os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
        logger.info(
            "Starting analysis loop for %d customer accounts with %d total campaigns",
            len(campaigns_by_customer),
            len(campaign_records),
        )
        for customer_id, campaigns in campaigns_by_customer.items():
            ga_campaign_ids = [campaign.campaign_id for campaign in campaigns]
            logger.info(
                "Processing customer %s with %d campaigns: %s",
                customer_id,
                len(campaigns),
                ga_campaign_ids,
            )
            logger.info(
                "Fetching landing pages for customer %s campaigns %s",
                customer_id,
                ga_campaign_ids,
            )
            landing_start = time.perf_counter()
            landing_pages = list(
                service.fetch_landing_pages(
                    customer_id, ga_campaign_ids, login_customer_id=login_header
                )
            )
            landing_elapsed = time.perf_counter() - landing_start
            landing_duration_total += landing_elapsed
            total_landing_urls += len(landing_pages)
            logger.info(
                "Fetched %d landing pages for customer %s in %.2fs",
                len(landing_pages),
                customer_id,
                landing_elapsed,
            )

            # Process landing pages in parallel
            max_workers = min(10, len(landing_pages) or 1)
            logger.info(
                "Starting parallel processing of %d landing pages with %d workers",
                len(landing_pages),
                max_workers,
            )
            customer_landing_pages = []  # Track landing pages processed for this customer
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_lp = {
                    executor.submit(get_or_create_landing_page, session, lp): lp
                    for lp in landing_pages
                }
                logger.info("Submitted %d landing page processing tasks", len(future_to_lp))
                completed_count = 0
                for future in as_completed(future_to_lp):
                    lp = future_to_lp[future]
                    try:
                        logger.debug("Processing completed future for landing page %s", lp.url)
                        landing_page = future.result()
                        landing_page_map.setdefault(lp.campaign_id or "*", landing_page)
                        customer_landing_pages.append(landing_page)  # Track this customer's landing pages
                        completed_count += 1
                        logger.info(
                            "Successfully processed landing page %s (%d/%d complete)",
                            lp.url,
                            completed_count,
                            len(landing_pages),
                        )
                    except Exception as exc:
                        logger.error("Failed to process landing page %s: %s", lp.url, exc, exc_info=True)
                        completed_count += 1
            logger.info("Completed parallel processing of all %d landing pages", len(landing_pages))

            # Update the "*" fallback to use a landing page from this customer's campaigns
            # Prefer landing pages with summaries
            customer_pages_with_summary = [
                lp for lp in customer_landing_pages
                if lp.summary  # Only consider landing pages with summaries
            ]
            if customer_pages_with_summary:
                landing_page_map["*"] = customer_pages_with_summary[0]
                logger.info("Set fallback landing page with summary for customer %s", customer_id)
            elif customer_landing_pages:
                # If no landing pages have summaries, use any available one as fallback
                landing_page_map["*"] = customer_landing_pages[0]
                logger.warning("Set fallback landing page without summary for customer %s", customer_id)

            logger.info(
                "Fetching search terms for customer %s campaigns %s",
                customer_id,
                ga_campaign_ids,
            )
            search_timer = time.perf_counter()
            search_rows = list(
                service.fetch_search_terms(
                    customer_id,
                    ga_campaign_ids,
                    start,
                    end,
                    login_customer_id=login_header,
                )
            )
            raw_duration = time.perf_counter() - search_timer
            search_duration_total += raw_duration
            total_raw_rows += len(search_rows)
            logger.info(
                "Fetched %d raw search term rows for customer %s in %.2fs",
                len(search_rows),
                customer_id,
                raw_duration,
            )
            aggregated_rows = aggregate_search_term_rows(search_rows)
            if not aggregated_rows:
                logger.info("No search terms returned for campaigns %s", ga_campaign_ids)
                continue
            total_aggregated_rows += len(aggregated_rows)
            logger.info(
                "Aggregated search terms down to %d rows for campaigns %s",
                len(aggregated_rows),
                ga_campaign_ids,
            )
            prioritised_rows = prioritise_search_terms(list(aggregated_rows))
            skipped = len(aggregated_rows) - len(prioritised_rows)
            total_skipped_rows += max(skipped, 0)
            if skipped > 0:
                logger.info(
                    "Skipping %d low-signal search terms (max_terms=%d, min_impressions=%d)",
                    skipped,
                    DEFAULT_MAX_TERMS,
                    MIN_METRIC_THRESHOLD,
                )
            persist_stats = persist_search_terms(session, campaigns, prioritised_rows)
            persist_duration_total += float(persist_stats["duration"])
            total_prioritised_rows += int(persist_stats["processed"])
            llm_stats = run_llm_analysis(
                session,
                analysis_run,
                campaigns,
                prioritised_rows,
                landing_page_map,
                auto_select_irrelevant=auto_select_irrelevant,
            )
            llm_duration_total += float(llm_stats["duration"])
            total_llm_results += int(llm_stats["results"])
            total_llm_suggestions += int(llm_stats["suggestions"])
    except Exception as exc:
        logger.exception("Analysis run failed: %s", exc)
        analysis_run.status = "failed"
        analysis_run.error_message = str(exc)
        session.add(analysis_run)
        error_message = "Analysis failed. Please check logs for details."
    else:
        analysis_run.status = "completed"
        analysis_run.completed_at = dt.datetime.now(dt.timezone.utc)
        session.add(analysis_run)

    session.flush()
    suggestions = (
        session.execute(
            select(Suggestion).join(Suggestion.analysis).where(
                SearchTermAnalysis.analysis_run_id == analysis_run.id
            )
        )
        .scalars()
        .all()
    )
    analysis_total_seconds = time.perf_counter() - analysis_timer
    logger.info(
        "Analysis %s produced %d suggestions in %.2fs",
        analysis_run.id,
        len(suggestions),
        analysis_total_seconds,
    )

    if landing_duration_total or total_landing_urls:
        landing_detail = (
            f"{total_landing_urls} URL{'s' if total_landing_urls != 1 else ''} prepared"
            if total_landing_urls
            else "Landing page summaries reused from cache"
        )
        analysis_steps.append(
            {
                "title": "Landing pages ready",
                "detail": landing_detail,
                "duration": _format_duration(landing_duration_total),
                "icon": "🌐",
            }
        )
    if total_raw_rows:
        analysis_steps.append(
            {
                "title": "Search terms retrieved",
                "detail": f"{total_raw_rows} rows downloaded from Google Ads",
                "duration": _format_duration(search_duration_total),
                "icon": "📥",
            }
        )
    if total_aggregated_rows:
        prioritised_detail = f"{total_prioritised_rows} analysed"
        if total_skipped_rows:
            prioritised_detail += f" · {total_skipped_rows} skipped"
        analysis_steps.append(
            {
                "title": "Search terms prioritised",
                "detail": prioritised_detail,
                "duration": _format_duration(persist_duration_total),
                "icon": "🧮",
            }
        )
    if total_llm_results:
        analysis_steps.append(
            {
                "title": "LLM scoring complete",
                "detail": f"{total_llm_results} queries evaluated",
                "duration": _format_duration(llm_duration_total),
                "icon": "🤖",
            }
        )
    other_duration = max(
        analysis_total_seconds
        - (landing_duration_total + search_duration_total + persist_duration_total + llm_duration_total),
        0.0,
    )
    detail_text = (
        f"{len(suggestions)} suggestion{'s' if len(suggestions) != 1 else ''} ready to review"
        if suggestions
        else "No suggestions generated for this run"
    )
    analysis_steps.append(
        {
            "title": "Review & export",
            "detail": detail_text,
            "duration": _format_duration(other_duration),
            "icon": "✅",
        }
    )

    analysis_summary = {
        "total_time": _format_duration(analysis_total_seconds),
        "campaign_count": len(campaign_ids),
        "terms_scored": total_llm_results,
        "suggestion_count": len(suggestions),
    }

    context = {
        "request": request,
        "suggestions": suggestions,
        "analysis": analysis_run,
        "error_message": error_message,
        "user": user,
        "analysis_steps": analysis_steps,
        "analysis_summary": analysis_summary,
        "confidence_default": 0.8,
    }
    return TEMPLATES.TemplateResponse("analyze.html", context)


def get_or_create_landing_page(session: Session, row: LandingPageRow) -> LandingPage:
    url = row.url
    logger.info("Starting get_or_create_landing_page for %s", url)
    try:
        logger.debug("Querying database for existing landing page %s", url)
        existing = session.execute(
            select(LandingPage).where(LandingPage.url == url)
        ).scalar_one_or_none()
        now = dt.datetime.now(dt.timezone.utc)
        if existing:
            landing_page = existing
            logger.debug("Found existing landing page record for %s", url)
            # SQLite returns naive datetimes, so we need to make them timezone-aware for comparison
            summary_created_at = landing_page.summary_created_at
            if summary_created_at and summary_created_at.tzinfo is None:
                summary_created_at = summary_created_at.replace(tzinfo=dt.timezone.utc)
            needs_refresh = (
                not summary_created_at
                or (now - summary_created_at) > SUMMARY_TTL
            )
            if needs_refresh:
                logger.info("Landing page %s summary expired; refreshing", url)
            else:
                logger.info("Using cached landing page summary for %s", url)
        else:
            landing_page = LandingPage(url=url)
            session.add(landing_page)
            needs_refresh = True
            logger.info("Discovered new landing page %s", url)

        if needs_refresh:
            fetch_start = time.perf_counter()
            logger.info("Fetching page content for %s", url)
            page_data = fetch_page(url)
            if page_data:
                logger.debug("Successfully fetched page data for %s", url)
                final_url, html = page_data
                logger.debug("Extracting page content for %s", url)
                content = extract_page_content(final_url, html)
                landing_page.canonical_url = content.canonical_url
                landing_page.title = content.title
                landing_page.meta_description = content.meta_description
                landing_page.h1 = ", ".join(content.h1s)
                landing_page.h2 = ", ".join(content.h2s)
                landing_page.text_excerpt = content.visible_text_excerpt
                landing_page.last_fetched_at = now
                landing_page.content_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()
                logger.info("Generating LLM summary for %s", url)
                summary = generate_page_summary(content)
                landing_page.summary = summary
                landing_page.summary_created_at = now
                logger.info(
                    "Generated landing page summary for %s in %.2fs",
                    url,
                    time.perf_counter() - fetch_start,
                )
            else:
                logger.warning("Skipping LLM summary for %s due to fetch failure", url)
        logger.debug("Adding/updating landing page in session for %s", url)
        session.add(landing_page)
        logger.info("Successfully completed get_or_create_landing_page for %s", url)
        return landing_page
    except Exception as exc:
        logger.error(
            "Exception in get_or_create_landing_page for %s: %s",
            url,
            exc,
            exc_info=True,
        )
        raise


def persist_search_terms(
    session: Session,
    campaigns: Iterable[Campaign],
    rows: Iterable[SearchTermRow],
) -> dict[str, float | int]:
    start = time.perf_counter()
    rows = list(rows)
    campaigns_by_id = {campaign.campaign_id: campaign for campaign in campaigns}
    processed = 0
    for row in rows:
        campaign = campaigns_by_id.get(row.campaign_id)
        if not campaign:
            continue
        date_value = None
        if row.date:
            if isinstance(row.date, str):
                date_value = dt.datetime.strptime(row.date, "%Y-%m-%d").date()
            else:
                date_value = row.date
        existing = (
            session.execute(
                select(SearchTerm)
                .where(SearchTerm.campaign_id == campaign.id)
                .where(SearchTerm.term == row.search_term)
            )
            .scalars()
            .first()
        )
        if existing:
            search_term = existing
        else:
            search_term = SearchTerm(campaign=campaign, term=row.search_term)
            session.add(search_term)

        search_term.date = date_value
        search_term.match_type = row.match_type
        search_term.match_source = row.match_source
        search_term.impressions = row.impressions
        search_term.clicks = row.clicks
        search_term.cost_micros = row.cost_micros
        search_term.conversions = row.conversions
        processed += 1
    session.flush()
    duration = time.perf_counter() - start
    logger.info(
        "Persisted %d search term rows in %.2fs",
        processed,
        duration,
    )
    return {"processed": processed, "duration": duration}


def get_or_create_search_term(session: Session, campaign: Campaign, term_string: str) -> SearchTerm:
    """Get existing search term or create a new one."""
    existing = session.execute(
        select(SearchTerm)
        .where(SearchTerm.campaign_id == campaign.id)
        .where(SearchTerm.term == term_string)
        .order_by(SearchTerm.id.desc())
    ).scalars().first()

    if existing:
        return existing

    new_term = SearchTerm(campaign=campaign, term=term_string)
    session.add(new_term)
    return new_term


def run_llm_analysis(
    session: Session,
    analysis_run: AnalysisRun,
    campaigns: Iterable[Campaign],
    rows: Iterable[SearchTermRow],
    landing_pages: Dict[str, LandingPage],
    auto_select_irrelevant: bool,
) -> dict[str, float | int]:
    start = time.perf_counter()
    rows = list(rows)
    logger.info("Running LLM analysis over %d aggregated search terms", len(rows))
    campaigns_by_id = {campaign.campaign_id: campaign for campaign in campaigns}
    grouped_terms: Dict[str, List[SearchTermRow]] = defaultdict(list)
    for row in rows:
        grouped_terms[row.campaign_id].append(row)

    campaigns_processed = 0
    results_returned = 0
    suggestions_created = 0
    per_campaign_durations: list[float] = []

    for campaign_id, term_rows in grouped_terms.items():
        logger.info(
            "Analysing %d search terms for campaign %s",
            len(term_rows),
            campaign_id,
        )
        landing_page = landing_pages.get(campaign_id) or landing_pages.get("*")
        if not landing_page or not landing_page.summary:
            logger.info("No landing page summary for campaign %s; skipping LLM analysis", campaign_id)
            continue
        campaign = campaigns_by_id.get(campaign_id)
        if not campaign:
            continue

        # Check for cached analyses (within last 7 days with same landing page)
        # Use naive datetime for SQLite comparison (SQLite stores datetimes as strings without tz)
        cache_cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)
        cache_cutoff = cache_cutoff.replace(tzinfo=None)
        cached_analyses: Dict[str, SearchTermAnalysis] = {}
        terms_needing_analysis: List[SearchTermRow] = []

        # Batch fetch all existing terms for this campaign at once
        term_strings = [row.search_term for row in term_rows]
        existing_terms = session.execute(
            select(SearchTerm)
            .where(SearchTerm.campaign_id == campaign.id)
            .where(SearchTerm.term.in_(term_strings))
        ).scalars().all()

        # Create lookup dict
        terms_by_string = {term.term: term for term in existing_terms}

        # Fetch all recent analyses for these terms at once
        term_ids = [term.id for term in existing_terms]
        if term_ids:
            recent_analyses = session.execute(
                select(SearchTermAnalysis)
                .where(SearchTermAnalysis.search_term_id.in_(term_ids))
                .where(SearchTermAnalysis.landing_page_id == landing_page.id)
                .where(SearchTermAnalysis.created_at >= cache_cutoff)
                .order_by(SearchTermAnalysis.search_term_id, SearchTermAnalysis.created_at.desc())
            ).scalars().all()

            # Build cache lookup (keep most recent per term)
            analyses_by_term_id = {}
            for analysis in recent_analyses:
                if analysis.search_term_id not in analyses_by_term_id:
                    analyses_by_term_id[analysis.search_term_id] = analysis

            # Map back to term strings
            for row in term_rows:
                existing_term = terms_by_string.get(row.search_term)
                if existing_term and existing_term.id in analyses_by_term_id:
                    cached_analyses[row.search_term] = analyses_by_term_id[existing_term.id]
                    logger.debug("Using cached analysis for term '%s'", row.search_term)
                else:
                    terms_needing_analysis.append(row)
        else:
            # No existing terms, all need analysis
            terms_needing_analysis = list(term_rows)

        logger.info(
            "Found %d cached analyses, %d terms need fresh analysis",
            len(cached_analyses),
            len(terms_needing_analysis),
        )

        # Only call LLM for terms without recent cached analyses
        results = []
        if terms_needing_analysis:
            term_payloads = [
                {
                    "query": r.search_term,
                    "impressions": r.impressions,
                    "clicks": r.clicks,
                    "cost_micros": r.cost_micros,
                    "conversions": r.conversions,
                }
                for r in terms_needing_analysis
            ]
            llm_start = time.perf_counter()
            results = analyze_search_terms(
                page_summary=landing_page.summary,
                campaign_context=f"Campaign: {campaign.name}",
                terms=term_payloads,
            )
            llm_elapsed = time.perf_counter() - llm_start
            per_campaign_durations.append(llm_elapsed)
            logger.info(
                "LLM returned %d results for campaign %s in %.2fs",
                len(results),
                campaign_id,
                llm_elapsed,
            )

        results_returned += len(results) + len(cached_analyses)
        campaigns_processed += 1

        if not results and not cached_analyses:
            logger.warning("No relevancy suggestions generated for campaign %s", campaign_id)
            continue

        # Process cached analyses by creating new references to them in this run
        for term, cached_analysis in cached_analyses.items():
            # Create a new analysis record linked to this run (reusing cached LLM results)
            # Reuse the term from terms_by_string lookup to avoid duplicate query
            existing_term = terms_by_string.get(term)
            if not existing_term:
                existing_term = SearchTerm(campaign=campaign, term=term)
                session.add(existing_term)
                session.flush()  # Flush to get ID and avoid duplicates
                terms_by_string[term] = existing_term

            analysis = SearchTermAnalysis(
                search_term=existing_term,
                landing_page=landing_page,
                analysis_run=analysis_run,
                relevancy_label=cached_analysis.relevancy_label,
                reason=cached_analysis.reason,
                confidence=cached_analysis.confidence,
                suggest_negative=cached_analysis.suggest_negative,
                suggested_match_type=cached_analysis.suggested_match_type,
                raw_response=cached_analysis.raw_response,
            )
            session.add(analysis)
            auto_approve = (
                auto_select_irrelevant
                and cached_analysis.suggest_negative
                and (cached_analysis.confidence or 0) >= 0.8
            )
            match_type_rationale = ""
            if cached_analysis.raw_response and isinstance(cached_analysis.raw_response, dict):
                match_type_rationale = cached_analysis.raw_response.get("match_type_rationale", "")
            suggestion = Suggestion(
                analysis=analysis,
                scope="campaign",
                rationale=match_type_rationale,
                approved=auto_approve,
            )
            session.add(suggestion)
            suggestions_created += 1
            logger.debug(
                "Reused cached analysis for term '%s' (negative=%s, confidence=%.2f)",
                term,
                cached_analysis.suggest_negative,
                cached_analysis.confidence or 0.0,
            )

        # Process fresh LLM results
        for result in results:
            search_term = get_or_create_search_term(session, campaign, result.query)

            analysis = SearchTermAnalysis(
                search_term=search_term,
                landing_page=landing_page,
                analysis_run=analysis_run,
                relevancy_label=result.relevancy_label,
                reason=result.reason,
                confidence=result.confidence,
                suggest_negative=result.suggest_negative,
                suggested_match_type=result.suggested_match_type,
                raw_response=result.model_dump(),
            )
            session.add(analysis)
            auto_approve = (
                auto_select_irrelevant
                and result.suggest_negative
                and (result.confidence or 0) >= 0.8
            )
            suggestion = Suggestion(
                analysis=analysis,
                scope="campaign",
                rationale=result.match_type_rationale,
                approved=auto_approve,
            )
            session.add(suggestion)
            suggestions_created += 1
            logger.debug(
                "Recorded suggestion for term '%s' (negative=%s, confidence=%.2f)",
                result.query,
                result.suggest_negative,
                result.confidence,
            )
    total_duration = time.perf_counter() - start
    logger.info("Completed LLM analysis phase in %.2fs", total_duration)
    return {
        "duration": total_duration,
        "campaigns": campaigns_processed,
        "results": results_returned,
        "suggestions": suggestions_created,
        "average_campaign_duration": (sum(per_campaign_durations) / len(per_campaign_durations))
        if per_campaign_durations
        else 0.0,
    }


@app.post("/suggestions/{suggestion_id}/toggle")
async def toggle_suggestion(
    suggestion_id: int,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    suggestion = session.get(Suggestion, suggestion_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    suggestion.approved = not suggestion.approved
    session.add(suggestion)
    return {"approved": suggestion.approved}


@app.get("/export")
def export_csv(
    request: Request,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    suggestions = (
        session.execute(
            select(Suggestion)
            .join(Suggestion.analysis)
            .join(SearchTermAnalysis.search_term)
            .join(SearchTerm.campaign)
            .where(Suggestion.approved.is_(True))
        )
        .scalars()
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["scope", "term", "match_type", "rationale"])
    for suggestion in suggestions:
        analysis = suggestion.analysis
        writer.writerow(
            [
                suggestion.scope,
                analysis.search_term.term,
                analysis.suggested_match_type,
                suggestion.rationale or analysis.reason,
            ]
        )
    output.seek(0)
    filename = f"negative-keyword-suggestions-{dt.datetime.now(dt.timezone.utc).date()}.csv"
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/apply-negatives")
async def apply_negatives(
    request: Request,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    if os.getenv("FEATURE_APPLY_NEGATIVES", "false").lower() != "true":
        raise HTTPException(status_code=403, detail="Applying negatives is disabled")
    # Placeholder implementation – integrate with Google Ads API negative keyword services here.
    return {"status": "not_implemented"}
