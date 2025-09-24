"""FastAPI application entrypoint."""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import io
import logging
import os
from collections import defaultdict, deque
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
    build_google_ads_client,
    build_oauth_flow,
    campaigns_to_dict,
    store_user_credentials,
)
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="Search Term Relevancy Assistant")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("APP_SECRET_KEY", "change-me"))

TEMPLATES = Jinja2Templates(directory="app/templates")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
SUMMARY_TTL = dt.timedelta(days=3)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


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

    flow = build_oauth_flow(BASE_URL, state=state)
    code_verifier = request.session.get("code_verifier")
    if code_verifier:
        flow.code_verifier = code_verifier
    try:
        flow.fetch_token(code=code)
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
        client = build_google_ads_client(user)
        service = GoogleAdsService(client)
        sync_customers(session, user, service)
    except GoogleAdsOAuthError as exc:
        session.rollback()
        logger.exception("Failed to sync Google Ads accounts: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


    request.session["user_id"] = user.id
    request.session.pop("oauth_state", None)
    request.session.pop("code_verifier", None)
    return RedirectResponse(url="/campaigns", status_code=status.HTTP_302_FOUND)


def sync_customers(session: Session, user: User, service: GoogleAdsService) -> None:
    accessible = service.list_accessible_customers()
    default_login_id = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")

    # Ensure we prefer the configured login customer ID if provided.
    if not accessible and default_login_id and not user.login_customer_id:
        user.login_customer_id = default_login_id
        session.add(user)
        return

    metadata: dict[str, CustomerClientSummary] = {}
    queue: deque[str] = deque()
    for entry in accessible:
        resource_name = entry.get("resource_name")
        if resource_name:
            queue.append(resource_name)

    if not queue:
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
        descriptive_name = summary.descriptive_name if summary else None
        currency_code = summary.currency_code if summary else None
        time_zone = summary.time_zone if summary else None
        is_manager = summary.manager if summary else False

        try:
            customer_data = service.get_customer(resource_name)
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

        try:
            child_customers = service.list_customer_clients(customer_id)
        except Exception as exc:  # pragma: no cover - network error fallback
            logger.debug("Failed to list child customers for %s: %s", customer_id, exc)
            child_customers = []

        for child in child_customers:
            if child.id in seen_ids:
                continue
            metadata.setdefault(child.resource_name, child)
            queue.append(child.resource_name)
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


@app.get("/campaigns", response_class=HTMLResponse)
async def list_campaigns(
    request: Request,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> HTMLResponse:
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
    customer_id = requested_customer or user.login_customer_id
    if not customer_id and customers:
        customer_id = customers[0].customer_id

    if not customer_id:
        raise HTTPException(status_code=400, detail="No customer ID selected")

    valid_customer_ids = {customer.customer_id for customer in customers}
    if customer_id not in valid_customer_ids:
        raise HTTPException(status_code=404, detail="Unknown customer ID")

    if user.login_customer_id != customer_id:
        user.login_customer_id = customer_id
        session.add(user)

    campaigns = service.list_campaigns(customer_id)
    persist_campaigns(session, user, customer_id, campaigns)

    customers = (
        session.execute(
            select(GoogleAdsCustomer).where(GoogleAdsCustomer.user_id == user.id)
        )
        .scalars()
        .all()
    )

    context = {
        "request": request,
        "campaigns": campaigns_to_dict(campaigns),
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
    for summary in campaigns:
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
        for customer_id, campaigns in campaigns_by_customer.items():
            ga_campaign_ids = [campaign.campaign_id for campaign in campaigns]
            landing_pages = list(service.fetch_landing_pages(customer_id, ga_campaign_ids))
            for lp in landing_pages:
                landing_page = get_or_create_landing_page(session, lp)
                landing_page_map.setdefault(lp.campaign_id or "*", landing_page)

            if "*" not in landing_page_map and landing_page_map:
                landing_page_map["*"] = next(iter(landing_page_map.values()))

            search_rows = list(
                service.fetch_search_terms(customer_id, ga_campaign_ids, start, end)
            )
            persist_search_terms(session, campaigns, search_rows)
            run_llm_analysis(
                session,
                analysis_run,
                campaigns,
                search_rows,
                landing_page_map,
                auto_select_irrelevant=auto_select_irrelevant,
            )
    except Exception as exc:
        logger.exception("Analysis run failed: %s", exc)
        analysis_run.status = "failed"
        analysis_run.error_message = str(exc)
        session.add(analysis_run)
        error_message = "Analysis failed. Please check logs for details."
    else:
        analysis_run.status = "completed"
        analysis_run.completed_at = dt.datetime.utcnow()
        session.add(analysis_run)

    suggestions = (
        session.execute(
            select(Suggestion).join(Suggestion.analysis).where(
                SearchTermAnalysis.analysis_run_id == analysis_run.id
            )
        )
        .scalars()
        .all()
    )

    context = {
        "request": request,
        "suggestions": suggestions,
        "analysis": analysis_run,
        "error_message": error_message,
        "user": user,
    }
    return TEMPLATES.TemplateResponse("analyze.html", context)


def get_or_create_landing_page(session: Session, row: LandingPageRow) -> LandingPage:
    url = row.url
    existing = session.execute(
        select(LandingPage).where(LandingPage.url == url)
    ).scalar_one_or_none()
    now = dt.datetime.utcnow()
    if existing:
        landing_page = existing
        needs_refresh = (
            not landing_page.summary_created_at
            or (now - landing_page.summary_created_at) > SUMMARY_TTL
        )
    else:
        landing_page = LandingPage(url=url)
        session.add(landing_page)
        needs_refresh = True

    if needs_refresh:
        page_data = fetch_page(url)
        if page_data:
            final_url, html = page_data
            content = extract_page_content(final_url, html)
            landing_page.canonical_url = content.canonical_url
            landing_page.title = content.title
            landing_page.meta_description = content.meta_description
            landing_page.h1 = ", ".join(content.h1s)
            landing_page.h2 = ", ".join(content.h2s)
            landing_page.text_excerpt = content.visible_text_excerpt
            landing_page.last_fetched_at = now
            landing_page.content_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()
            summary = generate_page_summary(content)
            landing_page.summary = summary
            landing_page.summary_created_at = now
        else:
            logger.info("Skipping LLM summary for %s due to fetch failure", url)
    session.add(landing_page)
    return landing_page


def persist_search_terms(
    session: Session,
    campaigns: Iterable[Campaign],
    rows: Iterable[SearchTermRow],
) -> None:
    campaigns_by_id = {campaign.campaign_id: campaign for campaign in campaigns}
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
        search_term = SearchTerm(
            campaign=campaign,
            term=row.search_term,
            date=date_value,
            match_type=row.match_type,
            match_source=row.match_source,
            impressions=row.impressions,
            clicks=row.clicks,
            cost_micros=row.cost_micros,
            conversions=row.conversions,
        )
        session.add(search_term)
    session.flush()


def run_llm_analysis(
    session: Session,
    analysis_run: AnalysisRun,
    campaigns: Iterable[Campaign],
    rows: Iterable[SearchTermRow],
    landing_pages: Dict[str, LandingPage],
    auto_select_irrelevant: bool,
) -> None:
    campaigns_by_id = {campaign.campaign_id: campaign for campaign in campaigns}
    grouped_terms: Dict[str, List[SearchTermRow]] = defaultdict(list)
    for row in rows:
        grouped_terms[row.campaign_id].append(row)

    for campaign_id, term_rows in grouped_terms.items():
        landing_page = landing_pages.get(campaign_id) or landing_pages.get("*")
        if not landing_page or not landing_page.summary:
            logger.info("No landing page summary for campaign %s; skipping LLM analysis", campaign_id)
            continue
        campaign = campaigns_by_id.get(campaign_id)
        if not campaign:
            continue
        term_payloads = [
            {
                "query": r.search_term,
                "impressions": r.impressions,
                "clicks": r.clicks,
                "cost_micros": r.cost_micros,
                "conversions": r.conversions,
            }
            for r in term_rows
        ]
        results = analyze_search_terms(
            page_summary=landing_page.summary,
            campaign_context=f"Campaign: {campaign.name}",
            terms=term_payloads,
        )
        for result in results:
            search_term = SearchTerm(
                campaign=campaign,
                term=result.query,
            )
            # Attempt to reuse existing search term entry
            existing_term = session.execute(
                select(SearchTerm)
                .where(SearchTerm.campaign_id == campaign.id)
                .where(SearchTerm.term == result.query)
                .order_by(SearchTerm.id.desc())
            ).scalars().first()
            if existing_term:
                search_term = existing_term
            else:
                session.add(search_term)
                session.flush()

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
    filename = f"negative-keyword-suggestions-{dt.datetime.utcnow().date()}.csv"
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
