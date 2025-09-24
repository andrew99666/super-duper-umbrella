"""Google Ads API OAuth and querying utilities."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Iterator, Sequence
from urllib.parse import urljoin

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from google.api_core.exceptions import MethodNotImplemented
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .models import User
from .security import get_encryption_manager

logger = logging.getLogger(__name__)

GOOGLE_ADS_SCOPE = "https://www.googleapis.com/auth/adwords"


@dataclass
class CampaignSummary:
    campaign_id: str
    name: str
    customer_id: str
    status: str | None = None
    advertising_channel_type: str | None = None


@dataclass
class CustomerClientSummary:
    """Lightweight representation of a child customer relationship."""

    resource_name: str
    id: str
    descriptive_name: str | None
    currency_code: str | None
    time_zone: str | None
    level: int | None
    manager: bool
    status: str | None
    hidden: bool | None = None


@dataclass
class SearchTermRow:
    campaign_id: str
    campaign_name: str
    search_term: str
    date: date | None
    match_type: str | None
    match_source: str | None
    impressions: int | None
    clicks: int | None
    cost_micros: int | None
    conversions: float | None


@dataclass
class LandingPageRow:
    campaign_id: str | None
    url: str
    canonical_url: str | None


class GoogleAdsOAuthError(RuntimeError):
    """Raised when OAuth flow cannot be completed."""


def _raise_api_version_error(exc: Exception) -> None:
    message = (
        "Google Ads API request failed because the client library is targeting a retired API version. "
        "Upgrade the installed `google-ads` dependency and rebuild the container, or set "
        "GOOGLE_ADS_API_VERSION to a supported release."
    )
    raise GoogleAdsOAuthError(message) from exc


def build_oauth_flow(base_url: str, state: str | None = None) -> Flow:
    """Create an OAuth flow for the Google Ads API."""
    client_id = os.getenv("GOOGLE_ADS_OAUTH_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_ADS_OAUTH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise GoogleAdsOAuthError("Google Ads OAuth client credentials are not configured.")

    config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    redirect_uri = urljoin(base_url.rstrip("/") + "/", "oauth/callback")
    flow = Flow.from_client_config(config, scopes=[GOOGLE_ADS_SCOPE], state=state)
    flow.redirect_uri = redirect_uri
    return flow


def store_user_credentials(user: User, credentials: Credentials) -> None:
    """Persist OAuth credentials on the user record."""
    encryption = get_encryption_manager()
    if not credentials.refresh_token:
        raise GoogleAdsOAuthError("No refresh token returned from Google OAuth flow.")

    user.google_refresh_token = encryption.encrypt(credentials.refresh_token)
    if credentials.token:
        user.google_access_token = encryption.encrypt(credentials.token)
    user.token_expiry = credentials.expiry


def get_refresh_token(user: User) -> str:
    encryption = get_encryption_manager()
    return encryption.decrypt(user.google_refresh_token)


def build_google_ads_client(user: User) -> GoogleAdsClient:
    developer_token = os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN")
    client_id = os.getenv("GOOGLE_ADS_OAUTH_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_ADS_OAUTH_CLIENT_SECRET")
    if not all([developer_token, client_id, client_secret]):
        raise GoogleAdsOAuthError("Google Ads API credentials are not fully configured.")

    config = {
        "developer_token": developer_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": get_refresh_token(user),
        "use_proto_plus": True,
    }
    login_customer_id = user.login_customer_id or os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
    if login_customer_id:
        config["login_customer_id"] = login_customer_id

    api_version = os.getenv("GOOGLE_ADS_API_VERSION", "").strip() or None
    try:
        if api_version:
            client = GoogleAdsClient.load_from_dict(config, version=api_version)
        else:
            client = GoogleAdsClient.load_from_dict(config)
    except MethodNotImplemented as exc:  # pragma: no cover - dependent on client internals
        _raise_api_version_error(exc)
    return client


def _google_ads_retry() -> callable:
    return retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        retry=retry_if_exception_type((GoogleAdsException, RefreshError)),
    )


def build_campaign_search_term_query(
    campaign_ids: Sequence[str], start_date: date, end_date: date
) -> str:
    """Build GAQL query for campaign_search_term_view."""
    id_list = ",".join(str(cid) for cid in campaign_ids)
    return f"""
    SELECT
      campaign.id,
      campaign.name,
      campaign_search_term_view.search_term,
      segments.date,
      segments.search_term_match_type,
      segments.search_term_match_source,
      metrics.impressions,
      metrics.clicks,
      metrics.cost_micros,
      metrics.conversions
    FROM campaign_search_term_view
    WHERE campaign.id IN ({id_list})
      AND segments.date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY metrics.impressions DESC
    """.strip()


def build_search_term_fallback_query(
    campaign_ids: Sequence[str], start_date: date, end_date: date
) -> str:
    id_list = ",".join(str(cid) for cid in campaign_ids)
    return f"""
    SELECT
      campaign.id,
      campaign.name,
      search_term_view.search_term,
      segments.date,
      segments.search_term_match_type,
      segments.search_term_match_source,
      metrics.impressions,
      metrics.clicks,
      metrics.cost_micros,
      metrics.conversions
    FROM search_term_view
    WHERE campaign.id IN ({id_list})
      AND segments.date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY metrics.impressions DESC
    """.strip()


def build_landing_page_query(campaign_ids: Sequence[str]) -> str:
    id_list = ",".join(str(cid) for cid in campaign_ids)
    return f"""
    SELECT
      campaign.id,
      ad_group_ad.ad.id,
      ad_group_ad.ad.final_urls,
      ad_group_ad.ad.final_mobile_urls
    FROM ad_group_ad
    WHERE campaign.id IN ({id_list})
    """.strip()


def build_landing_page_view_query(campaign_ids: Sequence[str]) -> str:
    id_list = ",".join(str(cid) for cid in campaign_ids)
    return f"""
    SELECT
      campaign.id,
      landing_page_view.unexpanded_final_url,
      landing_page_view.display_name
    FROM landing_page_view
    WHERE campaign.id IN ({id_list})
    """.strip()


class GoogleAdsService:
    """Wrapper around the Google Ads API client."""

    def __init__(self, client: GoogleAdsClient) -> None:
        self.client = client

    @_google_ads_retry()
    def list_accessible_customers(self) -> list[dict[str, str | None]]:
        customer_service = self.client.get_service("CustomerService")
        try:
            response = customer_service.list_accessible_customers()
        except MethodNotImplemented as exc:  # pragma: no cover - network version error
            _raise_api_version_error(exc)
        resource_names = response.resource_names
        result = []
        for resource_name in resource_names:
            result.append({"resource_name": resource_name})
        return result

    @_google_ads_retry()
    def get_customer(self, resource_name: str):
        customer_service = self.client.get_service("CustomerService")
        try:
            return customer_service.get_customer(resource_name=resource_name)
        except MethodNotImplemented as exc:  # pragma: no cover - network version error
            _raise_api_version_error(exc)

    @_google_ads_retry()
    def list_campaigns(self, customer_id: str) -> list[CampaignSummary]:
        ga_service = self.client.get_service("GoogleAdsService")
        query = (
            "SELECT campaign.id, campaign.name, campaign.status, "
            "campaign.advertising_channel_type FROM campaign WHERE campaign.status != 'REMOVED'"
        )
        campaigns: list[CampaignSummary] = []
        try:
            response = ga_service.search(customer_id=customer_id, query=query)
            for row in response:
                campaign = row.campaign
                campaigns.append(
                    CampaignSummary(
                        campaign_id=str(campaign.id),
                        name=str(campaign.name),
                        customer_id=customer_id,
                        status=str(campaign.status),
                        advertising_channel_type=str(campaign.advertising_channel_type),
                    )
                )
        except MethodNotImplemented as exc:  # pragma: no cover - network version error
            _raise_api_version_error(exc)
        except GoogleAdsException as exc:  # pragma: no cover - thin wrapper
            logger.exception("Failed to list campaigns: %s", exc)
            raise
        return campaigns

    @_google_ads_retry()
    def list_customer_clients(self, customer_id: str) -> list[CustomerClientSummary]:
        """Return the customer clients under the specified manager customer."""

        ga_service = self.client.get_service("GoogleAdsService")
        query = (
            "SELECT "
            "customer_client.client_customer, "
            "customer_client.descriptive_name, "
            "customer_client.currency_code, "
            "customer_client.time_zone, "
            "customer_client.level, "
            "customer_client.manager, "
            "customer_client.status, "
            "customer_client.hidden "
            "FROM customer_client "
            "WHERE customer_client.hidden = FALSE"
        )

        clients: list[CustomerClientSummary] = []
        try:
            response = ga_service.search(customer_id=customer_id, query=query)
        except MethodNotImplemented as exc:  # pragma: no cover - network version error
            _raise_api_version_error(exc)
        except GoogleAdsException as exc:  # pragma: no cover - thin wrapper
            logger.exception("Failed to list customer clients: %s", exc)
            raise

        for row in response:
            client = row.customer_client
            hidden = bool(getattr(client, "hidden", False))
            level_value = getattr(client, "level", 0)
            try:
                level = int(level_value) if level_value is not None else 0
            except (TypeError, ValueError):  # pragma: no cover - defensive
                level = 0
            if hidden or level == 0:
                # Skip the manager itself (level 0) and hidden accounts.
                continue

            resource_name = str(getattr(client, "client_customer", ""))
            if not resource_name:
                continue

            clients.append(
                CustomerClientSummary(
                    resource_name=resource_name,
                    id=resource_name.split("/")[-1],
                    descriptive_name=getattr(client, "descriptive_name", None),
                    currency_code=getattr(client, "currency_code", None),
                    time_zone=getattr(client, "time_zone", None),
                    level=level,
                    manager=bool(getattr(client, "manager", False)),
                    status=str(getattr(client, "status", None)) if getattr(client, "status", None) else None,
                    hidden=hidden,
                )
            )

        return clients

    @_google_ads_retry()
    def fetch_search_terms(
        self,
        customer_id: str,
        campaign_ids: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> Iterator[SearchTermRow]:
        ga_service = self.client.get_service("GoogleAdsService")
        query = build_campaign_search_term_query(campaign_ids, start_date, end_date)
        try:
            results = ga_service.search_stream(customer_id=customer_id, query=query)
            for batch in results:
                for row in batch.results:
                    yield SearchTermRow(
                        campaign_id=str(row.campaign.id),
                        campaign_name=str(row.campaign.name),
                        search_term=str(row.campaign_search_term_view.search_term),
                        date=row.segments.date,
                        match_type=str(row.segments.search_term_match_type),
                        match_source=str(row.segments.search_term_match_source),
                        impressions=row.metrics.impressions,
                        clicks=row.metrics.clicks,
                        cost_micros=row.metrics.cost_micros,
                        conversions=row.metrics.conversions,
                    )
        except MethodNotImplemented as exc:  # pragma: no cover - network version error
            _raise_api_version_error(exc)
        except GoogleAdsException as exc:
            if "campaign_search_term_view" in str(exc):
                logger.warning("Falling back to search_term_view due to API error: %s", exc)
                yield from self._fetch_search_terms_fallback(
                    customer_id, campaign_ids, start_date, end_date
                )
            else:
                logger.exception("Failed to fetch search terms: %s", exc)
                raise

    def _fetch_search_terms_fallback(
        self,
        customer_id: str,
        campaign_ids: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> Iterator[SearchTermRow]:
        ga_service = self.client.get_service("GoogleAdsService")
        query = build_search_term_fallback_query(campaign_ids, start_date, end_date)
        try:
            results = ga_service.search_stream(customer_id=customer_id, query=query)
        except MethodNotImplemented as exc:  # pragma: no cover - network version error
            _raise_api_version_error(exc)
        for batch in results:
            for row in batch.results:
                yield SearchTermRow(
                    campaign_id=str(row.campaign.id),
                    campaign_name=str(row.campaign.name),
                    search_term=str(row.search_term_view.search_term),
                    date=row.segments.date,
                    match_type=str(row.segments.search_term_match_type),
                    match_source=str(row.segments.search_term_match_source),
                    impressions=row.metrics.impressions,
                    clicks=row.metrics.clicks,
                    cost_micros=row.metrics.cost_micros,
                    conversions=row.metrics.conversions,
                )

    @_google_ads_retry()
    def fetch_landing_pages(
        self, customer_id: str, campaign_ids: Sequence[str]
    ) -> Iterator[LandingPageRow]:
        ga_service = self.client.get_service("GoogleAdsService")
        ad_query = build_landing_page_query(campaign_ids)
        urls_by_campaign: dict[str, set[str]] = {}
        try:
            response = ga_service.search_stream(customer_id=customer_id, query=ad_query)
        except MethodNotImplemented as exc:  # pragma: no cover - network version error
            _raise_api_version_error(exc)
        for batch in response:
            for row in batch.results:
                campaign_id = str(row.campaign.id)
                urls_by_campaign.setdefault(campaign_id, set())
                ad = row.ad_group_ad.ad
                for url in getattr(ad, "final_urls", []) or []:
                    urls_by_campaign[campaign_id].add(str(url))
                for url in getattr(ad, "final_mobile_urls", []) or []:
                    urls_by_campaign[campaign_id].add(str(url))

        lp_query = build_landing_page_view_query(campaign_ids)
        try:
            response = ga_service.search_stream(customer_id=customer_id, query=lp_query)
        except MethodNotImplemented as exc:  # pragma: no cover - network version error
            _raise_api_version_error(exc)
        for batch in response:
            for row in batch.results:
                campaign_id = str(row.campaign.id) if hasattr(row, "campaign") else None
                url = str(row.landing_page_view.unexpanded_final_url)
                if not url:
                    continue
                if campaign_id:
                    urls_by_campaign.setdefault(campaign_id, set()).add(url)
                else:
                    for cid in urls_by_campaign.keys():
                        urls_by_campaign[cid].add(url)

        for campaign_id, urls in urls_by_campaign.items():
            for url in urls:
                yield LandingPageRow(campaign_id=campaign_id, url=url, canonical_url=None)


def campaigns_to_dict(campaigns: Iterable[CampaignSummary]) -> list[dict[str, str | None]]:
    return [
        {
            "campaign_id": campaign.campaign_id,
            "name": campaign.name,
            "customer_id": campaign.customer_id,
            "status": campaign.status,
            "advertising_channel_type": campaign.advertising_channel_type,
        }
        for campaign in campaigns
    ]
