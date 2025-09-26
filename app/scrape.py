"""Landing page fetching and parsing utilities."""
from __future__ import annotations

import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .schemas import PageContent

logger = logging.getLogger(__name__)

USER_AGENT = "SearchTermRelevancyBot/1.0"
REQUEST_TIMEOUT = 5


def fetch_page(url: str) -> Optional[tuple[str, str]]:
    """Fetch a landing page.

    The application owns the domains it analyses, so we intentionally ignore
    robots.txt directives that would otherwise block crawling.
    """

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None

    if response.status_code != 200:
        logger.info("Skipping %s due to status %s", url, response.status_code)
        return None

    final_url = str(response.url)
    return final_url, response.text


def extract_page_content(url: str, html: str) -> PageContent:
    """Parse HTML content and extract text snippets for LLM consumption."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    meta_desc = None
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()

    canonical = None
    canonical_tag = soup.find("link", rel="canonical")
    if canonical_tag and canonical_tag.get("href"):
        canonical = canonical_tag["href"].strip()

    h1s = [tag.get_text(strip=True) for tag in soup.find_all("h1")][:5]
    h2s = [tag.get_text(strip=True) for tag in soup.find_all("h2")][:8]

    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    text_chunks = list(soup.stripped_strings)
    visible_text = " ".join(text_chunks)
    excerpt = visible_text[:2000]

    return PageContent(
        url=url,
        title=title,
        meta_description=meta_desc,
        h1s=h1s,
        h2s=h2s,
        visible_text_excerpt=excerpt,
        canonical_url=canonical,
    )
