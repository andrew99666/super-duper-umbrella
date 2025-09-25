"""Shared data structures used across modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class PageContent:
    url: str
    title: str | None
    meta_description: str | None
    h1s: List[str]
    h2s: List[str]
    visible_text_excerpt: str
    canonical_url: str | None = None
