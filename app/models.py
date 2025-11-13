"""SQLAlchemy ORM models for the application."""
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class TimestampMixin:
    """Mixin providing created/updated timestamps."""

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    google_refresh_token: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    google_access_token: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    token_expiry: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    login_customer_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    customers: Mapped[list[GoogleAdsCustomer]] = relationship("GoogleAdsCustomer", back_populates="user")
    analysis_runs: Mapped[list[AnalysisRun]] = relationship("AnalysisRun", back_populates="user")


class GoogleAdsCustomer(TimestampMixin, Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    resource_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    customer_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    descriptive_name: Mapped[Optional[str]] = mapped_column(String(255))
    currency_code: Mapped[Optional[str]] = mapped_column(String(10))
    time_zone: Mapped[Optional[str]] = mapped_column(String(100))
    manager_customer_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    user: Mapped[User] = relationship("User", back_populates="customers")
    campaigns: Mapped[list[Campaign]] = relationship("Campaign", back_populates="customer")


class Campaign(TimestampMixin, Base):
    __tablename__ = "campaigns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    campaign_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[Optional[str]] = mapped_column(String(50))
    advertising_channel_type: Mapped[Optional[str]] = mapped_column(String(50))

    customer: Mapped[GoogleAdsCustomer] = relationship("GoogleAdsCustomer", back_populates="campaigns")
    ads: Mapped[list[Ad]] = relationship("Ad", back_populates="campaign", cascade="all, delete-orphan")
    search_terms: Mapped[list[SearchTerm]] = relationship(
        "SearchTerm", back_populates="campaign", cascade="all, delete-orphan"
    )


class Ad(TimestampMixin, Base):
    __tablename__ = "ads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    ad_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    final_url: Mapped[Optional[str]] = mapped_column(String(2048))
    final_mobile_url: Mapped[Optional[str]] = mapped_column(String(2048))

    campaign: Mapped[Campaign] = relationship("Campaign", back_populates="ads")


class LandingPage(TimestampMixin, Base):
    __tablename__ = "landing_pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    canonical_url: Mapped[Optional[str]] = mapped_column(String(2048))
    title: Mapped[Optional[str]] = mapped_column(String(512))
    meta_description: Mapped[Optional[str]] = mapped_column(Text)
    h1: Mapped[Optional[str]] = mapped_column(Text)
    h2: Mapped[Optional[str]] = mapped_column(Text)
    text_excerpt: Mapped[Optional[str]] = mapped_column(Text)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64))
    last_fetched_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    summary: Mapped[Optional[str]] = mapped_column(Text)
    summary_created_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))

    analyses: Mapped[list[SearchTermAnalysis]] = relationship("SearchTermAnalysis", back_populates="landing_page")


class SearchTerm(TimestampMixin, Base):
    __tablename__ = "search_terms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    campaign_id: Mapped[int] = mapped_column(ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False)
    term: Mapped[str] = mapped_column(String(512), index=True, nullable=False)
    date: Mapped[Optional[dt.date]] = mapped_column(Date)
    match_type: Mapped[Optional[str]] = mapped_column(String(50))
    match_source: Mapped[Optional[str]] = mapped_column(String(50))
    impressions: Mapped[Optional[int]] = mapped_column(Integer)
    clicks: Mapped[Optional[int]] = mapped_column(Integer)
    cost_micros: Mapped[Optional[int]] = mapped_column(Integer)
    conversions: Mapped[Optional[float]] = mapped_column(Float)

    campaign: Mapped[Campaign] = relationship("Campaign", back_populates="search_terms")
    analyses: Mapped[list[SearchTermAnalysis]] = relationship(
        "SearchTermAnalysis", back_populates="search_term", cascade="all, delete-orphan"
    )


class AnalysisRun(TimestampMixin, Base):
    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now(), nullable=False)
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="running", nullable=False)
    campaign_ids: Mapped[str] = mapped_column(Text)
    start_date: Mapped[Optional[dt.date]] = mapped_column(Date)
    end_date: Mapped[Optional[dt.date]] = mapped_column(Date)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    user: Mapped[User] = relationship("User", back_populates="analysis_runs")
    search_term_analyses: Mapped[list[SearchTermAnalysis]] = relationship(
        "SearchTermAnalysis", back_populates="analysis_run", cascade="all, delete-orphan"
    )


class SearchTermAnalysis(TimestampMixin, Base):
    __tablename__ = "search_term_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    search_term_id: Mapped[int] = mapped_column(
        ForeignKey("search_terms.id", ondelete="CASCADE"), nullable=False
    )
    landing_page_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("landing_pages.id", ondelete="SET NULL"), nullable=True
    )
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False
    )
    relevancy_label: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    suggest_negative: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    suggested_match_type: Mapped[Optional[str]] = mapped_column(String(16))
    raw_response: Mapped[Optional[dict]] = mapped_column(JSON)

    search_term: Mapped[SearchTerm] = relationship("SearchTerm", back_populates="analyses")
    landing_page: Mapped[LandingPage] = relationship("LandingPage", back_populates="analyses")
    analysis_run: Mapped[AnalysisRun] = relationship("AnalysisRun", back_populates="search_term_analyses")
    suggestion: Mapped[Optional[Suggestion]] = relationship(
        "Suggestion", back_populates="analysis", uselist=False, cascade="all, delete-orphan"
    )


class Suggestion(TimestampMixin, Base):
    __tablename__ = "suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("search_term_analyses.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    scope: Mapped[str] = mapped_column(String(32), default="campaign", nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text)
    approved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    analysis: Mapped[SearchTermAnalysis] = relationship("SearchTermAnalysis", back_populates="suggestion")
