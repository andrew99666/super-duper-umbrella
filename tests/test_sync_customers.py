import os
from types import SimpleNamespace

import pytest

from sqlalchemy import select

from app.db import Base, SessionLocal, engine
from app.google_ads_client import CustomerClientSummary
from app.main import sync_customers
from app.models import GoogleAdsCustomer, User


class FakeGoogleAdsService:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def list_accessible_customers(self) -> list[dict[str, str]]:
        return [{"resource_name": "customers/1111111111"}]

    def get_customer(self, resource_name: str):
        if resource_name == "customers/1111111111":
            return SimpleNamespace(
                descriptive_name="Manager Account",
                currency_code="USD",
                time_zone="America/New_York",
                manager=True,
            )
        if resource_name == "customers/2222222222":
            return SimpleNamespace(
                descriptive_name="Child Account",
                currency_code="USD",
                time_zone="America/Los_Angeles",
                manager=False,
            )
        raise ValueError(f"Unexpected resource name: {resource_name}")

    def list_customer_clients(self, customer_id: str):
        self.requests.append(customer_id)
        if customer_id == "1111111111":
            return [
                CustomerClientSummary(
                    resource_name="customers/2222222222",
                    id="2222222222",
                    descriptive_name="Child Account",
                    currency_code="USD",
                    time_zone="America/Los_Angeles",
                    level=1,
                    manager=False,
                    status="ENABLED",
                    hidden=False,
                )
            ]
        return []


@pytest.fixture(autouse=True)
def clean_database():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def test_sync_customers_persists_manager_and_children(clean_database):
    os.environ.pop("GOOGLE_ADS_LOGIN_CUSTOMER_ID", None)
    service = FakeGoogleAdsService()
    session = SessionLocal()
    try:
        user = User(token_hash="hash", google_refresh_token=b"token")
        session.add(user)
        session.flush()

        sync_customers(session, user, service)
        session.commit()

        customers = (
            session.execute(select(GoogleAdsCustomer).order_by(GoogleAdsCustomer.customer_id))
            .scalars()
            .all()
        )
        assert [c.customer_id for c in customers] == ["1111111111", "2222222222"]
        names = {c.customer_id: c.descriptive_name for c in customers}
        assert names["2222222222"] == "Child Account"
        assert user.login_customer_id == "1111111111"
        # Ensure child lookup attempted for both manager and child accounts.
        assert service.requests.count("1111111111") == 1
        assert service.requests.count("2222222222") == 1
    finally:
        session.close()
