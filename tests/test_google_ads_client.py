import types

import pytest

from app.google_ads_client import (
    GoogleAdsOAuthError,
    GoogleAdsService,
    MethodNotImplemented,
    build_google_ads_client,
)


class _FakeCustomerService:
    def list_accessible_customers(self):
        raise MethodNotImplemented("test")


class _FakeClient:
    def get_service(self, name: str):
        assert name == "CustomerService"
        return _FakeCustomerService()


def test_list_accessible_customers_raises_version_error():
    service = GoogleAdsService(_FakeClient())
    with pytest.raises(GoogleAdsOAuthError) as excinfo:
        service.list_accessible_customers()
    assert "retired API version" in str(excinfo.value)


def test_build_google_ads_client_respects_api_version(monkeypatch):
    monkeypatch.setenv("GOOGLE_ADS_DEVELOPER_TOKEN", "dev")
    monkeypatch.setenv("GOOGLE_ADS_OAUTH_CLIENT_ID", "client")
    monkeypatch.setenv("GOOGLE_ADS_OAUTH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GOOGLE_ADS_API_VERSION", "v21")

    monkeypatch.setattr("app.google_ads_client.get_refresh_token", lambda user: "refresh")

    captured = {}

    def fake_load_from_dict(cls, config, version=None):
        captured["config"] = config
        captured["version"] = version
        return types.SimpleNamespace()

    monkeypatch.setattr(
        "app.google_ads_client.GoogleAdsClient.load_from_dict",
        classmethod(fake_load_from_dict),
    )

    user = types.SimpleNamespace(login_customer_id=None)
    client = build_google_ads_client(user)

    assert isinstance(client, types.SimpleNamespace)
    assert captured["config"]["refresh_token"] == "refresh"
    assert captured["version"] == "v21"
