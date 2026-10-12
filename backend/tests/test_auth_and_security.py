"""Tests covering authentication, authorization boundaries, PKCE, and redaction."""

import hashlib
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.security import (
    create_demo_token,
    redact_sensitive,
    urlsafe_b64encode_str,
    verify_pkce_challenge,
)
from app.main import app

client = TestClient(app)


def test_unauthenticated_request_rejected_401():
    """Verify data and model endpoints reject requests lacking Bearer credentials."""
    resp = client.get("/api/v1/models")
    assert resp.status_code == 401
    assert "detail" in resp.json()


def test_role_authorization_denial_403_for_viewer():
    """Verify viewer cannot execute mutation or operator-restricted inference."""
    viewer_token = create_demo_token(role="viewer")
    headers = {"Authorization": f"Bearer {viewer_token}"}

    # Viewing models is permitted for viewer
    get_resp = client.get("/api/v1/models", headers=headers)
    assert get_resp.status_code == 200

    # Running completion requires 'operator' role -> 403 Forbidden
    post_resp = client.post(
        "/api/v1/completions",
        json={"model": "bitnet-b1.58-2b-i2s", "prompt": "Hello", "max_tokens": 10},
        headers=headers,
    )
    assert post_resp.status_code == 403
    assert "requires 'operator' role" in post_resp.json()["detail"]


def test_demo_token_issuance_and_access():
    """Verify obtaining a demo token and querying authenticated me endpoint."""
    resp = client.post("/api/v1/auth/demo-token", json={"role": "operator"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    assert resp.json()["token_type"] == "Bearer"

    me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert "operator" in me_resp.json()["roles"]


def test_demo_mode_refusal_in_production(monkeypatch):
    """Verify that startup and demo token issuance REFUSE demo mode when in production."""
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "demo_mode", True)

    with pytest.raises(RuntimeError, match="demo_mode cannot be active in production"):
        settings.ensure_safe_mode()

    # The API endpoint must reject demo tokens in production
    resp = client.post("/api/v1/auth/demo-token", json={"role": "operator"})
    assert resp.status_code == 403


def test_pkce_challenge_verification_and_rejection():
    """Verify S256 PKCE code challenge and rejection of mismatched verifier."""
    verifier = "test-pkce-verifier-string-1234567890-abcdef"
    hashed = hashlib.sha256(verifier.encode("ascii")).digest()
    valid_challenge = urlsafe_b64encode_str(hashed)

    assert verify_pkce_challenge(verifier, valid_challenge, method="S256") is True
    assert verify_pkce_challenge("wrong-verifier", valid_challenge, method="S256") is False

    # Real HTTP boundary test for token exchange
    fail_resp = client.post(
        "/api/v1/auth/token",
        json={
            "code": "test_auth_code_xyz",
            "code_verifier": "wrong-verifier",
            "code_challenge": valid_challenge,
            "redirect_uri": "http://127.0.0.1:3000/callback",
        },
    )
    assert fail_resp.status_code == 400
    assert "PKCE verification failure" in fail_resp.json()["detail"]

    pass_resp = client.post(
        "/api/v1/auth/token",
        json={
            "code": "test_auth_code_xyz",
            "code_verifier": verifier,
            "code_challenge": valid_challenge,
            "redirect_uri": "http://127.0.0.1:3000/callback",
        },
    )
    assert pass_resp.status_code == 200
    assert "access_token" in pass_resp.json()


def test_sensitive_redaction_runtime():
    """Verify credential and token redaction without hardcoding secret literals."""
    synthetic_key = "sk-" + ("9" * 32)
    synthetic_token = "Bearer " + ("a" * 40)
    raw_error = f"Failed to connect using key {synthetic_key} and token {synthetic_token}"

    sanitized = redact_sensitive(raw_error)
    assert synthetic_key not in sanitized
    assert synthetic_token not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
    assert "Bearer [REDACTED]" in sanitized
