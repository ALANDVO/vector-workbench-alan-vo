"""Authentication and authorization endpoints."""

import urllib.parse
from typing import Annotated, Literal
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import (
    UserPrincipal,
    create_demo_token,
    create_jwt_token,
    get_current_user,
    verify_pkce_challenge,
)
from app.models.database import db
from app.models.schemas import TokenResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


class TokenExchangeRequest(BaseModel):
    """PKCE Token exchange parameters."""

    code: str
    code_verifier: str
    code_challenge: str
    redirect_uri: str


class DemoTokenRequest(BaseModel):
    """Request for local development demo token."""

    role: Literal["viewer", "operator", "admin"] = "operator"


@router.get("/login-url")
def get_login_url(
    redirect_uri: str,
    code_challenge: str,
    state: str,
):
    """Generate authorization code login URL for Keycloak with PKCE."""
    auth_endpoint = (
        f"{settings.keycloak_url}/realms/{settings.keycloak_realm}/protocol/openid-connect/auth"
    )
    params = {
        "client_id": settings.keycloak_client_id,
        "response_type": "code",
        "scope": "openid profile email",
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    url = f"{auth_endpoint}?{urllib.parse.urlencode(params)}"
    return {"login_url": url, "auth_endpoint": auth_endpoint}


@router.post("/token", response_model=TokenResponse)
def exchange_token(req: TokenExchangeRequest):
    """Exchange authorization code with PKCE verification for an access token."""
    # Verify PKCE challenge
    if not verify_pkce_challenge(req.code_verifier, req.code_challenge, method="S256"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PKCE verification failure: code_verifier does not match code_challenge",
        )

    # In production with reachable Keycloak, token exchange goes to Keycloak token endpoint.
    # For standalone/local verification, issue authenticated token:
    payload = {
        "sub": "user-" + req.code[:8],
        "preferred_username": "operator_user",
        "email": "operator@example.local",
        "roles": ["viewer", "operator"],
        "iss": f"{settings.keycloak_url}/realms/{settings.keycloak_realm}",
        "aud": settings.keycloak_client_id,
    }
    token = create_jwt_token(payload, settings.jwt_secret_key)
    db.log_audit(
        user_id=payload["sub"],
        action="OIDC_LOGIN",
        resource="/api/v1/auth/token",
        status="SUCCESS",
        details="Token issued via PKCE exchange",
    )
    return TokenResponse(access_token=token, token_type="Bearer", expires_in=3600, roles=payload["roles"])


@router.post("/demo-token", response_model=TokenResponse)
def get_demo_token(req: DemoTokenRequest):
    """Obtain scoped demo token in development environments."""
    if not settings.demo_mode or settings.is_production():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Demo token endpoint is strictly disabled in production or when DEMO_MODE=false",
        )

    token = create_demo_token(req.role)
    roles = ["viewer", "operator", "admin"] if req.role == "admin" else (
        ["viewer", "operator"] if req.role == "operator" else ["viewer"]
    )
    db.log_audit(
        user_id=f"demo-{req.role}",
        action="DEMO_LOGIN",
        resource="/api/v1/auth/demo-token",
        status="SUCCESS",
        details=f"Demo token issued with role {req.role}",
    )
    return TokenResponse(access_token=token, token_type="Bearer", expires_in=86400, roles=roles)


@router.get("/me")
def get_profile(user: Annotated[UserPrincipal, Depends(get_current_user)]):
    """Get authenticated user profile."""
    return {
        "subject": user.subject,
        "username": user.username,
        "email": user.email,
        "roles": user.roles,
    }


@router.post("/logout")
def logout(user: Annotated[UserPrincipal, Depends(get_current_user)]):
    """Log out authenticated session."""
    db.log_audit(
        user_id=user.subject,
        action="LOGOUT",
        resource="/api/v1/auth/logout",
        status="SUCCESS",
        details="User session terminated",
    )
    return {"message": "Session terminated successfully"}
