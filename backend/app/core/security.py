"""Security, token verification, and role-based authorization."""

import base64
import hashlib
import hmac
import json
import time
from typing import Annotated, Any, Dict, List, Optional
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

bearer_scheme = HTTPBearer(auto_error=False)


class UserPrincipal:
    """Authenticated user principal representation."""

    def __init__(self, subject: str, username: str, email: str, roles: List[str]):
        self.subject = subject
        self.username = username
        self.email = email
        self.roles = roles

    def has_role(self, role: str) -> bool:
        """Check if user possesses the required role or is admin."""
        if "admin" in self.roles:
            return True
        return role in self.roles


def urlsafe_b64encode_str(data: bytes) -> str:
    """Base64 URL-safe encode without trailing padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def urlsafe_b64decode_str(data: str) -> bytes:
    """Base64 URL-safe decode adding padding as needed."""
    rem = len(data) % 4
    if rem > 0:
        data += "=" * (4 - rem)
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def create_jwt_token(payload: Dict[str, Any], secret_key: str, expires_in_seconds: int = 3600) -> str:
    """Create a signed JWT token."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload_copy = dict(payload)
    now = int(time.time())
    payload_copy.setdefault("iat", now)
    payload_copy.setdefault("exp", now + expires_in_seconds)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_bytes = json.dumps(payload_copy, separators=(",", ":")).encode("utf-8")

    segment1 = urlsafe_b64encode_str(header_bytes)
    segment2 = urlsafe_b64encode_str(payload_bytes)
    to_sign = f"{segment1}.{segment2}".encode("utf-8")

    signature = hmac.new(secret_key.encode("utf-8"), to_sign, hashlib.sha256).digest()
    segment3 = urlsafe_b64encode_str(signature)

    return f"{segment1}.{segment2}.{segment3}"


def verify_jwt_token(token: str, secret_key: str) -> Dict[str, Any]:
    """Verify and decode a signed JWT token."""
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token header or payload",
        )

    segment1, segment2, segment3 = parts
    to_sign = f"{segment1}.{segment2}".encode("utf-8")
    expected_sig = hmac.new(secret_key.encode("utf-8"), to_sign, hashlib.sha256).digest()

    try:
        provided_sig = urlsafe_b64decode_str(segment3)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature encoding",
        )

    if not hmac.compare_digest(expected_sig, provided_sig):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token signature mismatch",
        )

    try:
        payload_bytes = urlsafe_b64decode_str(segment2)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload json",
        )

    now = int(time.time())
    if "exp" in payload and payload["exp"] < now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )

    return payload


def verify_pkce_challenge(code_verifier: str, code_challenge: str, method: str = "S256") -> bool:
    """Verify standard PKCE code challenge against code verifier."""
    if method == "plain":
        return code_verifier == code_challenge
    if method == "S256":
        hashed = hashlib.sha256(code_verifier.encode("ascii")).digest()
        computed = urlsafe_b64encode_str(hashed)
        return hmac.compare_digest(computed, code_challenge)
    return False


def create_demo_token(role: str = "operator") -> str:
    """Create a scoped token for local developer mode."""
    if settings.is_production():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Demo tokens are strictly forbidden in production mode",
        )

    roles = [role]
    if role == "admin":
        roles = ["viewer", "operator", "admin"]
    elif role == "operator":
        roles = ["viewer", "operator"]

    payload = {
        "sub": f"demo-user-{role}",
        "preferred_username": f"demo_{role}",
        "email": f"demo_{role}@example.local",
        "roles": roles,
        "iss": "vector-workbench-local",
        "aud": settings.keycloak_client_id,
    }
    return create_jwt_token(payload, settings.jwt_secret_key, expires_in_seconds=86400)


def get_current_user(
    auth: Annotated[Optional[HTTPAuthorizationCredentials], Security(bearer_scheme)]
) -> UserPrincipal:
    """Extract and validate user principal from Bearer header."""
    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: missing or empty Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth.credentials

    # Validate against our JWT secret
    try:
        payload = verify_jwt_token(token, settings.jwt_secret_key)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token verification failed: {str(exc)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    roles = payload.get("roles", [])
    if isinstance(roles, str):
        roles = [roles]

    realm_access = payload.get("realm_access", {})
    if isinstance(realm_access, dict):
        roles.extend(realm_access.get("roles", []))

    user = UserPrincipal(
        subject=payload.get("sub", "unknown"),
        username=payload.get("preferred_username", payload.get("sub", "unknown")),
        email=payload.get("email", ""),
        roles=list(set(roles)),
    )
    return user


def require_role(role_name: str):
    """Dependency factory checking user role."""

    def role_checker(user: Annotated[UserPrincipal, Depends(get_current_user)]) -> UserPrincipal:
        if not user.has_role(role_name):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires '{role_name}' role; possessed roles: {user.roles}",
            )
        return user

    return role_checker


def redact_sensitive(text: str) -> str:
    """Sanitize secrets, API keys, and bearer tokens from logs and error messages."""
    if not text:
        return ""
    import re

    sanitized = re.sub(
        r"(Bearer\s+)[A-Za-z0-9\-_.]+",
        r"\1[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"(sk-[A-Za-z0-9]{8,})",
        r"[REDACTED_API_KEY]",
        sanitized,
    )
    return sanitized
