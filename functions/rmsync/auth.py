"""Secrets (SSM SecureString) and reMarkable token exchange.

SSM Parameter Store rather than Secrets Manager: Secrets Manager bills $0.40 per
secret per month regardless of access, which for three secrets would be the
single largest line item in this stack. SecureString is free and uses the same
KMS encryption.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import boto3
import requests

logger = logging.getLogger(__name__)

AUTH_HOST = "https://webapp-prod.cloud.remarkable.engineering"
DEVICE_ENDPOINT = f"{AUTH_HOST}/token/json/2/device/new"
USER_ENDPOINT = f"{AUTH_HOST}/token/json/2/user/new"

HTTP_TIMEOUT = 30

# Module-scope caches. Warm containers reuse these for the invocation's
# lifetime; nothing is persisted across invocations.
_ssm = None
_secret_cache: dict[str, str] = {}
_user_token_cache: dict[str, str] = {}


class AuthError(RuntimeError):
    """reMarkable authentication failed. Deliberately loud - a silent auth
    failure looks identical to 'no new pages'."""


def _client() -> Any:
    global _ssm
    if _ssm is None:
        _ssm = boto3.client("ssm")
    return _ssm


def get_secret(name: str, *, prefix: str = "/rmsync", required: bool = True) -> str:
    """Read a SecureString parameter, cached per container."""
    full = name if name.startswith("/") else f"{prefix}/{name}"
    if full in _secret_cache:
        return _secret_cache[full]
    try:
        resp = _client().get_parameter(Name=full, WithDecryption=True)
    except Exception as exc:  # noqa: BLE001 - surfaced with context below
        if required:
            raise AuthError(
                f"Could not read SSM parameter {full}. Populate it out-of-band: "
                f"aws ssm put-parameter --name {full} --type SecureString --value ..."
            ) from exc
        return ""
    value = resp["Parameter"]["Value"]
    _secret_cache[full] = value
    return value


def get_user_token(device_token: str) -> str:
    """Exchange the long-lived device token for a short-lived user token."""
    cached = _user_token_cache.get(device_token)
    if cached:
        return cached
    resp = requests.post(
        USER_ENDPOINT,
        headers={"Authorization": f"Bearer {device_token}"},
        timeout=HTTP_TIMEOUT,
    )
    if not resp.ok:
        raise AuthError(
            f"Token refresh failed ({resp.status_code} {resp.reason}). The device "
            "token may have been revoked - re-register the device and update "
            "/rmsync/remarkable-token."
        )
    token = resp.text.strip()
    if not token:
        raise AuthError("Token refresh returned an empty body")
    _user_token_cache[device_token] = token
    return token


def register_device(code: str, device_desc: str = "desktop-macos") -> str:
    """One-time device registration. Run locally, then store the result in SSM.

    Get the eight-letter code from https://my.remarkable.com/device/desktop/connect
    """
    code = code.strip()
    if len(code) != 8:
        raise AuthError(f"Code should be 8 characters, got {len(code)}")
    resp = requests.post(
        DEVICE_ENDPOINT,
        headers={"Authorization": "Bearer"},
        json={"code": code, "deviceDesc": device_desc, "deviceID": str(uuid.uuid4())},
        timeout=HTTP_TIMEOUT,
    )
    if not resp.ok:
        raise AuthError(f"Device registration failed ({resp.status_code}): {resp.text[:200]}")
    return resp.text.strip()


def reset_caches() -> None:
    """Test hook."""
    _secret_cache.clear()
    _user_token_cache.clear()
