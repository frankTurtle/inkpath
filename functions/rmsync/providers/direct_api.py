"""Direct-to-vendor HTTPS provider, for a model that is not on Bedrock.

Only reach for this once you have actually picked a non-Bedrock provider: it
trades the IAM-only auth story for an API key stored in SSM.

Implemented against the Anthropic Messages API shape, which xAI also accepts at
its /v1/messages compatibility endpoint. Point base_url at your vendor.
"""

from __future__ import annotations

import base64
import logging
from urllib.parse import urlparse

import requests

from .base import (
    MAX_OUTPUT_TOKENS,
    RETRY_SUFFIX,
    TEMPERATURE,
    ProviderResult,
    build_prompt,
    fallback_result,
    parse_response,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
HTTP_TIMEOUT = 120

# Vendors that speak the Anthropic Messages shape but differ on the auth header.
# xAI is Anthropic-SDK-compatible at https://api.x.ai but authenticates with
# `Authorization: Bearer`, not `x-api-key`; sending the wrong one is a 401.
_BEARER_HOSTS = ("api.x.ai",)


def _auth_headers(base_url: str, api_key: str) -> dict[str, str]:
    """Pick the auth header the target vendor actually accepts."""
    host = urlparse(base_url).hostname or ""
    if any(host == h or host.endswith(f".{h}") for h in _BEARER_HOSTS):
        return {"Authorization": f"Bearer {api_key}"}
    return {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}


def _messages_url(base_url: str) -> str:
    """Accept either a bare origin or a full /v1/messages URL."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/messages"):
        return trimmed
    return f"{trimmed}/v1/messages"


class DirectApiProvider:
    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("AiModelId is required for the direct provider")
        if not api_key:
            raise ValueError(
                "AiProvider=direct requires an API key in SSM at /rmsync/ai-api-key"
            )
        self.model_id = model_id
        self._base_url = _messages_url(base_url or DEFAULT_BASE_URL)
        self._session = session or requests.Session()
        self._session.headers.update(
            {"content-type": "application/json", **_auth_headers(self._base_url, api_key)}
        )

    def _call(self, image_bytes: bytes, prompt: str) -> tuple[str, int, int]:
        payload = {
            "model": self.model_id,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        resp = self._session.post(self._base_url, json=payload, timeout=HTTP_TIMEOUT)
        if not resp.ok:
            raise RuntimeError(
                f"Provider call failed ({resp.status_code}): {resp.text[:200]}"
            )
        data = resp.json()
        text = "".join(
            b.get("text", "") for b in data.get("content", []) if isinstance(b, dict)
        )
        usage = data.get("usage", {}) or {}
        if data.get("stop_reason") == "max_tokens":
            logger.warning("Provider hit max_tokens - transcription may be truncated")
        return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    def extract_and_tag(
        self,
        image_bytes: bytes,
        existing_tags: list[str],
        existing_titles: list[str] | None = None,
    ) -> ProviderResult:
        prompt = build_prompt(existing_tags, existing_titles)
        raw, tin, tout = self._call(image_bytes, prompt)
        try:
            result = parse_response(raw)
        except ValueError:
            logger.warning("Unparseable response; retrying once")
            raw2, tin2, tout2 = self._call(image_bytes, prompt + RETRY_SUFFIX)
            tin, tout = tin + tin2, tout + tout2
            try:
                result = parse_response(raw2)
            except ValueError:
                result = fallback_result(raw2 or raw)
        result.input_tokens, result.output_tokens = tin, tout
        return result
