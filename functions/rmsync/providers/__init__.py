"""Provider registry.

enrich.py never imports a vendor SDK directly - it calls get() and works from
the common ProviderResult shape.
"""

from __future__ import annotations

from .base import (
    LINK_MODES,
    ProviderResult,
    VisionProvider,
    build_prompt,
    fallback_result,
    parse_response,
    sanitize_tag,
    sanitize_tags,
)

__all__ = [
    "LINK_MODES",
    "ProviderResult",
    "VisionProvider",
    "build_prompt",
    "fallback_result",
    "get",
    "parse_response",
    "sanitize_tag",
    "sanitize_tags",
]


def get(
    provider: str,
    model_id: str,
    *,
    api_key: str = "",
    base_url: str = "",
    link_mode: str = "related",
) -> VisionProvider:
    """Return the configured provider implementation."""
    if provider == "bedrock":
        from .bedrock import BedrockProvider

        return BedrockProvider(model_id, link_mode=link_mode)
    if provider == "direct":
        from .direct_api import DEFAULT_BASE_URL, DirectApiProvider

        return DirectApiProvider(
            model_id, api_key, base_url=base_url or DEFAULT_BASE_URL, link_mode=link_mode
        )
    raise ValueError(f"Unknown AiProvider {provider!r}; expected 'bedrock' or 'direct'")
