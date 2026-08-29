"""Generic Bedrock Converse implementation.

The Converse request/response shape is the same across every Bedrock-hosted
model with vision support, so this one file covers Anthropic, xAI, Amazon Nova,
Meta and the rest - a model swap is a config change, not new code.
"""

from __future__ import annotations

import logging

import boto3
from botocore.config import Config as BotoConfig

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

# Models exposing a reasoning knob: this is OCR plus tag selection, not
# multi-step analysis, so the lowest tier avoids latency and tokens we do not need.
_LOW_EFFORT_FIELDS: dict[str, dict] = {
    "grok": {"reasoning_effort": "low"},
}


def _additional_fields(model_id: str) -> dict:
    lowered = model_id.lower()
    for marker, fields in _LOW_EFFORT_FIELDS.items():
        if marker in lowered:
            return fields
    return {}


class BedrockProvider:
    def __init__(self, model_id: str, *, client=None) -> None:
        if not model_id:
            raise ValueError("AiModelId is required for the bedrock provider")
        self.model_id = model_id
        self._explicit_client = client

    @property
    def _client(self):
        """Built on first use, not at construction.

        Constructing the provider must not require AWS credentials or a region -
        only actually calling it should.
        """
        if self._explicit_client is None:
            self._explicit_client = boto3.client(
                "bedrock-runtime",
                config=BotoConfig(
                    read_timeout=120, connect_timeout=10, retries={"max_attempts": 3}
                ),
            )
        return self._explicit_client

    def _converse(self, image_bytes: bytes, prompt: str) -> tuple[str, int, int]:
        kwargs: dict = {
            "modelId": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                        {"text": prompt},
                    ],
                }
            ],
            "inferenceConfig": {
                "maxTokens": MAX_OUTPUT_TOKENS,
                "temperature": TEMPERATURE,
            },
        }
        extra = _additional_fields(self.model_id)
        if extra:
            kwargs["additionalModelRequestFields"] = extra

        resp = self._client.converse(**kwargs)
        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        usage = resp.get("usage", {}) or {}

        # Output-token limits vary by model; a truncated transcription would
        # otherwise silently produce a half-written note.
        if resp.get("stopReason") == "max_tokens":
            logger.warning(
                "Model %s hit maxTokens (%d) - transcription may be truncated",
                self.model_id,
                MAX_OUTPUT_TOKENS,
            )
        return text, int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))

    def extract_and_tag(
        self,
        image_bytes: bytes,
        existing_tags: list[str],
        existing_titles: list[str] | None = None,
    ) -> ProviderResult:
        prompt = build_prompt(existing_tags, existing_titles)
        raw, tin, tout = self._converse(image_bytes, prompt)
        try:
            result = parse_response(raw)
        except ValueError:
            logger.warning("Unparseable response from %s; retrying once", self.model_id)
            raw2, tin2, tout2 = self._converse(image_bytes, prompt + RETRY_SUFFIX)
            tin, tout = tin + tin2, tout + tout2
            try:
                result = parse_response(raw2)
            except ValueError:
                result = fallback_result(raw2 or raw)
        result.input_tokens, result.output_tokens = tin, tout
        return result
