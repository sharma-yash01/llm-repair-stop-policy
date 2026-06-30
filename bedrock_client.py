"""AWS Bedrock Converse API client (boto3). Isolated from OpenAI/OpenRouter paths."""

from __future__ import annotations

import logging
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from config import AWS_REGION, BEDROCK_DEFAULT_MAX_TOKENS, BEDROCK_DEFAULT_READ_TIMEOUT_SEC

logger = logging.getLogger(__name__)
_clients: dict[int, Any] = {}


def _get_bedrock_client(read_timeout_sec: int | None = None):
    """Return a lazily initialized bedrock-runtime client for the given read timeout."""
    timeout = read_timeout_sec or BEDROCK_DEFAULT_READ_TIMEOUT_SEC
    if timeout not in _clients:
        _clients[timeout] = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            config=Config(
                read_timeout=timeout,
                connect_timeout=10,
                retries={"max_attempts": 0},
            ),
        )
    return _clients[timeout]


def _to_converse_messages(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]] | None]:
    """
    Convert OpenAI-style messages to Bedrock Converse format.

    Args:
        messages: List of {role, content} dicts.

    Returns:
        Tuple of (converse_messages, system_blocks or None).
    """
    system_blocks: list[dict[str, str]] = []
    converse_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role == "system":
            system_blocks.append({"text": content})
            continue
        if role not in ("user", "assistant"):
            role = "user"
        converse_messages.append(
            {"role": role, "content": [{"text": content}]}
        )
    system = system_blocks or None
    return converse_messages, system


def _extract_text(response: dict[str, Any]) -> str | None:
    """Extract assistant text from a Converse response."""
    try:
        content_blocks = response["output"]["message"]["content"]
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for block in content_blocks:
            text = block.get("text", "")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
                continue
            reasoning = block.get("reasoningContent")
            if isinstance(reasoning, dict):
                reasoning_text = reasoning.get("reasoningText", {})
                if isinstance(reasoning_text, dict):
                    chunk = reasoning_text.get("text", "")
                    if isinstance(chunk, str) and chunk.strip():
                        reasoning_parts.append(chunk.strip())
        if text_parts:
            return "\n".join(text_parts)
        if reasoning_parts:
            logger.warning("bedrock_response_reasoning_only_no_text_block")
            return "\n".join(reasoning_parts)
    except (KeyError, TypeError, IndexError) as e:
        logger.warning("bedrock_empty_or_malformed_response %s", e)
    return None


def _bedrock_converse(
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int | None,
    additional_request_fields: dict[str, Any] | None,
    read_timeout_sec: int | None,
) -> tuple[str | None, int, int]:
    """Single Bedrock Converse call (no retry)."""
    converse_messages, system = _to_converse_messages(messages)
    if not converse_messages:
        raise ValueError("bedrock_chat requires at least one user/assistant message")

    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": converse_messages,
        "inferenceConfig": {"maxTokens": max_tokens or BEDROCK_DEFAULT_MAX_TOKENS},
    }
    if system is not None:
        kwargs["system"] = system
    if additional_request_fields is not None:
        kwargs["additionalModelRequestFields"] = additional_request_fields

    try:
        response = _get_bedrock_client(read_timeout_sec).converse(**kwargs)
    except (ClientError, BotoCoreError) as e:
        logger.exception("bedrock_chat failed model=%s: %s", model_id, e)
        raise

    usage = response.get("usage") or {}
    input_tokens = int(usage.get("inputTokens", 0))
    output_tokens = int(usage.get("outputTokens", 0))
    logger.debug(
        "bedrock_usage model=%s input_tokens=%s output_tokens=%s",
        model_id,
        input_tokens,
        output_tokens,
    )
    return _extract_text(response), input_tokens, output_tokens


def bedrock_chat(
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    additional_request_fields: dict[str, Any] | None = None,
    read_timeout_sec: int | None = None,
) -> tuple[str | None, int, int]:
    """
    Call Bedrock Converse API with OpenAI-style messages.

    Args:
        model_id: Bedrock model ID (e.g. anthropic.claude-3-5-haiku-20241022-v1:0).
        messages: OpenAI-style message list.
        max_tokens: Optional max output tokens.
        additional_request_fields: Optional reasoning/toggle fields for Converse.
        read_timeout_sec: Optional HTTP read timeout override.

    Returns:
        Tuple of (assistant text or None, input_tokens, output_tokens).
    """
    timeout = read_timeout_sec or BEDROCK_DEFAULT_READ_TIMEOUT_SEC
    wait_max = 60 if timeout >= 600 else 10

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=wait_max))
    def _call() -> tuple[str | None, int, int]:
        return _bedrock_converse(
            model_id,
            messages,
            max_tokens,
            additional_request_fields,
            read_timeout_sec,
        )

    return _call()
