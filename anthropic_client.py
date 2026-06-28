"""Anthropic Messages API client (native SDK). Isolated from Bedrock/OpenRouter paths."""

from __future__ import annotations

import logging
import os
from typing import Any

from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from config import ANTHROPIC_API_KEY_ENV

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 4096
_client: Anthropic | None = None


def _get_anthropic_client() -> Anthropic:
    """Return a lazily initialized Anthropic client."""
    global _client
    if _client is None:
        api_key = os.environ.get(ANTHROPIC_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {ANTHROPIC_API_KEY_ENV}."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def _split_messages(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    """
    Split OpenAI-style messages into system prompt and user/assistant turns.

    Args:
        messages: List of {role, content} dicts.

    Returns:
        Tuple of (system text or None, anthropic-style message list).
    """
    system_parts: list[str] = []
    api_messages: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role == "system":
            system_parts.append(content)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        api_messages.append({"role": role, "content": content})
    system = "\n\n".join(system_parts).strip() or None
    return system, api_messages


def _extract_text(response: Any) -> str | None:
    """Extract assistant text from a Messages API response."""
    try:
        parts: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return "\n".join(parts)
    except (AttributeError, TypeError) as e:
        logger.warning("anthropic_empty_or_malformed_response %s", e)
    return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def anthropic_chat(
    model_id: str,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
) -> tuple[str | None, int, int]:
    """
    Call Anthropic Messages API with OpenAI-style messages.

    Args:
        model_id: Anthropic model id (e.g. claude-haiku-4-5).
        messages: OpenAI-style message list.
        max_tokens: Optional max output tokens.

    Returns:
        Tuple of (assistant text or None, input_tokens, output_tokens).
    """
    system, api_messages = _split_messages(messages)
    if not api_messages:
        raise ValueError("anthropic_chat requires at least one user/assistant message")

    kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": api_messages,
        "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
    }
    if system is not None:
        kwargs["system"] = system

    try:
        response = _get_anthropic_client().messages.create(**kwargs)
    except Exception as e:
        logger.exception("anthropic_chat failed model=%s: %s", model_id, e)
        raise

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) if usage else 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) if usage else 0)
    logger.debug(
        "anthropic_usage model=%s input_tokens=%s output_tokens=%s",
        model_id,
        input_tokens,
        output_tokens,
    )
    return _extract_text(response), input_tokens, output_tokens
