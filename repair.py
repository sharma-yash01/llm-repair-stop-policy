"""Shared LLM utilities for repair strategies."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    COST_HARD_STOP_USD,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    GEMINI_API_BASE,
    GEMINI_API_KEY_ENV,
    LLM_PROVIDER,
    LLM_TIMEOUT_SEC,
    OPENROUTER_API_BASE,
    OPENROUTER_API_KEY_ENV,
    RATE_LIMIT_SLEEP,
    SELF_VERIFICATION_MODE,
)

logger = logging.getLogger(__name__)

running_cost_usd = 0.0
_client: OpenAI | None = None
_resolved_model_name: str | None = None


def _resolve_provider(model: str):
    """
    Resolve provider, base_url, and API key from config + environment.

    Args:
        model: Requested model id. May be empty.

    Returns:
        Tuple of (provider, base_url, api_key).
    """
    has_gemini_key = bool(os.environ.get(GEMINI_API_KEY_ENV))
    has_openrouter_key = bool(os.environ.get(OPENROUTER_API_KEY_ENV))

    provider = LLM_PROVIDER
    if provider not in ("auto", "gemini", "openrouter"):
        raise RuntimeError(
            f"Invalid LLM_PROVIDER={provider!r}. Use one of: auto, gemini, openrouter."
        )

    if provider == "auto":
        if has_gemini_key and not has_openrouter_key:
            provider = "gemini"
        elif has_openrouter_key and not has_gemini_key:
            provider = "openrouter"
        elif has_gemini_key and has_openrouter_key:
            # If both keys are present, infer from explicit model when possible.
            explicit = (model or "").strip()
            if explicit.startswith("gemini-"):
                provider = "gemini"
            elif explicit:
                provider = "openrouter"
            else:
                raise RuntimeError(
                    "Both GEMINI_API_KEY and OPENROUTER_API_KEY are set. "
                    "Set LLM_PROVIDER=gemini|openrouter to choose one."
                )
        else:
            raise RuntimeError(
                "No API key found. Export GEMINI_API_KEY or OPENROUTER_API_KEY."
            )

    if provider == "gemini":
        api_key = os.environ.get(GEMINI_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {GEMINI_API_KEY_ENV}."
            )
        return provider, GEMINI_API_BASE, api_key

    api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Missing API key environment variable: {OPENROUTER_API_KEY_ENV}."
        )
    return provider, OPENROUTER_API_BASE, api_key


def _resolve_model_name(model: str):
    """
    Resolve effective model name from explicit value or provider defaults.

    Args:
        model: Requested model id. May be empty.

    Returns:
        Effective model name.
    """
    explicit = (model or "").strip()
    if explicit:
        return explicit
    provider, _, _ = _resolve_provider(explicit)
    if provider == "gemini":
        return DEFAULT_GEMINI_MODEL
    return DEFAULT_OPENROUTER_MODEL


def _get_client(model: str):
    """Return a lazily initialized OpenAI client for the resolved provider."""
    global _client, _resolved_model_name
    resolved_model = _resolve_model_name(model)
    if _client is None or _resolved_model_name != resolved_model:
        _, base_url, api_key = _resolve_provider(resolved_model)
        _client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=LLM_TIMEOUT_SEC,
        )
        _resolved_model_name = resolved_model
    return _client


def _log_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
):
    """Log per-call model, tokens, and cost."""
    logger.info(
        "llm_call model=%s prompt_tokens=%s completion_tokens=%s cost_usd=%.6f",
        model,
        prompt_tokens,
        completion_tokens,
        cost_usd,
    )


def strip_code_fences(text: str):
    """
    Extract raw Python from LLM output that may be wrapped in markdown fences or prose.

    Args:
        text: Raw completion (may contain ```python...``` or ```...```, or prose before code).

    Returns:
        Clean Python source string.
    """
    text = text.strip()
    # Match ```python ... ``` or ``` ... ``` (optional language tag)
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(text)
    if blocks:
        # Take longest block (most likely the full function)
        code = max(blocks, key=len).strip()
        if code:
            return code
    # No fences: look for first def / from / import
    for start in ("def ", "from ", "import "):
        idx = text.find(start)
        if idx != -1:
            return text[idx:].strip()
    return text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(
    prompt: str,
    model: str,
    messages: list[dict[str, str]] | None = None,
):
    """
    Call LLM with retry, rate limit, and cost tracking.

    Args:
        prompt: User message content.
        model: Model name (e.g. gpt-4o-mini).
        messages: Optional full message list for multi-turn interactions.

    Returns:
        Assistant message content, or None when the provider returns null/empty content.

    Raises:
        RuntimeError: If running cost exceeds COST_HARD_STOP_USD.
    """
    global running_cost_usd
    resolved_model = _resolve_model_name(model)
    if running_cost_usd >= COST_HARD_STOP_USD:
        raise RuntimeError(f"Cost hard stop hit: ${running_cost_usd:.2f}")
    time.sleep(RATE_LIMIT_SLEEP)
    request_messages = messages or [{"role": "user", "content": prompt}]
    try:
        response = _get_client(resolved_model).chat.completions.create(
            model=resolved_model,
            messages=request_messages,
        )
    except (APIError, APIConnectionError, RateLimitError, Exception) as e:
        logger.exception("call_llm failed: %s", e)
        logger.warning("call_llm retry due to exception")
        raise
    try:
        content = response.choices[0].message.content
        if isinstance(content, str):
            content = content.strip() or None
        if content is None:
            _log_call(resolved_model, 0, 0, 0.0)
            return None
        cost = 0.0
        running_cost_usd += cost
        usage = getattr(response, "usage", None)
        if usage is not None and hasattr(usage, "prompt_tokens"):
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
        elif isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            prompt_tokens = completion_tokens = 0
        _log_call(resolved_model, prompt_tokens, completion_tokens, cost)
        return content
    except (IndexError, AttributeError, TypeError) as e:
        logger.warning("llm_bad_response_shape %s", e)
        _log_call(resolved_model, 0, 0, 0.0)
        return None


def _is_functional_problem(problem_dict: dict[str, Any] | None):
    """
    Determine whether a problem expects function-call style solutions.

    Args:
        problem_dict: Full problem metadata and tests.

    Returns:
        True when functional interface is expected.
    """
    if not problem_dict:
        return False
    metadata = problem_dict.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    func_name = metadata.get("func_name", "")
    if isinstance(func_name, str) and func_name.strip():
        return True
    for key in ("public_test_cases", "private_test_cases"):
        for tc in problem_dict.get(key) or []:
            if isinstance(tc, dict) and str(tc.get("testtype", "stdin")) == "functional":
                return True
    return False


def _build_initial_prompt(problem_prompt: str, problem_dict: dict[str, Any] | None) -> str:
    """
    Build an initial generation prompt tailored to test interface style.

    Args:
        problem_prompt: Full problem statement.
        problem_dict: Full problem metadata and tests.

    Returns:
        Prompt string.
    """
    if _is_functional_problem(problem_dict):
        return (
            "Solve the following competitive programming problem in Python.\n"
            "Implement the required function exactly as specified.\n"
            "Do not read from stdin or print to stdout unless the prompt explicitly asks for it.\n"
            "Return ONLY the Python code, no explanation.\n\n"
            f"{problem_prompt}"
        )
    return (
        "Solve the following competitive programming problem in Python.\n"
        "Read input from stdin and print output to stdout.\n"
        "Return ONLY the Python code, no explanation.\n\n"
        f"{problem_prompt}"
    )


def generate_initial(
    problem_prompt: str,
    model: str,
    problem_dict: dict[str, Any] | None = None,
):
    """
    Zero-shot initial code generation (iteration 0).

    Args:
        problem_prompt: Full problem prompt.
        model: Model name.
        problem_dict: Full problem metadata and tests.

    Returns:
        Raw completion (code string), or None if the provider returns null/empty content.
    """
    prompt = _build_initial_prompt(problem_prompt, problem_dict)
    return call_llm(prompt, model)


def _extract_yes_probability(response: Any):
    """
    Extract P(Yes) from logprobs (top_logprobs). Handles tokenizer-dependent Yes/No tokens.

    Args:
        response: Chat completion response with choices[0].logprobs.

    Returns:
        Probability of "Yes" in [0, 1], or 0.5 if not determinable.
    """
    import math
    try:
        choice = response.choices[0]
        logprobs = getattr(choice, "logprobs", None) or getattr(
            choice.message, "logprobs", None
        )
        if not logprobs:
            return 0.5
        content = getattr(logprobs, "content", None)
        if not content:
            return 0.5
        yes_logprob = None
        no_logprob = None
        for item in content if isinstance(content, list) else [content]:
            top = getattr(item, "top_logprobs", None) or (item if isinstance(item, dict) else None)
            if top is None:
                continue
            tokens_with_lp: list[tuple[str, float]] = []
            if isinstance(top, list):
                for t in top:
                    token = t.get("token", getattr(t, "token", "")) if isinstance(t, dict) else getattr(t, "token", "")
                    lp = t.get("logprob", getattr(t, "logprob", -999)) if isinstance(t, dict) else getattr(t, "logprob", -999)
                    tokens_with_lp.append((str(token), float(lp)))
            elif isinstance(top, dict):
                for token, lp in top.items():
                    tok = token if isinstance(token, str) else getattr(token, "token", "")
                    lp_val = lp if isinstance(lp, (int, float)) else getattr(lp, "logprob", -999)
                    tokens_with_lp.append((str(tok), float(lp_val)))
            for tok, lp in tokens_with_lp:
                t = tok.strip().lower()
                if t in ("yes", "yes."):
                    yes_logprob = lp if yes_logprob is None else max(yes_logprob, lp)
                elif t in ("no", "no."):
                    no_logprob = lp if no_logprob is None else max(no_logprob, lp)
        if yes_logprob is not None and no_logprob is not None:
            p_yes = math.exp(yes_logprob)
            p_no = math.exp(no_logprob)
            return float(p_yes / (p_yes + p_no))
        if yes_logprob is not None:
            return 1.0
        if no_logprob is not None:
            return 0.0
    except Exception as e:
        logger.debug("extract_yes_probability failed: %s", e)
    return 0.5


def get_self_verification_score(problem: str, code: str, model: str):
    """
    Return P(Yes) from logprobs for 'Will this code pass all tests? Yes/No'.
    Falls back to structured JSON confidence, then 0.5 if both fail.

    Args:
        problem: Problem description.
        code: Code string.
        model: Model name.

    Returns:
        Score in [0, 1].
    """
    prompt = (
        f"Problem:\n{problem}\n\nCode:\n```python\n{code}\n```\n\n"
        "Will this code pass all tests? Answer Yes or No only."
    )
    mode = (SELF_VERIFICATION_MODE or "auto").strip().lower()
    resolved_model = _resolve_model_name(model)
    if mode in ("auto", "logprobs"):
        try:
            response = _get_client(resolved_model).chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                logprobs=True,
                top_logprobs=5,
            )
            score = _extract_yes_probability(response)
            if abs(score - 0.5) > 1e-9:
                return score
            logger.info(
                "self-verification logprobs inconclusive (score=0.5), falling back"
            )
            if mode == "logprobs":
                return 0.5
        except Exception as e:
            logger.info("self-verification logprobs path failed, falling back: %s", e)
            if mode == "logprobs":
                return 0.5
    if mode in ("auto", "json"):
        try:
            response = _get_client(resolved_model).chat.completions.create(
                model=resolved_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt + ' Respond {"confidence": <float 0-1>}',
                    }
                ],
                max_tokens=20,
                response_format={"type": "json_object"},
            )
            raw = None
            try:
                raw = response.choices[0].message.content
            except (IndexError, AttributeError, TypeError):
                pass
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                logger.info(
                    "self-verification JSON content missing/empty, falling back to text"
                )
            else:
                try:
                    return float(json.loads(raw)["confidence"])
                except Exception as e:
                    logger.info(
                        "self-verification JSON parse failed, falling back to text: %s",
                        e,
                    )
        except Exception as e:
            logger.info("self-verification JSON path failed, falling back to text: %s", e)
            if mode == "json":
                return 0.5
    # Text-based fallback: plain Yes/No response (no logprobs/JSON)
    try:
        response = call_llm(prompt, model)
        raw = (response or "").strip().lower()
        yes_match = re.search(r"\byes\b", raw)
        no_match = re.search(r"\bno\b", raw)
        if yes_match and not no_match:
            return 1.0
        if no_match and not yes_match:
            return 0.0
        first_word = raw.split()[0].rstrip(".,!") if raw.split() else ""
        if first_word == "yes":
            return 1.0
        if first_word == "no":
            return 0.0
        logger.info("self-verification text fallback inconclusive; returning 0.5")
    except Exception as e:
        logger.info("self-verification text fallback failed; returning 0.5: %s", e)
    return 0.5
