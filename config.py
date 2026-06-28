"""All constants for the pilot. Do not hardcode these elsewhere."""

from __future__ import annotations

import os
from typing import Any

# Provider selection:
# - "auto": choose from available API keys (recommended; does not auto-select bedrock)
# - "gemini": force Google AI Studio Gemini endpoint
# - "openrouter": force OpenRouter endpoint
# - "bedrock": force AWS Bedrock Converse (boto3; requires IAM or AWS credentials)
# - "anthropic": force Anthropic Messages API (native SDK)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto").strip().lower()

# Optional explicit model override via env var MODEL.
# When empty, provider-specific default model is used.
MODEL = os.environ.get("MODEL", "").strip()

# Provider defaults (used when MODEL is empty).
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"

# AWS Bedrock (boto3 Converse)
AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("BEDROCK_REGION") or "us-west-2"

# OpenAI-compatible endpoints
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# API key env var names
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

# Bedrock reasoning toggle fields (edit here if console/API shape differs)
BEDROCK_REASONING_ON: dict[str, Any] = {
    "reasoning_config": {"type": "enabled", "budget_tokens": 2048},
}
BEDROCK_REASONING_OFF: dict[str, Any] = {
    "reasoning_config": {"type": "disabled"},
}

# Self-verification strategy:
# - "text": single cheap Yes/No request (best for Gemini free tier).
# - "auto": logprobs -> JSON -> text fallback.
SELF_VERIFICATION_MODE = "auto"
MAX_ITERATIONS = 5  # linear loop steps (kept for backward compatibility)
MAX_EVAL_STEPS = 5  # unified eval-step budget per problem for all strategies
N_PROBLEMS = 200  # number of problems to run (LCB: first N from release)
# LiveCodeBench: release_v1 (400) through release_v6 (1055)
LCB_RELEASE = "release_v1"
# Only include problems with this difficulty or harder (easy | medium | hard)
LCB_MIN_DIFFICULTY = "easy"
PASS_THRESHOLD = 0.8  # Oracle-First binary solve threshold
IMPROVEMENT_THRESHOLD = 0.05
SUBPROCESS_TIMEOUT = 10
RATE_LIMIT_SLEEP = float(os.environ.get("RATE_LIMIT_SLEEP", "2"))
# Max seconds for a single completion request; prevents indefinite hangs.
LLM_TIMEOUT_SEC = 120
MAX_RETRIES = 3

SUPPORTED_STRATEGIES = (
    "direct_fix",
    "self_debugging",
    "reflexion",
    "alphacodium",
    "codetree",
    "rex",
)
STRATEGY = os.environ.get("STRATEGY", "direct_fix").strip().lower()
if STRATEGY not in SUPPORTED_STRATEGIES:
    raise RuntimeError(
        f"Invalid STRATEGY={STRATEGY!r}. Use one of: {', '.join(SUPPORTED_STRATEGIES)}."
    )

MODELS = [
    "deepseek/deepseek-chat",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4o",
    "deepseek/deepseek-reasoner",
]

MODELS_BEDROCK = [
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
]

# Production model registry: label is the canonical id passed through the pipeline.
# route: bedrock | anthropic | (legacy openrouter/gemini via run.py MODELS)
# reasoning: None | "on" | "off" — applied via BEDROCK_REASONING_* for Bedrock entries
MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "label": "gemma-3-12b-it",
        "route": "bedrock",
        "model_id": "google.gemma-3-12b-it",
        "reasoning": None,
        "price_in": 0.10,
        "price_out": 0.25,
    },
    {
        "label": "qwen3-32b",
        "route": "bedrock",
        "model_id": "qwen.qwen3-32b-v1:0",
        "reasoning": "off",
        "price_in": 0.30,
        "price_out": 0.50,
    },
    {
        "label": "qwen3-coder-30b",
        "route": "bedrock",
        "model_id": "qwen.qwen3-coder-30b-a3b-v1:0",
        "reasoning": None,
        "price_in": 0.30,
        "price_out": 0.55,
    },
    {
        "label": "deepseek-v3.1-quick",
        "route": "bedrock",
        "model_id": "deepseek.v3-v1:0",
        "reasoning": "off",
        "price_in": 0.62,
        "price_out": 1.85,
    },
    {
        "label": "deepseek-v3.1-thinking",
        "route": "bedrock",
        "model_id": "deepseek.v3-v1:0",
        "reasoning": "on",
        "price_in": 0.62,
        "price_out": 1.85,
    },
    {
        "label": "claude-haiku-4.5",
        "route": "anthropic",
        "model_id": "claude-haiku-4-5",
        "reasoning": None,
        "price_in": 1.00,
        "price_out": 5.00,
    },
    {
        "label": "deepseek-r1",
        "route": "bedrock",
        "model_id": "us.deepseek.r1-v1:0",
        "reasoning": "on",
        "price_in": 1.35,
        "price_out": 5.40,
    },
    {
        "label": "claude-sonnet-4.6",
        "route": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "reasoning": None,
        "price_in": 3.00,
        "price_out": 15.00,
    },
    {
        "label": "claude-opus-4.8",
        "route": "anthropic",
        "model_id": "claude-opus-4-8",
        "reasoning": None,
        "price_in": 5.00,
        "price_out": 25.00,
    },
]


def get_reasoning_fields(reasoning: str | None) -> dict[str, Any] | None:
    """
    Map a reasoning toggle value to Bedrock additionalModelRequestFields.

    Args:
        reasoning: None, "on", or "off".

    Returns:
        Dict for additionalModelRequestFields, or None when not applicable.
    """
    if reasoning is None:
        return None
    if reasoning == "on":
        return BEDROCK_REASONING_ON
    if reasoning == "off":
        return BEDROCK_REASONING_OFF
    return None


def get_model_config(label: str) -> dict[str, Any] | None:
    """
    Look up a model registry entry by label.

    Args:
        label: Canonical model label (e.g. claude-haiku-4.5).

    Returns:
        Config dict or None if not in MODEL_CONFIGS.
    """
    key = (label or "").strip()
    if not key:
        return None
    for entry in MODEL_CONFIGS:
        if entry.get("label") == key:
            return entry
    return None


DATA_DIR = "data/trajectories"
STRATEGY_METADATA_DIR = "data/strategy_metadata"
RESULTS_DIR = "data/results"
FIGURES_DIR = "data/figures"

BOOTSTRAP_N_RESAMPLES = 1000
COST_HARD_STOP_USD = 200.0
