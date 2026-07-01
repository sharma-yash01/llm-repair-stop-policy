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

# Bedrock reasoning toggle defaults (Qwen3 / DeepSeek-V3.1 use reasoning_effort)
BEDROCK_REASONING_EFFORT_ON: dict[str, Any] = {"reasoning_effort": "medium"}
BEDROCK_REASONING_EFFORT_OFF: dict[str, Any] = {"reasoning_effort": "none"}

# Bedrock Converse maxTokens (router default; R1 needs headroom for reasoning blocks).
BEDROCK_DEFAULT_MAX_TOKENS = 4096
# R1 is always-reasoning; 4096 is exhausted by thinking before answer text is emitted.
BEDROCK_R1_MAX_TOKENS = int(os.environ.get("BEDROCK_R1_MAX_TOKENS", "16384"))
# Bedrock boto3 read timeout (default 60s is too short for R1 reasoning calls).
BEDROCK_DEFAULT_READ_TIMEOUT_SEC = int(
    os.environ.get("BEDROCK_DEFAULT_READ_TIMEOUT_SEC", "120")
)
BEDROCK_R1_READ_TIMEOUT_SEC = int(os.environ.get("BEDROCK_R1_READ_TIMEOUT_SEC", "900"))

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
# EXTENSION POINT: Reflexion episodic memory window (paper Omega=1-3).
REFLEXION_MEMORY_LIMIT = 3
# EXTENSION POINT: Self-Debugging rubber-duck explanation as separate LLM call.
SELF_DEBUG_SEPARATE_EXPLANATION = True
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

# Session-scoped ad-hoc configs registered via run.py --adhoc-* flags.
_EPHEMERAL_CONFIGS: dict[str, dict[str, Any]] = {}

# Base model specs. reasoning_toggle=True expands into separate off/on registry entries.
_MODEL_SPECS: list[dict[str, Any]] = [
    {
        "route": "bedrock",
        "model_id": "google.gemma-3-12b-it",
        "reasoning": None,
        "price_in": 0.10,
        "price_out": 0.25,
        "labels": {"single": "gemma-3-12b-it"},
    },
    {
        "route": "bedrock",
        "model_id": "qwen.qwen3-32b-v1:0",
        "reasoning_toggle": True,
        "reasoning_fields_off": {"reasoning_effort": "none"},
        "reasoning_fields_on": {"reasoning_effort": "medium"},
        "price_in": 0.30,
        "price_out": 0.50,
        "labels": {"off": "qwen3-32b", "on": "qwen3-32b-thinking"},
    },
    {
        "route": "bedrock",
        "model_id": "qwen.qwen3-coder-30b-a3b-v1:0",
        "reasoning": None,
        "price_in": 0.30,
        "price_out": 0.55,
        "labels": {"single": "qwen3-coder-30b"},
    },
    {
        "route": "bedrock",
        "model_id": "deepseek.v3-v1:0",
        "reasoning_toggle": True,
        "reasoning_fields_off": {"reasoning_effort": "none"},
        "reasoning_fields_on": {"reasoning_effort": "medium"},
        "price_in": 0.62,
        "price_out": 1.85,
        "labels": {"off": "deepseek-v3.1-quick", "on": "deepseek-v3.1-thinking"},
    },
    {
        "route": "anthropic",
        "model_id": "claude-haiku-4-5",
        "reasoning": None,
        "price_in": 1.00,
        "price_out": 5.00,
        "labels": {"single": "claude-haiku-4.5"},
    },
    {
        "route": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "reasoning": None,
        "price_in": 3.00,
        "price_out": 15.00,
        "labels": {"single": "claude-sonnet-4.6"},
    },
    {
        "route": "anthropic",
        "model_id": "claude-opus-4-8",
        "reasoning": None,
        "price_in": 5.00,
        "price_out": 25.00,
        "labels": {"single": "claude-opus-4.8"},
    },
    {
        "route": "bedrock",
        "model_id": "us.deepseek.r1-v1:0",
        "reasoning": None,
        "max_tokens": BEDROCK_R1_MAX_TOKENS,
        "read_timeout_sec": BEDROCK_R1_READ_TIMEOUT_SEC,
        "price_in": 1.35,
        "price_out": 5.40,
        "labels": {"single": "deepseek-r1"},
    },
]


def _expand_spec_entry(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expand one model spec into one or two flat MODEL_CONFIGS entries.

    Args:
        spec: Base spec from _MODEL_SPECS.

    Returns:
        List of registry dicts with label, route, model_id, reasoning, etc.
    """
    base = {k: v for k, v in spec.items() if k not in ("labels", "reasoning_toggle")}
    labels = spec.get("labels", {})

    if spec.get("reasoning_toggle"):
        off_label = labels.get("off")
        on_label = labels.get("on")
        if not off_label or not on_label:
            raise RuntimeError(f"reasoning_toggle spec missing off/on labels: {spec}")
        return [
            {**base, "label": off_label, "reasoning": "off"},
            {**base, "label": on_label, "reasoning": "on"},
        ]

    single = labels.get("single")
    if not single:
        raise RuntimeError(f"Model spec missing single label: {spec}")
    return [{**base, "label": single}]


def build_model_configs() -> list[dict[str, Any]]:
    """
    Build flat MODEL_CONFIGS from _MODEL_SPECS.

    Returns:
        Expanded registry list (reasoning-capable models get separate off/on entries).
    """
    configs: list[dict[str, Any]] = []
    for spec in _MODEL_SPECS:
        configs.extend(_expand_spec_entry(spec))
    return configs


def validate_reasoning_pairs(configs: list[dict[str, Any]] | None = None) -> None:
    """
    Assert every reasoning_toggle spec produced both off and on labels.

    Args:
        configs: Registry to validate (default: MODEL_CONFIGS).

    Raises:
        RuntimeError: If a toggle spec is missing an off or on entry.
    """
    registry = configs if configs is not None else MODEL_CONFIGS
    labels = {c["label"] for c in registry}
    for spec in _MODEL_SPECS:
        if not spec.get("reasoning_toggle"):
            continue
        off_label = spec["labels"]["off"]
        on_label = spec["labels"]["on"]
        if off_label not in labels or on_label not in labels:
            raise RuntimeError(
                f"Missing reasoning pair for spec {spec['model_id']}: "
                f"expected {off_label!r} and {on_label!r}"
            )


# Production model registry: label is the canonical id passed through the pipeline.
# route: bedrock | anthropic
# reasoning: None | "on" | "off" — Bedrock additionalModelRequestFields via get_reasoning_fields
MODEL_CONFIGS: list[dict[str, Any]] = build_model_configs()
REGISTRY_LABELS: list[str] = [c["label"] for c in MODEL_CONFIGS]
validate_reasoning_pairs(MODEL_CONFIGS)


def get_reasoning_fields(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """
    Map a model registry entry to Bedrock additionalModelRequestFields.

    Args:
        cfg: Model registry entry with optional reasoning and per-model overrides.

    Returns:
        Dict for additionalModelRequestFields, or None when not applicable.
    """
    reasoning = cfg.get("reasoning")
    if reasoning is None:
        return None
    if reasoning == "on":
        override = cfg.get("reasoning_fields_on")
        return override if override is not None else BEDROCK_REASONING_EFFORT_ON
    if reasoning == "off":
        override = cfg.get("reasoning_fields_off")
        return override if override is not None else BEDROCK_REASONING_EFFORT_OFF
    return None


def build_ephemeral_config(
    label: str,
    route: str,
    model_id: str,
    reasoning: str = "none",
    price_in: float = 0.0,
    price_out: float = 0.0,
    max_tokens: int | None = None,
    read_timeout_sec: int | None = None,
) -> dict[str, Any]:
    """
    Build a one-off registry entry for ad-hoc CLI runs.

    Args:
        label: Trajectory/summary slug.
        route: API route (bedrock | anthropic).
        model_id: Provider model id.
        reasoning: "on" | "off" | "none".
        price_in: USD per 1M input tokens.
        price_out: USD per 1M output tokens.
        max_tokens: Optional max output tokens.
        read_timeout_sec: Optional HTTP read timeout.

    Returns:
        Registry dict suitable for register_ephemeral_config.

    Raises:
        ValueError: If route or reasoning is invalid.
    """
    route_key = (route or "").strip().lower()
    if route_key not in ("bedrock", "anthropic"):
        raise ValueError(f"Unsupported route {route!r}. Use bedrock or anthropic.")

    reasoning_key = (reasoning or "none").strip().lower()
    if reasoning_key not in ("on", "off", "none"):
        raise ValueError(f"Invalid reasoning {reasoning!r}. Use on, off, or none.")

    cfg: dict[str, Any] = {
        "label": label.strip(),
        "route": route_key,
        "model_id": model_id.strip(),
        "reasoning": None if reasoning_key == "none" else reasoning_key,
        "price_in": price_in,
        "price_out": price_out,
    }
    if reasoning_key in ("on", "off"):
        cfg["reasoning_fields_off"] = BEDROCK_REASONING_EFFORT_OFF
        cfg["reasoning_fields_on"] = BEDROCK_REASONING_EFFORT_ON
    if max_tokens is not None:
        cfg["max_tokens"] = max_tokens
    if read_timeout_sec is not None:
        cfg["read_timeout_sec"] = read_timeout_sec
    return cfg


def register_ephemeral_config(cfg: dict[str, Any]) -> str:
    """
    Register a session-scoped ad-hoc model config.

    Args:
        cfg: Registry dict with at least label, route, model_id.

    Returns:
        The registered label.
    """
    label = str(cfg.get("label", "")).strip()
    if not label:
        raise ValueError("ephemeral config requires a non-empty label")
    _EPHEMERAL_CONFIGS[label] = cfg
    return label


def clear_ephemeral_configs() -> None:
    """Clear all session-scoped ad-hoc model configs."""
    _EPHEMERAL_CONFIGS.clear()


def get_model_config(label: str) -> dict[str, Any] | None:
    """
    Look up a model registry entry by label (ephemeral first, then MODEL_CONFIGS).

    Args:
        label: Canonical model label (e.g. claude-haiku-4.5).

    Returns:
        Config dict or None if not found.
    """
    key = (label or "").strip()
    if not key:
        return None
    if key in _EPHEMERAL_CONFIGS:
        return _EPHEMERAL_CONFIGS[key]
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
