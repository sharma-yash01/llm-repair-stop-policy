"""All constants for the pilot. Do not hardcode these elsewhere."""

import os

# Provider selection:
# - "auto": choose from available API keys (recommended)
# - "gemini": force Google AI Studio Gemini endpoint
# - "openrouter": force OpenRouter endpoint
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto").strip().lower()

# Optional explicit model override via env var MODEL.
# When empty, provider-specific default model is used.
MODEL = os.environ.get("MODEL", "").strip()

# Provider defaults (used when MODEL is empty).
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# OpenAI-compatible endpoints
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# API key env var names
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

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

DATA_DIR = "data/trajectories"
STRATEGY_METADATA_DIR = "data/strategy_metadata"
RESULTS_DIR = "data/results"
FIGURES_DIR = "data/figures"

BOOTSTRAP_N_RESAMPLES = 1000
COST_HARD_STOP_USD = 200.0
