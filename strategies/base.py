"""Shared strategy interfaces and trajectory/metadata utilities."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from config import DATA_DIR, STRATEGY_METADATA_DIR
from evaluate import run_tests
from features import compute_ast_levenshtein, extract_features
from repair import get_self_verification_score


def slugify_model(model: str) -> str:
    """Convert model id to filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model).strip("_") or "default_model"


def get_trajectory_path(model: str, strategy: str, task_id: str) -> str:
    """Return trajectory JSONL path for model/strategy/problem."""
    model_slug = slugify_model(model)
    task_slug = task_id.replace("/", "_")
    return os.path.join(DATA_DIR, model_slug, strategy, f"{task_slug}.jsonl")


def get_metadata_path(model: str, strategy: str, task_id: str) -> str:
    """Return strategy-metadata JSONL path for model/strategy/problem."""
    model_slug = slugify_model(model)
    task_slug = task_id.replace("/", "_")
    return os.path.join(STRATEGY_METADATA_DIR, model_slug, strategy, f"{task_slug}.jsonl")


def append_jsonl(path: str, payload: dict[str, Any]) -> None:
    """Append one JSON object as a line."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load JSONL file into memory."""
    items: list[dict[str, Any]] = []
    if not os.path.isfile(path):
        return items
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def is_complete(path: str, max_steps: int) -> bool:
    """Return True when trajectory has at least max_steps entries."""
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) >= max_steps


@dataclass
class StepEmission:
    """Container for step emission output."""

    step: dict[str, Any]
    test_results: dict[str, Any]


def build_step_dict(
    *,
    trajectory: list[dict[str, Any]],
    task_id: str,
    problem: str,
    model: str,
    strategy: str,
    code: str,
    step_number: int,
    problem_dict: dict[str, Any],
    llm_null_response: bool = False,
) -> StepEmission:
    """Evaluate code and build a normalized trajectory step dict."""
    test_results = run_tests(task_id, code, problem_dict)
    sv_score = get_self_verification_score(problem, code, model)

    prev_code = trajectory[-1]["code"] if trajectory else ""
    base = {
        "problem_id": task_id,
        "step_number": step_number,
        "iteration": step_number,
        "code": code,
        "pass_rate": test_results["pass_rate"],
        "passed": test_results["passed"],
        "total": test_results["total"],
        "error_types": test_results["error_types"],
        "patch_delta": compute_ast_levenshtein(prev_code, code),
        "self_verification_score": sv_score,
        "timestamp": time.time(),
        "strategy": strategy,
        "model": model,
        "llm_null_response": llm_null_response,
    }

    temp_trajectory = trajectory + [base]
    base["features"] = extract_features(temp_trajectory, step_number)
    return StepEmission(step=base, test_results=test_results)


class StrategyRunner(Protocol):
    """Protocol that every strategy implementation must satisfy."""

    strategy_name: str

    def run(
        self,
        task_id: str,
        problem: str,
        model: str,
        problem_dict: dict[str, Any],
        max_eval_steps: int,
    ) -> list[dict[str, Any]]:
        """Run strategy and return normalized trajectory steps."""
