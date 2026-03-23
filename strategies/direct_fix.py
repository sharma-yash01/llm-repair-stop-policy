"""Baseline direct-fix repair strategy (linear loop)."""

from __future__ import annotations

import logging
from typing import Any

from repair import _is_functional_problem, call_llm, generate_initial, strip_code_fences
from strategies.base import (
    build_step_dict,
    append_jsonl,
    get_trajectory_path,
    is_complete,
    load_jsonl,
)

logger = logging.getLogger(__name__)

MAX_REPAIR_HISTORY_MESSAGES = 6
MAX_REPAIR_ERROR_CHARS = 1500


def _trim_history(messages: list[dict[str, str]], max_messages: int) -> list[dict[str, str]]:
    """Keep most recent chat messages and avoid leading assistant role."""
    if len(messages) <= max_messages:
        return messages
    trimmed = messages[-max_messages:]
    if trimmed and trimmed[0].get("role") == "assistant":
        trimmed = trimmed[1:]
    return trimmed


def build_repair_prompt(
    problem: str,
    code: str,
    test_results: dict[str, Any],
    problem_dict: dict[str, Any] | None = None,
) -> str:
    """Build direct-fix prompt while preserving pilot hardening instructions."""

    def _truncate_error(err: Any) -> str:
        text = str(err).strip()
        return text[-MAX_REPAIR_ERROR_CHARS:] if len(text) > MAX_REPAIR_ERROR_CHARS else text

    errors = "\n".join(_truncate_error(err) for err in test_results["error_types"][:5])
    errors = errors or "No errors captured"
    interface_hint = (
        "This problem uses a function-call test harness. Keep the required callable interface."
        if _is_functional_problem(problem_dict)
        else "This problem uses stdin/stdout. Keep reading stdin and writing stdout."
    )
    return f"""Fix this Python solution.

Problem: {problem}
{interface_hint}

Current code ({test_results['passed']}/{test_results['total']} tests pass):
```python
{code}
```

Errors: {errors}

Do not break tests that already pass.
Make only the minimal changes needed to fix failing tests.

Return ONLY the fixed Python code, no explanation."""


class DirectFixStrategy:
    """Simple iterative direct-fix strategy."""

    strategy_name = "direct_fix"

    def run(
        self,
        task_id: str,
        problem: str,
        model: str,
        problem_dict: dict[str, Any],
        max_eval_steps: int,
    ) -> list[dict[str, Any]]:
        """Run linear direct-fix strategy and emit normalized steps."""
        out_path = get_trajectory_path(model, self.strategy_name, task_id)
        if is_complete(out_path, max_eval_steps):
            return load_jsonl(out_path)

        trajectory: list[dict[str, Any]] = []
        raw = generate_initial(problem, model, problem_dict=problem_dict)
        if raw is None:
            logger.warning("llm_null_content problem_id=%s stage=initial", task_id)
            raw = generate_initial(problem, model, problem_dict=problem_dict)
        if raw is None:
            raise RuntimeError(f"Initial LLM returned None for task_id={task_id}")

        code = strip_code_fences(raw)
        next_step_llm_null = False
        message_history: list[dict[str, str]] = []

        for step_number in range(max_eval_steps):
            emission = build_step_dict(
                trajectory=trajectory,
                task_id=task_id,
                problem=problem,
                model=model,
                strategy=self.strategy_name,
                code=code,
                step_number=step_number,
                problem_dict=problem_dict,
                llm_null_response=next_step_llm_null,
            )
            step = emission.step
            trajectory.append(step)
            append_jsonl(out_path, step)

            if step_number >= max_eval_steps - 1:
                continue

            repair_prompt = build_repair_prompt(
                problem,
                code,
                emission.test_results,
                problem_dict=problem_dict,
            )
            message_history.append({"role": "user", "content": repair_prompt})
            request_history = _trim_history(message_history, MAX_REPAIR_HISTORY_MESSAGES)
            raw = call_llm(repair_prompt, model, messages=request_history)
            if raw is None:
                raw = call_llm(repair_prompt, model, messages=request_history)

            if raw is None:
                if message_history and message_history[-1].get("role") == "user":
                    message_history.pop()
                next_step_llm_null = True
                continue

            code = strip_code_fences(raw)
            message_history.append({"role": "assistant", "content": code})
            message_history = _trim_history(message_history, MAX_REPAIR_HISTORY_MESSAGES)
            next_step_llm_null = False

        return trajectory
