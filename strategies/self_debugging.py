"""Self-Debugging (UT + rubber-duck code explanation), zero-shot, fixed-budget.

Harness-adapted from Chen et al. (ICLR 2024, arXiv:2304.05128): separate code
explanation step then UT-feedback revision. Uses ground-truth test feedback and a
linear fixed-step loop (not paper's early-stop on self-judged correctness).
"""

from __future__ import annotations

import re
from typing import Any

from config import SELF_DEBUG_SEPARATE_EXPLANATION
from repair import call_llm, generate_initial, strip_code_fences
from strategies.base import (
    append_jsonl,
    build_step_dict,
    get_metadata_path,
    get_trajectory_path,
    is_complete,
    load_jsonl,
)
from strategies.direct_fix import MAX_REPAIR_HISTORY_MESSAGES, _trim_history, build_repair_prompt


def _truncate_error(err: Any, max_chars: int = 1500) -> str:
    """Truncate long error strings for prompts."""
    text = str(err).strip()
    return text[-max_chars:] if len(text) > max_chars else text


def _build_code_explanation_prompt(
    problem: str,
    code: str,
    test_results: dict[str, Any],
) -> str:
    """Prompt for rubber-duck code explanation (paper Explanation step)."""
    errors = "\n".join(_truncate_error(err) for err in test_results.get("error_types", [])[:5])
    errors = errors or "No errors captured"
    passed = test_results.get("passed", 0)
    total = test_results.get("total", 0)
    return f"""Problem:
{problem}

Current code:
```python
{code}
```

Execution result: {passed}/{total} tests pass.
Observed failures:
{errors}

Explain the current code line by line: describe what each part does and how it
relates to the problem specification. Compare your explanation to what the
problem requires and identify any discrepancy between intended and actual
behavior. Do not write corrected code yet."""


def _build_revision_prompt(
    problem: str,
    code: str,
    test_results: dict[str, Any],
    code_explanation: str,
    problem_dict: dict[str, Any] | None = None,
) -> str:
    """UT feedback revision prompt augmented with the model's code explanation."""
    base = build_repair_prompt(problem, code, test_results, problem_dict=problem_dict)
    return (
        f"{base}\n\nYour prior code explanation:\n{code_explanation}\n\n"
        "Using your explanation and the test feedback above, write the corrected "
        "Python code. Return ONLY the fixed Python code, no explanation."
    )


def _split_explanation_and_code(raw: str) -> tuple[str, str]:
    """Extract explanation and code block from a mixed response (fallback path)."""
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    match = pattern.search(raw)
    if not match:
        return raw.strip(), strip_code_fences(raw)
    code = match.group(1).strip()
    explanation = raw[: match.start()].strip()
    return explanation, code


class SelfDebuggingStrategy:
    """Self-Debugging (UT + rubber-duck code explanation), zero-shot, fixed-budget."""

    strategy_name = "self_debugging"

    def run(
        self,
        task_id: str,
        problem: str,
        model: str,
        problem_dict: dict[str, Any],
        max_eval_steps: int,
    ) -> list[dict[str, Any]]:
        """Run self-debugging strategy and emit normalized steps."""
        out_path = get_trajectory_path(model, self.strategy_name, task_id)
        meta_path = get_metadata_path(model, self.strategy_name, task_id)
        if is_complete(out_path, max_eval_steps):
            return load_jsonl(out_path)

        trajectory: list[dict[str, Any]] = []
        raw = generate_initial(problem, model, problem_dict=problem_dict)
        if raw is None:
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

            code_explanation = ""
            if SELF_DEBUG_SEPARATE_EXPLANATION:
                explanation_prompt = _build_code_explanation_prompt(
                    problem,
                    code,
                    emission.test_results,
                )
                code_explanation = (call_llm(explanation_prompt, model) or "").strip()
                append_jsonl(
                    meta_path,
                    {
                        "step_number": step_number,
                        "phase": "explanation",
                        "code_explanation": code_explanation,
                        "separate_explanation": True,
                    },
                )
                repair_prompt = _build_revision_prompt(
                    problem,
                    code,
                    emission.test_results,
                    code_explanation or "(no explanation generated)",
                    problem_dict=problem_dict,
                )
            else:
                repair_prompt = (
                    build_repair_prompt(
                        problem,
                        code,
                        emission.test_results,
                        problem_dict=problem_dict,
                    )
                    + "\n\nFirst explain the current code line by line, then write "
                    + "the corrected code."
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
                append_jsonl(
                    meta_path,
                    {
                        "step_number": step_number,
                        "phase": "revision",
                        "code_explanation": code_explanation,
                        "null_response": True,
                    },
                )
                continue

            if SELF_DEBUG_SEPARATE_EXPLANATION:
                revised_code = strip_code_fences(raw)
            else:
                explanation, revised_code = _split_explanation_and_code(raw)
                code_explanation = explanation
            append_jsonl(
                meta_path,
                {
                    "step_number": step_number,
                    "phase": "revision",
                    "code_explanation": code_explanation,
                    "null_response": False,
                },
            )
            code = revised_code
            message_history.append({"role": "assistant", "content": code})
            message_history = _trim_history(message_history, MAX_REPAIR_HISTORY_MESSAGES)
            next_step_llm_null = False

        return trajectory
