"""Reflexion strategy with verbal memory accumulation."""

from __future__ import annotations

from typing import Any

from config import PASS_THRESHOLD
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


def _build_reflection_prompt(problem: str, code: str, error_types: list[str]) -> str:
    """Prompt model for concise lesson learned."""
    errors = "\n".join(error_types[:5]) or "No errors captured"
    return f"""Problem:
{problem}

Current code:
```python
{code}
```

Observed failures:
{errors}

Reflect on what went wrong. Summarize one concrete fix insight in 1-2 sentences."""


def _build_reflexion_repair_prompt(
    problem: str,
    code: str,
    test_results: dict[str, Any],
    reflections: list[str],
    problem_dict: dict[str, Any] | None = None,
) -> str:
    """Extend direct-fix prompt with accumulated reflections."""
    memory = "\n".join(f"- {r}" for r in reflections[-5:]) if reflections else "- (none yet)"
    return (
        build_repair_prompt(problem, code, test_results, problem_dict=problem_dict)
        + f"\n\nPast reflections:\n{memory}\n"
        + "Use these reflections to avoid repeated mistakes."
    )


class ReflexionStrategy:
    """Direct-fix loop augmented with reflection memory."""

    strategy_name = "reflexion"

    def run(
        self,
        task_id: str,
        problem: str,
        model: str,
        problem_dict: dict[str, Any],
        max_eval_steps: int,
    ) -> list[dict[str, Any]]:
        """Run reflexion strategy and emit normalized steps."""
        out_path = get_trajectory_path(model, self.strategy_name, task_id)
        meta_path = get_metadata_path(model, self.strategy_name, task_id)
        if is_complete(out_path, max_eval_steps):
            return load_jsonl(out_path)

        trajectory: list[dict[str, Any]] = []
        reflections: list[str] = []
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

            latest_reflection = ""
            if emission.test_results["pass_rate"] < PASS_THRESHOLD:
                latest_reflection = (
                    call_llm(
                        _build_reflection_prompt(
                            problem,
                            code,
                            emission.test_results.get("error_types", []),
                        ),
                        model,
                    )
                    or ""
                ).strip()
                if latest_reflection:
                    reflections.append(latest_reflection)

            append_jsonl(
                meta_path,
                {
                    "step_number": step_number,
                    "reflection": latest_reflection,
                    "accumulated_reflections": reflections[-5:],
                },
            )

            if step_number >= max_eval_steps - 1:
                continue

            repair_prompt = _build_reflexion_repair_prompt(
                problem,
                code,
                emission.test_results,
                reflections,
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
