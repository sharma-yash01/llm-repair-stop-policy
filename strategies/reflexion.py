"""Reflexion-faithful verbal reflection over a linear, ground-truth-evaluated loop.

Harness-adapted from Shinn et al. (NeurIPS 2023, arXiv:2303.11366): separate
self-reflection (M_sr) with bounded episodic memory, conditioned on prior
reflections + execution feedback. Intentional deviations for characterization
harness: ground-truth tests as evaluator (not self-generated tests) and no
episodic trial/reset (linear fixed-step loop).
"""

from __future__ import annotations

from typing import Any

from config import PASS_THRESHOLD, REFLEXION_MEMORY_LIMIT
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


def _memory_window(reflections: list[str]) -> list[str]:
    """Return the bounded reflection memory window (paper Omega=1-3)."""
    if REFLEXION_MEMORY_LIMIT <= 0:
        return []
    return reflections[-REFLEXION_MEMORY_LIMIT:]


def _build_reflection_prompt(
    problem: str,
    code: str,
    test_results: dict[str, Any],
    prior_reflections: list[str],
) -> str:
    """Paper-faithful self-reflection prompt (M_sr): credit assignment + memory."""
    errors = "\n".join(
        _truncate_error(err) for err in test_results.get("error_types", [])[:5]
    )
    errors = errors or "No errors captured"
    passed = test_results.get("passed", 0)
    total = test_results.get("total", 0)
    memory = "\n".join(f"- {r}" for r in _memory_window(prior_reflections))
    memory_block = memory if memory else "- (none yet)"
    return f"""You are reflecting on a failed code repair attempt.

Problem:
{problem}

Current code:
```python
{code}
```

Execution result: {passed}/{total} tests pass.
Observed failures:
{errors}

Past reflections from earlier attempts:
{memory_block}

Write a first-person reflection in 2-4 sentences. Identify what specifically
caused the failure (credit assignment), what you should do differently on the
next attempt, and how to avoid repeating mistakes noted in past reflections.
Do not write code."""


def _build_reflexion_repair_prompt(
    problem: str,
    code: str,
    test_results: dict[str, Any],
    reflections: list[str],
    problem_dict: dict[str, Any] | None = None,
) -> str:
    """Extend direct-fix prompt with bounded reflection memory."""
    memory = "\n".join(f"- {r}" for r in _memory_window(reflections))
    memory_block = memory if memory else "- (none yet)"
    return (
        build_repair_prompt(problem, code, test_results, problem_dict=problem_dict)
        + f"\n\nPast reflections:\n{memory_block}\n"
        + "Use these reflections to avoid repeated mistakes."
    )


class ReflexionStrategy:
    """Reflexion-faithful verbal reflection + bounded memory over linear repair."""

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
                            emission.test_results,
                            reflections,
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
                    "accumulated_reflections": _memory_window(reflections),
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
