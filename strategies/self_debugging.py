"""Self-debugging strategy with explain-then-fix prompting."""

from __future__ import annotations

import re
from typing import Any

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


def _split_explanation_and_code(raw: str) -> tuple[str, str]:
    """Extract explanation and code block from a mixed response."""
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    match = pattern.search(raw)
    if not match:
        return raw.strip(), strip_code_fences(raw)
    code = match.group(1).strip()
    explanation = raw[: match.start()].strip()
    return explanation, code


class SelfDebuggingStrategy:
    """Explain-first then patch strategy."""

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

            repair_prompt = (
                build_repair_prompt(problem, code, emission.test_results, problem_dict=problem_dict)
                + "\n\nFirst explain what went wrong and why in 2-3 sentences. "
                + "Then write the corrected code."
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
                append_jsonl(meta_path, {"step_number": step_number, "explanation": "", "null_response": True})
                continue

            explanation, code = _split_explanation_and_code(raw)
            append_jsonl(meta_path, {"step_number": step_number, "explanation": explanation, "null_response": False})
            message_history.append({"role": "assistant", "content": code})
            message_history = _trim_history(message_history, MAX_REPAIR_HISTORY_MESSAGES)
            next_step_llm_null = False

        return trajectory
