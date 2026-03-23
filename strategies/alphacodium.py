"""Lightweight AlphaCodium-style multi-phase strategy."""

from __future__ import annotations

import json
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
from strategies.direct_fix import build_repair_prompt


def _problem_reflection(problem: str, model: str) -> str:
    """Generate short plan and edge-case reflection."""
    prompt = (
        "Read the programming problem below and provide:\n"
        "1) core algorithm idea\n2) key edge cases\n3) likely implementation pitfalls.\n"
        "Keep it concise.\n\n"
        f"{problem}"
    )
    return (call_llm(prompt, model) or "").strip()


def _generate_ai_tests(problem: str, model: str) -> list[str]:
    """Generate candidate tests as plain text cases."""
    prompt = (
        "Generate 3 concise test cases for this programming problem.\n"
        'Return JSON: {"tests":[...]} where each test is a string.'
        f"\n\nProblem:\n{problem}"
    )
    raw = call_llm(prompt, model) or ""
    try:
        parsed = json.loads(raw)
        tests = parsed.get("tests", [])
        return [str(t) for t in tests][:3]
    except Exception:
        lines = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip()]
        return lines[:3]


def _score_ai_tests(code: str, ai_tests: list[str]) -> float:
    """Heuristic AI-test score for metadata/debugging (not primary metric)."""
    if not ai_tests:
        return 0.0
    score = 0
    for test in ai_tests:
        if any(tok in code for tok in ("if", "for", "while", "def")) and len(test) > 0:
            score += 1
    return score / len(ai_tests)


class AlphaCodiumStrategy:
    """Multi-phase strategy inspired by AlphaCodium flow."""

    strategy_name = "alphacodium"

    def run(
        self,
        task_id: str,
        problem: str,
        model: str,
        problem_dict: dict[str, Any],
        max_eval_steps: int,
    ) -> list[dict[str, Any]]:
        """Run multi-phase strategy and emit normalized flat evaluation steps."""
        out_path = get_trajectory_path(model, self.strategy_name, task_id)
        meta_path = get_metadata_path(model, self.strategy_name, task_id)
        if is_complete(out_path, max_eval_steps):
            return load_jsonl(out_path)

        trajectory: list[dict[str, Any]] = []
        reflection = _problem_reflection(problem, model)
        ai_tests = _generate_ai_tests(problem, model)
        append_jsonl(
            meta_path,
            {
                "step_number": -1,
                "phase": "setup",
                "problem_reflection": reflection,
                "ai_generated_tests": ai_tests,
            },
        )

        raw = generate_initial(problem, model, problem_dict=problem_dict)
        if raw is None:
            raw = generate_initial(problem, model, problem_dict=problem_dict)
        if raw is None:
            raise RuntimeError(f"Initial LLM returned None for task_id={task_id}")
        code = strip_code_fences(raw)

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
            )
            step = emission.step
            trajectory.append(step)
            append_jsonl(out_path, step)

            ai_test_pass_rate = _score_ai_tests(code, ai_tests)
            append_jsonl(
                meta_path,
                {
                    "step_number": step_number,
                    "phase": "initial" if step_number == 0 else "refinement",
                    "problem_reflection": reflection,
                    "ai_generated_tests": ai_tests,
                    "ai_test_pass_rate": ai_test_pass_rate,
                },
            )

            if step_number >= max_eval_steps - 1:
                continue

            repair_prompt = (
                build_repair_prompt(problem, code, emission.test_results, problem_dict=problem_dict)
                + "\n\nAdditional generated stress tests to consider:\n"
                + "\n".join(f"- {t}" for t in ai_tests)
                + "\nUse these tests while preserving correctness on official tests."
            )
            raw = call_llm(repair_prompt, model)
            if raw is None:
                raw = call_llm(repair_prompt, model)
            if raw is not None:
                code = strip_code_fences(raw)

        return trajectory
