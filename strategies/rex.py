"""Simplified REx-style strategy using Thompson sampling over candidates."""

from __future__ import annotations

import random
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


def _sample_beta(alpha: float, beta: float) -> float:
    """Draw one sample from Beta posterior."""
    return random.betavariate(alpha, beta)


class RExStrategy:
    """Population-based refinement with Thompson sampling."""

    strategy_name = "rex"

    def run(
        self,
        task_id: str,
        problem: str,
        model: str,
        problem_dict: dict[str, Any],
        max_eval_steps: int,
    ) -> list[dict[str, Any]]:
        """Run REx-style strategy and emit flat evaluated steps."""
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

        seed_code = strip_code_fences(raw)
        population: list[dict[str, Any]] = [
            {"id": idx, "code": seed_code, "alpha": 1.0, "beta": 1.0, "best_pass_rate": 0.0}
            for idx in range(3)
        ]

        for step_number in range(max_eval_steps):
            sampled = [(_sample_beta(c["alpha"], c["beta"]), c["id"]) for c in population]
            thompson_score, selected_id = max(sampled, key=lambda x: x[0])
            selected = next(c for c in population if c["id"] == selected_id)

            emission = build_step_dict(
                trajectory=trajectory,
                task_id=task_id,
                problem=problem,
                model=model,
                strategy=self.strategy_name,
                code=selected["code"],
                step_number=step_number,
                problem_dict=problem_dict,
            )
            step = emission.step
            trajectory.append(step)
            append_jsonl(out_path, step)

            passed = int(emission.test_results.get("passed", 0))
            total = int(emission.test_results.get("total", 1))
            failed = max(total - passed, 0)
            selected["alpha"] += passed
            selected["beta"] += failed
            selected["best_pass_rate"] = max(
                selected["best_pass_rate"], float(emission.test_results.get("pass_rate", 0.0))
            )

            append_jsonl(
                meta_path,
                {
                    "step_number": step_number,
                    "candidate_id": selected_id,
                    "thompson_score": thompson_score,
                    "population_state": [
                        {
                            "id": c["id"],
                            "alpha": c["alpha"],
                            "beta": c["beta"],
                            "best_pass_rate": c["best_pass_rate"],
                        }
                        for c in population
                    ],
                    "selected_for_refinement": True,
                },
            )

            if step_number >= max_eval_steps - 1:
                continue

            repair_prompt = build_repair_prompt(
                problem,
                selected["code"],
                emission.test_results,
                problem_dict=problem_dict,
            )
            raw = call_llm(repair_prompt, model)
            if raw is None:
                raw = call_llm(repair_prompt, model)
            if raw is not None:
                selected["code"] = strip_code_fences(raw)

        return trajectory
