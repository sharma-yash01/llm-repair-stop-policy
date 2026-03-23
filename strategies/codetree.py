"""Simplified CodeTree-style branching strategy."""

from __future__ import annotations

import uuid
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


def _think_fix_strategies(problem: str, code: str, errors: list[str], model: str) -> list[str]:
    """Produce 2-3 candidate fix directions."""
    error_text = "\n".join(errors[:5]) or "No errors captured"
    prompt = (
        "Given the problem, code, and failures, propose 3 distinct fix strategies.\n"
        "Return one strategy per line.\n\n"
        f"Problem:\n{problem}\n\nCode:\n```python\n{code}\n```\n\nErrors:\n{error_text}"
    )
    raw = call_llm(prompt, model) or ""
    lines = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip()]
    return lines[:3] or ["Apply minimal bug fix guided by failing tests."]


def _solver_generate(problem: str, code: str, strategy_text: str, model: str) -> str:
    """Generate candidate code for a chosen fix strategy."""
    prompt = (
        "Apply the following fix strategy to the current code.\n"
        "Return ONLY updated Python code.\n\n"
        f"Strategy: {strategy_text}\n\nProblem:\n{problem}\n\nCurrent code:\n```python\n{code}\n```"
    )
    raw = call_llm(prompt, model)
    return strip_code_fences(raw or code)


class CodeTreeStrategy:
    """BFS-like branching strategy using thinker/solver/critic decomposition."""

    strategy_name = "codetree"

    def run(
        self,
        task_id: str,
        problem: str,
        model: str,
        problem_dict: dict[str, Any],
        max_eval_steps: int,
    ) -> list[dict[str, Any]]:
        """Run simplified code tree search and emit flat evaluated steps."""
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

        root_code = strip_code_fences(raw)
        frontier: list[dict[str, Any]] = [
            {"branch_id": "root", "parent_branch_id": None, "code": root_code, "depth": 0}
        ]
        step_number = 0

        while step_number < max_eval_steps and frontier:
            node = frontier.pop(0)
            emission = build_step_dict(
                trajectory=trajectory,
                task_id=task_id,
                problem=problem,
                model=model,
                strategy=self.strategy_name,
                code=node["code"],
                step_number=step_number,
                problem_dict=problem_dict,
            )
            step = emission.step
            trajectory.append(step)
            append_jsonl(out_path, step)
            append_jsonl(
                meta_path,
                {
                    "step_number": step_number,
                    "branch_id": node["branch_id"],
                    "parent_branch_id": node["parent_branch_id"],
                    "agent_role": "solver",
                    "fix_strategy_text": node.get("strategy_text", "initial"),
                    "tree_depth": node["depth"],
                    "siblings_count": len(frontier),
                },
            )
            step_number += 1
            if step_number >= max_eval_steps:
                break

            strategy_texts = _think_fix_strategies(
                problem,
                node["code"],
                emission.test_results.get("error_types", []),
                model,
            )
            children: list[dict[str, Any]] = []
            for strategy_text in strategy_texts:
                child_code = _solver_generate(problem, node["code"], strategy_text, model)
                children.append(
                    {
                        "branch_id": str(uuid.uuid4())[:8],
                        "parent_branch_id": node["branch_id"],
                        "code": child_code,
                        "depth": node["depth"] + 1,
                        "strategy_text": strategy_text,
                    }
                )

            # Critic: rank by cheap heuristic (shorter error-prone code not preferred).
            children.sort(key=lambda c: len(c["code"]), reverse=False)
            frontier.extend(children[:2])

        return trajectory
