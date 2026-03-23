"""
Entrypoint: load HumanEval+, run repair loops, compute pilot metrics, save summary.
"""

from __future__ import annotations

import json
import os
import time

from tqdm import tqdm

from analyze import compute_ece, compute_feature_auc, compute_waste_rate, print_summary
from config import DATA_DIR, MODEL, N_PROBLEMS, PASS_THRESHOLD, RESULTS_DIR
from data_lcb import get_problems
from strategies.direct_fix import DirectFixStrategy


def _classify_problem_outcomes(
    trajectories: list[list[dict]],
) -> tuple[list[str], list[str], list[str]]:
    """
    Classify trajectory outcomes by first passing iteration.

    Args:
        trajectories: Per-problem trajectory steps from repair loops.

    Returns:
        Tuple of (solved_at_0, solved_later, never_passed) problem IDs.
    """
    solved_at_0: list[str] = []
    solved_later: list[str] = []
    never_passed: list[str] = []

    for traj in trajectories:
        if not traj:
            continue
        problem_id = str(traj[0].get("problem_id", "unknown"))
        first_passing_iter = next(
            (step["iteration"] for step in traj if step["pass_rate"] >= PASS_THRESHOLD),
            None,
        )
        if first_passing_iter is None:
            never_passed.append(problem_id)
        elif first_passing_iter == 0:
            solved_at_0.append(problem_id)
        else:
            solved_later.append(problem_id)

    return solved_at_0, solved_later, never_passed


def main():
    """Run pilot: repair loops on first N_PROBLEMS (LiveCodeBench), then compute and print metrics."""
    # Archive trajectory data only when explicitly requested (preserves crash-resume).
    if os.environ.get("PILOT_FORCE_CLEAN") == "1" and os.path.isdir(DATA_DIR):
        jsonl_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")]
        if jsonl_files:
            backup = f"data/trajectories_poisoned_{int(time.time())}"
            os.rename(DATA_DIR, backup)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n = int(os.environ.get("PILOT_N_PROBLEMS", N_PROBLEMS))
    problems = get_problems()[:n]
    strategy = DirectFixStrategy()
    trajectories = [
        strategy.run(
            task_id=p["task_id"],
            problem=p["prompt"],
            model=MODEL,
            problem_dict=p,
            max_eval_steps=5,
        )
        for p in tqdm(problems, desc="Repair loops")
    ]

    waste = compute_waste_rate(trajectories)
    ece = compute_ece(trajectories)
    auc_mean, auc_std = compute_feature_auc(trajectories)
    problems = [str(traj[0].get("problem_id", "unknown")) for traj in trajectories if traj]
    solved_at_0, solved_later, never_passed = _classify_problem_outcomes(trajectories)
    n_solved = len(solved_at_0) + len(solved_later)

    print_summary(waste, ece, auc_mean, auc_std, trajectories)
    with open(f"{RESULTS_DIR}/pilot_summary.json", "w") as f:
        json.dump(
            {
                "waste_rate": waste,
                "ece": ece,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "problems": problems,
                "solved_at_0": solved_at_0,
                "solved_later": solved_later,
                "never_passed": never_passed,
                "n_problems": len(problems),
                "n_solved": n_solved,
                "n_never_passed": len(never_passed),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
