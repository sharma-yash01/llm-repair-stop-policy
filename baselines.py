"""Post-hoc stopping baselines computed from saved trajectories."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from config import BOOTSTRAP_N_RESAMPLES, IMPROVEMENT_THRESHOLD, PASS_THRESHOLD

BASELINES = (
    "fixed_3",
    "fixed_5",
    "stop_on_success",
    "stop_on_plateau",
    "stop_on_duplicate",
    "oracle_first",
)


def _oracle_stop_step(trajectory: list[dict[str, Any]]) -> int:
    for i, step in enumerate(trajectory):
        if step.get("pass_rate", 0.0) >= PASS_THRESHOLD:
            return i
    return max(len(trajectory) - 1, 0)


def _stop_step_for_baseline(trajectory: list[dict[str, Any]], baseline: str) -> int:
    if not trajectory:
        return 0
    n = len(trajectory)
    if baseline == "fixed_3":
        return min(3, n - 1)
    if baseline == "fixed_5":
        return n - 1
    if baseline == "oracle_first":
        return _oracle_stop_step(trajectory)
    if baseline == "stop_on_success":
        for i, step in enumerate(trajectory):
            if float(step.get("self_verification_score", 0.0)) > 0.9:
                return i
        return n - 1
    if baseline == "stop_on_duplicate":
        streak = 0
        for i, step in enumerate(trajectory):
            if int(step.get("patch_delta", 0)) < 5:
                streak += 1
            else:
                streak = 0
            if streak >= 2:
                return i
        return n - 1
    if baseline == "stop_on_plateau":
        streak = 0
        for i in range(1, n):
            delta = trajectory[i]["pass_rate"] - trajectory[i - 1]["pass_rate"]
            if delta < IMPROVEMENT_THRESHOLD:
                streak += 1
            else:
                streak = 0
            if streak >= 2:
                return i
        return n - 1
    raise ValueError(f"Unknown baseline: {baseline}")


def apply_baseline(trajectory: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    """Simulate a stopping policy on one trajectory."""
    if not trajectory:
        return {
            "stop_step": 0,
            "final_code": "",
            "final_pass_rate": 0.0,
            "steps_used": 0,
            "waste_rate": 0.0,
            "oracle_stop_step": 0,
        }
    stop_step = _stop_step_for_baseline(trajectory, baseline)
    oracle = _oracle_stop_step(trajectory)
    total_steps = len(trajectory)
    tail_denom = max(total_steps - 1, 1)
    waste = max(stop_step - oracle, 0) / tail_denom
    chosen = trajectory[stop_step]
    return {
        "stop_step": stop_step,
        "final_code": chosen.get("code", ""),
        "final_pass_rate": float(chosen.get("pass_rate", 0.0)),
        "steps_used": stop_step + 1,
        "waste_rate": waste,
        "oracle_stop_step": oracle,
    }


def _bootstrap_ci(values: list[float], n_resamples: int = BOOTSTRAP_N_RESAMPLES) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[random.randint(0, len(values) - 1)] for _ in range(len(values))]
        means.append(float(np.mean(sample)))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def evaluate_all_baselines(trajectories: list[list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Compute baseline metrics on all trajectories with bootstrap CIs."""
    oracle_stats = [apply_baseline(traj, "oracle_first") for traj in trajectories if traj]
    oracle_pass = float(np.mean([s["final_pass_rate"] for s in oracle_stats])) if oracle_stats else 0.0

    summary: dict[str, dict[str, Any]] = {}
    for baseline in BASELINES:
        stats = [apply_baseline(traj, baseline) for traj in trajectories if traj]
        if not stats:
            summary[baseline] = {
                "waste_rate": 0.0,
                "final_pass_rate": 0.0,
                "steps_used": 0.0,
                "compute_savings": 0.0,
                "regret_vs_oracle": 0.0,
                "waste_ci_95": (0.0, 0.0),
                "pass_ci_95": (0.0, 0.0),
                "steps_ci_95": (0.0, 0.0),
            }
            continue

        waste_values = [s["waste_rate"] for s in stats]
        pass_values = [s["final_pass_rate"] for s in stats]
        step_values = [float(s["steps_used"]) for s in stats]
        mean_steps = float(np.mean(step_values))
        max_steps = max((len(traj) for traj in trajectories if traj), default=1)
        summary[baseline] = {
            "waste_rate": float(np.mean(waste_values)),
            "final_pass_rate": float(np.mean(pass_values)),
            "steps_used": mean_steps,
            "compute_savings": 1.0 - (mean_steps / max_steps),
            "regret_vs_oracle": oracle_pass - float(np.mean(pass_values)),
            "waste_ci_95": _bootstrap_ci(waste_values),
            "pass_ci_95": _bootstrap_ci(pass_values),
            "steps_ci_95": _bootstrap_ci(step_values),
            "per_problem": stats,
        }
    return summary
