"""Pilot and production analysis utilities."""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
from scipy.stats import chi2

from baselines import BASELINES, evaluate_all_baselines
from config import BOOTSTRAP_N_RESAMPLES, IMPROVEMENT_THRESHOLD, PASS_THRESHOLD


def compute_waste_rate(trajectories: list[list[dict[str, Any]]]) -> float:
    """Mean waste rate where waste starts after oracle-first stop."""
    wastes: list[float] = []
    for traj in trajectories:
        if not traj:
            continue
        oracle = next((i for i, s in enumerate(traj) if s["pass_rate"] >= PASS_THRESHOLD), len(traj) - 1)
        denom = max(len(traj) - 1, 1)
        wastes.append((len(traj) - 1 - oracle) / denom)
    return float(np.mean(wastes)) if wastes else 0.0


def compute_ece(trajectories: list[list[dict[str, Any]]], n_bins: int = 15) -> float:
    """Expected calibration error on self-verification scores."""
    confs: list[float] = []
    outcomes: list[int] = []
    for traj in trajectories:
        for step in traj:
            confs.append(float(step.get("self_verification_score", 0.5)))
            outcomes.append(int(step.get("pass_rate", 0.0) >= PASS_THRESHOLD))
    if not confs:
        return 0.0

    conf_arr = np.array(confs)
    out_arr = np.array(outcomes)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (conf_arr >= lo) & (conf_arr < hi)
        else:
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / len(conf_arr)) * abs(conf_arr[mask].mean() - out_arr[mask].mean())
    return float(ece)


def compute_feature_auc(trajectories: list[list[dict[str, Any]]]) -> tuple[float, float]:
    """AUC for predicting next-step improvement from extracted features."""
    X: list[list[float]] = []
    y: list[int] = []
    for traj in trajectories:
        for i in range(len(traj) - 1):
            features = traj[i].get("features", {})
            if not features:
                continue
            X.append([float(v) for v in features.values()])
            y.append(int(traj[i + 1]["pass_rate"] - traj[i]["pass_rate"] >= IMPROVEMENT_THRESHOLD))

    if not y:
        return 0.5, 0.0
    y_arr = np.array(y)
    n_pos = int(y_arr.sum())
    if n_pos == 0 or n_pos == len(y_arr):
        return 0.5, 0.0

    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        scale_pos_weight=max((len(y_arr) - n_pos) / max(n_pos, 1), 1),
        n_estimators=100,
        max_depth=4,
        random_state=42,
        eval_metric="auc",
    )
    aucs = cross_val_score(clf, np.array(X), y_arr, cv=5, scoring="roc_auc")
    return float(aucs.mean()), float(aucs.std())


def compute_bootstrap_ci(metric_fn, trajectories: list[list[dict[str, Any]]], n: int = BOOTSTRAP_N_RESAMPLES) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for a trajectory-level metric."""
    if not trajectories:
        return (0.0, 0.0)
    vals: list[float] = []
    for _ in range(n):
        sample = [trajectories[random.randint(0, len(trajectories) - 1)] for _ in range(len(trajectories))]
        vals.append(float(metric_fn(sample)))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def compute_mcnemar(baseline_a_results: list[dict[str, Any]], baseline_b_results: list[dict[str, Any]]) -> dict[str, float]:
    """McNemar significance test for paired binary outcomes (solved vs unsolved)."""
    b = 0
    c = 0
    for a, d in zip(baseline_a_results, baseline_b_results):
        a_ok = int(a.get("final_pass_rate", 0.0) >= PASS_THRESHOLD)
        d_ok = int(d.get("final_pass_rate", 0.0) >= PASS_THRESHOLD)
        if a_ok == 1 and d_ok == 0:
            b += 1
        elif a_ok == 0 and d_ok == 1:
            c += 1
    denom = max(b + c, 1)
    chi_sq = ((abs(b - c) - 1) ** 2) / denom
    p_value = float(1 - chi2.cdf(chi_sq, 1))
    return {"b": float(b), "c": float(c), "chi2": float(chi_sq), "p_value": p_value}


def print_baseline_comparison(baseline_results: dict[str, dict[str, Any]]) -> None:
    """Print concise baseline comparison table."""
    print("\nBaseline comparison:")
    print(f"{'Baseline':<20} {'Waste':>8} {'Pass@k':>8} {'Steps':>8} {'Regret':>8}")
    print("-" * 60)
    for name in BASELINES:
        row = baseline_results.get(name, {})
        print(
            f"{name:<20} "
            f"{row.get('waste_rate', 0.0):>8.3f} "
            f"{row.get('final_pass_rate', 0.0):>8.3f} "
            f"{row.get('steps_used', 0.0):>8.2f} "
            f"{row.get('regret_vs_oracle', 0.0):>8.3f}"
        )


def save_full_results(
    model: str,
    strategy: str,
    metrics: dict[str, Any],
    baseline_results: dict[str, dict[str, Any]],
    path: str,
) -> None:
    """Persist full analysis JSON."""
    payload = {
        "model": model,
        "strategy": strategy,
        "metrics": metrics,
        "baselines": baseline_results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_summary(
    waste_rate: float,
    ece: float,
    auc_mean: float,
    auc_std: float,
    trajectories: list[list[dict[str, Any]]],
    baseline_results: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Print summary table and optional baseline section."""
    rows = [
        ("Waste Rate", f"{waste_rate:.1%}", ">25%", waste_rate > 0.25),
        ("Self-Verification ECE", f"{ece:.3f}", ">0.200", ece > 0.2),
        (f"Feature AUC (±{auc_std:.3f})", f"{auc_mean:.3f}", ">0.650", auc_mean > 0.65),
    ]
    print("\n" + "=" * 58)
    print(f"{'Metric':<28} {'Value':>8}  {'Threshold':>9}  Status")
    print("-" * 58)
    for name, val, thresh, green in rows:
        print(f"{name:<28} {val:>8}  {thresh:>9}  {'GREEN' if green else 'RED'}")
    print("=" * 58)
    greens = sum(r[3] for r in rows)
    decision = {
        3: "\nDECISION: GREEN — Full COLM study, start immediately",
        2: "\nDECISION: YELLOW — Characterization paper, target TMLR",
        1: "\nDECISION: RED — Kill topic, move to next topic",
        0: "\nDECISION: RED — Kill topic, move to next topic",
    }
    print(decision[greens])
    if baseline_results is not None:
        print_baseline_comparison(baseline_results)


def run_full_analysis(trajectories: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Compute all core metrics, CIs, baselines, and pairwise tests."""
    waste_rate = compute_waste_rate(trajectories)
    ece = compute_ece(trajectories)
    auc_mean, auc_std = compute_feature_auc(trajectories)
    baseline_results = evaluate_all_baselines(trajectories)

    pairwise: dict[str, Any] = {}
    keys = list(BASELINES)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            pairwise[f"{a}__vs__{b}"] = compute_mcnemar(
                baseline_results[a].get("per_problem", []),
                baseline_results[b].get("per_problem", []),
            )

    metrics = {
        "waste_rate": waste_rate,
        "ece": ece,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "waste_ci_95": compute_bootstrap_ci(compute_waste_rate, trajectories),
        "ece_ci_95": compute_bootstrap_ci(compute_ece, trajectories),
    }
    return {
        "metrics": metrics,
        "baselines": baseline_results,
        "pairwise_tests": pairwise,
    }
