"""Figure generation for TMLR production experiments."""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import FIGURES_DIR, PASS_THRESHOLD


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_all_trajectories(data_dir: str) -> dict[tuple[str, str], list[list[dict[str, Any]]]]:
    grouped: dict[tuple[str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    pattern = os.path.join(data_dir, "*", "*", "*.jsonl")
    for path in glob.glob(pattern):
        parts = path.split(os.sep)
        model = parts[-3]
        strategy = parts[-2]
        grouped[(model, strategy)].append(_load_jsonl(path))
    return grouped


def plot_pass_rate_trajectories(grouped: dict[tuple[str, str], list[list[dict[str, Any]]]], out_dir: str) -> str:
    """Figure 1: mean pass-rate by step per model."""
    plt.figure(figsize=(10, 6))
    for (model, strategy), trajectories in grouped.items():
        max_len = max((len(t) for t in trajectories), default=0)
        if max_len == 0:
            continue
        means = []
        for i in range(max_len):
            vals = [t[i]["pass_rate"] for t in trajectories if i < len(t)]
            means.append(float(np.mean(vals)) if vals else 0.0)
        plt.plot(range(max_len), means, marker="o", label=f"{model} | {strategy}")
    plt.xlabel("Step")
    plt.ylabel("Mean pass rate")
    plt.title("Pass rate trajectories")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "pass_rate_trajectories.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_waste_decomposition(grouped: dict[tuple[str, str], list[list[dict[str, Any]]]], out_dir: str) -> str:
    """Figure 2: solved-at-0/solved-later/partial/never stacked bar."""
    labels = []
    solved0_vals = []
    solved_later_vals = []
    partial_vals = []
    never_vals = []
    for (model, strategy), trajectories in grouped.items():
        solved0 = solved_later = partial = never = 0
        for traj in trajectories:
            pass_rates = [s["pass_rate"] for s in traj]
            if not pass_rates:
                continue
            first = next((i for i, r in enumerate(pass_rates) if r >= PASS_THRESHOLD), None)
            if first == 0:
                solved0 += 1
            elif first is not None:
                solved_later += 1
            elif max(pass_rates) > 0:
                partial += 1
            else:
                never += 1
        total = max(len(trajectories), 1)
        labels.append(f"{model}\n{strategy}")
        solved0_vals.append(solved0 / total)
        solved_later_vals.append(solved_later / total)
        partial_vals.append(partial / total)
        never_vals.append(never / total)

    x = np.arange(len(labels))
    plt.figure(figsize=(12, 6))
    plt.bar(x, solved0_vals, label="Solved-at-0")
    plt.bar(x, solved_later_vals, bottom=solved0_vals, label="Solved-later")
    bottom2 = np.array(solved0_vals) + np.array(solved_later_vals)
    plt.bar(x, partial_vals, bottom=bottom2, label="Partial")
    bottom3 = bottom2 + np.array(partial_vals)
    plt.bar(x, never_vals, bottom=bottom3, label="Never-passed")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Fraction of problems")
    plt.title("Waste decomposition by model and strategy")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "waste_decomposition.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_calibration_diagram(grouped: dict[tuple[str, str], list[list[dict[str, Any]]]], out_dir: str, n_bins: int = 15) -> str:
    """Figure 3: reliability diagram."""
    plt.figure(figsize=(8, 8))
    for (model, strategy), trajectories in grouped.items():
        confs: list[float] = []
        outs: list[int] = []
        for traj in trajectories:
            for step in traj:
                confs.append(float(step.get("self_verification_score", 0.5)))
                outs.append(int(step.get("pass_rate", 0.0) >= PASS_THRESHOLD))
        if not confs:
            continue
        conf_arr = np.array(confs)
        out_arr = np.array(outs)
        edges = np.linspace(0, 1, n_bins + 1)
        x_vals, y_vals = [], []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (conf_arr >= lo) & (conf_arr < hi) if i < n_bins - 1 else (conf_arr >= lo) & (conf_arr <= hi)
            if not mask.any():
                continue
            x_vals.append(float(conf_arr[mask].mean()))
            y_vals.append(float(out_arr[mask].mean()))
        plt.plot(x_vals, y_vals, marker="o", label=f"{model}|{strategy}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical correctness")
    plt.title("Calibration reliability diagram")
    plt.legend(fontsize=7)
    plt.tight_layout()
    path = os.path.join(out_dir, "calibration_reliability.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_baseline_comparison(results_dir: str, out_dir: str) -> str:
    """Figure 4: baseline comparison chart from summary JSON files."""
    rows: list[dict[str, Any]] = []
    for path in glob.glob(os.path.join(results_dir, "*_summary.json")):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        baselines = payload.get("baselines", {})
        for name, vals in baselines.items():
            rows.append(
                {
                    "baseline": name,
                    "waste_rate": vals.get("waste_rate", 0.0),
                    "pass_rate": vals.get("final_pass_rate", 0.0),
                    "compute_savings": vals.get("compute_savings", 0.0),
                }
            )
    if not rows:
        return ""

    baseline_names = sorted({r["baseline"] for r in rows})
    waste = [np.mean([r["waste_rate"] for r in rows if r["baseline"] == b]) for b in baseline_names]
    pass_rate = [np.mean([r["pass_rate"] for r in rows if r["baseline"] == b]) for b in baseline_names]
    savings = [np.mean([r["compute_savings"] for r in rows if r["baseline"] == b]) for b in baseline_names]

    x = np.arange(len(baseline_names))
    width = 0.25
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, waste, width, label="Waste")
    plt.bar(x, pass_rate, width, label="Pass@k")
    plt.bar(x + width, savings, width, label="Compute savings")
    plt.xticks(x, baseline_names, rotation=30, ha="right")
    plt.title("Baseline comparison")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "baseline_comparison.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_state_transitions(grouped: dict[tuple[str, str], list[list[dict[str, Any]]]], out_dir: str) -> str:
    """Figure 5: state transition heatmap (aggregated)."""
    states = ["zero", "partial", "solved", "regressed"]
    idx = {s: i for i, s in enumerate(states)}
    mat = np.zeros((4, 4), dtype=float)

    def state_of(pass_rate: float) -> str:
        if pass_rate == 0:
            return "zero"
        if pass_rate >= PASS_THRESHOLD:
            return "solved"
        return "partial"

    for trajectories in grouped.values():
        for traj in trajectories:
            for i in range(len(traj) - 1):
                curr = state_of(float(traj[i]["pass_rate"]))
                nxt = state_of(float(traj[i + 1]["pass_rate"]))
                if curr == "solved" and nxt in ("zero", "partial"):
                    nxt = "regressed"
                mat[idx[curr], idx[nxt]] += 1

    plt.figure(figsize=(7, 6))
    sns.heatmap(mat, annot=True, fmt=".0f", xticklabels=states, yticklabels=states, cmap="Blues")
    plt.xlabel("To state")
    plt.ylabel("From state")
    plt.title("State transition heatmap")
    plt.tight_layout()
    path = os.path.join(out_dir, "state_transition_heatmap.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_feature_importance(trajectories_dir: str, out_dir: str) -> str:
    """Figure 6: XGBoost feature importance from saved trajectories."""
    X: list[list[float]] = []
    y: list[int] = []
    feature_names: list[str] = []
    for path in glob.glob(os.path.join(trajectories_dir, "*", "*", "*.jsonl")):
        traj = _load_jsonl(path)
        for i in range(len(traj) - 1):
            feats = traj[i].get("features", {})
            if not feats:
                continue
            if not feature_names:
                feature_names = list(feats.keys())
            X.append([float(v) for v in feats.values()])
            y.append(int(traj[i + 1]["pass_rate"] - traj[i]["pass_rate"] >= 0.05))
    if not X or not feature_names:
        return ""

    from xgboost import XGBClassifier

    clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="auc")
    clf.fit(np.array(X), np.array(y))
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar([feature_names[i] for i in order], importances[order])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Feature importance")
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def generate_all_figures(trajectories_dir: str, results_dir: str) -> dict[str, str]:
    """Generate all required figures and return output paths."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    grouped = _load_all_trajectories(trajectories_dir)
    outputs = {
        "pass_rate_trajectories": plot_pass_rate_trajectories(grouped, FIGURES_DIR),
        "waste_decomposition": plot_waste_decomposition(grouped, FIGURES_DIR),
        "calibration_reliability": plot_calibration_diagram(grouped, FIGURES_DIR),
        "baseline_comparison": plot_baseline_comparison(results_dir, FIGURES_DIR),
        "state_transition_heatmap": plot_state_transitions(grouped, FIGURES_DIR),
        "feature_importance": plot_feature_importance(trajectories_dir, FIGURES_DIR),
    }
    return outputs
