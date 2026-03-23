"""Production entrypoint for multi-model and multi-strategy experiments."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

from tqdm import tqdm

from analyze import print_summary, run_full_analysis, save_full_results
from config import MAX_EVAL_STEPS, MODELS, N_PROBLEMS, RESULTS_DIR, SUPPORTED_STRATEGIES
from data_lcb import get_problems
from figures import generate_all_figures
from strategies import STRATEGY_REGISTRY
from strategies.base import load_jsonl, slugify_model


def _load_combo_trajectories(model: str, strategy: str) -> list[list[dict[str, Any]]]:
    model_slug = slugify_model(model)
    root = os.path.join("data/trajectories", model_slug, strategy, "*.jsonl")
    return [load_jsonl(p) for p in glob.glob(root)]


def _run_combo(model: str, strategy_name: str, problems: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runner = STRATEGY_REGISTRY[strategy_name]()
    trajectories: list[list[dict[str, Any]]] = []
    for problem in tqdm(problems, desc=f"{model} | {strategy_name}"):
        traj = runner.run(
            task_id=problem["task_id"],
            problem=problem["prompt"],
            model=model,
            problem_dict=problem,
            max_eval_steps=MAX_EVAL_STEPS,
        )
        trajectories.append(traj)
    return trajectories


def _analyze_combo(model: str, strategy_name: str, trajectories: list[list[dict[str, Any]]]) -> dict[str, Any]:
    analysis = run_full_analysis(trajectories)
    metrics = analysis["metrics"]
    baselines = analysis["baselines"]
    print_summary(
        waste_rate=metrics["waste_rate"],
        ece=metrics["ece"],
        auc_mean=metrics["auc_mean"],
        auc_std=metrics["auc_std"],
        trajectories=trajectories,
        baseline_results=baselines,
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_name = f"{model.replace('/', '_')}__{strategy_name}_summary.json"
    out_path = os.path.join(RESULTS_DIR, out_name)
    save_full_results(model, strategy_name, metrics, baselines, out_path)
    return {"summary_path": out_path, "pairwise_tests": analysis["pairwise_tests"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production repair-stop experiment.")
    parser.add_argument("--model", default=None, help="Single model id override.")
    parser.add_argument("--strategy", default=None, help="Single strategy override.")
    parser.add_argument("--analyze-only", action="store_true", help="Skip LLM calls, analyze saved trajectories.")
    parser.add_argument("--figures-only", action="store_true", help="Only regenerate figures from saved outputs.")
    args = parser.parse_args()

    os.makedirs("data/trajectories", exist_ok=True)
    os.makedirs("data/strategy_metadata", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.figures_only:
        outputs = generate_all_figures("data/trajectories", RESULTS_DIR)
        print(json.dumps(outputs, indent=2))
        return

    selected_models = [args.model] if args.model else MODELS
    selected_strategies = [args.strategy] if args.strategy else list(SUPPORTED_STRATEGIES)
    for strategy_name in selected_strategies:
        if strategy_name not in STRATEGY_REGISTRY:
            raise RuntimeError(f"Unsupported strategy: {strategy_name}")

    problems = get_problems()[:N_PROBLEMS]
    run_report: dict[str, Any] = {"runs": []}
    for model in selected_models:
        for strategy_name in selected_strategies:
            if args.analyze_only:
                trajectories = _load_combo_trajectories(model, strategy_name)
            else:
                trajectories = _run_combo(model, strategy_name, problems)
            info = _analyze_combo(model, strategy_name, trajectories)
            run_report["runs"].append(
                {"model": model, "strategy": strategy_name, "summary_path": info["summary_path"]}
            )

    outputs = generate_all_figures("data/trajectories", RESULTS_DIR)
    run_report["figures"] = outputs
    with open(os.path.join(RESULTS_DIR, "run_report.json"), "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)
    print(json.dumps(run_report, indent=2))


if __name__ == "__main__":
    main()
