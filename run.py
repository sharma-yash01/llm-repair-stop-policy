"""Production entrypoint for multi-model and multi-strategy experiments."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import traceback
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from tqdm import tqdm

from analyze import print_summary, run_full_analysis, save_full_results
from config import (
    ANTHROPIC_API_KEY_ENV,
    AWS_REGION,
    MAX_EVAL_STEPS,
    MODEL_CONFIGS,
    N_PROBLEMS,
    REGISTRY_LABELS,
    RESULTS_DIR,
    SUPPORTED_STRATEGIES,
    build_ephemeral_config,
    clear_ephemeral_configs,
    get_model_config,
    register_ephemeral_config,
)
from data_lcb import get_problems
from figures import generate_all_figures
from strategies import STRATEGY_REGISTRY
from strategies.base import load_jsonl, slugify_model

logger = logging.getLogger(__name__)


def _load_combo_trajectories(model: str, strategy: str) -> list[list[dict[str, Any]]]:
    """Load saved trajectory JSONL files for one model/strategy combo."""
    model_slug = slugify_model(model)
    root = os.path.join("data/trajectories", model_slug, strategy, "*.jsonl")
    return [load_jsonl(p) for p in glob.glob(root)]


def _run_combo(model: str, strategy_name: str, problems: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Run repair loop for all problems with one model/strategy."""
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
    """Compute metrics and persist summary JSON for one model/strategy combo."""
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


def _has_aws_credentials() -> bool:
    """
    Return True when AWS credentials are available for Bedrock.

    Returns:
        True if STS get_caller_identity succeeds.
    """
    try:
        boto3.client("sts", region_name=AWS_REGION).get_caller_identity()
        return True
    except (NoCredentialsError, ClientError, BotoCoreError):
        return False


def _select_configs(model_filter: list[str] | None) -> list[dict[str, Any]]:
    """
    Select MODEL_CONFIGS entries, optionally filtered by label.

    Args:
        model_filter: Optional list of labels to include.

    Returns:
        Filtered config list.

    Raises:
        RuntimeError: If a requested label is unknown.
    """
    if not model_filter:
        return list(MODEL_CONFIGS)
    selected: list[dict[str, Any]] = []
    for label in model_filter:
        cfg = get_model_config(label)
        if cfg is None:
            raise RuntimeError(f"Unknown model label: {label!r}")
        selected.append(cfg)
    return selected


def _preflight(cfg: dict[str, Any]) -> tuple[bool, str]:
    """
    Check whether credentials exist for a model route.

    Args:
        cfg: Model registry entry.

    Returns:
        Tuple of (ready, reason_if_skipped).
    """
    route = str(cfg.get("route", ""))
    if route == "bedrock":
        if _has_aws_credentials():
            return True, ""
        return False, "missing AWS credentials for Bedrock"
    if route == "anthropic":
        if os.environ.get(ANTHROPIC_API_KEY_ENV):
            return True, ""
        return False, f"missing {ANTHROPIC_API_KEY_ENV}"
    return False, f"unsupported route {route!r}"


def _print_dry_run(configs: list[dict[str, Any]], strategies: list[str]) -> None:
    """Print planned runs and credential requirements."""
    print(f"AWS_REGION={AWS_REGION}")
    print(f"Strategies={', '.join(strategies)}")
    print(f"Problems={N_PROBLEMS}")
    for cfg in configs:
        ready, reason = _preflight(cfg)
        status = "ready" if ready else f"skip ({reason})"
        reasoning = cfg.get("reasoning")
        reasoning_note = f" reasoning={reasoning}" if reasoning else ""
        for strategy_name in strategies:
            print(
                f"- {cfg['label']} | {strategy_name}: route={cfg['route']} "
                f"model_id={cfg['model_id']}{reasoning_note} [{status}]"
            )


def _parse_model_filter(args: argparse.Namespace) -> list[str] | None:
    """
    Resolve model labels from --model / --models CLI flags.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of labels, or None for all REGISTRY_LABELS.
    """
    if args.model:
        return [args.model.strip()]
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    return None


def _parse_strategies(args: argparse.Namespace) -> list[str]:
    """
    Resolve strategy names from --strategy / --strategies CLI flags.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of strategy names.

    Raises:
        RuntimeError: If both --strategy and --strategies are set, or name is invalid.
    """
    if args.strategy and args.strategies:
        raise RuntimeError("Use only one of --strategy or --strategies.")
    if args.strategies:
        names = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
    elif args.strategy:
        names = [args.strategy.strip().lower()]
    else:
        names = ["direct_fix"]
    for name in names:
        if name not in SUPPORTED_STRATEGIES:
            raise RuntimeError(
                f"Invalid strategy {name!r}. Use one of: {', '.join(SUPPORTED_STRATEGIES)}."
            )
        if name not in STRATEGY_REGISTRY:
            raise RuntimeError(f"Strategy not registered: {name}")
    return names


def _register_adhoc_config(args: argparse.Namespace) -> list[dict[str, Any]]:
    """
    Build and register an ephemeral config from ad-hoc CLI flags.

    Args:
        args: Parsed CLI arguments with adhoc fields set.

    Returns:
        Single-element config list for the ad-hoc label.
    """
    if not args.route or not args.model_id:
        raise RuntimeError("--adhoc-label requires --route and --model-id.")
    clear_ephemeral_configs()
    cfg = build_ephemeral_config(
        label=args.adhoc_label,
        route=args.route,
        model_id=args.model_id,
        reasoning=args.reasoning or "none",
        price_in=float(args.price_in or 0.0),
        price_out=float(args.price_out or 0.0),
        max_tokens=int(args.max_tokens) if args.max_tokens else None,
        read_timeout_sec=int(args.read_timeout_sec) if args.read_timeout_sec else None,
    )
    register_ephemeral_config(cfg)
    return [cfg]


def main() -> None:
    """Run registry and/or ad-hoc model experiments."""
    parser = argparse.ArgumentParser(description="Run production repair-stop experiment.")
    parser.add_argument("--model", default=None, help="Single registry label (alias for --models).")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated registry labels (default: all MODEL_CONFIGS).",
    )
    parser.add_argument("--strategy", default=None, help="Single strategy (default: direct_fix).")
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategies (mutually exclusive with --strategy).",
    )
    parser.add_argument("--analyze-only", action="store_true", help="Skip LLM calls, analyze saved trajectories.")
    parser.add_argument("--figures-only", action="store_true", help="Only regenerate figures from saved outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print run plan and credential checks without executing.")
    parser.add_argument(
        "--adhoc-label",
        default=None,
        help="Ad-hoc run label (requires --route and --model-id; not in MODEL_CONFIGS).",
    )
    parser.add_argument("--route", default=None, help="Ad-hoc API route: bedrock | anthropic.")
    parser.add_argument("--model-id", default=None, help="Ad-hoc provider model id.")
    parser.add_argument(
        "--reasoning",
        default="none",
        choices=("on", "off", "none"),
        help="Ad-hoc reasoning mode (default: none).",
    )
    parser.add_argument("--price-in", type=float, default=None, help="Ad-hoc USD per 1M input tokens.")
    parser.add_argument("--price-out", type=float, default=None, help="Ad-hoc USD per 1M output tokens.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Ad-hoc max output tokens.")
    parser.add_argument("--read-timeout-sec", type=int, default=None, help="Ad-hoc HTTP read timeout.")
    args = parser.parse_args()

    os.makedirs("data/trajectories", exist_ok=True)
    os.makedirs("data/strategy_metadata", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.figures_only:
        outputs = generate_all_figures("data/trajectories", RESULTS_DIR)
        print(json.dumps(outputs, indent=2))
        return

    strategies = _parse_strategies(args)

    if args.adhoc_label:
        configs = _register_adhoc_config(args)
    else:
        model_filter = _parse_model_filter(args)
        configs = _select_configs(model_filter)

    if args.dry_run:
        _print_dry_run(configs, strategies)
        return

    problems = get_problems()[:N_PROBLEMS]
    run_report: dict[str, Any] = {
        "aws_region": AWS_REGION,
        "strategies": strategies,
        "registry_labels": REGISTRY_LABELS,
        "runs": [],
        "skipped": [],
        "failed": [],
    }

    for cfg in configs:
        label = str(cfg["label"])
        ready, reason = _preflight(cfg)
        if not ready and not args.analyze_only:
            print(f"Skipping {label}: {reason}")
            run_report["skipped"].append({"model": label, "reason": reason})
            continue

        for strategy_name in strategies:
            print(f"Running {label} via {cfg['route']} (model_id={cfg['model_id']}) | {strategy_name}")
            try:
                if args.analyze_only:
                    trajectories = _load_combo_trajectories(label, strategy_name)
                else:
                    trajectories = _run_combo(label, strategy_name, problems)
                info = _analyze_combo(label, strategy_name, trajectories)
                run_report["runs"].append(
                    {
                        "model": label,
                        "route": cfg["route"],
                        "model_id": cfg["model_id"],
                        "strategy": strategy_name,
                        "summary_path": info["summary_path"],
                    }
                )
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                logger.exception("Run failed for %s | %s", label, strategy_name)
                print(f"FAILED {label} | {strategy_name}: {err_msg}")
                run_report["failed"].append(
                    {
                        "model": label,
                        "route": cfg["route"],
                        "model_id": cfg["model_id"],
                        "strategy": strategy_name,
                        "error": err_msg,
                        "traceback": traceback.format_exc(),
                    }
                )

    outputs = generate_all_figures("data/trajectories", RESULTS_DIR)
    run_report["figures"] = outputs
    report_path = os.path.join(RESULTS_DIR, "run_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)
    print(json.dumps(run_report, indent=2))


if __name__ == "__main__":
    main()
