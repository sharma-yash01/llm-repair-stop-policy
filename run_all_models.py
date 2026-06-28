"""Run all registry models (MODEL_CONFIGS) against a single strategy."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from config import (
    ANTHROPIC_API_KEY_ENV,
    AWS_REGION,
    MODEL_CONFIGS,
    N_PROBLEMS,
    RESULTS_DIR,
    SUPPORTED_STRATEGIES,
    get_model_config,
)
from data_lcb import get_problems
from figures import generate_all_figures
from run import _analyze_combo, _run_combo
from strategies import STRATEGY_REGISTRY


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


def _print_dry_run(configs: list[dict[str, Any]], strategy: str) -> None:
    """Print planned runs and credential requirements."""
    print(f"AWS_REGION={AWS_REGION}")
    print(f"Strategy={strategy}")
    print(f"Problems={N_PROBLEMS}")
    for cfg in configs:
        ready, reason = _preflight(cfg)
        status = "ready" if ready else f"skip ({reason})"
        reasoning = cfg.get("reasoning")
        reasoning_note = f" reasoning={reasoning}" if reasoning else ""
        print(
            f"- {cfg['label']}: route={cfg['route']} model_id={cfg['model_id']}"
            f"{reasoning_note} [{status}]"
        )


def main() -> None:
    """Run all MODEL_CONFIGS entries sequentially for one strategy."""
    parser = argparse.ArgumentParser(
        description="Run all registry models for a single repair strategy."
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of model labels (default: all MODEL_CONFIGS).",
    )
    parser.add_argument(
        "--strategy",
        default="direct_fix",
        help="Strategy to run (default: direct_fix).",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip LLM calls; analyze saved trajectories only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print run plan and credential checks without executing.",
    )
    args = parser.parse_args()

    strategy_name = (args.strategy or "direct_fix").strip().lower()
    if strategy_name not in SUPPORTED_STRATEGIES:
        raise RuntimeError(
            f"Invalid strategy {strategy_name!r}. "
            f"Use one of: {', '.join(SUPPORTED_STRATEGIES)}."
        )
    if strategy_name not in STRATEGY_REGISTRY:
        raise RuntimeError(f"Strategy not registered: {strategy_name}")

    model_filter = None
    if args.models:
        model_filter = [m.strip() for m in args.models.split(",") if m.strip()]
    configs = _select_configs(model_filter)

    if args.dry_run:
        _print_dry_run(configs, strategy_name)
        return

    os.makedirs("data/trajectories", exist_ok=True)
    os.makedirs("data/strategy_metadata", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    problems = get_problems()[:N_PROBLEMS]
    run_report: dict[str, Any] = {
        "aws_region": AWS_REGION,
        "strategy": strategy_name,
        "runs": [],
        "skipped": [],
    }

    for cfg in configs:
        label = str(cfg["label"])
        ready, reason = _preflight(cfg)
        if not ready and not args.analyze_only:
            print(f"Skipping {label}: {reason}")
            run_report["skipped"].append({"model": label, "reason": reason})
            continue

        print(f"Running {label} via {cfg['route']} (model_id={cfg['model_id']})")
        if args.analyze_only:
            from run import _load_combo_trajectories

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

    outputs = generate_all_figures("data/trajectories", RESULTS_DIR)
    run_report["figures"] = outputs
    report_path = os.path.join(RESULTS_DIR, "run_all_models_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2)
    print(json.dumps(run_report, indent=2))


if __name__ == "__main__":
    main()
