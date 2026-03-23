"""
Feature extraction from repair trajectories. Standalone — importable by other projects.
"""

from __future__ import annotations

import ast
from typing import Any

import Levenshtein
import numpy as np
from scipy.stats import entropy

from config import IMPROVEMENT_THRESHOLD

# EXTENSION POINT — add feature groups here for downstream papers


def ast_normalize(code: str) -> str:
    """
    Strip comments and normalize whitespace via AST round-trip.

    Args:
        code: Raw Python source string.

    Returns:
        Normalized source string, or code.strip() on SyntaxError.
    """
    try:
        return ast.unparse(ast.parse(code))
    except SyntaxError:
        return code.strip()


def compute_ast_levenshtein(code_a: str, code_b: str) -> int:
    """
    AST-normalized edit distance. Raw string distance is too noisy.

    Args:
        code_a: First code string.
        code_b: Second code string.

    Returns:
        Levenshtein distance between normalized strings.
    """
    return Levenshtein.distance(ast_normalize(code_a), ast_normalize(code_b))


def extract_features(trajectory: list[dict[str, Any]], current_idx: int) -> dict[str, Any]:
    """
    Extract features for predicting whether the next iteration improves pass_rate.

    Args:
        trajectory: Steps recorded so far (before current_idx); each step has
            pass_rate, error_types, patch_delta, code, self_verification_score.
        current_idx: Current iteration number (0-indexed).

    Returns:
        Feature dict with pass_rate, deltas, error stats, patch/history/self-verification.
    """
    if current_idx == 0 or not trajectory:
        return _zero_features()

    curr = trajectory[-1]
    prev = trajectory[-2] if len(trajectory) >= 2 else None
    older = trajectory[-3] if len(trajectory) >= 3 else None

    pass_rate = curr["pass_rate"]
    pass_rate_delta = pass_rate - (prev["pass_rate"] if prev else 0.0)
    pass_rate_delta_2 = pass_rate - (older["pass_rate"] if older else 0.0)

    error_types = curr["error_types"]
    error_vec = np.array(
        [
            error_types.count(e)
            for e in [
                "SyntaxError",
                "TypeError",
                "AssertionError",
                "TimeoutError",
                "RuntimeError",
            ]
        ],
        dtype=float,
    )
    error_vec_norm = error_vec / error_vec.sum() if error_vec.sum() > 0 else error_vec
    error_ent = float(entropy(error_vec_norm + 1e-9))

    patch_lev = curr["patch_delta"]
    is_dup = int(patch_lev < 5)
    is_osc = int(
        older is not None
        and compute_ast_levenshtein(curr["code"], older["code"]) < 10
    )

    no_improve = 0
    for step in reversed(trajectory):
        if step["pass_rate"] < pass_rate - IMPROVEMENT_THRESHOLD:
            break
        no_improve += 1

    return {
        # EXTENSION POINT — pass rate features
        "pass_rate": pass_rate,
        "pass_rate_delta": pass_rate_delta,
        "pass_rate_delta_2": pass_rate_delta_2,
        # EXTENSION POINT — error features
        "error_type_entropy": error_ent,
        "syntax_error_count": int(error_vec[0]),
        "assertion_error_count": int(error_vec[2]),
        "timeout_count": int(error_vec[3]),
        # EXTENSION POINT — patch features
        "patch_levenshtein": patch_lev,
        "is_duplicate": is_dup,
        "is_oscillating": is_osc,
        # EXTENSION POINT — history features
        "iteration_number": current_idx,
        "consecutive_no_improvement": no_improve,
        "max_pass_rate_so_far": max(s["pass_rate"] for s in trajectory),
        # EXTENSION POINT — self-verification
        "self_verification_score": curr["self_verification_score"],
    }


def _zero_features() -> dict[str, Any]:
    """Return feature dict with all keys set to 0 for iteration 0 or empty trajectory."""
    keys = [
        "pass_rate",
        "pass_rate_delta",
        "pass_rate_delta_2",
        "error_type_entropy",
        "syntax_error_count",
        "assertion_error_count",
        "timeout_count",
        "patch_levenshtein",
        "is_duplicate",
        "is_oscillating",
        "iteration_number",
        "consecutive_no_improvement",
        "max_pass_rate_so_far",
        "self_verification_score",
    ]
    return dict.fromkeys(keys, 0)
