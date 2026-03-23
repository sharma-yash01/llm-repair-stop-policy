"""
Test execution via LiveCodeBench-style runs: subprocess, stdin/stdout, no exec/eval.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from typing import Callable

from config import SUBPROCESS_TIMEOUT

try:
    import resource
except ImportError:  # pragma: no cover - non-posix fallback
    resource = None

MAX_ERROR_CHARS = 500
TMP_FILE_GLOB = "/tmp/tmp*.py"
TMP_FILE_MAX_AGE_SEC = 60
_TYPING_PREAMBLE = "from typing import *\n"
_PYTHON_BIN = shutil.which("python3") or shutil.which("python") or sys.executable
_SAFE_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "LANG": "en_US.UTF-8",
    "LC_ALL": "en_US.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}


def _normalize_output(text: str) -> str:
    """
    Normalize output text for robust comparisons.

    Args:
        text: Raw output string.

    Returns:
        Normalized output with stripped lines and collapsed whitespace.
    """
    lines = []
    for line in text.strip().splitlines():
        compact = " ".join(line.split()).strip()
        if compact:
            lines.append(compact)
    return "\n".join(lines)


def _outputs_match(actual: str, expected: str) -> bool:
    """
    Compare outputs after whitespace normalization.

    Args:
        actual: Candidate output.
        expected: Expected output.

    Returns:
        True when outputs should be treated as equivalent.
    """
    if actual == expected:
        return True
    return _normalize_output(actual) == _normalize_output(expected)


def _truncate_text(text: str, limit: int = MAX_ERROR_CHARS) -> str:
    """
    Truncate long text to the trailing `limit` characters.

    Args:
        text: Input string.
        limit: Character limit.

    Returns:
        Trailing substring no longer than `limit`.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]


def _format_error_from_stderr(stderr: str, fallback: str) -> str:
    """
    Build an actionable error string from stderr.

    Args:
        stderr: Subprocess stderr content.
        fallback: Fallback label when stderr is empty.

    Returns:
        Best-available exception text with fallback runtime summary.
    """
    # Python tracebacks usually end with "<ExceptionType>: <message>".
    exception_pattern = re.compile(
        r"^(\w+Error|\w+Exception|KeyboardInterrupt)\b(?::\s*(.*))?$"
    )
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    for line in reversed(lines):
        match = exception_pattern.match(line)
        if not match:
            continue
        exc_type = match.group(1)
        exc_msg = (match.group(2) or "").strip()
        formatted = f"{exc_type}: {exc_msg}" if exc_msg else exc_type
        return _truncate_text(formatted)

    detail = _truncate_text(stderr)
    if detail:
        return f"RuntimeError: {detail}"
    return f"RuntimeError: {fallback}"


def _subprocess_preexec() -> None:
    """
    Apply process limits for sandboxed execution on POSIX hosts.

    Returns:
        None.
    """
    if resource is None:
        return
    mb = 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (512 * mb, 512 * mb))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * mb, 10 * mb))


def _get_preexec_fn() -> Callable[[], None] | None:
    """
    Return a preexec function for POSIX subprocess limits.

    Returns:
        Callable for preexec_fn on POSIX, else None.
    """
    if os.name != "posix" or resource is None:
        return None
    return _subprocess_preexec


_PREEXEC_FN = _get_preexec_fn()


def _cleanup_tmp_py_files() -> None:
    """
    Remove stale temporary python files in /tmp.

    Returns:
        None.
    """
    now = time.time()
    for path in glob.glob(TMP_FILE_GLOB):
        try:
            if now - os.path.getmtime(path) > TMP_FILE_MAX_AGE_SEC:
                os.unlink(path)
        except Exception:
            continue


def run_tests(
    task_id: str,
    code: str,
    problem: dict[str, Any],
    timeout: int | None = None,
) -> dict:
    """
    Run tests for a solution using LiveCodeBench test cases (subprocess, stdin/stdout).

    Args:
        task_id: Problem identifier (e.g. LCB question_id).
        code: Full solution code (run as script; stdin = test input).
        problem: Dict with public_test_cases (and optionally private_test_cases),
                 each a list of {"input": str, "output": str}.
        timeout: Seconds per test run (from config).

    Returns:
        dict with pass_rate (float), passed (int), total (int), error_types (list[str]).
    """
    if timeout is None:
        timeout = SUBPROCESS_TIMEOUT
    return _run_tests_lcb(code, problem, timeout)


def _run_tests_lcb(
    code: str,
    problem: dict[str, Any],
    timeout: int,
) -> dict:
    public = problem.get("public_test_cases") or []
    private = problem.get("private_test_cases") or []
    test_cases = public + private
    if not test_cases:
        return {
            "pass_rate": 0.0,
            "passed": 0,
            "total": 0,
            "error_types": ["NoTestCases"],
        }

    metadata = problem.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    func_name = metadata.get("func_name", "")

    passed = 0
    error_types: list[str] = []
    for tc in test_cases:
        test_in = tc.get("input", tc.get("input_text", ""))
        test_out = tc.get("output", tc.get("output_text", ""))
        testtype = tc.get("testtype", "stdin")
        if not isinstance(test_in, str):
            test_in = str(test_in)
        if not isinstance(test_out, str):
            test_out = str(test_out)

        if testtype == "functional":
            ok, err = _run_one_test_functional(
                code, test_in, test_out, timeout, func_name
            )
        else:
            ok, err = _run_one_test(code, test_in, test_out, timeout)
        if ok:
            passed += 1
        else:
            error_types.append(err)

    total = len(test_cases)
    pass_rate = passed / total if total > 0 else 0.0
    _cleanup_tmp_py_files()
    return {
        "pass_rate": pass_rate,
        "passed": passed,
        "total": total,
        "error_types": error_types,
    }


def _run_one_test(code: str, test_in: str, expected_out: str, timeout: int) -> tuple[bool, str]:
    """
    Run code in a subprocess with test_in on stdin; compare stdout to expected_out.

    Returns:
        (True, "") if output matches (after stripping); else (False, error_type).
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        dir="/tmp",
        encoding="utf-8",
    ) as f:
        f.write(_TYPING_PREAMBLE + code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [_PYTHON_BIN, tmp_path],
            input=test_in,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
            env=_SAFE_ENV,
            preexec_fn=_PREEXEC_FN,
        )
        out = (result.stdout or "").strip()
        expected = expected_out.strip()
        if result.returncode != 0:
            return False, _format_error_from_stderr(result.stderr or "", "non-zero return code")
        if not _outputs_match(out, expected):
            return False, "AssertionError"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "TimeoutError"
    except Exception as exc:
        return False, _format_error_from_stderr("", f"{type(exc).__name__}: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _run_one_test_functional(
    code: str,
    test_in: str,
    expected_out: str,
    timeout: int,
    func_name: str,
) -> tuple[bool, str]:
    """
    Run solution as a module: call func_name(*json.loads(test_in)) and compare
    printed result to expected_out. Used for LeetCode-style call-based tests.
    """
    if not func_name:
        return False, "NoFuncName"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        dir="/tmp",
        encoding="utf-8",
    ) as f:
        f.write(_TYPING_PREAMBLE + code)
        solution_path = f.name
    runner_path = solution_path + "_runner.py"
    runner_dir = os.path.dirname(solution_path)
    runner_code = f"""import sys
import json
import inspect
sys.path.insert(0, {repr(runner_dir)})
import importlib.util
spec = importlib.util.spec_from_file_location("solution", {repr(solution_path)})
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)
_raw = sys.stdin.read().strip()
try:
    args = json.loads(_raw) if _raw else []
except json.JSONDecodeError:
    args = [json.loads(line) for line in _raw.splitlines() if line.strip()]
_fn = getattr(solution, {repr(func_name)}, None)
if _fn is None and hasattr(solution, "Solution"):
    _instance = solution.Solution()
    _fn = getattr(_instance, {repr(func_name)}, None)
if _fn is None:
    raise AttributeError("missing callable: " + {repr(func_name)})
_n_params = len(inspect.signature(_fn).parameters)
if isinstance(args, list) and len(args) == _n_params:
    result = _fn(*args)
else:
    result = _fn(args)
print(result)
"""
    try:
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(runner_code)
        result = subprocess.run(
            [_PYTHON_BIN, runner_path],
            input=test_in,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
            env=_SAFE_ENV,
            preexec_fn=_PREEXEC_FN,
        )
        out = (result.stdout or "").strip()
        expected = expected_out.strip()
        if result.returncode != 0:
            return False, _format_error_from_stderr(result.stderr or "", "non-zero return code")
        if not _outputs_match(out, expected):
            return False, "AssertionError"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "TimeoutError"
    except Exception as exc:
        return False, _format_error_from_stderr("", f"{type(exc).__name__}: {exc}")
    finally:
        for p in (solution_path, runner_path):
            try:
                os.unlink(p)
            except Exception:
                pass
