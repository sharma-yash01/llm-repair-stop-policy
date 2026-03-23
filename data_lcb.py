"""
Load LiveCodeBench code_generation_lite from HuggingFace for the pilot.
Returns problems in a format compatible with run_repair_loop and evaluate.
"""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from typing import Any

from config import LCB_MIN_DIFFICULTY, LCB_RELEASE, N_PROBLEMS

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def get_problems() -> list[dict[str, Any]]:
    """
    Load LiveCodeBench problems (first N_PROBLEMS) for the configured release.

    Returns:
        List of dicts: task_id (str), prompt (str), public_test_cases (list),
        private_test_cases (list), starter_code (str), question_title (str).
        test_cases items are {"input": str, "output": str}.
    """
    from datasets import load_dataset

    # Config name matches version_tag, e.g. release_v1, release_v2.
    # trust_remote_code required to run the dataset's loading script (datasets<4.0).
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        name=LCB_RELEASE,
        split="test",
        trust_remote_code=True,
    )
    min_level = DIFFICULTY_ORDER.get(LCB_MIN_DIFFICULTY, 0)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if len(out) >= N_PROBLEMS:
            break
        difficulty = row.get("difficulty", "easy")
        if DIFFICULTY_ORDER.get(difficulty, 0) < min_level:
            continue
        task_id = str(row.get("question_id", row.get("id", f"LCB_{i}")))
        question_content = row.get("question_content", "")
        question_title = row.get("question_title", "")
        starter_code = row.get("starter_code") or ""
        if starter_code and not starter_code.strip().endswith("\n"):
            starter_code = starter_code.strip() + "\n"

        # Build prompt: title + content + optional starter code
        prompt_parts = [f"# {question_title}\n\n", question_content.strip()]
        if starter_code:
            prompt_parts.append(f"\n\nStarter code:\n```python\n{starter_code}```")
        prompt = "\n".join(prompt_parts)

        def parse_tests(raw: Any, is_private: bool = False) -> list[dict[str, Any]]:
            """Parse test cases. Public are JSON; private may be base64/zlib/pickle encoded."""
            if raw is None:
                return []
            if isinstance(raw, list):
                return [_norm_tc(x) for x in raw]
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    if is_private:
                        try:
                            decoded = base64.b64decode(raw.encode("utf-8"))
                            data = pickle.loads(zlib.decompress(decoded))
                            if isinstance(data, str):
                                data = json.loads(data)
                        except Exception:
                            return []
                    else:
                        return []
                if isinstance(data, list):
                    return [_norm_tc(t) for t in data]
                if isinstance(data, dict) and "input" in data and "output" in data:
                    return [_norm_tc(data)]
            return []

        def _norm_tc(t: Any) -> dict[str, Any]:
            d = t if isinstance(t, dict) else {}
            inp = str(d.get("input", d.get("input_text", "")))
            out = str(d.get("output", d.get("output_text", "")))
            testtype = d.get("testtype", "stdin")
            testtype = testtype if isinstance(testtype, str) else "stdin"
            return {"input": inp, "output": out, "testtype": testtype}

        public = parse_tests(row.get("public_test_cases"), is_private=False)
        private = parse_tests(row.get("private_test_cases"), is_private=True)

        raw_meta = row.get("metadata")
        if isinstance(raw_meta, dict):
            metadata = raw_meta
        elif isinstance(raw_meta, str):
            try:
                metadata = json.loads(raw_meta) if raw_meta.strip() else {}
            except json.JSONDecodeError:
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
        else:
            metadata = {}
        out.append({
            "task_id": task_id,
            "prompt": prompt,
            "public_test_cases": public,
            "private_test_cases": private,
            "starter_code": starter_code,
            "question_title": question_title,
            "difficulty": difficulty,
            "metadata": metadata,
        })
    return out
