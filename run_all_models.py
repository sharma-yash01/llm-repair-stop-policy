"""Deprecated: use run.py instead."""

from __future__ import annotations

import warnings

warnings.warn(
    "run_all_models.py is deprecated; use run.py (same flags: --models, --strategy, --dry-run).",
    DeprecationWarning,
    stacklevel=1,
)

from run import main

if __name__ == "__main__":
    main()
