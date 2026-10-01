"""Minimal .env loading helpers for local training workflows."""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(start_path: str | Path | None = None) -> None:
    """Load a simple KEY=VALUE .env file from repo root if present.

    Existing environment variables are not overwritten.
    """
    if start_path is None:
        base = Path.cwd()
    else:
        base = Path(start_path)

    candidates = [base / ".env", Path(__file__).resolve().parent.parent / ".env"]
    env_path = None
    for candidate in candidates:
        if candidate.exists():
            env_path = candidate
            break

    if env_path is None:
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
