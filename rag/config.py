from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            key, default = match.group(1), match.group(2)
            return os.environ.get(key, default if default is not None else "")

        return ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@lru_cache
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("CONFIG_PATH", ROOT / "config.yaml"))
    with cfg_path.open() as f:
        raw = yaml.safe_load(f) or {}
    return _expand_env(raw)
