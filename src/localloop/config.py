from __future__ import annotations

import os
from pathlib import Path

from localloop.types import AgentConfig

DEFAULT_BASE_URL = "https://token.bayesdl.com/api/maas/v1"


class ConfigError(ValueError):
    pass


def load_config(
    workspace: str | Path,
    *,
    max_steps: int = 20,
    max_duration_seconds: int = 600,
    auto_approve: bool = False,
    require_model: bool = True,
) -> AgentConfig:
    root = Path(workspace).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ConfigError(f"Workspace is not a directory: {root}")
    if not 1 <= max_steps <= 100:
        raise ConfigError("max_steps must be between 1 and 100")
    if not 1 <= max_duration_seconds <= 86_400:
        raise ConfigError("max_duration_seconds must be between 1 and 86400")

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = os.environ.get("LLM_MODEL", "").strip()
    if not api_key:
        raise ConfigError("LLM_API_KEY is not set")
    if not base_url.startswith(("https://", "http://")):
        raise ConfigError("LLM_BASE_URL must start with https:// or http://")
    if require_model and not model:
        raise ConfigError("LLM_MODEL is not set; run `localloop doctor` to list models")

    return AgentConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        workspace=root,
        max_steps=max_steps,
        max_duration_seconds=max_duration_seconds,
        auto_approve=auto_approve,
    )
