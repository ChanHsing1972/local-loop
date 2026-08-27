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
        raise ConfigError(f"工作区不是有效目录：{root}")
    if not 1 <= max_steps <= 100:
        raise ConfigError("max_steps 必须介于 1 和 100 之间")
    if not 1 <= max_duration_seconds <= 86_400:
        raise ConfigError("max_duration_seconds 必须介于 1 和 86400 之间")

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = os.environ.get("LLM_MODEL", "").strip()
    if not api_key:
        raise ConfigError("尚未设置 LLM_API_KEY")
    if not base_url.startswith(("https://", "http://")):
        raise ConfigError("LLM_BASE_URL 必须以 https:// 或 http:// 开头")
    if require_model and not model:
        raise ConfigError("尚未设置 LLM_MODEL；请运行 `localloop doctor` 查看可用模型")

    return AgentConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        workspace=root,
        max_steps=max_steps,
        max_duration_seconds=max_duration_seconds,
        auto_approve=auto_approve,
    )
