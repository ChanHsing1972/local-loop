from __future__ import annotations

import os
from pathlib import Path

from localloop.types import AgentConfig

DEFAULT_BASE_URL = "https://token.bayesdl.com/api/maas/v1"


class ConfigError(ValueError):
    pass


def _parse_dotenv(path: Path) -> dict[str, str]:
    """读取最常用的 .env 语法，不修改进程环境，也不展开变量。"""

    if not path.is_file():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件 {path}：{exc}") from exc
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or not name.replace("_", "a").isalnum() or name[0].isdigit():
            raise ConfigError(f"{path} 第 {line_number} 行不是有效的 KEY=VALUE")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[name] = value
    return values


def load_environment(env_file: str | Path | None = None) -> dict[str, str]:
    """返回环境变量与 .env 的合并视图；已导出的环境变量具有最高优先级。"""

    path = Path(env_file).expanduser().resolve() if env_file else Path.cwd() / ".env"
    values = _parse_dotenv(path)
    return {**values, **os.environ}


def load_config(
    workspace: str | Path,
    *,
    max_steps: int = 20,
    max_duration_seconds: int = 600,
    auto_approve: bool = False,
    require_model: bool = True,
    env_file: str | Path | None = None,
) -> AgentConfig:
    root = Path(workspace).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ConfigError(f"工作区不是有效目录：{root}")
    if not 1 <= max_steps <= 100:
        raise ConfigError("max_steps 必须介于 1 和 100 之间")
    if not 1 <= max_duration_seconds <= 86_400:
        raise ConfigError("max_duration_seconds 必须介于 1 和 86400 之间")

    environment = load_environment(env_file)
    api_key = environment.get("LLM_API_KEY", "").strip()
    base_url = environment.get("LLM_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = environment.get("LLM_MODEL", "").strip()
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
