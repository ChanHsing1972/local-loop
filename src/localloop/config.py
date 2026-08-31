"""读取当前工作区的模型配置。"""

from __future__ import annotations

import os
from pathlib import Path

from localloop.types import AgentConfig

DEFAULT_BASE_URL = "https://token.bayesdl.com/api/maas/v1"


class ConfigError(ValueError):
    pass


def _parse_dotenv(path: Path) -> dict[str, str]:
    """读取简单的 KEY=VALUE；不展开变量，也不执行 shell。"""

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


def load_config(workspace: str | Path) -> AgentConfig:
    root = Path(workspace).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ConfigError(f"工作区不是有效目录：{root}")

    environment = {**_parse_dotenv(root / ".env"), **os.environ}
    api_key = environment.get("LLM_API_KEY", "").strip()
    base_url = environment.get("LLM_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = environment.get("LLM_MODEL", "").strip()
    if not api_key:
        raise ConfigError("尚未设置 LLM_API_KEY")
    if not base_url.startswith(("https://", "http://")):
        raise ConfigError("LLM_BASE_URL 必须以 https:// 或 http:// 开头")
    if not model:
        raise ConfigError("尚未设置 LLM_MODEL")

    return AgentConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        workspace=root,
    )
