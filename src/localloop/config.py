"""读取并校验 LocalLoop 的运行配置。

配置层只负责把命令行选项、环境变量和 ``.env`` 文件整理成 ``AgentConfig``。
它不会主动连接模型网关，也不会把 ``.env`` 的值写回 ``os.environ``，因而读取配置
本身没有网络和全局状态副作用。
"""

from __future__ import annotations

import os
from pathlib import Path

from localloop.types import AgentConfig

DEFAULT_BASE_URL = "https://token.bayesdl.com/api/maas/v1"


class ConfigError(ValueError):
    """表示用户可以通过修改配置自行解决的错误。"""


def _parse_dotenv(path: Path) -> dict[str, str]:
    """读取项目需要的简化版 ``.env`` 语法。

    支持空行、注释、``export KEY=value`` 和成对引号。这里刻意不支持变量展开或执行
    shell 语句，因为配置文件应当只是数据，不能借读取配置之机运行代码。
    """

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
        # partition 只按第一个等号拆分，所以值本身仍然可以包含等号。
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
    """返回环境变量与 ``.env`` 的合并副本。

    ``{**dotenv, **os.environ}`` 的后者优先，因此终端中显式导出的值可以覆盖文件值。
    返回副本而不修改真实进程环境，方便测试，也减少秘密意外扩散到子进程的机会。
    """

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
    """构造一份经过边界校验、可供各模块共享的不可变运行配置。"""

    # resolve 把 ``.``、``..`` 和符号链接折叠成明确的绝对工作区路径。
    root = Path(workspace).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ConfigError(f"工作区不是有效目录：{root}")
    if not 1 <= max_steps <= 100:
        raise ConfigError("max_steps 必须介于 1 和 100 之间")
    if not 1 <= max_duration_seconds <= 86_400:
        raise ConfigError("max_duration_seconds 必须介于 1 和 86400 之间")

    environment = load_environment(env_file)
    # 密钥只保存在内存中的配置对象里，AgentConfig 的 repr 也会主动隐藏它。
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
