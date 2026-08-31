"""唯一的命令行入口：加载当前目录配置并启动交互界面。"""

from __future__ import annotations

import sys

from localloop.config import ConfigError, load_config
from localloop.interactive import InteractiveShell
from localloop.session import SessionError


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        print("用法：localloop（不支持参数或子命令）", file=sys.stderr)
        return 2
    try:
        config = load_config(".")
        return InteractiveShell(config).run()
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
    except (SessionError, OSError) as exc:
        print(f"本地状态错误：{exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n已退出 LocalLoop。")
        return 130
    return 2
