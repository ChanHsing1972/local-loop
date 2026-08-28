"""LocalLoop：小型、可审计的交互式编程智能体。"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("localloop-agent")
except PackageNotFoundError:  # pragma: no cover - 未安装时直接从源码运行
    __version__ = "0.3.0"

__all__ = ["__version__"]
