"""LocalLoop 包的最外层入口。

这个文件只公开版本号，不创建模型客户端，也不执行任何本地工具。保持入口轻量，
可以避免仅仅执行 ``import localloop`` 时就读取配置、访问网络或修改文件。
"""

from importlib.metadata import PackageNotFoundError, version

try:
    # 正常安装后，版本号以 pyproject.toml 生成的包元数据为准。
    __version__ = version("localloop-agent")
except PackageNotFoundError:  # pragma: no cover - 未安装时直接从源码运行
    # 开发者也可能直接把 src 加入 PYTHONPATH，此时没有已安装的包元数据。
    __version__ = "0.3.0"

__all__ = ["__version__"]
