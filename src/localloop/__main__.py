"""支持通过 ``python -m localloop`` 启动程序。

真正的命令行解析位于 :mod:`localloop.cli`；这里仅把 Python 模块入口转交给它。
"""

from localloop.cli import main

if __name__ == "__main__":
    # main 返回整数退出码；SystemExit 会把它交给操作系统。
    raise SystemExit(main())
