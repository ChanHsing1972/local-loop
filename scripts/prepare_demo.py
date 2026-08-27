from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

DEMO_FILES = ("README.md", "pricing.py", "test_pricing.py")


def prepare_demo(destination: Path | None = None) -> Path:
    repository = Path(__file__).resolve().parents[1]
    source = repository / "demo" / "price_project"
    if destination is None:
        destination = Path(tempfile.mkdtemp(prefix="localloop-price-demo-", dir="/tmp"))
    else:
        destination = destination.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise ValueError(f"目标目录必须为空：{destination}")
    for name in DEMO_FILES:
        shutil.copy2(source / name, destination / name)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="创建不含缓存和本机状态的干净演示项目")
    parser.add_argument("--destination", type=Path, help="可选的空目标目录")
    args = parser.parse_args()
    try:
        destination = prepare_demo(args.destination)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
