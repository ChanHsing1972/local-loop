from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
REQUIRED_FILES = {
    "README.md",
    "README.txt",
    "LICENSE",
    "LICENSE.zh-CN.md",
    "docs/答辩准备.md",
    "docs/英文介绍.md",
    "docs/视频脚本.md",
    "docs/设计日志.md",
}
SECRET_PATTERNS = {
    "通用 API 密钥": re.compile(rb"sk-[A-Za-z0-9_-]{16,}"),
    "GitHub 令牌": re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    "Google API 密钥": re.compile(rb"AIza[A-Za-z0-9_-]{30,}"),
    "私钥头": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
IDENTITY_PATH_PATTERN = re.compile(rb"/(?:Users|home)/[^/\s]+/")


def git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git 命令失败：{detail}")
    return completed.stdout


def check_patterns(label: str, content: bytes, failures: list[str]) -> None:
    for pattern_name, pattern in SECRET_PATTERNS.items():
        if pattern.search(content):
            failures.append(f"{label}命中疑似{pattern_name}")
    if IDENTITY_PATH_PATTERN.search(content):
        failures.append(f"{label}包含用户主目录绝对路径")


def main() -> int:
    failures: list[str] = []
    tracked = {
        item.decode("utf-8")
        for item in git_bytes("ls-files", "-z").split(b"\0")
        if item
    }
    missing = sorted(REQUIRED_FILES - tracked)
    if missing:
        failures.append("缺少必需文件：" + "、".join(missing))
    forbidden = sorted(
        path
        for path in tracked
        if path == ".env" or path.startswith(".localloop/") or path.startswith(".venv/")
    )
    if forbidden:
        failures.append("运行时或凭据文件被提交：" + "、".join(forbidden))

    for path in sorted(tracked):
        file_path = REPOSITORY / path
        if file_path.is_file():
            check_patterns(f"文件 {path} ", file_path.read_bytes(), failures)
    history = git_bytes("log", "--all", "--format=fuller", "-p", "--no-ext-diff")
    check_patterns("完整 Git 历史", history, failures)

    readme_text = (REPOSITORY / "README.txt").read_text(encoding="utf-8")
    if len(readme_text) > 1_000:
        failures.append(f"README.txt 共 {len(readme_text)} 字符，超过 1000 字限制")
    if "PUBLIC_REPOSITORY_URL" in readme_text or "<PUBLIC" in readme_text:
        failures.append("README.txt 仍含仓库地址占位符")
    if not re.search(r"https://(?:github\.com|gitee\.com)/\S+", readme_text):
        failures.append("README.txt 没有 GitHub 或 Gitee 仓库地址")

    authors = {
        line
        for line in git_bytes("log", "--format=%an <%ae>").decode().splitlines()
        if line
    }
    expected_author = "Candidate <candidate@users.noreply.github.com>"
    unexpected_authors = sorted(authors - {expected_author})
    if unexpected_authors:
        failures.append("提交历史存在非中性作者信息：" + "、".join(unexpected_authors))

    if failures:
        print("发布验收失败：")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(
        f"发布验收通过：{len(tracked)} 个受版本控制文件；"
        f"README.txt {len(readme_text)} 字符；未发现疑似密钥、身份绝对路径或运行时文件。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
