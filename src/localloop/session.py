"""把对话和运行事件持久化为工作区内的 JSONL 会话文件。

JSONL 是“一行一个 JSON 对象”的文本格式。追加写入比反复改写整个历史简单，也使程序
崩溃时通常只会损坏最后一行。每个工作区各自保存 ``.localloop/sessions``，会话不会跨
项目混用。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from localloop.types import JsonObject, Message, SessionEvent

SESSION_VERSION = 1
SESSION_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")


class SessionError(RuntimeError):
    """表示会话编号、路径或文件内容不可信，不能继续使用。"""


@dataclass(frozen=True, slots=True)
class SessionData:
    """从 JSONL 中重建出的会话元数据和模型消息历史。"""

    metadata: JsonObject
    messages: list[Message]


class SessionStore:
    """负责单个会话文件的创建、追加、读取和删除。"""

    def __init__(self, workspace: Path, session_id: str) -> None:
        """计算会话路径，并防止伪造编号或符号链接把存储位置引出工作区。"""

        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise SessionError("Invalid session id")
        self.workspace = workspace.resolve()
        self.session_id = session_id
        raw_directory = self.workspace / ".localloop" / "sessions"
        # strict=False 允许目录尚未创建，同时仍解析已经存在的符号链接部分。
        self.directory = raw_directory.resolve(strict=False)
        try:
            self.directory.relative_to(self.workspace)
        except ValueError as exc:
            raise SessionError("会话目录不能通过符号链接越过工作区") from exc
        self.path = self.directory / f"{session_id}.jsonl"

    @classmethod
    def create(cls, workspace: Path, *, task: str, model: str) -> SessionStore:
        """生成随机短编号并写入会话的第一条 metadata 事件。"""

        store = cls(workspace, uuid.uuid4().hex[:12])
        store.directory.mkdir(parents=True, exist_ok=True)
        store.append(
            "metadata",
            {
                "task": task,
                "model": model,
                "workspace_name": workspace.resolve().name,
            },
        )
        return store

    @classmethod
    def open(cls, workspace: Path, session_id: str) -> SessionStore:
        """打开已有会话；普通文件之外的对象和符号链接均不接受。"""

        store = cls(workspace, session_id)
        if store.path.is_symlink():
            raise SessionError(f"Session path must not be a symbolic link: {session_id}")
        if not store.path.is_file():
            raise SessionError(f"Session not found: {session_id}")
        return store

    def append(self, kind: str, data: JsonObject) -> None:
        """以仅限当前用户的权限，可靠追加并立刻刷新一条事件。"""

        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.directory, 0o700)
        if self.path.is_symlink():
            raise SessionError("会话文件不能是符号链接")
        event = SessionEvent(
            version=SESSION_VERSION,
            kind=kind,
            data=data,
            timestamp=datetime.now(UTC).isoformat(),
        )
        line = json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            # flush 只送到操作系统缓冲；fsync 再要求操作系统尽量落盘。
            os.fsync(stream.fileno())
        os.chmod(self.path, 0o600)

    def append_message(self, message: Message) -> None:
        """把模型协议消息包装成统一的 ``message`` 类型事件。"""

        self.append("message", {"message": message})

    def delete(self) -> None:
        """删除单个会话文件；调用方应先取得用户确认。"""

        if self.path.is_symlink():
            raise SessionError("会话文件不能是符号链接")
        try:
            self.path.unlink()
        except FileNotFoundError as exc:
            raise SessionError(f"Session not found: {self.session_id}") from exc
        except OSError as exc:
            detail = exc.strerror or type(exc).__name__
            raise SessionError(f"Cannot delete session: {detail}") from exc

    def load(self) -> SessionData:
        """顺序校验事件并重建对话；仅容忍崩溃留下的最后一条未写完记录。"""

        metadata: JsonObject | None = None
        messages: list[Message] = []
        try:
            raw_text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SessionError(f"Cannot read session: {exc}") from exc
        lines = raw_text.splitlines()
        for line_number, line in enumerate(lines, start=1):
            try:
                raw: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                # 只有“文件最后没有换行且最后一行残缺”符合追加写中途崩溃的特征。
                # 中间坏行仍然报错，避免悄悄丢失一段历史后继续执行工具。
                if line_number == len(lines) and not raw_text.endswith("\n"):
                    break
                raise SessionError(f"Invalid JSON at session line {line_number}") from exc
            if raw.get("version") != SESSION_VERSION:
                raise SessionError(f"Unsupported session version at line {line_number}")
            kind = raw.get("kind")
            data = raw.get("data")
            if not isinstance(data, dict):
                raise SessionError(f"Invalid event data at line {line_number}")
            if kind == "metadata" and metadata is None:
                metadata = data
            elif kind == "message":
                message = data.get("message")
                if not isinstance(message, dict) or "role" not in message:
                    raise SessionError(f"Invalid message at line {line_number}")
                messages.append(message)
        if metadata is None:
            raise SessionError("Session metadata is missing")
        return SessionData(metadata=metadata, messages=messages)
