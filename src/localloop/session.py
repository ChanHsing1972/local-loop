"""在工作区内以追加式 JSONL 保存和恢复会话。"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from localloop.types import JsonObject, Message

SESSION_VERSION = 1
SESSION_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")


class SessionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SessionData:
    metadata: JsonObject
    messages: list[Message]


class SessionStore:
    def __init__(self, workspace: Path, session_id: str) -> None:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise SessionError("Invalid session id")
        self.workspace = workspace.resolve()
        self.session_id = session_id
        raw_directory = self.workspace / ".localloop" / "sessions"
        self.directory = raw_directory.resolve(strict=False)
        try:
            self.directory.relative_to(self.workspace)
        except ValueError as exc:
            raise SessionError("会话目录不能通过符号链接越过工作区") from exc
        self.path = self.directory / f"{session_id}.jsonl"

    @classmethod
    def create(cls, workspace: Path, *, task: str, model: str) -> SessionStore:
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
        store = cls(workspace, session_id)
        if store.path.is_symlink():
            raise SessionError(f"Session path must not be a symbolic link: {session_id}")
        if not store.path.is_file():
            raise SessionError(f"Session not found: {session_id}")
        return store

    def append(self, kind: str, data: JsonObject) -> None:
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.directory, 0o700)
        if self.path.is_symlink():
            raise SessionError("会话文件不能是符号链接")
        event = {
            "version": SESSION_VERSION,
            "kind": kind,
            "data": data,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(self.path, 0o600)

    def append_message(self, message: Message) -> None:
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
                # 追加写崩溃只会留下最后一行不完整；中间坏行仍必须报错。
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
