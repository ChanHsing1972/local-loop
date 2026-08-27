from __future__ import annotations

import json
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
        self.directory = self.workspace / ".localloop" / "sessions"
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
        if not store.path.is_file():
            raise SessionError(f"Session not found: {session_id}")
        return store

    def append(self, kind: str, data: JsonObject) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
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

    def append_message(self, message: Message) -> None:
        self.append("message", {"message": message})

    def load(self) -> SessionData:
        metadata: JsonObject | None = None
        messages: list[Message] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise SessionError(f"Cannot read session: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            try:
                raw: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
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
