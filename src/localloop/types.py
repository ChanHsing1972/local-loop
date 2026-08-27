from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

JsonObject = dict[str, Any]
Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """运行配置；对象表示会主动排除密钥。"""

    api_key: str = field(repr=False)
    base_url: str
    model: str
    workspace: Path
    max_steps: int = 20
    max_duration_seconds: int = 600
    max_context_chars: int = 60_000
    recent_groups: int = 6
    auto_approve: bool = False


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_message_item(self) -> JsonObject:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: JsonObject | None = None

    def as_message(self) -> Message:
        message: Message = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [call.as_message_item() for call in self.tool_calls]
        return message


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    ok: bool
    content: str

    def as_message(self) -> Message:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
        }


class RunStatus(StrEnum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    TIMED_OUT = "timed_out"
    REPEATED_CALL = "repeated_call"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    final_output: str
    session_id: str
    steps: int


@dataclass(frozen=True, slots=True)
class SessionEvent:
    version: int
    kind: str
    data: JsonObject
    timestamp: str


class ChatProvider(Protocol):
    def complete(
        self,
        messages: list[Message],
        tools: list[JsonObject],
        *,
        tool_choice: str | JsonObject = "auto",
    ) -> AssistantTurn: ...


class ApprovalPolicy(Protocol):
    def approve(self, action: str, details: str, preview: str = "") -> bool: ...
