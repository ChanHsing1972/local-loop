"""模块之间共享的消息、结果和接口类型。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

JsonObject = dict[str, Any]
Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """交互进程配置；repr 不显示 API 密钥。"""

    api_key: str = field(repr=False)
    base_url: str
    model: str
    workspace: Path
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
    steps: int


class ChatProvider(Protocol):
    def stream(
        self,
        messages: list[Message],
        tools: list[JsonObject],
        *,
        tool_choice: str | JsonObject = "auto",
        on_text_delta: Callable[[str], None],
        on_retry: Callable[[str], None] | None = None,
    ) -> AssistantTurn: ...


class ApprovalPolicy(Protocol):
    def approve(self, action: str, details: str, preview: str = "") -> bool: ...
