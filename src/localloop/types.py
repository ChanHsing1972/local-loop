"""跨模块共享的数据结构与接口约定。

可以把本文件理解为项目内部的“共同语言”：模型层产生 ``AssistantTurn``，工具层产生
``ToolResult``，编排层最终产生 ``RunResult``。这些类型不做网络或文件操作，避免业务
模块彼此依赖具体实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

# OpenAI Chat Completions 协议中的 JSON 对象和消息字段具有动态结构，因此保留 Any。
JsonObject = dict[str, Any]
Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """一次运行所需的全部配置。

    ``frozen=True`` 防止模块运行途中误改配置，``slots=True`` 减少动态属性错误。
    ``api_key`` 的 ``repr=False`` 可避免调试打印整个对象时泄露密钥。
    """

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
    """模型请求调用一个本地函数时返回的标准化数据。"""

    id: str
    name: str
    arguments: str

    def as_message_item(self) -> JsonObject:
        """转换回 OpenAI 协议要求的 assistant.tool_calls 元素。"""

        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """一次模型响应，可能是最终文本，也可能包含一个或多个工具调用。"""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: JsonObject | None = None

    def as_message(self) -> Message:
        """转换为可追加到对话历史的 assistant 消息。"""

        message: Message = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [call.as_message_item() for call in self.tool_calls]
        return message


@dataclass(frozen=True, slots=True)
class ToolResult:
    """一次本地工具执行结果及其对应的模型调用编号。"""

    tool_call_id: str
    name: str
    ok: bool
    content: str

    def as_message(self) -> Message:
        """转换为模型能够与原工具调用配对的 tool 消息。"""

        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
        }


class RunStatus(StrEnum):
    """Agent 主循环所有可能的终止原因。"""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    TIMED_OUT = "timed_out"
    REPEATED_CALL = "repeated_call"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RunResult:
    """一轮 Agent 执行结束后交给命令行界面的摘要。"""

    status: RunStatus
    final_output: str
    session_id: str
    steps: int


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """写入 JSONL 会话文件的一条事件。"""

    version: int
    kind: str
    data: JsonObject
    timestamp: str


class ChatProvider(Protocol):
    """AgentEngine 依赖的最小模型接口，便于替换网关和编写假模型测试。"""

    def complete(
        self,
        messages: list[Message],
        tools: list[JsonObject],
        *,
        tool_choice: str | JsonObject = "auto",
    ) -> AssistantTurn: ...


class ApprovalPolicy(Protocol):
    """工具层询问“本次危险操作是否允许”的统一接口。"""

    def approve(self, action: str, details: str, preview: str = "") -> bool: ...
