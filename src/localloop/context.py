from __future__ import annotations

import json
from copy import deepcopy

from localloop.types import Message


class ContextBudgetError(ValueError):
    pass


def message_chars(messages: list[Message]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))


def _group_messages(messages: list[Message]) -> list[list[Message]]:
    groups: list[list[Message]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        group = [message]
        index += 1
        if message.get("role") == "assistant" and message.get("tool_calls"):
            while index < len(messages) and messages[index].get("role") == "tool":
                group.append(messages[index])
                index += 1
        groups.append(group)
    return groups


def _compact_group(group: list[Message]) -> Message:
    first = group[0]
    if first.get("role") == "assistant" and first.get("tool_calls"):
        names = [
            str(call.get("function", {}).get("name", "unknown"))
            for call in first.get("tool_calls", [])
        ]
        results = []
        for message in group[1:]:
            content = str(message.get("content", ""))
            try:
                payload = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                payload = None
            ok = payload.get("ok") if isinstance(payload, dict) else None
            status = "成功" if ok is True else "失败" if ok is False else "未知"
            preview = content[:120].replace("\n", " ")
            results.append(
                f"{message.get('name', 'tool')}:{status}:{len(content)}字符:预览={preview!r}"
            )
        summary = "；".join(results) or "没有记录工具结果"
        return {
            "role": "assistant",
            "content": f"[较早工具交互已压缩：调用={names}；结果={summary}]",
        }
    content = str(first.get("content", ""))
    preview = content[:240].replace("\n", " ")
    return {
        "role": first.get("role", "assistant"),
        "content": f"[较早消息已压缩：{len(content)} 字符；预览={preview!r}]",
    }


class ContextManager:
    """不调用第二个模型、不会产生摘要幻觉的确定性上下文压缩器。"""

    def __init__(self, max_chars: int = 60_000, recent_groups: int = 6) -> None:
        if max_chars < 1_000:
            raise ValueError("max_chars must be at least 1000")
        if recent_groups < 1:
            raise ValueError("recent_groups must be positive")
        self.max_chars = max_chars
        self.recent_groups = recent_groups

    def prepare(self, messages: list[Message]) -> list[Message]:
        copied = deepcopy(messages)
        if message_chars(copied) <= self.max_chars:
            return copied
        if len(copied) < 2:
            raise ContextBudgetError("固定提示词超过上下文预算")

        pinned = copied[:2]
        if message_chars(pinned) > self.max_chars:
            raise ContextBudgetError("系统提示词与原始任务超过上下文预算")
        groups = _group_messages(copied[2:])
        split = max(0, len(groups) - self.recent_groups)
        prepared_groups = [[_compact_group(group)] for group in groups[:split]] + groups[split:]

        def flatten() -> list[Message]:
            return pinned + [message for group in prepared_groups for message in group]

        omitted = False
        while message_chars(flatten()) > self.max_chars and len(prepared_groups) > 2:
            prepared_groups.pop(1 if omitted else 0)
            if not omitted:
                prepared_groups.insert(
                    0,
                    [
                        {
                            "role": "assistant",
                            "content": "[为满足上下文预算，已省略更早的压缩交互]",
                        }
                    ],
                )
                omitted = True
        result = flatten()
        if message_chars(result) > self.max_chars:
            raise ContextBudgetError(
                "最近的工具交互超过上下文预算，请开始新会话"
            )
        return result
