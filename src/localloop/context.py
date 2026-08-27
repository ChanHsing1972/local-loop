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
            results.append(f"{message.get('name', 'tool')}:{len(content)} chars")
        summary = ", ".join(results) or "no recorded result"
        return {
            "role": "assistant",
            "content": f"[Compacted earlier tool interaction: calls={names}; results={summary}]",
        }
    content = str(first.get("content", ""))
    preview = content[:240].replace("\n", " ")
    return {
        "role": first.get("role", "assistant"),
        "content": f"[Compacted earlier message: {len(content)} chars; preview={preview!r}]",
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
            raise ContextBudgetError("The pinned prompt exceeds the context budget")

        pinned = copied[:2]
        if message_chars(pinned) > self.max_chars:
            raise ContextBudgetError("System prompt and task exceed the context budget")
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
                            "content": (
                                "[Older compacted interactions omitted to fit the context budget]"
                            ),
                        }
                    ],
                )
                omitted = True
        result = flatten()
        if message_chars(result) > self.max_chars:
            raise ContextBudgetError(
                "Recent tool interactions exceed the context budget; start a new session"
            )
        return result
