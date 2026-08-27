from __future__ import annotations

import json

import pytest

from localloop.context import ContextBudgetError, ContextManager, message_chars


def tool_group(index: int, result_size: int = 600):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call-{index}",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "x"})},
                }
            ],
        },
        {
            "role": "tool",
            "name": "read_file",
            "tool_call_id": f"call-{index}",
            "content": "x" * result_size,
        },
    ]


def test_prepare_returns_copy_when_under_budget():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    prepared = ContextManager(max_chars=1_000).prepare(messages)
    assert prepared == messages
    assert prepared is not messages


def test_compacts_whole_tool_groups_and_preserves_recent_call_ids():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    messages += tool_group(1) + tool_group(2) + tool_group(3)
    prepared = ContextManager(max_chars=1_800, recent_groups=1).prepare(messages)
    serialized = json.dumps(prepared)
    assert "Compacted earlier tool interaction" in serialized
    assert "call-3" in serialized
    assert "call-1" not in serialized
    assert message_chars(prepared) <= 1_800


def test_omits_old_summaries_until_budget_fits():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    for index in range(12):
        messages += tool_group(index, 1_000)
    prepared = ContextManager(max_chars=2_200, recent_groups=2).prepare(messages)
    assert message_chars(prepared) <= 2_200
    assert any("omitted" in str(message.get("content")) for message in prepared)


def test_rejects_tiny_budget_and_oversized_pinned_prompt():
    with pytest.raises(ValueError, match="at least"):
        ContextManager(max_chars=999)
    with pytest.raises(ValueError, match="positive"):
        ContextManager(recent_groups=0)
    messages = [
        {"role": "system", "content": "s" * 900},
        {"role": "user", "content": "u" * 900},
        {"role": "assistant", "content": "extra"},
    ]
    with pytest.raises(ContextBudgetError, match="System prompt"):
        ContextManager(max_chars=1_000).prepare(messages)

