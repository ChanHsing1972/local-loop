from __future__ import annotations

import hashlib

from conftest import make_call

from localloop.agent import AgentEngine, create_new_session, resume_session
from localloop.context import ContextManager
from localloop.policy import AlwaysApprovePolicy
from localloop.provider import ProviderError
from localloop.session import SessionStore
from localloop.tools import LocalTools
from localloop.types import AssistantTurn, RunStatus


class FakeProvider:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def complete(self, messages, tools, *, tool_choice="auto"):
        self.requests.append((messages, tools, tool_choice))
        next_item = self.turns.pop(0)
        if isinstance(next_item, BaseException):
            raise next_item
        return next_item


def engine(tmp_path, provider, *, max_steps=20, clock=lambda: 0.0):
    return AgentEngine(
        provider=provider,
        tools=LocalTools(tmp_path, AlwaysApprovePolicy()),
        context=ContextManager(),
        max_steps=max_steps,
        max_duration_seconds=600,
        output_fn=lambda _text: None,
        clock=clock,
    )


def test_complete_multi_turn_run_and_resume_history(tmp_path):
    target = tmp_path / "bug.py"
    target.write_text("bad\n")
    digest = hashlib.sha256(b"bad\n").hexdigest()
    provider = FakeProvider(
        [
            AssistantTurn("", (make_call("read_file", {"path": "bug.py"}, "read"),)),
            AssistantTurn(
                "",
                (
                    make_call(
                        "write_file",
                        {"path": "bug.py", "content": "good\n", "expected_sha256": digest},
                        "write",
                    ),
                    make_call(
                        "run_command",
                        {"args": ["python3", "-c", "print('tests passed')"]},
                        "test",
                    ),
                ),
            ),
            AssistantTurn("Fixed the bug; tests passed."),
        ]
    )
    store, messages = create_new_session(workspace=tmp_path, task="fix bug", model="fake")
    result = engine(tmp_path, provider).run(messages, store)
    assert result.status is RunStatus.COMPLETED
    assert result.steps == 3
    assert target.read_text() == "good\n"
    resumed_store, resumed_messages = resume_session(
        workspace=tmp_path, session_id=store.session_id
    )
    assert resumed_store.session_id == store.session_id
    assert [message["role"] for message in resumed_messages].count("tool") == 3
    second_request = provider.requests[1][0]
    assert second_request[-1]["tool_call_id"] == "read"


def test_malformed_tool_error_is_returned_to_model(tmp_path):
    provider = FakeProvider(
        [
            AssistantTurn("", (type(make_call("read_file", {}))("bad", "read_file", "{"),)),
            AssistantTurn("Stopped after diagnosing the malformed call."),
        ]
    )
    store, messages = create_new_session(workspace=tmp_path, task="read", model="fake")
    result = engine(tmp_path, provider).run(messages, store)
    assert result.status is RunStatus.COMPLETED
    assert "Invalid JSON" in provider.requests[1][0][-1]["content"]


def test_three_identical_calls_stop_before_third_execution(tmp_path):
    calls = [
        type(make_call("list_files", {}))(
            f"id-{index}",
            "list_files",
            '{"path":".","max_depth":1}' if index % 2 else '{"max_depth":1,"path":"."}',
        )
        for index in range(3)
    ]
    provider = FakeProvider([AssistantTurn("", (call,)) for call in calls])
    store, messages = create_new_session(workspace=tmp_path, task="loop", model="fake")
    result = engine(tmp_path, provider).run(messages, store)
    assert result.status is RunStatus.REPEATED_CALL
    assert result.steps == 3


def test_max_steps_empty_response_provider_error_and_timeout(tmp_path):
    provider = FakeProvider(
        [
            AssistantTurn("", (make_call("list_files", {"path": ".", "max_depth": 1}),)),
            AssistantTurn("", (make_call("list_files", {"path": ".", "max_depth": 2}),)),
        ]
    )
    store, messages = create_new_session(workspace=tmp_path, task="loop", model="fake")
    maxed = engine(tmp_path, provider, max_steps=2).run(messages, store)
    assert maxed.status is RunStatus.MAX_STEPS

    empty_store, empty_messages = create_new_session(
        workspace=tmp_path, task="empty", model="fake"
    )
    empty = engine(tmp_path, FakeProvider([AssistantTurn("", finish_reason="stop")])).run(
        empty_messages, empty_store
    )
    assert empty.status is RunStatus.ERROR
    assert "neither text" in empty.final_output

    length_store, length_messages = create_new_session(
        workspace=tmp_path, task="length", model="fake"
    )
    length = engine(
        tmp_path, FakeProvider([AssistantTurn("partial", finish_reason="length")])
    ).run(length_messages, length_store)
    assert length.status is RunStatus.ERROR
    assert "truncated" in length.final_output

    error_store, error_messages = create_new_session(
        workspace=tmp_path, task="error", model="fake"
    )
    errored = engine(tmp_path, FakeProvider([ProviderError("gateway down")])).run(
        error_messages, error_store
    )
    assert errored.status is RunStatus.ERROR
    assert errored.final_output == "gateway down"

    times = iter([0.0, 700.0])
    timeout_store, timeout_messages = create_new_session(
        workspace=tmp_path, task="timeout", model="fake"
    )
    timed = engine(
        tmp_path, FakeProvider([]), clock=lambda: next(times)
    ).run(timeout_messages, timeout_store)
    assert timed.status is RunStatus.TIMED_OUT


def test_create_and_resume_reject_invalid_history(tmp_path):
    try:
        create_new_session(workspace=tmp_path, task="  ", model="fake")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    store = SessionStore.create(tmp_path, task="x", model="fake")
    store.append_message({"role": "system", "content": "only one"})
    try:
        resume_session(workspace=tmp_path, session_id=store.session_id)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
