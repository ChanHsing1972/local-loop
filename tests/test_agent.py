from __future__ import annotations

import hashlib

from conftest import Approval, make_call

from localloop.agent import AgentEngine, _describe_tool_result, create_new_session, resume_session
from localloop.context import ContextManager
from localloop.provider import ProviderError
from localloop.session import SessionStore
from localloop.tools import LocalTools
from localloop.types import AssistantTurn, RunStatus


class FakeProvider:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def stream(self, messages, tools, *, tool_choice="auto", on_text_delta, on_retry=None):
        self.requests.append((messages, tools, tool_choice))
        next_item = self.turns.pop(0)
        if isinstance(next_item, BaseException):
            raise next_item
        if next_item.content:
            on_text_delta(next_item.content)
        return next_item


def engine(tmp_path, provider, *, max_steps=20):
    return AgentEngine(
        provider=provider,
        tools=LocalTools(tmp_path, Approval()),
        context=ContextManager(),
        stream_fn=lambda _text: None,
        stream_end_fn=lambda: None,
        max_steps=max_steps,
        output_fn=lambda _text: None,
    )


def test_multi_turn_run_and_resume_history(tmp_path):
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
    events = []
    agent = engine(tmp_path, provider)
    agent.output_fn = events.append
    result = agent.run(messages, store)
    assert result.status is RunStatus.COMPLETED
    assert result.steps == 3
    assert target.read_text() == "good\n"
    assert events[0] == "[模型] 思考中（1）…"
    assert any("[工具] 读取 bug.py 第 1-400 行" in event for event in events)
    assert any("[工具成功] 已读取 bug.py 第 1-1 行" in event for event in events)
    assert any("运行 python3 -c" in event for event in events)
    assert any("退出码 0；输出：\n    tests passed" in event for event in events)
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
    tool_messages = [message for message in messages if message["role"] == "tool"]
    assert len(tool_messages) == 3
    assert "本次未执行" in tool_messages[-1]["content"]


def test_max_steps_empty_response_provider_error_and_timeout(tmp_path, monkeypatch):
    provider = FakeProvider(
        [
            AssistantTurn("", (make_call("list_files", {"path": ".", "max_depth": 1}),)),
            AssistantTurn("", (make_call("list_files", {"path": ".", "max_depth": 2}),)),
        ]
    )
    store, messages = create_new_session(workspace=tmp_path, task="loop", model="fake")
    maxed = engine(tmp_path, provider, max_steps=2).run(messages, store)
    assert maxed.status is RunStatus.MAX_STEPS

    empty_store, empty_messages = create_new_session(workspace=tmp_path, task="empty", model="fake")
    empty = engine(tmp_path, FakeProvider([AssistantTurn("", finish_reason="stop")])).run(
        empty_messages, empty_store
    )
    assert empty.status is RunStatus.ERROR
    assert "既未返回文本" in empty.final_output

    length_store, length_messages = create_new_session(
        workspace=tmp_path, task="length", model="fake"
    )
    length = engine(tmp_path, FakeProvider([AssistantTurn("partial", finish_reason="length")])).run(
        length_messages, length_store
    )
    assert length.status is RunStatus.ERROR
    assert "被截断" in length.final_output

    error_store, error_messages = create_new_session(workspace=tmp_path, task="error", model="fake")
    errored = engine(tmp_path, FakeProvider([ProviderError("gateway down")])).run(
        error_messages, error_store
    )
    assert errored.status is RunStatus.ERROR
    assert errored.final_output == "gateway down"

    times = iter([0.0, 700.0])
    timeout_store, timeout_messages = create_new_session(
        workspace=tmp_path, task="timeout", model="fake"
    )
    monkeypatch.setattr("localloop.agent.time.monotonic", lambda: next(times))
    timed = engine(tmp_path, FakeProvider([])).run(timeout_messages, timeout_store)
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


def test_interrupt_during_tool_fills_all_pending_results(tmp_path):
    first = make_call("list_files", {"path": "."}, "first")
    second = make_call("read_file", {"path": "missing"}, "second")
    provider = FakeProvider([AssistantTurn("", (first, second))])
    store, messages = create_new_session(workspace=tmp_path, task="interrupt", model="fake")
    agent = engine(tmp_path, provider)

    def interrupt(_call):
        raise KeyboardInterrupt

    agent.tools.execute = interrupt
    result = agent.run(messages, store)
    assert result.status is RunStatus.INTERRUPTED
    tool_messages = [message for message in messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["first", "second"]
    assert all("用户中断" in message["content"] for message in tool_messages)


def test_resume_repairs_trailing_incomplete_tool_calls_once(tmp_path):
    store, messages = create_new_session(workspace=tmp_path, task="crash", model="fake")
    first = make_call("list_files", {"path": "."}, "first")
    second = make_call("read_file", {"path": "a.py"}, "second")
    assistant = AssistantTurn("", (first, second)).as_message()
    first_result = {
        "role": "tool",
        "tool_call_id": "first",
        "name": "list_files",
        "content": '{"ok":true}',
    }
    store.append_message(assistant)
    store.append_message(first_result)

    resumed_store, resumed = resume_session(
        workspace=tmp_path,
        session_id=store.session_id,
    )
    assert resumed_store.session_id == store.session_id
    assert [message.get("tool_call_id") for message in resumed[-2:]] == ["first", "second"]
    assert "执行状态未知" in resumed[-1]["content"]

    _store_again, resumed_again = resume_session(
        workspace=tmp_path,
        session_id=store.session_id,
    )
    assert [message.get("tool_call_id") for message in resumed_again].count("second") == 1


def test_command_result_summary_preserves_lines_and_failure_details():
    success = _describe_tool_result(
        "run_command",
        '{"ok":true,"exit_code":0,"stdout":"first\\nsecond\\n","stderr":""}',
    )
    assert "输出：\n    first\n    second" in success

    failure = _describe_tool_result(
        "run_command",
        '{"ok":false,"exit_code":1,"stdout":"","stderr":"linker error\\nmore detail"}',
    )
    assert failure == "退出码 1；输出：\n    linker error\n    more detail"

    both_streams = _describe_tool_result(
        "run_command",
        '{"ok":false,"exit_code":2,"stdout":"partial","stderr":"actual error"}',
    )
    assert "stdout:\n    partial\n    stderr:\n    actual error" in both_streams

    ordinary_error = _describe_tool_result("write_file", '{"ok":false,"error":"file changed"}')
    assert ordinary_error == "write_file · file changed"
