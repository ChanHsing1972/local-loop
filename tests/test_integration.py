from __future__ import annotations

import hashlib
import subprocess
import sys

from conftest import Approval, make_call

from localloop.agent import AgentEngine, create_new_session
from localloop.context import ContextManager
from localloop.tools import LocalTools
from localloop.types import AssistantTurn, RunStatus


class ScriptedProvider:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def stream(self, messages, tools, *, tool_choice="auto", on_text_delta, on_retry=None):
        self.requests.append(list(messages))
        turn = self.turns.pop(0)
        if turn.content:
            on_text_delta(turn.content)
        return turn


def test_agent_repairs_bug_and_runs_real_tests_in_temporary_git_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "calculator.py"
    source.write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
    test_file = tmp_path / "test_calculator.py"
    test_file.write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    provider = ScriptedProvider(
        [
            AssistantTurn("", (make_call("list_files", {"path": "."}, "list"),)),
            AssistantTurn(
                "",
                (
                    make_call("read_file", {"path": "calculator.py"}, "read-source"),
                    make_call("read_file", {"path": "test_calculator.py"}, "read-test"),
                ),
            ),
            AssistantTurn(
                "",
                (
                    make_call(
                        "write_file",
                        {
                            "path": "calculator.py",
                            "content": "def add(left, right):\n    return left + right\n",
                            "expected_sha256": digest,
                        },
                        "write",
                    ),
                ),
            ),
            AssistantTurn(
                "",
                (
                    make_call(
                        "run_command",
                        {
                            "args": [sys.executable, "-m", "pytest", "-q"],
                            "timeout_seconds": 30,
                        },
                        "test",
                    ),
                ),
            ),
            AssistantTurn("缺陷已修复，测试通过。"),
        ]
    )
    store, messages = create_new_session(
        workspace=tmp_path,
        task="修复加法缺陷并运行测试",
        model="fake",
    )
    engine = AgentEngine(
        provider=provider,
        tools=LocalTools(tmp_path, Approval()),
        context=ContextManager(),
        stream_fn=lambda _text: None,
        stream_end_fn=lambda: None,
        max_steps=10,
        max_duration_seconds=120,
        output_fn=lambda _text: None,
    )

    result = engine.run(messages, store)

    assert result.status is RunStatus.COMPLETED
    assert result.steps == 5
    assert "left + right" in source.read_text(encoding="utf-8")
    test_result = next(message for message in messages if message.get("tool_call_id") == "test")
    assert '"exit_code":0' in test_result["content"]
    assert "1 passed" in test_result["content"]
    assert [request[-1]["role"] for request in provider.requests] == [
        "user",
        "tool",
        "tool",
        "tool",
        "tool",
    ]
