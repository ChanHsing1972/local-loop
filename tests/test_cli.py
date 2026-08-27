from __future__ import annotations

import pytest

from localloop.agent import create_new_session
from localloop.cli import main
from localloop.provider import ProviderError
from localloop.types import AssistantTurn, RunResult, RunStatus, ToolCall


def test_doctor_without_key_is_actionable(monkeypatch, capsys):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert main(["doctor"]) == 2
    captured = capsys.readouterr()
    assert "API key: missing" in captured.out
    assert "newly issued key" in captured.err


def test_run_requires_exactly_task_or_resume(capsys):
    assert main(["run"]) == 2
    assert "TASK is required" in capsys.readouterr().err
    assert main(["run", "task", "--resume", "abcdef123456"]) == 2
    assert "omit TASK" in capsys.readouterr().err


def test_run_reports_missing_configuration(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert main(["run", "task", "--workspace", str(tmp_path)]) == 2
    assert "LLM_API_KEY" in capsys.readouterr().err


def configured(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fresh-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "model-a")


def test_doctor_lists_models_and_can_skip_paid_check(monkeypatch, capsys):
    configured(monkeypatch)
    monkeypatch.setattr("localloop.cli.probe_models", lambda **_kwargs: ["model-a", "model-b"])
    assert main(["doctor", "--skip-tool-check"]) == 0
    output = capsys.readouterr().out
    assert "model-a *" in output
    assert "tool calling check: skipped" in output


def test_doctor_native_tool_check_success_and_failure(monkeypatch, capsys):
    configured(monkeypatch)
    monkeypatch.setattr("localloop.cli.probe_models", lambda **_kwargs: ["model-a"])

    class DoctorProvider:
        def __init__(self, **_kwargs):
            pass

        def complete(self, *_args, **_kwargs):
            return AssistantTurn(
                "", (ToolCall("id", "doctor_echo", '{"value":"ok"}'),)
            )

    monkeypatch.setattr("localloop.cli.OpenAIChatProvider", DoctorProvider)
    assert main(["doctor"]) == 0
    assert "tool calling check: passed" in capsys.readouterr().out

    class WrongProvider(DoctorProvider):
        def complete(self, *_args, **_kwargs):
            return AssistantTurn("plain text")

    monkeypatch.setattr("localloop.cli.OpenAIChatProvider", WrongProvider)
    assert main(["doctor"]) == 2
    assert "did not return" in capsys.readouterr().err


def test_doctor_probe_and_tool_errors_are_actionable(monkeypatch, capsys):
    configured(monkeypatch)

    def bad_probe(**_kwargs):
        raise ProviderError("bad gateway")

    monkeypatch.setattr("localloop.cli.probe_models", bad_probe)
    assert main(["doctor"]) == 2
    assert "bad gateway" in capsys.readouterr().err

    monkeypatch.setattr("localloop.cli.probe_models", lambda **_kwargs: ["model-a"])

    class BrokenProvider:
        def __init__(self, **_kwargs):
            pass

        def complete(self, *_args, **_kwargs):
            raise ProviderError("tool probe broke")

    monkeypatch.setattr("localloop.cli.OpenAIChatProvider", BrokenProvider)
    assert main(["doctor"]) == 2
    assert "tool probe broke" in capsys.readouterr().err


@pytest.mark.parametrize("status", [RunStatus.COMPLETED, RunStatus.ERROR])
def test_run_constructs_engine_and_maps_status(tmp_path, monkeypatch, capsys, status):
    configured(monkeypatch)

    class FakeEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, _messages, session):
            output = "done" if status is RunStatus.COMPLETED else "failed"
            return RunResult(status, output, session.session_id, 2)

    monkeypatch.setattr("localloop.cli.OpenAIChatProvider", lambda **_kwargs: object())
    monkeypatch.setattr("localloop.cli.AgentEngine", FakeEngine)
    code = main(
        ["run", "task", "--workspace", str(tmp_path), "--auto-approve", "--max-steps", "3"]
    )
    assert code == (0 if status is RunStatus.COMPLETED else 1)
    captured = capsys.readouterr()
    assert "session:" in captured.out
    assert "auto-approve" in captured.out
    if status is RunStatus.ERROR:
        assert "failed" in captured.err


def test_run_can_resume_existing_session(tmp_path, monkeypatch):
    configured(monkeypatch)
    store, _messages = create_new_session(workspace=tmp_path, task="task", model="model-a")

    class FakeEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, _messages, session):
            return RunResult(RunStatus.COMPLETED, "done", session.session_id, 1)

    monkeypatch.setattr("localloop.cli.OpenAIChatProvider", lambda **_kwargs: object())
    monkeypatch.setattr("localloop.cli.AgentEngine", FakeEngine)
    assert (
        main(["run", "--workspace", str(tmp_path), "--resume", store.session_id]) == 0
    )
