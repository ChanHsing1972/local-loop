from __future__ import annotations

import pytest

from localloop.cli import main
from localloop.session import SessionError


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "fresh-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "model-a")


def test_no_arguments_launches_interactive_shell(tmp_path, monkeypatch, configured):
    monkeypatch.chdir(tmp_path)
    calls = []

    class FakeShell:
        def __init__(self, config):
            calls.append(config.workspace)

        def run(self):
            return 0

    monkeypatch.setattr("localloop.cli.InteractiveShell", FakeShell)
    assert main([]) == 0
    assert calls == [tmp_path.resolve()]


@pytest.mark.parametrize("arguments", [["run"], ["doctor"], ["chat"], ["--help"]])
def test_arguments_and_old_subcommands_are_rejected(arguments, capsys):
    assert main(arguments) == 2
    assert "不支持参数或子命令" in capsys.readouterr().err


def test_configuration_and_local_state_errors_are_clear(tmp_path, monkeypatch, configured, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY")
    assert main([]) == 2
    assert "配置错误" in capsys.readouterr().err

    monkeypatch.setenv("LLM_API_KEY", "fresh-key")

    class BrokenShell:
        def __init__(self, _config):
            raise SessionError("状态目录越界")

    monkeypatch.setattr("localloop.cli.InteractiveShell", BrokenShell)
    assert main([]) == 2
    assert "本地状态错误" in capsys.readouterr().err
