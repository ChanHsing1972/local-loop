from __future__ import annotations

from io import StringIO
from threading import Event, Thread

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from localloop.interactive import (
    COMMANDS,
    InteractiveShell,
    SlashCommandCompleter,
    _command_key_bindings,
    _complete_commands_while_typing,
    _is_command_prefix,
)
from localloop.provider import ProviderError
from localloop.session import SessionError
from localloop.types import AgentConfig, AssistantTurn


class FakeProvider:
    def __init__(self, answers=None):
        self.answers = list(answers or ["完成。"])
        self.requests = []

    def stream(self, messages, tools, *, tool_choice="auto", on_text_delta, on_retry=None):
        self.requests.append((list(messages), tools, tool_choice))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        for character in answer:
            on_text_delta(character)
        return AssistantTurn(answer)


class ScriptedPrompt:
    def __init__(self, lines):
        self.lines = list(lines)

    def prompt(self, *_args, **_kwargs):
        item = self.lines.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def completions(text):
    return list(
        SlashCommandCompleter().get_completions(
            Document(text=text, cursor_position=len(text)),
            CompleteEvent(completion_requested=False),
        )
    )


def test_slash_opens_all_commands_with_descriptions():
    items = completions("/")
    assert [item.text for item in items] == [command.name for command in COMMANDS]
    assert all(item.display_meta_text for item in items)
    assert "/resume ID" in {item.display_text for item in items}


def test_slash_completion_filters_prefix_and_ignores_arguments():
    assert [item.text for item in completions("/mo")] == ["/models", "/model"]
    assert completions("普通任务") == []
    assert completions("/model ") == []


def test_prompt_only_enables_menu_space_for_command_prefix(tmp_path):
    config = AgentConfig(
        api_key="fresh-key",
        base_url="https://example.test/v1",
        model="model-a",
        workspace=tmp_path,
    )
    shell = InteractiveShell(config, provider_factory=lambda _config: FakeProvider())
    assert shell.prompt_session.reserve_space_for_menu == len(COMMANDS)
    assert _is_command_prefix("") is False
    assert _is_command_prefix("普通任务") is False
    assert _is_command_prefix("/") is True
    assert _is_command_prefix("/mo") is True
    assert _is_command_prefix("/model ") is False


def test_initial_slash_key_binding_inserts_and_opens_menu():
    bindings = _command_key_bindings()
    matching = [binding for binding in bindings.bindings if binding.keys == ("/",)]
    assert len(matching) == 1

    class Buffer:
        text = ""
        started_with = None

        def insert_text(self, value):
            self.text += value

        def start_completion(self, *, select_first):
            self.started_with = select_first

    buffer = Buffer()
    matching[0].handler(type("Event", (), {"current_buffer": buffer})())
    assert buffer.text == "/"
    assert buffer.started_with is True

    buffer = Buffer()
    buffer.text = "path"
    matching[0].handler(type("Event", (), {"current_buffer": buffer})())
    assert buffer.text == "path/"
    assert buffer.started_with is None


def test_real_prompt_session_renders_commands_and_executes_selection():
    """覆盖真实按键处理与渲染链路，防止只测试 Completer 却漏掉界面集成。"""

    class RecordingOutput(DummyOutput):
        def __init__(self):
            self.parts = []
            self.menu_rendered = Event()

        def write(self, data):
            self.parts.append(data)
            if "/quit" in "".join(self.parts):
                self.menu_rendered.set()

        write_raw = write

    with create_pipe_input() as pipe_input:
        output = RecordingOutput()
        session = PromptSession(
            input=pipe_input,
            output=output,
            completer=SlashCommandCompleter(),
            key_bindings=_command_key_bindings(),
            complete_while_typing=_complete_commands_while_typing(),
            reserve_space_for_menu=len(COMMANDS),
        )
        result = []
        prompt_thread = Thread(target=lambda: result.append(session.prompt("› ")))
        prompt_thread.start()
        pipe_input.send_text("/")
        menu_rendered = output.menu_rendered.wait(timeout=2)
        if not menu_rendered:
            pipe_input.send_text("\x03")
            prompt_thread.join(timeout=2)
        assert menu_rendered
        pipe_input.send_text("\r")
        prompt_thread.join(timeout=2)

    assert not prompt_thread.is_alive()
    assert result == ["/help"]
    rendered = "".join(output.parts)
    assert all(command.usage in rendered for command in COMMANDS)


def make_shell(tmp_path, *, answers=None, prompts=None, auto=False, provider=None):
    config = AgentConfig(
        api_key="fresh-key",
        base_url="https://example.test/v1",
        model="model-a",
        workspace=tmp_path,
        auto_approve=auto,
    )
    provider = provider or FakeProvider(answers)
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    shell = InteractiveShell(
        config,
        console=console,
        prompt_session=ScriptedPrompt(prompts or ["/exit"]),
        provider_factory=lambda _config: provider,
        approval_input=lambda _prompt: "y",
    )
    return shell, provider, output


def test_interactive_loop_keeps_multi_turn_history(tmp_path):
    shell, provider, output = make_shell(
        tmp_path,
        answers=["第一轮完成。", "第二轮完成。"],
        prompts=["修复问题", "继续补充测试", "/status", "/exit"],
    )
    assert shell.run() == 0
    assert len(provider.requests) == 2
    second_messages = provider.requests[1][0]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert second_messages[-1]["content"] == "继续补充测试"
    rendered = output.getvalue()
    assert "交互式编程智能体" in rendered
    assert "第一轮完成" in rendered
    assert "当前状态" in rendered


def test_interactive_shell_renders_streamed_text_once(tmp_path):
    provider = FakeProvider(["逐字输出成功。"])
    shell, _provider, output = make_shell(tmp_path, provider=provider)
    shell.submit_task("测试流式输出")
    rendered = output.getvalue()
    assert rendered.count("逐字输出成功。") == 1
    assert "已完成 · 1 步" in rendered


def test_resume_replays_user_and_assistant_messages(tmp_path):
    shell, _provider, output = make_shell(tmp_path, answers=["历史回复。"])
    shell.submit_task("历史任务")
    session_id = shell.session.session_id
    shell.handle_command("/new")
    output.seek(0)
    output.truncate(0)

    assert shell.resume(session_id) is True
    rendered = output.getvalue()
    assert "历史对话" in rendered
    assert "› 历史任务" in rendered
    assert "历史回复。" in rendered
    assert "你是 LocalLoop" not in rendered
    assert "历史已恢复，可继续输入任务" in rendered


def test_all_local_commands_and_resume(tmp_path, monkeypatch):
    shell, _provider, output = make_shell(tmp_path, answers=["完成。"])
    shell.submit_task("创建会话")
    session_id = shell.session.session_id
    assert shell.handle_command("/help") is False
    assert shell.handle_command("/sessions") is False
    assert shell.handle_command("/status") is False
    assert shell.handle_command("/model") is False
    assert shell.handle_command("/model model-b") is False
    assert shell.config.model == "model-b"
    assert shell.handle_command("/approval invalid") is False
    assert shell.handle_command("/approval auto") is False
    assert shell.config.auto_approve is True
    assert shell.handle_command("/approval ask") is False
    assert shell.config.auto_approve is False
    assert shell.handle_command("/new") is False
    assert shell.session is None
    assert shell.handle_command("/resume") is False
    assert shell.handle_command(f"/resume {session_id}") is False
    assert shell.session.session_id == session_id
    assert shell.handle_command("/delete") is False
    shell.approval_input = lambda _prompt: "n"
    assert shell.handle_command(f"/delete {session_id}") is False
    assert shell.session.session_id == session_id
    assert (tmp_path / ".localloop" / "sessions" / f"{session_id}.jsonl").exists()
    shell.approval_input = lambda _prompt: "y"
    assert shell.handle_command(f"/delete {session_id}") is False
    assert shell.session is None
    assert not (tmp_path / ".localloop" / "sessions" / f"{session_id}.jsonl").exists()
    assert shell.handle_command(f"/delete {session_id}") is False
    assert shell.handle_command("/does-not-exist") is False

    monkeypatch.setattr(
        "localloop.interactive.probe_models", lambda **_kwargs: ["model-a", "model-b"]
    )
    assert shell.handle_command("/models") is False
    monkeypatch.setattr("localloop.interactive.probe_models", lambda **_kwargs: [])
    shell.handle_command("/models")

    def broken_models(**_kwargs):
        raise ProviderError("网关失败")

    monkeypatch.setattr("localloop.interactive.probe_models", broken_models)
    shell.handle_command("/models")
    assert shell.handle_command("/clear") is False
    assert shell.handle_command("/quit") is True
    rendered = output.getvalue()
    assert "可用命令" in rendered
    assert "最近会话" in rendered
    assert "已切换模型" in rendered
    assert "已删除会话" in rendered
    assert "网关失败" in rendered


def test_empty_sessions_bad_resume_toolbar_and_output_styles(tmp_path):
    shell, _provider, output = make_shell(tmp_path)
    shell.handle_command("/sessions")
    assert shell.resume("abcdef123456") is False
    toolbar = shell._bottom_toolbar()
    assert "model-a" in toolbar[0][1]
    shell._agent_output("[模型] 正在分析任务…")
    shell._agent_output("[重试] 1 秒后重试")
    shell._agent_output("[工具] read_file")
    shell._agent_output("[工具成功] read_file")
    shell._agent_output("[工具失败] write_file")
    shell._agent_output("## 最终回复")
    shell._approval_output("\n[需要批准] write_file：a.py")
    shell._approval_output("[已自动批准]")
    shell._approval_output("diff")
    rendered = output.getvalue()
    assert "还没有会话记录" in rendered
    assert "无法恢复会话" in rendered
    assert "最终回复" in rendered


def test_eof_and_keyboard_interrupt_at_prompt(tmp_path):
    shell, _provider, output = make_shell(
        tmp_path,
        prompts=[KeyboardInterrupt(), EOFError()],
    )
    assert shell.run() == 0
    assert "Ctrl-D" in output.getvalue()
    assert "再见" in output.getvalue()


def test_interactive_shell_displays_provider_error(tmp_path):
    shell, _provider, output = make_shell(
        tmp_path,
        answers=[ProviderError("网关响应格式异常")],
    )
    shell.submit_task("列出文件")
    rendered = output.getvalue()
    assert "运行错误" in rendered
    assert "网关响应格式异常" in rendered


def test_interactive_history_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-history"
    outside.mkdir()
    (tmp_path / ".localloop").symlink_to(outside, target_is_directory=True)
    config = AgentConfig(
        api_key="fresh-key",
        base_url="https://example.test/v1",
        model="model-a",
        workspace=tmp_path,
    )
    with pytest.raises(SessionError, match="越过工作区"):
        InteractiveShell(
            config,
            provider_factory=lambda _config: FakeProvider(),
        )
