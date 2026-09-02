"""持续接收任务、显示流式结果并处理斜杠命令的终端界面。"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from localloop.agent import AgentEngine, create_new_session, resume_session
from localloop.checkpoint import CheckpointError, CheckpointStore
from localloop.context import ContextManager
from localloop.memory import WorkspaceMemoryError, WorkspaceMemoryStore
from localloop.provider import OpenAIChatProvider, ProviderError, probe_models
from localloop.session import SessionError, SessionStore
from localloop.tools import InteractiveApprovalPolicy, LocalTools
from localloop.types import AgentConfig, ChatProvider, Message, RunStatus


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    usage: str
    description: str


COMMANDS = (
    CommandSpec("/help", "/help", "显示全部交互命令"),
    CommandSpec("/status", "/status", "显示模型、工作区、审批模式和当前会话"),
    CommandSpec("/new", "/new", "结束当前上下文，下一条输入创建新会话"),
    CommandSpec("/resume", "/resume ID", "恢复指定会话"),
    CommandSpec("/sessions", "/sessions", "列出最近会话"),
    CommandSpec("/delete", "/delete ID", "删除指定会话记录"),
    CommandSpec("/remember", "/remember TEXT", "保存一条工作区记忆（下个新会话生效）"),
    CommandSpec("/memory", "/memory", "查看当前工作区记忆"),
    CommandSpec("/forget", "/forget ID", "遗忘指定工作区记忆"),
    CommandSpec("/checkpoints", "/checkpoints", "列出最近文件检查点"),
    CommandSpec("/undo", "/undo", "撤销最近一次 LocalLoop 文件写入"),
    CommandSpec("/models", "/models", "从网关获取可用模型"),
    CommandSpec("/model", "/model ID", "切换当前交互会话使用的模型"),
    CommandSpec("/approval", "/approval ask|auto", "切换逐次确认或自动批准"),
    CommandSpec("/clear", "/clear", "清屏并重新显示启动信息"),
    CommandSpec("/exit", "/exit", "退出 LocalLoop"),
    CommandSpec("/quit", "/quit", "退出 LocalLoop（/exit 的别名）"),
)
COMMAND_MENU_ROWS = len(COMMANDS) + 1  # 额外一行供 prompt-toolkit 绘制滚动区域。


class SlashCommandCompleter(Completer):
    """输入斜杠时展示所有命令及说明，并支持按前缀筛选。"""

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ):
        del complete_event
        text = document.text_before_cursor
        if not text.startswith("/") or any(character.isspace() for character in text):
            return
        for command in COMMANDS:
            if command.name.startswith(text):
                display: AnyFormattedText = [("class:completion.command", command.usage)]
                yield Completion(
                    command.name,
                    start_position=-len(text),
                    display=display,
                    display_meta=command.description,
                )


def _command_key_bindings() -> KeyBindings:
    """在首字符 `/` 被输入的同一个按键事件中打开补全菜单。

    `complete_while_typing` 只在部分终端和输入后端中可靠触发；显式绑定
    `/` 可以保证菜单不会因为输入事件或终端刷新时序而静默消失。
    """

    bindings = KeyBindings()

    @bindings.add("/", eager=True)
    def open_command_menu(event) -> None:
        buffer = event.current_buffer
        buffer.insert_text("/")
        if len(buffer.text) == 1:
            buffer.start_completion(select_first=True)

    @bindings.add("enter", eager=True)
    def execute_selected_command(event) -> None:
        """选中补全项后按一次 Enter，补全并立即提交该命令。"""

        buffer = event.current_buffer
        state = buffer.complete_state
        if state is not None and state.completions:
            # 菜单刚打开时可能还没有 current_completion，默认采用第一项。
            selected = state.current_completion or state.completions[0]
            buffer.apply_completion(selected)
        buffer.validate_and_handle()

    return bindings


def _is_command_prefix(text: str) -> bool:
    return text.startswith("/") and not any(character.isspace() for character in text)


def _complete_commands_while_typing() -> Condition:
    """只在输入斜杠命令时启用动态补全与菜单空间。"""

    return Condition(lambda: _is_command_prefix(get_app().current_buffer.text))


_STATUS_LABELS = {
    RunStatus.COMPLETED: "已完成",
    RunStatus.MAX_STEPS: "达到步数限制",
    RunStatus.TIMED_OUT: "运行超时",
    RunStatus.REPEATED_CALL: "重复调用终止",
    RunStatus.INTERRUPTED: "已中断",
    RunStatus.ERROR: "运行错误",
}


class InteractiveShell:
    def __init__(
        self,
        config: AgentConfig,
        *,
        console: Console | None = None,
        prompt_session: Any | None = None,
        provider_factory: Callable[[AgentConfig], ChatProvider] | None = None,
        approval_input: Callable[[str], str] = input,
    ) -> None:
        self.config = config
        self.console = console or Console()
        self.provider_factory = provider_factory
        self.approval_input = approval_input
        self.session: SessionStore | None = None
        self.messages: list[Message] | None = None
        self._stream_open = False
        history_dir = (config.workspace / ".localloop").resolve(strict=False)
        try:
            history_dir.relative_to(config.workspace)
        except ValueError as exc:
            raise SessionError("输入历史目录不能通过符号链接越过工作区") from exc
        history_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(history_dir, 0o700)
        history_path = history_dir / "input_history"
        if history_path.is_symlink():
            raise SessionError("输入历史文件不能是符号链接")
        history_path.touch(mode=0o600, exist_ok=True)
        os.chmod(history_path, 0o600)
        try:
            self.memory = WorkspaceMemoryStore(config.workspace)
            self.checkpoints = CheckpointStore(config.workspace)
        except (WorkspaceMemoryError, CheckpointError) as exc:
            raise SessionError(str(exc)) from exc
        self.prompt_session = prompt_session or PromptSession(
            history=FileHistory(str(history_path)),
            completer=SlashCommandCompleter(),
            key_bindings=_command_key_bindings(),
            complete_while_typing=_complete_commands_while_typing(),
            complete_style=CompleteStyle.MULTI_COLUMN,
            # 只有上面的动态条件成立或菜单已打开时，prompt-toolkit 才使用这些行。
            reserve_space_for_menu=COMMAND_MENU_ROWS,
            style=Style.from_dict(
                {
                    "prompt": "ansicyan bold",
                    "bottom-toolbar": "bg:#eeeeee #555555",
                    "completion-menu.completion": "bg:#f4f4f4 #222222",
                    "completion-menu.completion.current": "bg:#00a6b2 #ffffff bold",
                    "completion-menu.meta.completion": "bg:#f4f4f4 #777777",
                    "completion-menu.meta.completion.current": "bg:#00a6b2 #ffffff",
                    "completion.command": "ansicyan bold",
                }
            ),
        )
        self._rebuild_runtime()

    def _rebuild_runtime(self) -> None:
        if self.provider_factory:
            provider = self.provider_factory(self.config)
        else:
            provider = OpenAIChatProvider(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                model=self.config.model,
            )
        self.approval_policy = InteractiveApprovalPolicy(
            auto_approve=self.config.auto_approve,
            input_fn=self.approval_input,
            output_fn=self._approval_output,
        )
        tools = LocalTools(self.config.workspace, self.approval_policy, self.checkpoints)
        self.engine = AgentEngine(
            provider=provider,
            tools=tools,
            context=ContextManager(),
            output_fn=self._agent_output,
            stream_fn=self._agent_stream,
            stream_end_fn=self._agent_stream_end,
        )

    def run(self) -> int:
        self.show_banner()
        while True:
            try:
                line = self.prompt_session.prompt(
                    [("class:prompt", "› ")],
                    bottom_toolbar=self._bottom_toolbar,
                )
            except EOFError:
                self.console.print("\n再见。", style="dim")
                return 0
            except KeyboardInterrupt:
                self.console.print("按 Ctrl-D 或输入 /exit 退出。", style="yellow")
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if self.handle_command(line):
                    return 0
                continue
            self.submit_task(line)

    def show_banner(self) -> None:
        approval = "自动批准（仅限可信工作区）" if self.config.auto_approve else "每次确认"
        lines = Text()
        lines.append("LocalLoop\n", style="bold cyan")
        lines.append("模型：", style="dim")
        lines.append(self.config.model + "\n", style="bold")
        lines.append("目录：", style="dim")
        lines.append(str(self.config.workspace) + "\n")
        lines.append("审批：", style="dim")
        lines.append(approval)
        self.console.print(
            Panel(
                lines,
                title="交互式编程智能体",
                border_style="cyan",
                expand=False,
            )
        )
        self.console.print("直接输入编程任务；输入 [cyan]/help[/cyan] 查看命令。")

    def submit_task(self, task: str) -> None:
        checkpoint = None
        try:
            if self.session is None or self.messages is None:
                memories = self.memory.list_active()
                self.session, self.messages = create_new_session(
                    workspace=self.config.workspace,
                    task=task,
                    model=self.config.model,
                    workspace_memory=memories,
                )
                self.console.print(f"新会话：[cyan]{self.session.session_id}[/cyan]")
                if memories:
                    self.console.print(f"已加载 {len(memories)} 条工作区记忆。", style="dim cyan")
            else:
                message: Message = {"role": "user", "content": task}
                self.messages.append(message)
                self.session.append_message(message)
            self.checkpoints.begin(session_id=self.session.session_id, task=task)
            try:
                result = self.engine.run(self.messages, self.session)
            finally:
                checkpoint = self.checkpoints.finish()
        except (SessionError, WorkspaceMemoryError, CheckpointError, ValueError) as exc:
            self.console.print(f"会话错误：{exc}", style="bold red")
            return
        label = _STATUS_LABELS[result.status]
        style = "green" if result.status is RunStatus.COMPLETED else "yellow"
        self.console.print(
            f"{label} · {result.steps} 步 · 会话 {self.session.session_id}",
            style=f"dim {style}",
        )
        if result.status is not RunStatus.COMPLETED:
            error_style = "bold red" if result.status is RunStatus.ERROR else "yellow"
            self.console.print(result.final_output, style=error_style)
        if checkpoint:
            self.console.print(
                f"已创建文件检查点 [cyan]{checkpoint.id}[/cyan]，记录 "
                f"{len(checkpoint.files)} 个文件；输入 [cyan]/undo[/cyan] 可撤销。",
                style="green",
            )

    def handle_command(self, line: str) -> bool:
        command, _, argument = line.partition(" ")
        argument = argument.strip()
        if command in {"/exit", "/quit"}:
            self.console.print("再见。", style="dim")
            return True
        if command == "/help":
            self._show_help()
        elif command == "/status":
            self._show_status()
        elif command == "/new":
            self.session = None
            self.messages = None
            self.console.print("已清空当前上下文；下一条输入会创建新会话。", style="green")
        elif command == "/resume":
            if not argument:
                self.console.print("用法：/resume 会话编号", style="yellow")
            else:
                self.resume(argument)
        elif command == "/sessions":
            self._show_sessions()
        elif command == "/delete":
            if not argument:
                self.console.print("用法：/delete 会话编号", style="yellow")
            else:
                self._delete_session(argument)
        elif command == "/remember":
            self._remember(argument)
        elif command == "/memory":
            self._show_memory()
        elif command == "/forget":
            self._forget(argument)
        elif command == "/checkpoints":
            self._show_checkpoints()
        elif command == "/undo":
            self._undo_latest()
        elif command == "/models":
            self._show_models()
        elif command == "/model":
            self._change_model(argument)
        elif command == "/approval":
            self._change_approval(argument)
        elif command == "/clear":
            self.console.clear()
            self.show_banner()
        else:
            self.console.print(f"未知命令：{command}；输入 /help 查看帮助。", style="yellow")
        return False

    def resume(self, session_id: str) -> bool:
        try:
            store, messages = resume_session(
                workspace=self.config.workspace,
                session_id=session_id,
            )
        except (SessionError, ValueError) as exc:
            self.console.print(f"无法恢复会话：{exc}", style="bold red")
            return False
        self.session = store
        self.messages = messages
        self.console.print(f"已恢复会话 [cyan]{session_id}[/cyan]。")
        self._show_conversation_history(messages)
        self.console.print("历史已恢复，可继续输入任务。", style="green")
        return True

    def _show_conversation_history(self, messages: list[Message]) -> None:
        visible = [
            (str(message.get("role")), str(message.get("content")))
            for message in messages
            if message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
            and str(message.get("content")).strip()
        ]
        self.console.print("── 历史对话 ──", style="dim")
        for role, content in visible:
            if role == "user":
                self.console.print(f"› {content}", style="bold cyan", markup=False)
            else:
                self.console.print(Markdown(content))
        self.console.print("── 历史结束 ──", style="dim")

    def _show_help(self) -> None:
        table = Table(title="可用命令", show_header=False, box=None)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column()
        for command in COMMANDS:
            table.add_row(command.usage, command.description)
        self.console.print(table)

    def _show_status(self) -> None:
        table = Table(title="当前状态", show_header=False)
        table.add_column(style="dim")
        table.add_column()
        table.add_row("模型", self.config.model)
        table.add_row("工作区", str(self.config.workspace))
        table.add_row("审批", "自动" if self.config.auto_approve else "逐次确认")
        table.add_row("会话", self.session.session_id if self.session else "尚未创建")
        table.add_row("步数上限", str(self.engine.max_steps))
        self.console.print(table)

    def _show_sessions(self) -> None:
        session_dir = self.config.workspace / ".localloop" / "sessions"
        paths = sorted(
            session_dir.glob("*.jsonl"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not paths:
            self.console.print("当前工作区还没有会话记录。", style="dim")
            return
        table = Table(title="最近会话")
        table.add_column("会话编号", style="cyan")
        table.add_column("任务")
        table.add_column("模型")
        for path in paths[:10]:
            session_id = path.stem
            try:
                data = SessionStore.open(self.config.workspace, session_id).load()
                task = str(data.metadata.get("task", ""))
                model = str(data.metadata.get("model", ""))
            except SessionError:
                task, model = "记录损坏", "-"
            table.add_row(session_id, task[:48], model)
        self.console.print(table)

    def _delete_session(self, session_id: str) -> None:
        try:
            store = SessionStore.open(self.config.workspace, session_id)
        except SessionError as exc:
            self.console.print(f"无法删除会话：{exc}", style="bold red")
            return
        try:
            answer = self.approval_input(f"确定删除会话 {session_id}？[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in {"y", "yes", "是"}:
            self.console.print("已取消删除。", style="dim")
            return
        try:
            store.delete()
        except SessionError as exc:
            self.console.print(f"无法删除会话：{exc}", style="bold red")
            return
        if self.session and self.session.session_id == session_id:
            self.session = None
            self.messages = None
        self.console.print(f"已删除会话 [cyan]{session_id}[/cyan]。", style="green")

    def _remember(self, text: str) -> None:
        if not text:
            self.console.print("用法：/remember 要长期保存的项目约定", style="yellow")
            return
        try:
            entry = self.memory.remember(text)
        except WorkspaceMemoryError as exc:
            self.console.print(f"无法保存记忆：{exc}", style="bold red")
            return
        self.console.print(
            f"已保存工作区记忆 [cyan]{entry.id}[/cyan]；将在下一个新会话生效。",
            style="green",
        )

    def _show_memory(self) -> None:
        try:
            entries = self.memory.list_active()
        except WorkspaceMemoryError as exc:
            self.console.print(f"无法读取记忆：{exc}", style="bold red")
            return
        if not entries:
            self.console.print("当前工作区还没有记忆。", style="dim")
            return
        table = Table(title="工作区记忆")
        table.add_column("编号", style="cyan", no_wrap=True)
        table.add_column("内容")
        for entry in entries:
            table.add_row(entry.id, entry.text)
        self.console.print(table)

    def _forget(self, memory_id: str) -> None:
        if not memory_id:
            self.console.print("用法：/forget 记忆编号", style="yellow")
            return
        try:
            entry = self.memory.forget(memory_id)
        except WorkspaceMemoryError as exc:
            self.console.print(f"无法遗忘记忆：{exc}", style="bold red")
            return
        self.console.print(
            f"已遗忘工作区记忆 [cyan]{entry.id}[/cyan]；当前会话的记忆快照不会改变。",
            style="green",
        )

    def _show_checkpoints(self) -> None:
        try:
            checkpoints = self.checkpoints.list_checkpoints()
        except CheckpointError as exc:
            self.console.print(f"无法读取检查点：{exc}", style="bold red")
            return
        if not checkpoints:
            self.console.print("当前工作区还没有文件检查点。", style="dim")
            return
        labels = {"completed": "可撤销", "in_progress": "中断", "restored": "已撤销"}
        table = Table(title="最近文件检查点")
        table.add_column("编号", style="cyan", no_wrap=True)
        table.add_column("状态", no_wrap=True)
        table.add_column("任务")
        table.add_column("文件")
        for checkpoint in checkpoints:
            table.add_row(
                checkpoint.id,
                labels.get(checkpoint.status, checkpoint.status),
                checkpoint.task[:48],
                ", ".join(checkpoint.files),
            )
        self.console.print(table)

    def _undo_latest(self) -> None:
        try:
            checkpoint = self.checkpoints.undo_latest(self.approval_policy)
        except CheckpointError as exc:
            self.console.print(f"无法撤销：{exc}", style="yellow")
            return
        self.session = None
        self.messages = None
        self.console.print(
            f"已撤销检查点 [cyan]{checkpoint.id}[/cyan]，恢复 {len(checkpoint.files)} 个文件。",
            style="green",
        )
        self.console.print("当前对话上下文已清空；下一条输入会创建新会话。", style="dim")

    def _show_models(self) -> None:
        self.console.print("正在查询模型…", style="dim cyan")
        try:
            models = probe_models(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        except ProviderError as exc:
            self.console.print(f"模型查询失败：{exc}", style="bold red")
            return
        if not models:
            self.console.print("网关没有返回模型编号。", style="yellow")
            return
        for model in models:
            marker = "  ← 当前" if model == self.config.model else ""
            self.console.print(f"  {model}{marker}")

    def _change_model(self, model: str) -> None:
        if not model:
            self.console.print(f"当前模型：{self.config.model}；用法：/model 模型编号")
            return
        self.config = replace(self.config, model=model)
        self._rebuild_runtime()
        if self.session:
            self.session.append("runtime_change", {"model": model})
        self.console.print(f"已切换模型：[bold cyan]{model}[/bold cyan]")

    def _change_approval(self, mode: str) -> None:
        normalized = mode.lower()
        if normalized not in {"ask", "auto"}:
            self.console.print("用法：/approval ask|auto", style="yellow")
            return
        auto = normalized == "auto"
        self.config = replace(self.config, auto_approve=auto)
        self._rebuild_runtime()
        if self.session:
            self.session.append("runtime_change", {"auto_approve": auto})
        if auto:
            self.console.print(
                "已启用自动批准；请仅在可信、可恢复的工作区使用。", style="bold yellow"
            )
        else:
            self.console.print("已恢复写入和命令逐次确认。", style="green")

    def _bottom_toolbar(self) -> list[tuple[str, str]]:
        approval = "自动批准" if self.config.auto_approve else "逐次确认"
        session = self.session.session_id if self.session else "新会话"
        text = f" {self.config.model} · {self.config.workspace.name} · {approval} · {session} "
        return [("class:bottom-toolbar", text)]

    def _agent_output(self, text: str) -> None:
        if text.startswith("[模型]"):
            self.console.print("● " + text, style="dim cyan")
        elif text.startswith("[重试]"):
            self.console.print("  ↻ " + text, style="yellow")
        elif text.startswith("[工具成功]"):
            self.console.print("  ✓ " + text, style="green")
        elif text.startswith("[工具失败]"):
            self.console.print("  ✗ " + text, style="red")
        elif text.startswith("[工具]"):
            self.console.print("  → " + text, style="cyan")
        else:
            self.console.print()
            self.console.print(Markdown(text))

    def _agent_stream(self, text: str) -> None:
        if not self._stream_open:
            self.console.print()
            self.console.print("● ", style="cyan", end="")
            self._stream_open = True
        self.console.print(text, markup=False, end="", soft_wrap=True)

    def _agent_stream_end(self) -> None:
        if self._stream_open:
            self.console.print()
            self._stream_open = False

    def _approval_output(self, text: str) -> None:
        if text.startswith("[需要批准]") or text.startswith("\n[需要批准]"):
            self.console.print(text, style="bold yellow", markup=False)
        elif text == "[已自动批准]":
            self.console.print(text, style="yellow")
        else:
            self.console.print(text, markup=False)
