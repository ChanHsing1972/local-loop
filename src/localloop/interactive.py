from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from localloop import __version__
from localloop.agent import AgentEngine, create_new_session, resume_session
from localloop.context import ContextManager
from localloop.policy import InteractiveApprovalPolicy
from localloop.provider import OpenAIChatProvider, ProviderError, probe_models
from localloop.session import SessionError, SessionStore
from localloop.tools import LocalTools
from localloop.types import AgentConfig, ChatProvider, Message, RunStatus

COMMANDS = (
    "/help",
    "/status",
    "/new",
    "/resume",
    "/sessions",
    "/models",
    "/model",
    "/approval",
    "/clear",
    "/exit",
    "/quit",
)

STATUS_LABELS = {
    RunStatus.COMPLETED: "已完成",
    RunStatus.MAX_STEPS: "达到步数限制",
    RunStatus.TIMED_OUT: "运行超时",
    RunStatus.REPEATED_CALL: "重复调用终止",
    RunStatus.INTERRUPTED: "已中断",
    RunStatus.ERROR: "运行错误",
}


def _default_provider(config: AgentConfig) -> ChatProvider:
    return OpenAIChatProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
    )


class InteractiveShell:
    """保持会话历史的交互式命令行外壳。"""

    def __init__(
        self,
        config: AgentConfig,
        *,
        console: Console | None = None,
        prompt_session: Any | None = None,
        provider_factory: Callable[[AgentConfig], ChatProvider] = _default_provider,
        approval_input: Callable[[str], str] = input,
    ) -> None:
        self.config = config
        self.console = console or Console()
        self.provider_factory = provider_factory
        self.approval_input = approval_input
        self.session: SessionStore | None = None
        self.messages: list[Message] | None = None
        history_dir = config.workspace / ".localloop"
        history_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_session = prompt_session or PromptSession(
            history=FileHistory(str(history_dir / "input_history")),
            completer=WordCompleter(COMMANDS, sentence=True),
            complete_while_typing=False,
            style=Style.from_dict(
                {
                    "prompt": "ansicyan bold",
                    "bottom-toolbar": "bg:#eeeeee #555555",
                }
            ),
        )
        self._rebuild_runtime()

    def _rebuild_runtime(self) -> None:
        provider = self.provider_factory(self.config)
        policy = InteractiveApprovalPolicy(
            auto_approve=self.config.auto_approve,
            input_fn=self.approval_input,
            output_fn=self._approval_output,
        )
        tools = LocalTools(self.config.workspace, policy)
        context = ContextManager(
            max_chars=self.config.max_context_chars,
            recent_groups=self.config.recent_groups,
        )
        self.engine = AgentEngine(
            provider=provider,
            tools=tools,
            context=context,
            max_steps=self.config.max_steps,
            max_duration_seconds=self.config.max_duration_seconds,
            output_fn=self._agent_output,
        )

    def run(self, *, initial_resume: str | None = None) -> int:
        self.show_banner()
        if initial_resume and not self.resume(initial_resume):
            return 2
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
        lines.append(f"LocalLoop v{__version__}\n", style="bold cyan")
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
        try:
            if self.session is None or self.messages is None:
                self.session, self.messages = create_new_session(
                    workspace=self.config.workspace,
                    task=task,
                    model=self.config.model,
                )
                self.console.print(f"新会话：[cyan]{self.session.session_id}[/cyan]")
            else:
                message: Message = {"role": "user", "content": task}
                self.messages.append(message)
                self.session.append_message(message)
            result = self.engine.run(self.messages, self.session)
        except (SessionError, ValueError) as exc:
            self.console.print(f"会话错误：{exc}", style="bold red")
            return
        label = STATUS_LABELS[result.status]
        style = "green" if result.status is RunStatus.COMPLETED else "yellow"
        self.console.print(
            f"{label} · {result.steps} 步 · 会话 {result.session_id}", style=f"dim {style}"
        )
        if result.status is not RunStatus.COMPLETED:
            error_style = "bold red" if result.status is RunStatus.ERROR else "yellow"
            self.console.print(result.final_output, style=error_style)

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
        self.console.print(f"已恢复会话 [cyan]{session_id}[/cyan]，可继续输入任务。")
        return True

    def _show_help(self) -> None:
        table = Table(title="可用命令", show_header=False, box=None)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column()
        rows = (
            ("/status", "显示模型、工作区、审批模式和当前会话"),
            ("/new", "结束当前上下文，下一条输入创建新会话"),
            ("/resume ID", "恢复指定会话"),
            ("/sessions", "列出最近会话"),
            ("/models", "从网关获取可用模型"),
            ("/model ID", "切换当前交互会话使用的模型"),
            ("/approval ask|auto", "切换逐次确认或自动批准"),
            ("/clear", "清屏并重新显示启动信息"),
            ("/exit", "退出 LocalLoop"),
        )
        for row in rows:
            table.add_row(*row)
        self.console.print(table)

    def _show_status(self) -> None:
        table = Table(title="当前状态", show_header=False)
        table.add_column(style="dim")
        table.add_column()
        table.add_row("模型", self.config.model)
        table.add_row("工作区", str(self.config.workspace))
        table.add_row("审批", "自动" if self.config.auto_approve else "逐次确认")
        table.add_row("会话", self.session.session_id if self.session else "尚未创建")
        table.add_row("步数上限", str(self.config.max_steps))
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
        if text.startswith("[步骤"):
            self.console.print("● " + text, style="dim cyan")
        elif text.startswith("[工具成功]"):
            self.console.print("  ✓ " + text, style="green")
        elif text.startswith("[工具失败]"):
            self.console.print("  ✗ " + text, style="red")
        elif text.startswith("[工具]"):
            self.console.print("  → " + text, style="cyan")
        else:
            self.console.print()
            self.console.print(Markdown(text))

    def _approval_output(self, text: str) -> None:
        if text.startswith("[需要批准]") or text.startswith("\n[需要批准]"):
            self.console.print(text, style="bold yellow", markup=False)
        elif text == "[已自动批准]":
            self.console.print(text, style="yellow")
        else:
            self.console.print(text, markup=False)
