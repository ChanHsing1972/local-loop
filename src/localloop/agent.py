from __future__ import annotations

import json
import shlex
import time
from collections.abc import Callable, Iterable

from localloop.context import ContextBudgetError, ContextManager
from localloop.provider import ProviderError
from localloop.session import SessionStore
from localloop.tools import LocalTools
from localloop.types import (
    AssistantTurn,
    ChatProvider,
    Message,
    RunResult,
    RunStatus,
    ToolCall,
    ToolResult,
)

SYSTEM_PROMPT = """你是 LocalLoop，一个仅在指定工作区内工作的编程智能体。
请使用提供的本地工具检查项目、完成聚焦的修改并验证结果。始终使用用户的语言回答。
规则：
1. 只有工具结果能够证明文件已修改或命令已通过，绝不能凭空宣称成功。
2. 更新现有文件前必须先读取，并把最近读取所得的 SHA-256 传给 write_file。
3. 优先进行小而可审查的修改，完成后运行最相关的测试。
4. 不得尝试访问工作区之外的路径或敏感凭据。
5. 工具返回错误时，应诊断原因并修正调用；无法解决时要明确说明阻碍。
6. 任务完成后，不再调用工具，简洁总结修改内容与已运行的测试。
7. 当前工作区已由程序确定，不要再次询问路径；安全控制只能称为防误操作机制，不能称为完整沙箱。
8. 寒暄、能力说明或不依赖项目事实的问题应直接回答，不要调用本地工具。
"""


class AgentEngine:
    def __init__(
        self,
        *,
        provider: ChatProvider,
        tools: LocalTools,
        context: ContextManager,
        max_steps: int,
        max_duration_seconds: int,
        output_fn: Callable[[str], None] = print,
        stream_fn: Callable[[str], None] | None = None,
        stream_end_fn: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.context = context
        self.max_steps = max_steps
        self.max_duration_seconds = max_duration_seconds
        self.output_fn = output_fn
        self.stream_fn = stream_fn
        self.stream_end_fn = stream_end_fn or (lambda: None)
        self.clock = clock

    def run(self, messages: list[Message], session: SessionStore) -> RunResult:
        start = self.clock()
        last_signature = ""
        repeated = 0
        for step in range(1, self.max_steps + 1):
            if self.clock() - start >= self.max_duration_seconds:
                return self._finish(
                    session,
                    RunStatus.TIMED_OUT,
                    "智能体已达到总时间限制并停止。",
                    step - 1,
                )
            self.output_fn(f"[模型] 思考中（{step}）…")
            try:
                turn = self._request_model(messages)
            except (ProviderError, ContextBudgetError) as exc:
                return self._finish(session, RunStatus.ERROR, str(exc), step - 1)
            except KeyboardInterrupt:
                return self._finish(
                    session,
                    RunStatus.INTERRUPTED,
                    "运行已中断；可使用 --resume " + session.session_id + " 继续",
                    step - 1,
                )

            assistant_message = turn.as_message()
            messages.append(assistant_message)
            session.append_message(assistant_message)
            if turn.usage:
                session.append("usage", {"step": step, "usage": turn.usage})

            if not turn.tool_calls:
                return self._finish_text_turn(turn, session, step)

            signature = _tool_call_signature(turn.tool_calls)
            if signature == last_signature:
                repeated += 1
            else:
                last_signature = signature
                repeated = 1
            if repeated >= 3:
                _append_failed_tool_results(
                    messages,
                    session,
                    turn.tool_calls,
                    "相同工具调用已连续出现三次，本次未执行并终止循环",
                )
                return self._finish(
                    session,
                    RunStatus.REPEATED_CALL,
                    "智能体连续三次重复相同工具调用，已停止运行。",
                    step,
                )

            if not self._execute_tools(turn.tool_calls, messages, session):
                return self._finish(
                    session,
                    RunStatus.INTERRUPTED,
                    "运行已中断；可使用 --resume " + session.session_id + " 继续",
                    step,
                )

        return self._finish(
            session,
            RunStatus.MAX_STEPS,
            f"智能体达到 {self.max_steps} 步限制，已停止运行。",
            self.max_steps,
        )

    def _request_model(self, messages: list[Message]) -> AssistantTurn:
        request_messages = self.context.prepare(messages)
        stream_method = getattr(self.provider, "stream", None)
        if self.stream_fn is None or not callable(stream_method):
            return self.provider.complete(request_messages, self.tools.definitions)
        try:
            return stream_method(
                request_messages,
                self.tools.definitions,
                on_text_delta=self.stream_fn,
                on_retry=lambda detail: self.output_fn(f"[重试] {detail}"),
            )
        finally:
            self.stream_end_fn()

    def _finish_text_turn(
        self,
        turn: AssistantTurn,
        session: SessionStore,
        step: int,
    ) -> RunResult:
        if turn.finish_reason == "length":
            return self._finish(session, RunStatus.ERROR, "模型输出在任务完成前被截断。", step)
        if turn.content.strip():
            if self.stream_fn is None or not callable(getattr(self.provider, "stream", None)):
                self.output_fn(turn.content)
            return self._finish(session, RunStatus.COMPLETED, turn.content.strip(), step)
        reason = f" (finish_reason={turn.finish_reason})" if turn.finish_reason else ""
        return self._finish(
            session,
            RunStatus.ERROR,
            "模型既未返回文本，也未返回工具调用" + reason,
            step,
        )

    def _execute_tools(
        self,
        calls: tuple[ToolCall, ...],
        messages: list[Message],
        session: SessionStore,
    ) -> bool:
        for index, call in enumerate(calls):
            self.output_fn(f"[工具] {_describe_tool_call(call.name, call.arguments)}")
            try:
                result = self.tools.execute(call)
            except KeyboardInterrupt:
                _append_failed_tool_results(
                    messages,
                    session,
                    calls[index:],
                    "用户中断了工具执行；当前或本批次剩余调用未正常完成",
                )
                return False
            result_message = result.as_message()
            messages.append(result_message)
            session.append_message(result_message)
            status = "成功" if result.ok else "失败"
            self.output_fn(f"[工具{status}] {_describe_tool_result(call.name, result.content)}")
        return True

    @staticmethod
    def _finish(
        session: SessionStore,
        status: RunStatus,
        output: str,
        steps: int,
    ) -> RunResult:
        session.append("terminal", {"status": status.value, "output": output, "steps": steps})
        return RunResult(
            status=status,
            final_output=output,
            session_id=session.session_id,
            steps=steps,
        )


def _describe_tool_call(name: str, raw_arguments: str) -> str:
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        return f"{name} · 参数 JSON 无法解析"
    if not isinstance(arguments, dict):
        return f"{name} · 参数不是对象"
    if name == "read_file":
        start = arguments.get("start_line", 1)
        end = arguments.get("end_line", start + 399 if isinstance(start, int) else "?")
        return f"读取 {arguments.get('path', '?')} 第 {start}-{end} 行"
    if name == "list_files":
        return f"列出 {arguments.get('path', '.')}（深度 {arguments.get('max_depth', 3)}）"
    if name == "search_text":
        return f"在 {arguments.get('path', '.')} 搜索 {arguments.get('query', '')!r}"
    if name == "write_file":
        content = arguments.get("content", "")
        size = len(content.encode("utf-8")) if isinstance(content, str) else "?"
        return f"写入 {arguments.get('path', '?')}（{size} 字节）"
    if name == "run_command":
        args = arguments.get("args", [])
        command = (
            shlex.join(args)
            if isinstance(args, list) and all(isinstance(x, str) for x in args)
            else "?"
        )
        return f"在 {arguments.get('cwd', '.')} 运行 {command}"
    return name


def _describe_tool_result(name: str, raw_content: str) -> str:
    try:
        payload = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return f"{name} · 返回内容无法解析"
    if not isinstance(payload, dict):
        return f"{name} · 返回内容不是对象"
    if payload.get("ok") is False:
        if name == "run_command" and "exit_code" in payload:
            return _command_result_summary(payload)
        detail = str(payload.get("error", "未知错误")).replace("\n", " ")
        output = str(payload.get("output", "")).strip()
        if output:
            detail += "\n" + output
        return _error_result_summary(name, detail)
    if name == "read_file":
        return (
            f"已读取 {payload.get('path', '?')} 第 {payload.get('start_line', '?')}-"
            f"{payload.get('end_line', '?')} 行，共 {payload.get('total_lines', '?')} 行"
        )
    if name == "list_files":
        return f"{payload.get('path', '.')} 中发现 {len(payload.get('entries', []))} 个条目"
    if name == "search_text":
        return f"找到 {len(payload.get('matches', []))} 条匹配"
    if name == "write_file":
        action = "已创建" if payload.get("created") else "已更新"
        return f"{action} {payload.get('path', '?')}，写入 {payload.get('bytes_written', '?')} 字节"
    if name == "run_command":
        return _command_result_summary(payload)
    return name


def _command_result_summary(payload: dict) -> str:
    stdout = str(payload.get("stdout") or "").strip()
    stderr = str(payload.get("stderr") or "").strip()
    output = f"stdout:\n{stdout}\nstderr:\n{stderr}" if stdout and stderr else stdout or stderr
    suffix = f"；输出：\n{_indented_preview(output)}" if output else ""
    return f"退出码 {payload.get('exit_code', '?')}{suffix}"


def _error_result_summary(name: str, detail: str) -> str:
    preview = detail[:600]
    if len(detail) > 600:
        preview += "\n…"
    first, *rest = preview.splitlines() or [""]
    suffix = "".join(f"\n    {line}" for line in rest)
    return f"{name} · {first}{suffix}"


def _indented_preview(text: str, limit: int = 600) -> str:
    preview = text[:limit]
    if len(text) > limit:
        preview += "\n…"
    return "\n".join(f"    {line}" for line in preview.splitlines())


def _tool_call_signature(calls: Iterable[ToolCall]) -> str:
    normalized = []
    for call in calls:
        try:
            arguments = json.dumps(
                json.loads(call.arguments),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except json.JSONDecodeError:
            arguments = call.arguments
        normalized.append((call.name, arguments))
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _append_failed_tool_results(
    messages: list[Message],
    session: SessionStore,
    calls: Iterable[ToolCall],
    error: str,
) -> None:
    for call in calls:
        message = ToolResult(
            tool_call_id=call.id,
            name=call.name,
            ok=False,
            content=json.dumps({"ok": False, "error": error}, ensure_ascii=False),
        ).as_message()
        messages.append(message)
        session.append_message(message)


def create_new_session(
    *, workspace, task: str, model: str
) -> tuple[SessionStore, list[Message]]:
    clean_task = task.strip()
    if not clean_task:
        raise ValueError("任务不能为空")
    store = SessionStore.create(workspace, task=clean_task, model=model)
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": clean_task},
    ]
    for message in messages:
        store.append_message(message)
    return store, messages


def resume_session(*, workspace, session_id: str) -> tuple[SessionStore, list[Message]]:
    store = SessionStore.open(workspace, session_id)
    data = store.load()
    if len(data.messages) < 2:
        raise ValueError("会话中没有有效的提示历史")
    messages = data.messages
    _repair_trailing_tool_calls(store, messages)
    return store, messages


def _repair_trailing_tool_calls(store: SessionStore, messages: list[Message]) -> None:
    """为进程意外退出后缺失的尾部工具结果追加明确错误。"""

    assistant_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            assistant_index = index
            break
        if message.get("role") != "tool":
            return
    if assistant_index is None:
        return
    trailing = messages[assistant_index + 1 :]
    if any(message.get("role") != "tool" for message in trailing):
        return
    completed_ids = {str(message.get("tool_call_id", "")) for message in trailing}
    calls = messages[assistant_index].get("tool_calls", [])
    for call in calls:
        call_id = str(call.get("id", ""))
        if not call_id or call_id in completed_ids:
            continue
        function = call.get("function", {})
        name = str(function.get("name", "unknown"))
        result = ToolResult(
            tool_call_id=call_id,
            name=name,
            ok=False,
            content=json.dumps(
                {
                    "ok": False,
                    "error": "上次进程在工具返回结果前退出；该调用的执行状态未知，请重新检查",
                },
                ensure_ascii=False,
            ),
        ).as_message()
        messages.append(result)
        store.append_message(result)
