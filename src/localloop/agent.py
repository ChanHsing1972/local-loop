from __future__ import annotations

import json
import time
from collections.abc import Callable

from localloop.context import ContextBudgetError, ContextManager
from localloop.provider import ProviderError
from localloop.session import SessionStore
from localloop.tools import LocalTools
from localloop.types import ChatProvider, Message, RunResult, RunStatus, ToolResult

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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.context = context
        self.max_steps = max_steps
        self.max_duration_seconds = max_duration_seconds
        self.output_fn = output_fn
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
            self.output_fn(f"[步骤 {step}/{self.max_steps}] 正在请求模型…")
            try:
                request_messages = self.context.prepare(messages)
                turn = self.provider.complete(request_messages, self.tools.definitions)
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
                if turn.finish_reason == "length":
                    return self._finish(
                        session,
                        RunStatus.ERROR,
                        "模型输出在任务完成前被截断。",
                        step,
                    )
                if turn.content.strip():
                    self.output_fn(turn.content)
                    return self._finish(
                        session, RunStatus.COMPLETED, turn.content.strip(), step
                    )
                reason = f" (finish_reason={turn.finish_reason})" if turn.finish_reason else ""
                return self._finish(
                    session,
                    RunStatus.ERROR,
                    "模型既未返回文本，也未返回工具调用" + reason,
                    step,
                )

            normalized_calls = []
            for call in turn.tool_calls:
                try:
                    normalized_arguments = json.dumps(
                        json.loads(call.arguments),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                except json.JSONDecodeError:
                    normalized_arguments = call.arguments
                normalized_calls.append((call.name, normalized_arguments))
            signature = json.dumps(
                normalized_calls,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if signature == last_signature:
                repeated += 1
            else:
                last_signature = signature
                repeated = 1
            if repeated >= 3:
                return self._finish(
                    session,
                    RunStatus.REPEATED_CALL,
                    "智能体连续三次重复相同工具调用，已停止运行。",
                    step,
                )

            for call_index, call in enumerate(turn.tool_calls):
                self.output_fn(f"[工具] {call.name}")
                try:
                    result = self.tools.execute(call)
                except KeyboardInterrupt:
                    for pending_call in turn.tool_calls[call_index:]:
                        interrupted_result = ToolResult(
                            tool_call_id=pending_call.id,
                            name=pending_call.name,
                            ok=False,
                            content=json.dumps(
                                {
                                    "ok": False,
                                    "error": "用户中断了工具执行；当前或本批次剩余调用未正常完成",
                                },
                                ensure_ascii=False,
                            ),
                        )
                        interrupted_message = interrupted_result.as_message()
                        messages.append(interrupted_message)
                        session.append_message(interrupted_message)
                    return self._finish(
                        session,
                        RunStatus.INTERRUPTED,
                        "运行已中断；可使用 --resume " + session.session_id + " 继续",
                        step,
                    )
                result_message = result.as_message()
                messages.append(result_message)
                session.append_message(result_message)
                status = "成功" if result.ok else "失败"
                self.output_fn(f"[工具{status}] {call.name}")

        return self._finish(
            session,
            RunStatus.MAX_STEPS,
            f"智能体达到 {self.max_steps} 步限制，已停止运行。",
            self.max_steps,
        )

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
    return store, data.messages
