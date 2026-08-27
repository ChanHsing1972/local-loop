from __future__ import annotations

import json
import time
from collections.abc import Callable

from localloop.context import ContextBudgetError, ContextManager
from localloop.provider import ProviderError
from localloop.session import SessionStore
from localloop.tools import LocalTools
from localloop.types import ChatProvider, Message, RunResult, RunStatus

SYSTEM_PROMPT = """You are LocalLoop, a coding agent operating inside one workspace.
Use the provided local tools to inspect the project, make focused changes, and verify them.
Rules:
1. Never claim that a file changed or a command passed unless a tool result proves it.
2. Read an existing file before updating it, then pass its latest SHA-256 to write_file.
3. Prefer small, reviewable edits and run the most relevant tests after changes.
4. Do not attempt to access paths outside the workspace or sensitive credentials.
5. If a tool returns an error, diagnose it and either correct the call or explain the blocker.
6. When the task is complete, respond with a concise summary and tests run, without a tool call.
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
                    "Agent stopped after reaching the total time limit.",
                    step - 1,
                )
            self.output_fn(f"[step {step}/{self.max_steps}] asking model...")
            try:
                request_messages = self.context.prepare(messages)
                turn = self.provider.complete(request_messages, self.tools.definitions)
            except (ProviderError, ContextBudgetError) as exc:
                return self._finish(session, RunStatus.ERROR, str(exc), step - 1)
            except KeyboardInterrupt:
                return self._finish(
                    session,
                    RunStatus.INTERRUPTED,
                    "Interrupted; resume with --resume " + session.session_id,
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
                        "Model output was truncated before the task could finish.",
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
                    "Model returned neither text nor tool calls" + reason,
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
                    "Agent stopped after repeating the same tool call three times.",
                    step,
                )

            for call in turn.tool_calls:
                self.output_fn(f"[tool] {call.name}")
                try:
                    result = self.tools.execute(call)
                except KeyboardInterrupt:
                    return self._finish(
                        session,
                        RunStatus.INTERRUPTED,
                        "Interrupted; resume with --resume " + session.session_id,
                        step,
                    )
                result_message = result.as_message()
                messages.append(result_message)
                session.append_message(result_message)
                status = "ok" if result.ok else "error"
                self.output_fn(f"[tool {status}] {call.name}")

        return self._finish(
            session,
            RunStatus.MAX_STEPS,
            f"Agent stopped after reaching the {self.max_steps}-step limit.",
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
        raise ValueError("Task must not be empty")
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
        raise ValueError("Session does not contain a valid prompt history")
    return store, data.messages
