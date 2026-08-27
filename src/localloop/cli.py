from __future__ import annotations

import argparse
import os
import sys

from localloop import __version__
from localloop.agent import AgentEngine, create_new_session, resume_session
from localloop.config import DEFAULT_BASE_URL, ConfigError, load_config
from localloop.context import ContextManager
from localloop.policy import InteractiveApprovalPolicy
from localloop.provider import OpenAIChatProvider, ProviderError, probe_models
from localloop.session import SessionError
from localloop.tools import LocalTools
from localloop.types import RunStatus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localloop",
        description="A small coding agent with auditable, locally executed tools.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check gateway configuration and tool calling")
    doctor.add_argument(
        "--skip-tool-check",
        action="store_true",
        help="Only list models; do not spend a model request testing function calling",
    )

    run = subparsers.add_parser("run", help="Run a coding task")
    run.add_argument("task", nargs="?", help="Task for a new session; omit when resuming")
    run.add_argument("--workspace", default=".", help="Workspace root (default: current directory)")
    run.add_argument("--resume", metavar="SESSION_ID", help="Resume an interrupted session")
    run.add_argument("--max-steps", type=int, default=20)
    run.add_argument("--max-duration", type=int, default=600, metavar="SECONDS")
    run.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip write/command confirmations; use only in a disposable workspace",
    )
    return parser


def _doctor(skip_tool_check: bool) -> int:
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = os.environ.get("LLM_MODEL", "").strip()
    print(f"base URL: {base_url}")
    print(f"API key: {'set' if api_key else 'missing'}")
    print(f"model: {model or 'not set'}")
    if not api_key:
        print("error: set LLM_API_KEY to a newly issued key; never commit it", file=sys.stderr)
        return 2
    try:
        models = probe_models(api_key=api_key, base_url=base_url)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if models:
        print("available models:")
        for model_id in models:
            marker = " *" if model_id == model else ""
            print(f"  {model_id}{marker}")
    else:
        print("warning: endpoint returned no model IDs")
    if not model:
        print("error: choose a model and set LLM_MODEL", file=sys.stderr)
        return 2
    if model not in models and models:
        print("warning: configured model is not present in the returned list")
    if skip_tool_check:
        print("tool calling check: skipped")
        return 0

    provider = OpenAIChatProvider(api_key=api_key, base_url=base_url, model=model)
    test_tool = {
        "type": "function",
        "function": {
            "name": "doctor_echo",
            "description": "Return a diagnostic value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
    try:
        turn = provider.complete(
            [
                {"role": "system", "content": "Call the required diagnostic function."},
                {"role": "user", "content": "Call doctor_echo with value 'ok'."},
            ],
            [test_tool],
            tool_choice={"type": "function", "function": {"name": "doctor_echo"}},
        )
    except ProviderError as exc:
        print(f"error: tool calling probe failed: {exc}", file=sys.stderr)
        return 2
    if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != "doctor_echo":
        print("error: model did not return a native function tool call", file=sys.stderr)
        return 2
    print("tool calling check: passed")
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.resume and args.task:
        print("error: omit TASK when using --resume", file=sys.stderr)
        return 2
    if not args.resume and not args.task:
        print("error: TASK is required for a new session", file=sys.stderr)
        return 2
    try:
        config = load_config(
            args.workspace,
            max_steps=args.max_steps,
            max_duration_seconds=args.max_duration,
            auto_approve=args.auto_approve,
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if config.auto_approve:
        print("WARNING: --auto-approve allows model-requested writes and commands.")
        print("Use it only inside a disposable, backed-up workspace.")

    try:
        if args.resume:
            session, messages = resume_session(
                workspace=config.workspace, session_id=args.resume
            )
        else:
            session, messages = create_new_session(
                workspace=config.workspace, task=args.task, model=config.model
            )
    except (SessionError, ValueError) as exc:
        print(f"session error: {exc}", file=sys.stderr)
        return 2

    print(f"workspace: {config.workspace}")
    print(f"session: {session.session_id}")
    provider = OpenAIChatProvider(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
    )
    policy = InteractiveApprovalPolicy(auto_approve=config.auto_approve)
    tools = LocalTools(config.workspace, policy)
    context = ContextManager(
        max_chars=config.max_context_chars, recent_groups=config.recent_groups
    )
    engine = AgentEngine(
        provider=provider,
        tools=tools,
        context=context,
        max_steps=config.max_steps,
        max_duration_seconds=config.max_duration_seconds,
    )
    result = engine.run(messages, session)
    print(f"status: {result.status.value}; steps: {result.steps}; session: {result.session_id}")
    if result.status is not RunStatus.COMPLETED:
        print(result.final_output, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.skip_tool_check)
    if args.command == "run":
        return _run(args)
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
