from __future__ import annotations

import argparse
import os
import sys

from localloop import __version__
from localloop.agent import AgentEngine, create_new_session, resume_session
from localloop.config import DEFAULT_BASE_URL, ConfigError, load_config
from localloop.context import ContextManager
from localloop.interactive import STATUS_LABELS, InteractiveShell
from localloop.policy import InteractiveApprovalPolicy
from localloop.provider import OpenAIChatProvider, ProviderError, probe_models
from localloop.session import SessionError
from localloop.tools import LocalTools
from localloop.types import RunStatus


class ChineseArgumentParser(argparse.ArgumentParser):
    """把 argparse 自动生成的帮助界面统一为中文。"""

    def __init__(self, *args, **kwargs):
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self.add_argument("-h", "--help", action="help", help="显示帮助并退出")

    def format_help(self) -> str:
        return (
            super()
            .format_help()
            .replace("usage:", "用法：")
            .replace("positional arguments:", "位置参数：")
            .replace("options:", "选项：")
        )


def _add_runtime_options(parser: argparse.ArgumentParser, *, include_resume: bool = True) -> None:
    parser.add_argument("--workspace", "-C", default=".", help="工作区根目录")
    if include_resume:
        parser.add_argument("--resume", metavar="SESSION_ID", help="恢复已有会话")
    parser.add_argument("--max-steps", type=int, default=20, help="每轮最大模型调用步数")
    parser.add_argument(
        "--max-duration",
        type=int,
        default=600,
        metavar="SECONDS",
        help="每轮最长运行秒数",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="跳过写入和命令确认；仅用于可信、可恢复的工作区",
    )


def _parser() -> argparse.ArgumentParser:
    parser = ChineseArgumentParser(
        prog="localloop",
        description="具有本地工具执行能力的交互式编程智能体。",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="显示版本并退出",
    )
    _add_runtime_options(parser)
    subparsers = parser.add_subparsers(dest="command", parser_class=ChineseArgumentParser)

    chat = subparsers.add_parser("chat", help="显式启动交互模式")
    _add_runtime_options(chat)

    doctor = subparsers.add_parser("doctor", help="检查网关配置和原生工具调用")
    doctor.add_argument(
        "--skip-tool-check",
        action="store_true",
        help="只列出模型，不发送函数调用检查请求",
    )

    run = subparsers.add_parser("run", help="以非交互方式执行单个编程任务")
    run.add_argument("task", nargs="?", help="新会话任务；恢复会话时省略")
    _add_runtime_options(run)
    return parser


def _doctor(skip_tool_check: bool) -> int:
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = os.environ.get("LLM_MODEL", "").strip()
    print(f"接口地址：{base_url}")
    print(f"API 密钥：{'已设置' if api_key else '未设置'}")
    print(f"当前模型：{model or '未设置'}")
    if not api_key:
        print("错误：请设置新签发的 LLM_API_KEY，切勿将其提交到仓库", file=sys.stderr)
        return 2
    try:
        models = probe_models(api_key=api_key, base_url=base_url)
    except ProviderError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    if models:
        print("可用模型：")
        for model_id in models:
            marker = " *" if model_id == model else ""
            print(f"  {model_id}{marker}")
    else:
        print("警告：接口没有返回模型编号")
    if not model:
        print("错误：请选择模型并设置 LLM_MODEL", file=sys.stderr)
        return 2
    if model not in models and models:
        print("警告：当前模型不在接口返回的模型列表中")
    if skip_tool_check:
        print("原生工具调用检查：已跳过")
        return 0

    provider = OpenAIChatProvider(api_key=api_key, base_url=base_url, model=model)
    test_tool = {
        "type": "function",
        "function": {
            "name": "doctor_echo",
            "description": "返回诊断值。",
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
                {"role": "system", "content": "请调用指定的诊断函数。"},
                {"role": "user", "content": "调用 doctor_echo，并把 value 设为 ok。"},
            ],
            [test_tool],
            tool_choice={"type": "function", "function": {"name": "doctor_echo"}},
        )
    except ProviderError as exc:
        print(f"错误：原生工具调用检查失败：{exc}", file=sys.stderr)
        return 2
    if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != "doctor_echo":
        print("错误：模型没有返回原生函数调用", file=sys.stderr)
        return 2
    print("原生工具调用检查：通过")
    return 0


def _load_runtime_config(args: argparse.Namespace):
    return load_config(
        args.workspace,
        max_steps=args.max_steps,
        max_duration_seconds=args.max_duration,
        auto_approve=args.auto_approve,
    )


def _interactive(args: argparse.Namespace) -> int:
    try:
        config = _load_runtime_config(args)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    try:
        return InteractiveShell(config).run(initial_resume=args.resume)
    except KeyboardInterrupt:
        print("\n已退出 LocalLoop。")
        return 130


def _run(args: argparse.Namespace) -> int:
    if args.resume and args.task:
        print("错误：使用 --resume 时请不要再提供 TASK", file=sys.stderr)
        return 2
    if not args.resume and not args.task:
        print("错误：新会话必须提供 TASK", file=sys.stderr)
        return 2
    try:
        config = _load_runtime_config(args)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    if config.auto_approve:
        print("警告：--auto-approve 会自动批准模型请求的写入和命令。")
        print("请仅在可信、可恢复的一次性工作区中使用。")

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
        print(f"会话错误：{exc}", file=sys.stderr)
        return 2

    print(f"工作区：{config.workspace}")
    print(f"会话：{session.session_id}")
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
    print(
        f"状态：{STATUS_LABELS[result.status]}；步骤：{result.steps}；"
        f"会话：{result.session_id}"
    )
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
    if args.command in {None, "chat"}:
        return _interactive(args)
    parser.error(f"未知命令：{args.command}")
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
