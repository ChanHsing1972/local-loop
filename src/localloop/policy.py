from __future__ import annotations

from collections.abc import Callable


class InteractiveApprovalPolicy:
    def __init__(
        self,
        *,
        auto_approve: bool = False,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self.auto_approve = auto_approve
        self.input_fn = input_fn
        self.output_fn = output_fn

    def approve(self, action: str, details: str, preview: str = "") -> bool:
        self.output_fn(f"\n[需要批准] {action}：{details}")
        if preview:
            self.output_fn(preview)
        if self.auto_approve:
            self.output_fn("[已自动批准]")
            return True
        try:
            answer = self.input_fn("是否批准？[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.output_fn("已拒绝。")
            return False
        return answer in {"y", "yes", "是"}


class AlwaysApprovePolicy:
    """供确定性测试使用的非交互式批准策略。"""

    def approve(self, action: str, details: str, preview: str = "") -> bool:
        return True


class AlwaysDenyPolicy:
    def approve(self, action: str, details: str, preview: str = "") -> bool:
        return False
