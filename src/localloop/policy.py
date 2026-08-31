"""写文件和运行命令之前使用的审批策略。

审批与工具实现分离后，真实终端可以询问用户，自动化测试则可以确定性地批准或拒绝；
工具层无需知道答案来自键盘、测试桩还是其他界面。
"""

from __future__ import annotations

from collections.abc import Callable


class InteractiveApprovalPolicy:
    """默认的终端审批器：先展示动作与预览，再采用默认拒绝原则询问。"""

    def __init__(
        self,
        *,
        auto_approve: bool = False,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        """注入输入输出函数，使相同逻辑既可用于终端，也便于单元测试。"""

        self.auto_approve = auto_approve
        self.input_fn = input_fn
        self.output_fn = output_fn

    def approve(self, action: str, details: str, preview: str = "") -> bool:
        """返回用户是否明确批准；空输入、中断和文件结束都视为拒绝。"""

        self.output_fn(f"\n[需要批准] {action}：{details}")
        if preview:
            self.output_fn(preview)
        if self.auto_approve:
            # 自动批准仍然打印提示，让用户知道安全确认已被跳过。
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
    """供拒绝路径测试使用，不进行真实交互。"""

    def approve(self, action: str, details: str, preview: str = "") -> bool:
        return False
