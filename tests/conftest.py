from __future__ import annotations

import json

from localloop.types import ToolCall


class Approval:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed

    def approve(self, _action: str, _details: str, _preview: str = "") -> bool:
        return self.allowed


def make_call(name: str, arguments: dict, call_id: str = "call-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))
