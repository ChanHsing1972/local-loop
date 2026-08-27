from __future__ import annotations

import json

from localloop.types import ToolCall


def make_call(name: str, arguments: dict, call_id: str = "call-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))

