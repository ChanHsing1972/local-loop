from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from localloop.types import AssistantTurn, JsonObject, Message, ToolCall


class ProviderError(RuntimeError):
    """经过脱敏、可直接展示给用户的模型接口错误。"""


def _safe_error(error: BaseException, api_key: str) -> str:
    text = str(error).replace(api_key, "[REDACTED]")
    return text[:1_000]


class OpenAIChatProvider:
    """仅负责接口传输；所有编排逻辑均由 AgentEngine 自行实现。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.sleeper = sleeper
        self.client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[JsonObject],
        *,
        tool_choice: str | JsonObject = "auto",
    ) -> AssistantTurn:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                return self._parse_response(response)
            except (AuthenticationError, PermissionDeniedError, BadRequestError) as exc:
                raise ProviderError(_safe_error(exc, self.api_key)) from exc
            except RateLimitError as exc:
                if attempt >= self.max_retries:
                    raise ProviderError(_safe_error(exc, self.api_key)) from exc
            except APIStatusError as exc:
                if exc.status_code not in {408, 409, 429} and exc.status_code < 500:
                    raise ProviderError(_safe_error(exc, self.api_key)) from exc
                if attempt >= self.max_retries:
                    raise ProviderError(_safe_error(exc, self.api_key)) from exc
            except (APIConnectionError, APITimeoutError) as exc:
                if attempt >= self.max_retries:
                    raise ProviderError(_safe_error(exc, self.api_key)) from exc
            except (AttributeError, TypeError, ValueError) as exc:
                raise ProviderError(
                    "The gateway returned an incompatible Chat Completions response: "
                    + _safe_error(exc, self.api_key)
                ) from exc
            self.sleeper(min(2**attempt, 8))
        raise AssertionError("retry loop exhausted")  # pragma: no cover

    @staticmethod
    def _parse_response(response: Any) -> AssistantTurn:
        if not response.choices:
            raise ValueError("response has no choices")
        choice = response.choices[0]
        message = choice.message
        calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            if getattr(call, "type", "function") != "function":
                raise ValueError(f"unsupported tool call type: {call.type}")
            calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
            )
        usage = None
        if getattr(response, "usage", None) is not None:
            raw_usage = response.usage
            usage = (
                raw_usage.model_dump(exclude_none=True)
                if hasattr(raw_usage, "model_dump")
                else dict(raw_usage)
            )
        content = message.content if isinstance(message.content, str) else ""
        return AssistantTurn(
            content=content,
            tool_calls=tuple(calls),
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
        )


def probe_models(*, api_key: str, base_url: str, timeout: float = 15.0) -> list[str]:
    """探测兼容 OpenAI 的模型列表接口，并识别被包装的业务错误。"""

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            status = response.status
            raw = response.read(2_000_000)
    except urllib.error.HTTPError as exc:
        body = exc.read(20_000).decode("utf-8", errors="replace")
        body = body.replace(api_key, "[REDACTED]")
        raise ProviderError(f"Model probe failed with HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Cannot reach model endpoint: {exc.reason}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Model endpoint returned non-JSON data (HTTP {status})") from exc
    if not isinstance(payload, dict):
        raise ProviderError("Model endpoint returned an unexpected JSON shape")

    if "error" in payload:
        error = payload["error"]
        detail = error.get("message", error) if isinstance(error, dict) else error
        safe_detail = str(detail).replace(api_key, "[REDACTED]")
        raise ProviderError(f"Model endpoint error: {safe_detail[:500]}")
    if "code" in payload and str(payload["code"]).lower() not in {"0", "200", "success"}:
        detail = payload.get("msg") or payload.get("message") or "unknown business error"
        safe_detail = str(detail).replace(api_key, "[REDACTED]")
        raise ProviderError(f"Model endpoint business error {payload['code']}: {safe_detail}")

    data = payload.get("data", [])
    if isinstance(data, dict):
        data = data.get("data") or data.get("models") or data.get("items") or []
    if not isinstance(data, list):
        raise ProviderError("Model endpoint did not return a model list")
    model_ids = []
    for item in data:
        if isinstance(item, str):
            model_ids.append(item)
        elif isinstance(item, dict) and item.get("id"):
            model_ids.append(str(item["id"]))
    return sorted(set(model_ids))
