"""OpenAI 兼容网关的流式请求、重试与响应解析。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx
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
    pass


class _RetryableResponseError(ValueError):
    pass


def _safe_error(error: BaseException, api_key: str) -> str:
    text = str(error).replace(api_key, "[REDACTED]")
    return text[:1_000]


class OpenAIChatProvider:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 2,
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

    def stream(
        self,
        messages: list[Message],
        tools: list[JsonObject],
        *,
        tool_choice: str | JsonObject = "auto",
        on_text_delta: Callable[[str], None],
        on_retry: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        """流式返回正文，同时在本地拼装完整文本和原生工具调用。"""
        for attempt in range(self.max_retries + 1):
            stream_state = [False]
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    stream=True,
                )
                return self._parse_stream(response, on_text_delta, stream_state)
            except (AuthenticationError, PermissionDeniedError, BadRequestError) as exc:
                raise ProviderError(_safe_error(exc, self.api_key)) from exc
            except RateLimitError as exc:
                if stream_state[0]:
                    raise ProviderError(
                        "模型输出流在传输中断开；已保留已显示内容，请重试本轮任务"
                    ) from exc
                if attempt >= self.max_retries:
                    raise ProviderError(_safe_error(exc, self.api_key)) from exc
            except APIStatusError as exc:
                if stream_state[0]:
                    raise ProviderError(
                        "模型输出流在传输中断开；已保留已显示内容，请重试本轮任务"
                    ) from exc
                if exc.status_code not in {408, 409, 429} and exc.status_code < 500:
                    raise ProviderError(_safe_error(exc, self.api_key)) from exc
                if attempt >= self.max_retries:
                    raise ProviderError(_safe_error(exc, self.api_key)) from exc
            except (APIConnectionError, APITimeoutError) as exc:
                if stream_state[0]:
                    raise ProviderError(
                        "模型输出流在传输中断开；已保留已显示内容，请重试本轮任务"
                    ) from exc
                if attempt >= self.max_retries:
                    raise ProviderError(_safe_error(exc, self.api_key)) from exc
            except _RetryableResponseError as exc:
                if attempt >= self.max_retries:
                    raise ProviderError(
                        f"网关连续 {attempt + 1} 次返回空响应；"
                        "请检查网关配置，或使用 /model 切换模型"
                    ) from exc
            except (AttributeError, TypeError, ValueError) as exc:
                raise ProviderError(
                    "网关返回了不兼容的 Chat Completions 响应：" + _safe_error(exc, self.api_key)
                ) from exc
            delay = min(2**attempt, 8)
            if on_retry:
                on_retry(f"第 {attempt + 1} 次请求未得到可用响应，{delay} 秒后重试")
            self.sleeper(delay)
        raise AssertionError("retry loop exhausted")  # pragma: no cover

    @staticmethod
    def _parse_stream(
        response: Any,
        on_text_delta: Callable[[str], None],
        stream_state: list[bool],
    ) -> AssistantTurn:
        content_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        try:
            for chunk in response:
                if isinstance(chunk, Mapping):
                    OpenAIChatProvider._raise_payload_error(chunk)
                choices = OpenAIChatProvider._field(chunk, "choices", []) or []
                for choice in choices:
                    # 已经收到输出后不再自动重试，以免重复显示或执行工具。
                    stream_state[0] = True
                    reason = OpenAIChatProvider._field(choice, "finish_reason")
                    if reason:
                        finish_reason = str(reason)
                    delta = OpenAIChatProvider._field(choice, "delta")
                    if delta is None:
                        continue
                    content = OpenAIChatProvider._field(delta, "content", "")
                    if isinstance(content, str) and content:
                        content_parts.append(content)
                        on_text_delta(content)
                    for fallback_index, call in enumerate(
                        OpenAIChatProvider._field(delta, "tool_calls", []) or []
                    ):
                        index = OpenAIChatProvider._field(call, "index", fallback_index)
                        if not isinstance(index, int):
                            raise ValueError("流式工具调用缺少有效 index")
                        aggregate = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        call_id = OpenAIChatProvider._field(call, "id", "")
                        if isinstance(call_id, str) and call_id:
                            aggregate["id"] = call_id
                        function = OpenAIChatProvider._field(call, "function")
                        if function is not None:
                            name = OpenAIChatProvider._field(function, "name", "")
                            arguments = OpenAIChatProvider._field(function, "arguments", "")
                            if isinstance(name, str) and name:
                                aggregate["name"] += name
                            if isinstance(arguments, str) and arguments:
                                aggregate["arguments"] += arguments
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        tool_calls = tuple(
            ToolCall(id=item["id"], name=item["name"], arguments=item["arguments"])
            for _, item in sorted(calls.items())
        )
        call_ids: set[str] = set()
        for call in tool_calls:
            if not call.id or not call.name:
                raise ValueError("流式工具调用缺少编号或名称")
            if call.id in call_ids:
                raise ValueError(f"工具调用编号重复：{call.id}")
            call_ids.add(call.id)
        turn = AssistantTurn(
            content="".join(content_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
        if not turn.content.strip() and not turn.tool_calls:
            raise _RetryableResponseError("流中没有文本或工具调用")
        return turn

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    @staticmethod
    def _raise_payload_error(payload: Mapping[str, Any]) -> None:
        if "error" in payload:
            error = payload["error"]
            detail = error.get("message", error) if isinstance(error, Mapping) else error
            raise ValueError(f"接口错误：{detail}")
        if "code" in payload and str(payload["code"]).lower() not in {
            "0",
            "200",
            "success",
        }:
            detail = payload.get("msg") or payload.get("message") or "未知业务错误"
            raise ValueError(f"接口业务错误 {payload['code']}：{detail}")


def probe_models(
    *,
    api_key: str,
    base_url: str,
    timeout: float = 15.0,
    max_retries: int = 2,
    sleeper: Callable[[float], None] = time.sleep,
    client: Any | None = None,
) -> list[str]:
    """读取兼容 OpenAI 的模型列表，供交互命令 `/models` 使用。"""

    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout)
    try:
        for attempt in range(max_retries + 1):
            try:
                response = http_client.get(
                    f"{base_url.rstrip('/')}/models",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                    },
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= max_retries:
                    raise ProviderError(
                        "无法连接模型列表接口：" + _safe_error(exc, api_key)
                    ) from exc
                sleeper(min(2**attempt, 8))
                continue
            status = int(response.status_code)
            # 限制异常网关响应，避免模型列表请求占用无界内存。
            raw = bytes(response.content[:2_000_000])
            if (status in {408, 409, 429} or status >= 500) and attempt < max_retries:
                sleeper(min(2**attempt, 8))
                continue
            if not 200 <= status < 300:
                body = raw[:20_000].decode("utf-8", errors="replace")
                safe_body = body.replace(api_key, "[REDACTED]")
                raise ProviderError(f"模型列表接口返回 HTTP {status}：{safe_body[:500]}")
            break
        else:  # pragma: no cover
            raise AssertionError("模型列表重试循环耗尽")
    finally:
        if owns_client:
            http_client.close()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"模型列表接口返回非 JSON 数据（HTTP {status}）") from exc
    if not isinstance(payload, dict):
        raise ProviderError("模型列表接口返回了意外的 JSON 结构")

    if "error" in payload:
        error = payload["error"]
        detail = error.get("message", error) if isinstance(error, dict) else error
        safe_detail = str(detail).replace(api_key, "[REDACTED]")
        raise ProviderError(f"模型列表接口错误：{safe_detail[:500]}")
    if "code" in payload and str(payload["code"]).lower() not in {"0", "200", "success"}:
        detail = payload.get("msg") or payload.get("message") or "未知业务错误"
        safe_detail = str(detail).replace(api_key, "[REDACTED]")
        raise ProviderError(f"模型列表接口业务错误 {payload['code']}：{safe_detail}")

    data = payload.get("data", [])
    # 兼容标准列表和常见的 data.models/data.items 二次包装。
    if isinstance(data, dict):
        data = data.get("data") or data.get("models") or data.get("items") or []
    if not isinstance(data, list):
        raise ProviderError("模型列表接口没有返回模型列表")
    model_ids = []
    for item in data:
        if isinstance(item, str):
            model_ids.append(item)
        elif isinstance(item, dict) and item.get("id"):
            model_ids.append(str(item["id"]))
    return sorted(set(model_ids))
