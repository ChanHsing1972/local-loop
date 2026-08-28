from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError

from localloop.provider import OpenAIChatProvider, ProviderError, probe_models


def response_with_tool():
    call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="read_file", arguments='{"path":"a.py"}'),
    )
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    usage = SimpleNamespace(model_dump=lambda **_kwargs: {"total_tokens": 12})
    return SimpleNamespace(choices=[choice], usage=usage)


class FakeCompletions:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, items):
        self.completions = FakeCompletions(items)
        self.chat = SimpleNamespace(completions=self.completions)


class FakeStream:
    def __init__(self, items):
        self.items = list(items)
        self.closed = False

    def __iter__(self):
        for item in self.items:
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self):
        self.closed = True


def stream_chunk(*, content="", tool_calls=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_parses_tool_calls_usage_and_request_shape():
    client = FakeClient([response_with_tool()])
    provider = OpenAIChatProvider(
        api_key="secret", base_url="https://example.test/v1", model="model", client=client
    )
    turn = provider.complete([{"role": "user", "content": "x"}], [])
    assert turn.tool_calls[0].name == "read_file"
    assert turn.usage == {"total_tokens": 12}
    assert client.completions.calls[0]["tool_choice"] == "auto"


def test_retries_timeout_then_succeeds_without_leaking_key():
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    sleeps = []
    client = FakeClient([timeout, response_with_tool()])
    provider = OpenAIChatProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model",
        client=client,
        sleeper=sleeps.append,
    )
    assert provider.complete([], []).tool_calls
    assert sleeps == [1]


def test_incompatible_response_is_sanitized():
    client = FakeClient([SimpleNamespace(choices=[])] * 3)
    provider = OpenAIChatProvider(
        api_key="secret", base_url="https://example.test/v1", model="model", client=client
    )
    with pytest.raises(ProviderError, match="连续 3 次返回空响应") as error:
        provider.complete([], [])
    assert "secret" not in str(error.value)


def test_accepts_plain_text_and_json_string_gateway_responses():
    plain = OpenAIChatProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model",
        client=FakeClient(["工具执行完成，已列出文件。"]),
    )
    assert plain.complete([], []).content == "工具执行完成，已列出文件。"

    encoded = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": "JSON 字符串响应正常。", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 9},
        },
        ensure_ascii=False,
    )
    provider = OpenAIChatProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model",
        client=FakeClient([encoded]),
    )
    turn = provider.complete([], [])
    assert turn.content == "JSON 字符串响应正常。"
    assert turn.finish_reason == "stop"
    assert turn.usage == {"total_tokens": 9}


def test_rejects_business_error_inside_json_string():
    provider = OpenAIChatProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model",
        client=FakeClient(['{"code": 401, "msg": "bad auth"}']),
    )
    with pytest.raises(ProviderError, match="业务错误 401"):
        provider.complete([], [])


@pytest.mark.parametrize(
    ("tool_calls", "expected"),
    [
        (
            [{"id": "", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            "调用编号",
        ),
        (
            [
                {"id": "same", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "same", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ],
            "编号重复",
        ),
        (
            [{"id": "id", "type": "function", "function": {"name": "", "arguments": "{}"}}],
            "函数名称",
        ),
        (
            [{"id": "id", "type": "function", "function": {"name": "read_file", "arguments": {}}}],
            "不是 JSON 字符串",
        ),
    ],
)
def test_rejects_ambiguous_tool_call_shapes(tool_calls, expected):
    response = {
        "choices": [
            {
                "message": {"content": None, "tool_calls": tool_calls},
                "finish_reason": "tool_calls",
            }
        ]
    }
    provider = OpenAIChatProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model",
        client=FakeClient([response]),
    )
    with pytest.raises(ProviderError, match=expected):
        provider.complete([], [])


def test_retries_empty_string_response_then_succeeds():
    sleeps = []
    client = FakeClient(["", "   ", '"重试后成功。"'])
    provider = OpenAIChatProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model",
        client=client,
        max_retries=2,
        sleeper=sleeps.append,
    )
    assert provider.complete([], []).content == "重试后成功。"
    assert sleeps == [1, 2]


def test_empty_string_response_exhausts_retries_with_clear_error():
    client = FakeClient(["", ""])
    provider = OpenAIChatProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model",
        client=client,
        max_retries=1,
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(ProviderError, match="连续 2 次返回空响应"):
        provider.complete([], [])


def test_streams_text_and_assembles_fragmented_tool_call():
    text_stream = FakeStream(
        [
            stream_chunk(content="你"),
            stream_chunk(content="好", finish_reason="stop"),
        ]
    )
    text_client = FakeClient([text_stream])
    deltas = []
    provider = OpenAIChatProvider(
        api_key="secret", base_url="https://example.test/v1", model="model", client=text_client
    )
    turn = provider.stream([], [], on_text_delta=deltas.append)
    assert turn.content == "你好"
    assert deltas == ["你", "好"]
    assert text_stream.closed is True
    request = text_client.completions.calls[0]
    assert request["stream"] is True
    assert "stream_options" not in request

    call_head = SimpleNamespace(
        index=0,
        id="call-1",
        function=SimpleNamespace(name="read_file", arguments='{"path":'),
    )
    call_tail = SimpleNamespace(
        index=0,
        id=None,
        function=SimpleNamespace(name=None, arguments='"agent.py"}'),
    )
    tool_provider = OpenAIChatProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model",
        client=FakeClient(
            [
                FakeStream(
                    [
                        stream_chunk(tool_calls=[call_head]),
                        stream_chunk(tool_calls=[call_tail], finish_reason="tool_calls"),
                    ]
                )
            ]
        ),
    )
    tool_turn = tool_provider.stream([], [], on_text_delta=lambda _text: None)
    assert tool_turn.tool_calls[0].name == "read_file"
    assert json.loads(tool_turn.tool_calls[0].arguments) == {"path": "agent.py"}


def test_empty_stream_retries_with_observable_status():
    client = FakeClient([FakeStream([]), FakeStream([stream_chunk(content="ok")])])
    sleeps = []
    retries = []
    provider = OpenAIChatProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model",
        client=client,
        sleeper=sleeps.append,
    )
    turn = provider.stream(
        [], [], on_text_delta=lambda _text: None, on_retry=retries.append
    )
    assert turn.content == "ok"
    assert sleeps == [1]
    assert retries == ["第 1 次请求未得到可用响应，1 秒后重试"]


def test_partial_stream_failure_is_not_retried_or_duplicated():
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    client = FakeClient([FakeStream([stream_chunk(content="partial"), timeout])])
    deltas = []
    provider = OpenAIChatProvider(
        api_key="secret", base_url="https://example.test/v1", model="model", client=client
    )
    with pytest.raises(ProviderError, match="传输中断开"):
        provider.stream([], [], on_text_delta=deltas.append)
    assert deltas == ["partial"]
    assert len(client.completions.calls) == 1


class FakeProbeResponse:
    def __init__(self, payload, status=200):
        self.content = json.dumps(payload).encode()
        self.status_code = status


class FakeProbeClient:
    def __init__(self, items):
        self.items = list(items)

    def get(self, *_args, **_kwargs):
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def test_probe_models_standard_and_wrapped():
    assert probe_models(
        api_key="key",
        base_url="https://example.test/v1",
        client=FakeProbeClient([FakeProbeResponse({"data": [{"id": "b"}, {"id": "a"}]})]),
    ) == ["a", "b"]
    assert probe_models(
        api_key="key",
        base_url="https://example.test/v1",
        client=FakeProbeClient([FakeProbeResponse({"code": 200, "data": ["model"]})]),
    ) == ["model"]


def test_probe_models_retries_network_and_server_errors():
    request = httpx.Request("GET", "https://example.test/v1/models")
    sleeps = []
    client = FakeProbeClient(
        [
            httpx.ConnectTimeout("timeout", request=request),
            FakeProbeResponse({"error": "busy"}, status=503),
            FakeProbeResponse({"data": ["model"]}),
        ]
    )
    assert probe_models(
        api_key="key",
        base_url="https://example.test/v1",
        client=client,
        max_retries=2,
        sleeper=sleeps.append,
    ) == ["model"]
    assert sleeps == [1, 2]


def test_probe_models_business_error_and_bad_json():
    with pytest.raises(ProviderError, match="业务错误 401"):
        probe_models(
            api_key="key",
            base_url="https://example.test/v1",
            client=FakeProbeClient([FakeProbeResponse({"code": 401, "msg": "bad auth"})]),
        )
    bad = FakeProbeResponse({})
    bad.content = b"not json"
    with pytest.raises(ProviderError, match="非 JSON"):
        probe_models(
            api_key="key",
            base_url="https://example.test/v1",
            client=FakeProbeClient([bad]),
        )
