from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError

from localloop.provider import OpenAIChatProvider, ProviderError, probe_models


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


def stream_chunk(*, content="", tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def provider(items, **kwargs):
    return OpenAIChatProvider(
        api_key="secret",
        base_url="https://example.test/v1",
        model="model",
        client=FakeClient(items),
        **kwargs,
    )


def test_streams_text_and_assembles_fragmented_tool_call():
    text_stream = FakeStream(
        [stream_chunk(content="你"), stream_chunk(content="好", finish_reason="stop")]
    )
    text_provider = provider([text_stream])
    deltas = []
    turn = text_provider.stream([], [], on_text_delta=deltas.append)
    assert turn.content == "你好"
    assert deltas == ["你", "好"]
    assert text_stream.closed is True
    request = text_provider.client.completions.calls[0]
    assert request["stream"] is True
    assert request["tool_choice"] == "auto"

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
    tool_provider = provider(
        [
            FakeStream(
                [
                    stream_chunk(tool_calls=[call_head]),
                    stream_chunk(tool_calls=[call_tail], finish_reason="tool_calls"),
                ]
            )
        ]
    )
    tool_turn = tool_provider.stream([], [], on_text_delta=lambda _text: None)
    assert tool_turn.tool_calls[0].name == "read_file"
    assert json.loads(tool_turn.tool_calls[0].arguments) == {"path": "agent.py"}


def test_retries_before_output_and_reports_empty_stream():
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    sleeps = []
    retries = []
    chat = provider(
        [timeout, FakeStream([]), FakeStream([stream_chunk(content="ok")])],
        sleeper=sleeps.append,
    )
    turn = chat.stream([], [], on_text_delta=lambda _text: None, on_retry=retries.append)
    assert turn.content == "ok"
    assert sleeps == [1, 2]
    assert retries == [
        "第 1 次请求未得到可用响应，1 秒后重试",
        "第 2 次请求未得到可用响应，2 秒后重试",
    ]

    empty = provider([FakeStream([]), FakeStream([])], max_retries=1, sleeper=lambda _seconds: None)
    with pytest.raises(ProviderError, match="连续 2 次返回空响应") as error:
        empty.stream([], [], on_text_delta=lambda _text: None)
    assert "secret" not in str(error.value)


def test_partial_stream_is_not_retried_or_duplicated():
    timeout = APITimeoutError(request=httpx.Request("POST", "https://example.test"))
    chat = provider([FakeStream([stream_chunk(content="partial"), timeout])])
    deltas = []
    with pytest.raises(ProviderError, match="传输中断开"):
        chat.stream([], [], on_text_delta=deltas.append)
    assert deltas == ["partial"]
    assert len(chat.client.completions.calls) == 1


def test_rejects_invalid_stream_payloads_and_duplicate_call_ids():
    bad_payload = provider([FakeStream([{"code": 401, "msg": "bad auth"}])])
    with pytest.raises(ProviderError, match="业务错误 401"):
        bad_payload.stream([], [], on_text_delta=lambda _text: None)

    calls = [
        SimpleNamespace(
            index=index,
            id="same",
            function=SimpleNamespace(name="read_file", arguments="{}"),
        )
        for index in range(2)
    ]
    duplicate = provider([FakeStream([stream_chunk(tool_calls=calls)])])
    with pytest.raises(ProviderError, match="编号重复"):
        duplicate.stream([], [], on_text_delta=lambda _text: None)


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
            api_key="key", base_url="https://example.test/v1", client=FakeProbeClient([bad])
        )
