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
    client = FakeClient([SimpleNamespace(choices=[])])
    provider = OpenAIChatProvider(
        api_key="secret", base_url="https://example.test/v1", model="model", client=client
    )
    with pytest.raises(ProviderError, match="incompatible") as error:
        provider.complete([], [])
    assert "secret" not in str(error.value)


class FakeHTTPResponse:
    def __init__(self, payload, status=200):
        self.payload = json.dumps(payload).encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return self.payload


def test_probe_models_standard_and_wrapped(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeHTTPResponse({"data": [{"id": "b"}, {"id": "a"}]}),
    )
    assert probe_models(api_key="key", base_url="https://example.test/v1") == ["a", "b"]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeHTTPResponse({"code": 200, "data": ["model"]}),
    )
    assert probe_models(api_key="key", base_url="https://example.test/v1") == ["model"]


def test_probe_models_business_error_and_bad_json(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeHTTPResponse({"code": 401, "msg": "bad auth"}),
    )
    with pytest.raises(ProviderError, match="business error 401"):
        probe_models(api_key="key", base_url="https://example.test/v1")
    bad = FakeHTTPResponse({})
    bad.payload = b"not json"
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: bad)
    with pytest.raises(ProviderError, match="non-JSON"):
        probe_models(api_key="key", base_url="https://example.test/v1")

