from __future__ import annotations

import pytest

from localloop.config import ConfigError, load_config


def test_load_config_and_secret_is_not_in_repr(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    config = load_config(tmp_path, max_steps=7, max_duration_seconds=9, auto_approve=True)
    assert config.base_url == "https://example.test/v1"
    assert config.model == "test-model"
    assert config.max_steps == 7
    assert config.auto_approve is True
    assert "super-secret" not in repr(config)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"LLM_API_KEY": ""}, "LLM_API_KEY"),
        ({"LLM_MODEL": ""}, "LLM_MODEL"),
        ({"LLM_BASE_URL": "ftp://bad"}, "LLM_BASE_URL"),
    ],
)
def test_invalid_environment(tmp_path, monkeypatch, updates, message):
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL", "model")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    for name, value in updates.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=message):
        load_config(tmp_path)


def test_invalid_workspace_and_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL", "model")
    with pytest.raises(ConfigError, match="工作区"):
        load_config(tmp_path / "missing")
    with pytest.raises(ConfigError, match="max_steps"):
        load_config(tmp_path, max_steps=0)
    with pytest.raises(ConfigError, match="max_duration"):
        load_config(tmp_path, max_duration_seconds=0)
