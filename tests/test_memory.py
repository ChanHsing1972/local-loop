from __future__ import annotations

import json
import stat

import pytest

from localloop.agent import create_new_session, resume_session
from localloop.memory import WorkspaceMemoryError, WorkspaceMemoryStore, memory_prompt


def test_memory_round_trip_forget_and_session_snapshot(tmp_path):
    store = WorkspaceMemoryStore(tmp_path)
    first = store.remember("项目使用 C++17")
    second = store.remember("修改源码后必须运行测试")
    assert [entry.text for entry in store.list_active()] == [
        "项目使用 C++17",
        "修改源码后必须运行测试",
    ]
    assert "项目使用 C++17" in memory_prompt(store.list_active())

    session, messages = create_new_session(
        workspace=tmp_path,
        task="补全 test.cpp",
        model="fake",
        workspace_memory=store.list_active(),
    )
    store.forget(first.id)
    assert [entry.id for entry in store.list_active()] == [second.id]

    _resumed_store, resumed = resume_session(workspace=tmp_path, session_id=session.session_id)
    assert messages[0] == resumed[0]
    assert "项目使用 C++17" in resumed[0]["content"]
    assert "修改源码后必须运行测试" in resumed[0]["content"]
    assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_memory_rejects_invalid_sensitive_and_oversized_content(tmp_path):
    store = WorkspaceMemoryStore(tmp_path)
    with pytest.raises(WorkspaceMemoryError, match="不能为空"):
        store.remember("   ")
    with pytest.raises(WorkspaceMemoryError, match="500"):
        store.remember("x" * 501)
    with pytest.raises(WorkspaceMemoryError, match="密钥"):
        store.remember("api_key=sk-this-is-a-secret-value")
    with pytest.raises(WorkspaceMemoryError, match="格式"):
        store.forget("../escape")


def test_memory_recovers_truncated_tail_but_rejects_corrupt_middle(tmp_path):
    store = WorkspaceMemoryStore(tmp_path)
    entry = store.remember("稳定记忆")
    with store.path.open("a", encoding="utf-8") as stream:
        stream.write('{"version":1')
    assert [item.id for item in store.list_active()] == [entry.id]

    store.path.write_text(
        "not-json\n"
        + json.dumps(
            {
                "version": 1,
                "kind": "remember",
                "id": "m-abcdef",
                "text": "x",
                "timestamp": "now",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkspaceMemoryError, match="第 1 行"):
        store.list_active()


def test_memory_file_rejects_symlink(tmp_path):
    directory = tmp_path / ".localloop"
    directory.mkdir()
    outside = tmp_path.parent / "outside-memory.jsonl"
    outside.write_text("keep", encoding="utf-8")
    (directory / "memory.jsonl").symlink_to(outside)
    store = WorkspaceMemoryStore(tmp_path)
    with pytest.raises(WorkspaceMemoryError, match="符号链接"):
        store.remember("不要写到外面")
    assert outside.read_text(encoding="utf-8") == "keep"


def test_memory_directory_rejects_runtime_symlink_replacement(tmp_path):
    store = WorkspaceMemoryStore(tmp_path)
    outside = tmp_path.parent / "outside-memory-state"
    outside.mkdir()
    (tmp_path / ".localloop").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceMemoryError, match="符号链接"):
        store.remember("不会写入外部目录")
