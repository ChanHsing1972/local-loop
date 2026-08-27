from __future__ import annotations

import json
import stat

import pytest

from localloop.session import SessionError, SessionStore


def test_session_round_trip(tmp_path):
    store = SessionStore.create(tmp_path, task="fix bug", model="test-model")
    store.append_message({"role": "system", "content": "rules"})
    store.append_message({"role": "user", "content": "fix bug"})
    loaded = SessionStore.open(tmp_path, store.session_id).load()
    assert loaded.metadata["task"] == "fix bug"
    assert loaded.metadata["workspace_name"] == tmp_path.name
    assert [message["role"] for message in loaded.messages] == ["system", "user"]
    assert str(tmp_path.resolve()) not in store.path.read_text()
    assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_invalid_and_missing_session(tmp_path):
    with pytest.raises(SessionError, match="Invalid"):
        SessionStore(tmp_path, "../escape")
    with pytest.raises(SessionError, match="not found"):
        SessionStore.open(tmp_path, "abcdef123456")


def test_corrupt_session_is_rejected(tmp_path):
    store = SessionStore.create(tmp_path, task="task", model="model")
    with store.path.open("a") as stream:
        stream.write("not-json\n")
    with pytest.raises(SessionError, match="Invalid JSON"):
        store.load()


def test_truncated_final_event_is_ignored_for_crash_recovery(tmp_path):
    store = SessionStore.create(tmp_path, task="task", model="model")
    store.append_message({"role": "system", "content": "rules"})
    with store.path.open("a", encoding="utf-8") as stream:
        stream.write('{"version":1,"kind":"message"')
    loaded = store.load()
    assert loaded.metadata["task"] == "task"
    assert loaded.messages == [{"role": "system", "content": "rules"}]


def test_unsupported_version_and_invalid_message_are_rejected(tmp_path):
    store = SessionStore.create(tmp_path, task="task", model="model")
    store.path.write_text(
        json.dumps({"version": 99, "kind": "metadata", "data": {}, "timestamp": "x"})
        + "\n"
    )
    with pytest.raises(SessionError, match="Unsupported"):
        store.load()
    store.path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "metadata",
                "data": {"task": "x"},
                "timestamp": "x",
            }
        )
        + "\n"
        + json.dumps(
            {"version": 1, "kind": "message", "data": {"message": []}, "timestamp": "x"}
        )
        + "\n"
    )
    with pytest.raises(SessionError, match="Invalid message"):
        store.load()


def test_session_storage_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-state"
    outside.mkdir()
    (tmp_path / ".localloop").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SessionError, match="越过工作区"):
        SessionStore.create(tmp_path, task="task", model="model")


def test_session_file_rejects_symbolic_link(tmp_path):
    session_id = "abcdef123456"
    directory = tmp_path / ".localloop" / "sessions"
    directory.mkdir(parents=True)
    outside = tmp_path.parent / "outside-session.jsonl"
    outside.write_text("keep", encoding="utf-8")
    (directory / f"{session_id}.jsonl").symlink_to(outside)
    with pytest.raises(SessionError, match="symbolic link"):
        SessionStore.open(tmp_path, session_id)
