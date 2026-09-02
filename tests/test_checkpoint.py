from __future__ import annotations

import hashlib

import pytest
from conftest import Approval, make_call

from localloop.checkpoint import CheckpointError, CheckpointStore
from localloop.tools import LocalTools


class RecordingApproval(Approval):
    def __init__(self, allowed: bool = True) -> None:
        super().__init__(allowed)
        self.requests: list[tuple[str, str, str]] = []

    def approve(self, action: str, details: str, preview: str = "") -> bool:
        self.requests.append((action, details, preview))
        return self.allowed


def write(tools, path, content, old=b""):
    arguments = {"path": path, "content": content}
    if old:
        arguments["expected_sha256"] = hashlib.sha256(old).hexdigest()
    result = tools.execute(make_call("write_file", arguments))
    assert result.ok is True


def test_checkpoint_groups_existing_and_new_files_then_undoes(tmp_path):
    original = b"old\n"
    (tmp_path / "test.cpp").write_bytes(original)
    checkpoints = CheckpointStore(tmp_path)
    policy = RecordingApproval()
    tools = LocalTools(tmp_path, policy, checkpoints)

    checkpoints.begin(session_id="abcdef123456", task="补全 test.cpp")
    write(tools, "test.cpp", "new\n", original)
    write(tools, "created.txt", "created\n")
    info = checkpoints.finish()

    assert info is not None
    assert info.files == ("test.cpp", "created.txt")
    assert (tmp_path / "test.cpp").read_text() == "new\n"
    assert (tmp_path / "created.txt").exists()
    assert checkpoints.list_checkpoints()[0].status == "completed"

    restored = checkpoints.undo_latest(policy)
    assert restored.id == info.id
    assert (tmp_path / "test.cpp").read_bytes() == original
    assert not (tmp_path / "created.txt").exists()
    assert checkpoints.list_checkpoints()[0].status == "restored"
    undo_requests = [request for request in policy.requests if request[0] == "undo"]
    assert len(undo_requests) == 1
    assert "current/test.cpp" in undo_requests[0][2]
    assert "restored/created.txt" in undo_requests[0][2]


def test_checkpoint_keeps_original_across_multiple_writes(tmp_path):
    original = b"one\n"
    target = tmp_path / "a.txt"
    target.write_bytes(original)
    checkpoints = CheckpointStore(tmp_path)
    tools = LocalTools(tmp_path, Approval(), checkpoints)
    checkpoints.begin(session_id="abcdef123456", task="多次修改")
    write(tools, "a.txt", "two\n", original)
    write(tools, "a.txt", "three\n", b"two\n")
    info = checkpoints.finish()
    assert info is not None
    assert info.files == ("a.txt",)
    checkpoints.undo_latest(Approval())
    assert target.read_bytes() == original


def test_checkpoint_refuses_to_overwrite_newer_user_change(tmp_path):
    original = b"old\n"
    target = tmp_path / "a.txt"
    target.write_bytes(original)
    checkpoints = CheckpointStore(tmp_path)
    tools = LocalTools(tmp_path, Approval(), checkpoints)
    checkpoints.begin(session_id="abcdef123456", task="修改")
    write(tools, "a.txt", "agent\n", original)
    checkpoints.finish()
    target.write_text("user changed\n")

    with pytest.raises(CheckpointError, match="又发生了变化"):
        checkpoints.undo_latest(Approval())
    assert target.read_text() == "user changed\n"


def test_checkpoint_denied_undo_and_empty_task(tmp_path):
    checkpoints = CheckpointStore(tmp_path)
    checkpoints.begin(session_id="abcdef123456", task="只读任务")
    assert checkpoints.finish() is None
    with pytest.raises(CheckpointError, match="没有可撤销"):
        checkpoints.undo_latest(Approval())

    original = b"old"
    (tmp_path / "a.txt").write_bytes(original)
    tools = LocalTools(tmp_path, Approval(), checkpoints)
    checkpoints.begin(session_id="abcdef123456", task="修改")
    write(tools, "a.txt", "new", original)
    checkpoints.finish()
    with pytest.raises(CheckpointError, match="取消"):
        checkpoints.undo_latest(Approval(False))
    assert (tmp_path / "a.txt").read_text() == "new"


def test_in_progress_checkpoint_can_be_recovered_after_restart(tmp_path):
    original = b"before\n"
    target = tmp_path / "a.txt"
    target.write_bytes(original)
    first_process = CheckpointStore(tmp_path)
    tools = LocalTools(tmp_path, Approval(), first_process)
    first_process.begin(session_id="abcdef123456", task="进程中断")
    write(tools, "a.txt", "after\n", original)

    restarted = CheckpointStore(tmp_path)
    assert restarted.list_checkpoints()[0].status == "in_progress"
    restarted.undo_latest(Approval())
    assert target.read_bytes() == original


def test_checkpoint_directory_rejects_runtime_symlink_replacement(tmp_path):
    (tmp_path / ".localloop").mkdir()
    checkpoints = CheckpointStore(tmp_path)
    outside = tmp_path.parent / "outside-checkpoints"
    outside.mkdir()
    checkpoints.directory.symlink_to(outside, target_is_directory=True)
    checkpoints.begin(session_id="abcdef123456", task="写文件")
    with pytest.raises(CheckpointError, match="符号链接"):
        checkpoints.capture_before(
            tmp_path / "a.txt",
            previous=None,
            new_content=b"new",
            previous_mode=None,
        )
