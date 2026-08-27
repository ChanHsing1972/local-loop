from __future__ import annotations

import hashlib
import json

import pytest
from conftest import make_call

from localloop.policy import AlwaysApprovePolicy, AlwaysDenyPolicy
from localloop.tools import LocalTools
from localloop.types import ToolCall


def payload(result):
    return json.loads(result.content)


def test_definitions_and_unknown_or_malformed_call(tmp_path):
    tools = LocalTools(tmp_path, AlwaysApprovePolicy())
    assert [tool["function"]["name"] for tool in tools.definitions] == [
        "list_files",
        "read_file",
        "search_text",
        "write_file",
        "run_command",
    ]
    unknown = tools.execute(make_call("missing", {}))
    assert unknown.ok is False
    malformed = tools.execute(ToolCall("id", "read_file", "{"))
    assert "Invalid JSON" in malformed.content
    array_args = tools.execute(ToolCall("id", "read_file", "[]"))
    assert "JSON object" in array_args.content


def test_list_read_create_update_and_stale_write(tmp_path):
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "a.py"
    target.write_text("one\ntwo\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / "build").mkdir()
    (tmp_path / "package.egg-info").mkdir()
    (tmp_path / ".env").write_text("secret")
    (tmp_path / ".coverage").write_text("generated")
    tools = LocalTools(tmp_path, AlwaysApprovePolicy())

    listed = payload(tools.execute(make_call("list_files", {"path": ".", "max_depth": 2})))
    assert "src/a.py" in listed["entries"]
    assert not any(".git" in entry or entry == ".env" for entry in listed["entries"])
    assert not any(
        generated in entry
        for entry in listed["entries"]
        for generated in (".pytest_cache", "build", ".egg-info", ".coverage")
    )

    read = payload(tools.execute(make_call("read_file", {"path": "src/a.py"})))
    assert read["content"] == "one\ntwo\n"
    assert read["sha256"] == hashlib.sha256(b"one\ntwo\n").hexdigest()

    no_hash = tools.execute(make_call("write_file", {"path": "src/a.py", "content": "new\n"}))
    assert no_hash.ok is False
    stale = tools.execute(
        make_call(
            "write_file",
            {"path": "src/a.py", "content": "new\n", "expected_sha256": "bad"},
        )
    )
    assert "changed since" in stale.content
    updated = payload(
        tools.execute(
            make_call(
                "write_file",
                {
                    "path": "src/a.py",
                    "content": "new\n",
                    "expected_sha256": read["sha256"],
                },
            )
        )
    )
    assert updated["created"] is False
    assert target.read_text() == "new\n"
    created = payload(
        tools.execute(make_call("write_file", {"path": "new/empty.txt", "content": ""}))
    )
    assert created["created"] is True
    assert (tmp_path / "new" / "empty.txt").exists()


def test_denied_write_does_not_change_file(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("old")
    digest = hashlib.sha256(b"old").hexdigest()
    tools = LocalTools(tmp_path, AlwaysDenyPolicy())
    result = tools.execute(
        make_call(
            "write_file",
            {"path": "a.txt", "content": "new", "expected_sha256": digest},
        )
    )
    assert result.ok is False
    assert target.read_text() == "old"


@pytest.mark.parametrize("path", ["../outside", "/tmp/outside", ".env", "key.pem", ".git/config"])
def test_blocked_paths(tmp_path, path):
    tools = LocalTools(tmp_path, AlwaysApprovePolicy())
    result = tools.execute(make_call("read_file", {"path": path}))
    assert result.ok is False


def test_symlink_escape_and_binary_file_are_blocked(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside")
    (tmp_path / "link").symlink_to(outside)
    (tmp_path / "binary").write_bytes(b"a\x00b")
    tools = LocalTools(tmp_path, AlwaysApprovePolicy())
    assert tools.execute(make_call("read_file", {"path": "link"})).ok is False
    assert "Binary" in tools.execute(make_call("read_file", {"path": "binary"})).content


def test_read_line_limits_and_invalid_arguments(tmp_path):
    (tmp_path / "many.txt").write_text("".join(f"{i}\n" for i in range(500)))
    tools = LocalTools(tmp_path, AlwaysApprovePolicy())
    result = payload(
        tools.execute(
            make_call("read_file", {"path": "many.txt", "start_line": 2, "end_line": 3})
        )
    )
    assert result["content"] == "1\n2\n"
    too_many = tools.execute(
        make_call("read_file", {"path": "many.txt", "start_line": 1, "end_line": 401})
    )
    assert too_many.ok is False
    extra = tools.execute(make_call("read_file", {"path": "many.txt", "unknown": 1}))
    assert "Invalid arguments" in extra.content


def test_search_with_python_fallback(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("alpha\nneedle here\n")
    (tmp_path / "b.txt").write_text("needle ignored by glob\n")
    monkeypatch.setattr("localloop.tools.shutil.which", lambda *args, **kwargs: None)
    tools = LocalTools(tmp_path, AlwaysApprovePolicy())
    result = payload(
        tools.execute(
            make_call("search_text", {"query": "needle", "path": ".", "glob": "*.py"})
        )
    )
    assert result["engine"] == "python"
    assert result["matches"] == ["a.py:2:needle here"]


def test_run_command_success_failure_timeout_block_and_secret_removal(tmp_path, monkeypatch):
    tools = LocalTools(tmp_path, AlwaysApprovePolicy())
    monkeypatch.setenv("LLM_API_KEY", "must-not-leak")
    success = payload(
        tools.execute(
            make_call(
                "run_command",
                {
                    "args": [
                        "python3",
                        "-c",
                        "import os; print(os.getenv('LLM_API_KEY', 'clean'))",
                    ]
                },
            )
        )
    )
    assert success["ok"] is True
    assert success["stdout"].strip() == "clean"
    failure = payload(
        tools.execute(
            make_call("run_command", {"args": ["python3", "-c", "raise SystemExit(3)"]})
        )
    )
    assert failure["exit_code"] == 3
    assert failure["ok"] is False
    timeout = payload(
        tools.execute(
            make_call(
                "run_command",
                {"args": ["python3", "-c", "import time; time.sleep(2)"], "timeout_seconds": 1},
            )
        )
    )
    assert "timed out" in timeout["error"]
    assert tools.execute(make_call("run_command", {"args": ["rm", "x"]})).ok is False
    assert tools.execute(make_call("run_command", {"args": ["git", "reset"]})).ok is False


def test_denied_and_missing_command(tmp_path):
    denied = LocalTools(tmp_path, AlwaysDenyPolicy()).execute(
        make_call("run_command", {"args": ["python3", "--version"]})
    )
    assert denied.ok is False
    missing = LocalTools(tmp_path, AlwaysApprovePolicy()).execute(
        make_call("run_command", {"args": ["definitely-not-a-command"]})
    )
    assert "not found" in missing.content


def test_command_output_is_truncated(tmp_path):
    tools = LocalTools(tmp_path, AlwaysApprovePolicy())
    result = payload(
        tools.execute(
            make_call(
                "run_command",
                {
                    "args": [
                        "python3",
                        "-c",
                        "import sys; print('o' * 30000); print('e' * 30000, file=sys.stderr)",
                    ]
                },
            )
        )
    )
    assert result["ok"] is True
    assert result["truncated"] is True
    assert "output truncated" in result["stdout"]
    assert "output truncated" in result["stderr"]


def test_operating_system_error_becomes_structured_tool_result(tmp_path, monkeypatch):
    target = tmp_path / "blocked.txt"
    target.write_text("content", encoding="utf-8")

    def denied(_path):
        raise PermissionError(13, "permission denied", "/private/identity/blocked.txt")

    monkeypatch.setattr("pathlib.Path.read_bytes", denied)
    tools = LocalTools(tmp_path, AlwaysApprovePolicy())
    result = tools.execute(make_call("read_file", {"path": "blocked.txt"}))
    assert result.ok is False
    assert "permission denied" in result.content
    assert "/private/identity" not in result.content
