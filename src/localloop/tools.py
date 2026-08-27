from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from localloop.types import ApprovalPolicy, JsonObject, ToolCall, ToolResult

MAX_FILE_BYTES = 1_048_576
MAX_COMMAND_OUTPUT_CHARS = 20_000
MAX_SEARCH_RESULTS = 200
BLOCKED_COMMANDS = {
    "chown",
    "dd",
    "halt",
    "kill",
    "launchctl",
    "mkfs",
    "mount",
    "pkill",
    "poweroff",
    "reboot",
    "rm",
    "shutdown",
    "sudo",
}
BLOCKED_GIT_SUBCOMMANDS = {"clean", "commit", "push", "rebase", "reset", "restore"}
SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
IGNORED_DIRECTORIES = {
    ".git",
    ".localloop",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
}
IGNORED_FILES = {".coverage", ".DS_Store"}


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: JsonObject

    def as_api_tool(self) -> JsonObject:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _safe_environment() -> dict[str, str]:
    allowed_exact = {"PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV"}
    result: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if any(marker in upper for marker in SECRET_ENV_MARKERS):
            continue
        if name in allowed_exact or name.startswith("LC_"):
            result[name] = value
    result.setdefault("PATH", os.defpath)
    result.setdefault("LANG", "C.UTF-8")
    return result


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = limit * 3 // 4
    tail = limit - head
    return text[:head] + "\n...[output truncated]...\n" + text[-tail:], True


class LocalTools:
    def __init__(self, workspace: Path, policy: ApprovalPolicy) -> None:
        self.workspace = workspace.resolve()
        self.policy = policy
        self._definitions = self._build_definitions()
        self._handlers = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "search_text": self.search_text,
            "write_file": self.write_file,
            "run_command": self.run_command,
        }

    @property
    def definitions(self) -> list[JsonObject]:
        return [definition.as_api_tool() for definition in self._definitions]

    def execute(self, call: ToolCall) -> ToolResult:
        handler = self._handlers.get(call.name)
        if handler is None:
            return self._error_result(call, f"Unknown tool: {call.name}")
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            return self._error_result(call, f"Invalid JSON arguments: {exc.msg}")
        if not isinstance(arguments, dict):
            return self._error_result(call, "Tool arguments must be a JSON object")
        try:
            payload = handler(**arguments)
            ok = bool(payload.get("ok", True))
        except ToolError as exc:
            return self._error_result(call, str(exc))
        except OSError as exc:
            detail = exc.strerror or type(exc).__name__
            return self._error_result(call, f"本地操作系统错误：{detail}")
        except TypeError as exc:
            return self._error_result(call, f"Invalid arguments for {call.name}: {exc}")
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            ok=ok,
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )

    @staticmethod
    def _error_result(call: ToolCall, message: str) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            ok=False,
            content=json.dumps({"ok": False, "error": message}, ensure_ascii=False),
        )

    def _validate_path(self, raw_path: str, *, must_exist: bool = False) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path must be a non-empty string")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ToolError("Absolute paths and '..' are not allowed")
        for part in relative.parts:
            lower = part.lower()
            if lower in {".git", ".localloop", ".env", "id_rsa", "id_ed25519"}:
                raise ToolError(f"Sensitive path is blocked: {raw_path}")
            if lower.startswith(".env.") and lower != ".env.example":
                raise ToolError(f"Sensitive path is blocked: {raw_path}")
            if lower.endswith((".pem", ".p12", ".pfx", ".key")):
                raise ToolError(f"Credential-like file is blocked: {raw_path}")
        candidate = self.workspace / relative
        try:
            resolved = candidate.resolve(strict=must_exist)
        except FileNotFoundError as exc:
            raise ToolError(f"Path does not exist: {raw_path}") from exc
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ToolError(f"Path escapes the workspace: {raw_path}") from exc
        return resolved

    @staticmethod
    def _read_text(path: Path) -> tuple[bytes, str]:
        if not path.is_file():
            raise ToolError(f"Not a regular file: {path.name}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ToolError(f"File is too large ({size} bytes; limit is {MAX_FILE_BYTES})")
        raw = path.read_bytes()
        if b"\x00" in raw:
            raise ToolError("Binary files are not supported")
        try:
            return raw, raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("File is not valid UTF-8 text") from exc

    def list_files(self, path: str = ".", max_depth: int = 3) -> JsonObject:
        if not isinstance(max_depth, int) or not 0 <= max_depth <= 8:
            raise ToolError("max_depth must be an integer between 0 and 8")
        root = self._validate_path(path, must_exist=True)
        if not root.is_dir():
            raise ToolError(f"Not a directory: {path}")
        entries: list[str] = []
        for current, directories, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            depth = len(current_path.relative_to(root).parts)
            directories[:] = sorted(
                name
                for name in directories
                if name not in IGNORED_DIRECTORIES
                and not name.endswith(".egg-info")
                and not name.startswith(".env")
            )
            if depth >= max_depth:
                directories[:] = []
            for name in directories:
                entries.append((current_path / name).relative_to(self.workspace).as_posix() + "/")
            for name in sorted(filenames):
                if (
                    name in IGNORED_FILES
                    or name == ".env"
                    or (name.startswith(".env.") and name != ".env.example")
                ):
                    continue
                entries.append((current_path / name).relative_to(self.workspace).as_posix())
            if len(entries) >= 500:
                break
        entries = entries[:500]
        return {"ok": True, "path": path, "entries": entries, "truncated": len(entries) == 500}

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> JsonObject:
        if not isinstance(start_line, int) or start_line < 1:
            raise ToolError("start_line must be a positive integer")
        if end_line is None:
            end_line = start_line + 399
        if not isinstance(end_line, int) or end_line < start_line:
            raise ToolError("end_line must be an integer greater than or equal to start_line")
        if end_line - start_line + 1 > 400:
            raise ToolError("At most 400 lines may be read at once")
        resolved = self._validate_path(path, must_exist=True)
        raw, text = self._read_text(resolved)
        lines = text.splitlines(keepends=True)
        selected = "".join(lines[start_line - 1 : end_line])
        return {
            "ok": True,
            "path": path,
            "start_line": start_line,
            "end_line": min(end_line, len(lines)),
            "total_lines": len(lines),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "content": selected,
            "truncated": end_line < len(lines),
        }

    def search_text(self, query: str, path: str = ".", glob: str | None = None) -> JsonObject:
        if not isinstance(query, str) or not query:
            raise ToolError("query must be a non-empty string")
        if len(query) > 500:
            raise ToolError("query is too long")
        root = self._validate_path(path, must_exist=True)
        if not root.is_dir():
            raise ToolError("search path must be a directory")
        rg = shutil.which("rg", path=_safe_environment().get("PATH"))
        if rg:
            command = [
                rg,
                "--line-number",
                "--no-heading",
                "--color=never",
                "--max-count",
                str(MAX_SEARCH_RESULTS),
                "--glob",
                "!.git/**",
                "--glob",
                "!.localloop/**",
                "--glob",
                "!.env",
                "--glob",
                "!.env.*",
            ]
            if glob:
                command.extend(["--glob", glob])
            relative_root = root.relative_to(self.workspace).as_posix() or "."
            command.extend(["--", query, relative_root])
            completed = subprocess.run(
                command,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_safe_environment(),
                timeout=15,
                check=False,
            )
            if completed.returncode not in {0, 1}:
                raise ToolError(f"rg failed: {completed.stderr[:500]}")
            matches = completed.stdout.splitlines()[:MAX_SEARCH_RESULTS]
            return {"ok": True, "query": query, "matches": matches, "engine": "rg"}
        return self._python_search(query=query, root=root, glob=glob)

    def _python_search(self, *, query: str, root: Path, glob: str | None) -> JsonObject:
        matches: list[str] = []
        for current, directories, filenames in os.walk(root, followlinks=False):
            directories[:] = [
                name
                for name in directories
                if name not in IGNORED_DIRECTORIES
                and not name.endswith(".egg-info")
                and not name.startswith(".env")
            ]
            for name in filenames:
                if glob and not fnmatch.fnmatch(name, glob):
                    continue
                candidate = Path(current) / name
                try:
                    _raw, text = self._read_text(candidate)
                except (OSError, ToolError):
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if query in line:
                        relative = candidate.relative_to(self.workspace).as_posix()
                        matches.append(f"{relative}:{line_number}:{line}")
                        if len(matches) >= MAX_SEARCH_RESULTS:
                            return {
                                "ok": True,
                                "query": query,
                                "matches": matches,
                                "engine": "python",
                                "truncated": True,
                            }
        return {"ok": True, "query": query, "matches": matches, "engine": "python"}

    def write_file(
        self,
        path: str,
        content: str,
        expected_sha256: str | None = None,
    ) -> JsonObject:
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_FILE_BYTES:
            raise ToolError(f"New content exceeds the {MAX_FILE_BYTES}-byte limit")
        resolved = self._validate_path(path)
        existed = resolved.exists()
        previous = ""
        previous_mode: int | None = None
        if existed:
            raw, previous = self._read_text(resolved)
            actual_hash = hashlib.sha256(raw).hexdigest()
            if not expected_sha256:
                raise ToolError("expected_sha256 is required when updating an existing file")
            if expected_sha256 != actual_hash:
                raise ToolError("File changed since it was read; read it again before writing")
            previous_mode = stat.S_IMODE(resolved.stat().st_mode)
        elif expected_sha256:
            raise ToolError("expected_sha256 was supplied, but the target file does not exist")

        diff = "".join(
            difflib.unified_diff(
                previous.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        preview, preview_truncated = _truncate(diff, 12_000)
        if not self.policy.approve("write_file", path, preview):
            raise ToolError("User denied file write")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=resolved.parent, prefix=f".{resolved.name}.", delete=False
            ) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
                temp_name = stream.name
            if previous_mode is not None:
                os.chmod(temp_name, previous_mode)
            os.replace(temp_name, resolved)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
        return {
            "ok": True,
            "path": path,
            "bytes_written": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "created": not existed,
            "diff_truncated": preview_truncated,
        }

    def run_command(
        self,
        args: list[str],
        cwd: str = ".",
        timeout_seconds: int = 30,
    ) -> JsonObject:
        if not isinstance(args, list) or not args or len(args) > 64:
            raise ToolError("args must be a non-empty array with at most 64 strings")
        if any(not isinstance(item, str) or not item or "\x00" in item for item in args):
            raise ToolError("Every command argument must be a non-empty string without NUL bytes")
        if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 120:
            raise ToolError("timeout_seconds must be between 1 and 120")
        directory = self._validate_path(cwd, must_exist=True)
        if not directory.is_dir():
            raise ToolError("cwd must be a directory")
        executable = Path(args[0]).name.lower()
        if executable in BLOCKED_COMMANDS:
            raise ToolError(f"Blocked destructive command: {executable}")
        if executable == "git" and len(args) > 1 and args[1].lower() in BLOCKED_GIT_SUBCOMMANDS:
            raise ToolError(f"Blocked state-changing git command: git {args[1]}")
        display = shlex.join(args)
        if not self.policy.approve("run_command", f"cwd={cwd} command={display}"):
            raise ToolError("User denied command execution")
        try:
            completed = subprocess.run(
                args,
                cwd=directory,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=_safe_environment(),
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            combined, truncated = _truncate(stdout + stderr, MAX_COMMAND_OUTPUT_CHARS)
            return {
                "ok": False,
                "error": f"Command timed out after {timeout_seconds} seconds",
                "output": combined,
                "truncated": truncated,
            }
        except FileNotFoundError as exc:
            raise ToolError(f"Command not found: {args[0]}") from exc
        stdout, stdout_truncated = _truncate(completed.stdout, MAX_COMMAND_OUTPUT_CHARS // 2)
        stderr, stderr_truncated = _truncate(completed.stderr, MAX_COMMAND_OUTPUT_CHARS // 2)
        return {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": stdout_truncated or stderr_truncated,
        }

    @staticmethod
    def _build_definitions() -> tuple[ToolDefinition, ...]:
        return (
            ToolDefinition(
                "list_files",
                (
                    "List files under a workspace-relative directory. "
                    "Sensitive directories are omitted."
                ),
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "max_depth": {"type": "integer", "minimum": 0, "maximum": 8},
                    },
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                "read_file",
                "Read up to 400 lines of a UTF-8 file and return its SHA-256 for safe updates.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                "search_text",
                "Search literal text in workspace files using ripgrep with a Python fallback.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                        "glob": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                "write_file",
                (
                    "Create or atomically replace a UTF-8 file. "
                    "Existing files require the last read SHA-256."
                ),
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "expected_sha256": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                "run_command",
                "Run an argv array without a shell, with a sanitized environment and timeout.",
                {
                    "type": "object",
                    "properties": {
                        "args": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "cwd": {"type": "string", "default": "."},
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 120,
                            "default": 30,
                        },
                    },
                    "required": ["args"],
                    "additionalProperties": False,
                },
            ),
        )
