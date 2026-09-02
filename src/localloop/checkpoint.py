"""记录并安全撤销 LocalLoop 通过 write_file 产生的文件修改。"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from localloop.types import ApprovalPolicy

CHECKPOINT_VERSION = 1
CHECKPOINT_ID_PATTERN = re.compile(r"^cp-[a-f0-9]{6}$")
MAX_UNDO_DIFF_CHARS = 12_000


class CheckpointError(RuntimeError):
    """表示检查点损坏、发生冲突或用户取消撤销。"""


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    id: str
    task: str
    created_at: str
    status: str
    files: tuple[str, ...]


class CheckpointStore:
    """按一次用户任务归组保存 write_file 修改前的内容。"""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        raw_directory = self.workspace / ".localloop" / "checkpoints"
        self.directory = raw_directory.resolve(strict=False)
        try:
            self.directory.relative_to(self.workspace)
        except ValueError as exc:
            raise CheckpointError("检查点目录不能通过符号链接越过工作区") from exc
        self._active: dict[str, Any] | None = None

    def begin(self, *, session_id: str, task: str) -> None:
        """开始一次任务；没有文件写入时不会创建磁盘记录。"""

        if self._active is not None:
            raise CheckpointError("已有尚未结束的检查点任务")
        checkpoint_id = self._new_id()
        self._active = {
            "version": CHECKPOINT_VERSION,
            "id": checkpoint_id,
            "session_id": session_id,
            "task": task.strip()[:500],
            "created_at": datetime.now(UTC).isoformat(),
            "status": "in_progress",
            "files": [],
        }

    def capture_before(
        self,
        target: Path,
        *,
        previous: bytes | None,
        new_content: bytes,
        previous_mode: int | None,
    ) -> None:
        """在真正写文件前持久化旧版本和计划写入版本的哈希。"""

        if self._active is None:
            return
        relative = self._relative_target(target)
        files: list[dict[str, Any]] = self._active["files"]
        existing = next((item for item in files if item["path"] == relative), None)
        after_sha256 = hashlib.sha256(new_content).hexdigest()
        if existing is not None:
            existing["after_sha256"] = after_sha256
            history = existing.setdefault("after_sha256_history", [])
            if after_sha256 not in history:
                history.append(after_sha256)
            self._write_manifest(self._active)
            return

        checkpoint_id = self._active["id"]
        backup: str | None = None
        before_sha256: str | None = None
        if previous is not None:
            backup = f"before/{relative}"
            backup_path = self._checkpoint_directory(checkpoint_id) / backup
            self._atomic_write(backup_path, previous, mode=0o600)
            before_sha256 = hashlib.sha256(previous).hexdigest()
        files.append(
            {
                "path": relative,
                "existed_before": previous is not None,
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "after_sha256_history": [after_sha256],
                "backup": backup,
                "previous_mode": previous_mode,
            }
        )
        self._write_manifest(self._active)

    def finish(self) -> CheckpointInfo | None:
        """结束当前任务；返回实际记录了文件修改的检查点。"""

        if self._active is None:
            return None
        manifest = self._active
        self._active = None
        if not manifest["files"]:
            return None
        manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        self._write_manifest(manifest)
        return self._to_info(manifest)

    def list_checkpoints(self, limit: int = 10) -> list[CheckpointInfo]:
        """按创建时间从新到旧列出可读检查点。"""

        self._ensure_directory_safe()
        if self.directory.is_symlink():
            raise CheckpointError("检查点目录不能是符号链接")
        if not self.directory.exists():
            return []
        manifests: list[dict[str, Any]] = []
        for path in self.directory.glob("cp-*/manifest.json"):
            checkpoint_id = path.parent.name
            try:
                manifests.append(self._read_manifest(checkpoint_id))
            except CheckpointError:
                continue
        manifests.sort(key=lambda item: str(item["created_at"]), reverse=True)
        return [self._to_info(item) for item in manifests[:limit]]

    def undo_latest(self, policy: ApprovalPolicy) -> CheckpointInfo:
        """经冲突检查和审批后撤销最近一个尚未撤销的检查点。"""

        checkpoint = next(
            (
                item
                for item in self.list_checkpoints(limit=100)
                if item.status in {"completed", "in_progress"}
            ),
            None,
        )
        if checkpoint is None:
            raise CheckpointError("当前工作区没有可撤销的文件检查点")
        manifest = self._read_manifest(checkpoint.id)
        changes = self._prepare_restore(manifest)
        if not changes:
            manifest["status"] = "restored"
            manifest["restored_at"] = datetime.now(UTC).isoformat()
            self._write_manifest(manifest)
            return self._to_info(manifest)

        diff = self._restore_diff(changes)
        if not policy.approve(
            "undo",
            f"{checkpoint.id}（{len(changes)} 个文件）",
            diff,
        ):
            raise CheckpointError("用户取消了撤销")

        # 审批期间文件仍可能被其他程序修改，因此执行前再次做同样的哈希检查。
        changes = self._prepare_restore(manifest)
        for entry, target, before, _current in changes:
            if entry["existed_before"]:
                mode = entry.get("previous_mode")
                self._atomic_write(target, before, mode=mode)
            else:
                with suppress(FileNotFoundError):
                    target.unlink()
        manifest["status"] = "restored"
        manifest["restored_at"] = datetime.now(UTC).isoformat()
        self._write_manifest(manifest)
        return self._to_info(manifest)

    def _prepare_restore(
        self, manifest: dict[str, Any]
    ) -> list[tuple[dict[str, Any], Path, bytes, bytes]]:
        changes: list[tuple[dict[str, Any], Path, bytes, bytes]] = []
        checkpoint_directory = self._checkpoint_directory(manifest["id"])
        for entry in manifest["files"]:
            target = self._target_from_relative(entry["path"])
            existed_before = bool(entry["existed_before"])
            if not target.exists():
                if not existed_before:
                    continue
                raise CheckpointError(f"无法撤销 {entry['path']}：当前文件已不存在")
            if target.is_symlink() or not target.is_file():
                raise CheckpointError(f"无法撤销 {entry['path']}：目标不再是普通文件")
            try:
                current = target.read_bytes()
            except OSError as exc:
                raise CheckpointError(f"无法读取待撤销文件 {entry['path']}：{exc}") from exc
            current_sha256 = hashlib.sha256(current).hexdigest()

            before = b""
            if existed_before:
                backup = entry.get("backup")
                if not isinstance(backup, str):
                    raise CheckpointError(f"检查点缺少 {entry['path']} 的原始内容")
                backup_path = (checkpoint_directory / backup).resolve(strict=False)
                try:
                    backup_path.relative_to(checkpoint_directory / "before")
                except ValueError as exc:
                    raise CheckpointError("检查点备份路径无效") from exc
                try:
                    before = backup_path.read_bytes()
                except OSError as exc:
                    raise CheckpointError(f"无法读取 {entry['path']} 的检查点备份") from exc
                if hashlib.sha256(before).hexdigest() != entry.get("before_sha256"):
                    raise CheckpointError(f"{entry['path']} 的检查点备份校验失败")
                if current_sha256 == entry.get("before_sha256"):
                    continue
            accepted_hashes = set(entry.get("after_sha256_history", []))
            accepted_hashes.add(entry.get("after_sha256"))
            if current_sha256 not in accepted_hashes:
                raise CheckpointError(
                    f"无法撤销 {entry['path']}：创建检查点后文件又发生了变化"
                )
            changes.append((entry, target, before, current))
        return changes

    @staticmethod
    def _restore_diff(changes: list[tuple[dict[str, Any], Path, bytes, bytes]]) -> str:
        parts: list[str] = []
        for entry, _target, before, current in changes:
            try:
                old_text = current.decode("utf-8")
                new_text = before.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CheckpointError(f"无法生成 {entry['path']} 的文本撤销预览") from exc
            parts.extend(
                difflib.unified_diff(
                    old_text.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=f"current/{entry['path']}",
                    tofile=f"restored/{entry['path']}",
                )
            )
        diff = "".join(parts)
        if len(diff) <= MAX_UNDO_DIFF_CHARS:
            return diff
        return diff[:MAX_UNDO_DIFF_CHARS] + "\n...[撤销预览已截断]...\n"

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        checkpoint_directory = self._checkpoint_directory(manifest["id"])
        checkpoint_directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(checkpoint_directory, 0o700)
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        self._atomic_write(checkpoint_directory / "manifest.json", payload, mode=0o600)

    def _read_manifest(self, checkpoint_id: str) -> dict[str, Any]:
        checkpoint_directory = self._checkpoint_directory(checkpoint_id)
        manifest_path = checkpoint_directory / "manifest.json"
        if manifest_path.is_symlink():
            raise CheckpointError("检查点清单不能是符号链接")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"无法读取检查点 {checkpoint_id}") from exc
        if not isinstance(manifest, dict) or manifest.get("version") != CHECKPOINT_VERSION:
            raise CheckpointError(f"检查点 {checkpoint_id} 格式无效")
        if manifest.get("id") != checkpoint_id or not isinstance(manifest.get("files"), list):
            raise CheckpointError(f"检查点 {checkpoint_id} 内容无效")
        if not isinstance(manifest.get("created_at"), str) or not isinstance(
            manifest.get("status"), str
        ):
            raise CheckpointError(f"检查点 {checkpoint_id} 元数据无效")
        for entry in manifest["files"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise CheckpointError(f"检查点 {checkpoint_id} 文件记录无效")
            if not isinstance(entry.get("existed_before"), bool):
                raise CheckpointError(f"检查点 {checkpoint_id} 文件状态无效")
            if not isinstance(entry.get("after_sha256"), str):
                raise CheckpointError(f"检查点 {checkpoint_id} 文件哈希无效")
            history = entry.get("after_sha256_history")
            if not isinstance(history, list) or not all(isinstance(item, str) for item in history):
                raise CheckpointError(f"检查点 {checkpoint_id} 哈希历史无效")
            if entry.get("backup") is not None and not isinstance(entry.get("backup"), str):
                raise CheckpointError(f"检查点 {checkpoint_id} 备份记录无效")
            if entry.get("previous_mode") is not None and not isinstance(
                entry.get("previous_mode"), int
            ):
                raise CheckpointError(f"检查点 {checkpoint_id} 权限记录无效")
        return manifest

    def _checkpoint_directory(self, checkpoint_id: str) -> Path:
        self._ensure_directory_safe()
        if not CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_id):
            raise CheckpointError("检查点编号格式无效")
        path = self.directory / checkpoint_id
        if path.is_symlink():
            raise CheckpointError("检查点目录不能是符号链接")
        return path

    def _ensure_directory_safe(self) -> None:
        resolved = self.directory.resolve(strict=False)
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise CheckpointError("检查点目录不能通过符号链接越过工作区") from exc
        if resolved != self.directory:
            raise CheckpointError("检查点目录在运行期间被替换为符号链接")

    def _relative_target(self, target: Path) -> str:
        try:
            return target.resolve(strict=False).relative_to(self.workspace).as_posix()
        except ValueError as exc:
            raise CheckpointError("检查点目标越过工作区") from exc

    def _target_from_relative(self, raw_path: str) -> Path:
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts or ".localloop" in relative.parts:
            raise CheckpointError("检查点中的文件路径无效")
        target = (self.workspace / relative).resolve(strict=False)
        try:
            target.relative_to(self.workspace)
        except ValueError as exc:
            raise CheckpointError("检查点目标通过符号链接越过工作区") from exc
        return target

    def _new_id(self) -> str:
        while True:
            checkpoint_id = f"cp-{uuid.uuid4().hex[:6]}"
            if not (self.directory / checkpoint_id).exists():
                return checkpoint_id

    @staticmethod
    def _to_info(manifest: dict[str, Any]) -> CheckpointInfo:
        return CheckpointInfo(
            id=str(manifest["id"]),
            task=str(manifest.get("task", "")),
            created_at=str(manifest.get("created_at", "")),
            status=str(manifest.get("status", "unknown")),
            files=tuple(str(entry["path"]) for entry in manifest["files"]),
        )

    @staticmethod
    def _atomic_write(path: Path, content: bytes, mode: int | None) -> None:
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temp_name = stream.name
            if mode is not None:
                os.chmod(temp_name, stat.S_IMODE(mode))
            os.replace(temp_name, path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
