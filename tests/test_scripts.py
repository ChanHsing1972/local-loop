from __future__ import annotations

import pytest

from scripts.prepare_demo import DEMO_FILES, prepare_demo


def test_prepare_demo_copies_only_tracked_fixture_files(tmp_path):
    destination = tmp_path / "demo"
    prepared = prepare_demo(destination)
    assert prepared == destination.resolve()
    assert {path.name for path in prepared.iterdir()} == set(DEMO_FILES)
    assert not any(path.name.startswith(".") for path in prepared.iterdir())


def test_prepare_demo_rejects_nonempty_destination(tmp_path):
    destination = tmp_path / "demo"
    destination.mkdir()
    (destination / "existing").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="必须为空"):
        prepare_demo(destination)
