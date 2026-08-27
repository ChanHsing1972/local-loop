from __future__ import annotations

import httpx
import pytest

from scripts.prepare_demo import DEMO_FILES, prepare_demo
from scripts.release_check import check_online_repository


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


def test_online_release_check_accepts_public_neutral_github_account():
    def handler(request: httpx.Request) -> httpx.Response:
        if "/repos/" in request.url.path:
            return httpx.Response(200, json={"private": False})
        return httpx.Response(
            200,
            json={"name": None, "bio": None, "company": None, "location": None, "blog": ""},
        )

    failures = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        check_online_repository(
            "仓库地址：https://github.com/neutral/local-loop",
            failures,
            client=client,
        )
    assert failures == []


def test_online_release_check_reports_private_repo_and_profile_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        if "/repos/" in request.url.path:
            return httpx.Response(404, json={"message": "Not Found"})
        return httpx.Response(200, json={"name": "identity", "company": "school"})

    failures = []
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        check_online_repository(
            "仓库地址：https://github.com/account/local-loop",
            failures,
            client=client,
        )
    assert any("无法匿名读取" in failure for failure in failures)
    assert any("name" in failure and "company" in failure for failure in failures)
