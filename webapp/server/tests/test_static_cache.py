"""Cache headers of the static frontend mount (protocol spec section 4).

A cached ``index.html`` pointed the browser at a stale bundle after a rebuild (Group 11 browser
check), so the page must be revalidated while content-hashed assets may be cached forever.
"""
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from webapp.server.app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "textures").mkdir()
    (tmp_path / "textures" / "earth.jpg").write_bytes(b"\xff\xd8\xff\xe0fake")
    with TestClient(create_app(frontend_dist=tmp_path)) as c:
        yield c


@pytest.mark.parametrize("url", ["/", "/index.html", "/textures/earth.jpg"])
def test_page_and_unhashed_files_are_revalidated(client, url):
    r = client.get(url)
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"


def test_hashed_assets_are_immutable(client):
    r = client.get("/assets/index-abc123.js")
    assert r.status_code == 200
    assert "immutable" in r.headers["cache-control"]


def test_api_unaffected(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"ok": True}
