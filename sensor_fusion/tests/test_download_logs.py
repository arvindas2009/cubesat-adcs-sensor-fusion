"""Tests for ``sensor_fusion.download_logs`` (spec ``docs/specs/sensor_fusion_data.md`` section 1).

No network: ``download_log`` is driven through a fake ``session`` object and ``tmp_path``;
``time.sleep`` is patched so the retry back-off does not slow the tests.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

import pytest
import requests

from sensor_fusion import download_logs as dl

MAGIC = b"ULog\x01\x12\x35"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
LOG_ID = "11111111-2222-3333-4444-555555555555"


class FakeResponse:
    def __init__(self, body: bytes = b"", status: int = 200):
        self.status_code = status
        self.content = body
        self.headers = {"Content-Length": str(len(body))}
        self.ok = status == 200

    def iter_content(self, chunk_size=1 << 16):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        pass


class FakeSession:
    """Plays back a list of responses / exceptions, one per ``get`` call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def sleeps(monkeypatch):
    rec = []
    monkeypatch.setattr(dl.time, "sleep", lambda s: rec.append(s))
    return rec


# ---------------------------------------------------------------------------- PINNED_LOGS
def _entries():
    p = dl.PINNED_LOGS
    return list(p.values()) if isinstance(p, dict) else list(p)


def test_pinned_logs_three_wellformed_unique_entries():
    entries = _entries()
    assert len(entries) == 3
    for e in entries:
        assert {"id", "role", "notes"} <= set(e)
        assert UUID_RE.match(e["id"]), e["id"]
        assert e["role"] in {"hover", "aggressive", "other"}
        assert isinstance(e["notes"], str) and len(e["notes"].strip()) > 10
    assert len({e["id"] for e in entries}) == 3


def test_pinned_logs_include_hover_and_aggressive_roles():
    roles = [e["role"] for e in _entries()]
    assert "hover" in roles and "aggressive" in roles


def test_pinned_ids_match_notes_md():
    notes = (Path(dl.DATA_DIR) / "NOTES.md").read_text(encoding="utf-8")
    for e in _entries():
        assert e["id"] in notes, f"{e['id']} pinned but not documented in data/NOTES.md"


def test_data_dir_is_sensor_fusion_data_and_ulg_gitignored():
    repo = Path(__file__).resolve().parents[2]
    assert Path(dl.DATA_DIR).resolve() == (repo / "sensor_fusion" / "data").resolve()
    gi = (repo / ".gitignore").read_text()
    assert "sensor_fusion/data/*.ulg" in gi or "*.ulg" in gi


def test_ulog_magic_constant_is_spec_value():
    assert dl.ULOG_MAGIC == MAGIC


def test_pinned_files_on_disk_if_present_are_ulogs():
    for e in _entries():
        p = Path(dl.DATA_DIR) / f"{e['id']}.ulg"
        if not p.exists():
            pytest.skip(f"{p.name} not downloaded")
        with open(p, "rb") as fh:
            assert fh.read(7) == MAGIC
        assert p.stat().st_size < 40e6 * 1.05  # spec: < ~40 MB each


# ---------------------------------------------------------------------------- download_log
def test_download_writes_file_with_payload_and_returns_path(tmp_path, sleeps):
    payload = MAGIC + b"\x00" * 1000 + bytes(range(256))
    sess = FakeSession([FakeResponse(payload)])
    out = dl.download_log(LOG_ID, tmp_path, session=sess)
    assert Path(out) == tmp_path / f"{LOG_ID}.ulg"
    assert out.read_bytes() == payload
    assert len(sess.calls) == 1
    url, kw = sess.calls[0]
    assert "review.px4.io/download" in url
    assert LOG_ID in url + str(kw.get("params", ""))
    assert kw.get("timeout") is not None  # spec: timeout
    assert not list(tmp_path.glob("*.part"))  # no temp leftovers


def test_download_creates_dest_dir(tmp_path):
    dest = tmp_path / "a" / "b"
    sess = FakeSession([FakeResponse(MAGIC + b"x")])
    out = dl.download_log(LOG_ID, dest, session=sess)
    assert out.exists()


def test_download_skips_existing_nonempty_file(tmp_path):
    existing = tmp_path / f"{LOG_ID}.ulg"
    existing.write_bytes(b"existing-content")  # deliberately not even a ULog: must not be touched
    sess = FakeSession([RuntimeError("network must not be used")])
    out = dl.download_log(LOG_ID, tmp_path, session=sess)
    assert Path(out) == existing
    assert existing.read_bytes() == b"existing-content"
    assert sess.calls == []


def test_download_replaces_empty_existing_file(tmp_path):
    existing = tmp_path / f"{LOG_ID}.ulg"
    existing.write_bytes(b"")
    payload = MAGIC + b"abc"
    out = dl.download_log(LOG_ID, tmp_path, session=FakeSession([FakeResponse(payload)]))
    assert out.read_bytes() == payload


def test_download_rejects_file_without_ulog_magic(tmp_path):
    sess = FakeSession([FakeResponse(b"<html>Not Found</html>" + b"\x00" * 100)])
    with pytest.raises((ValueError, RuntimeError, OSError)):
        dl.download_log(LOG_ID, tmp_path, session=sess)
    # a corrupt file must not be left where the "skip existing" logic would accept it
    assert not (tmp_path / f"{LOG_ID}.ulg").exists()


def test_download_rejects_truncated_magic(tmp_path):
    with pytest.raises((ValueError, RuntimeError, OSError)):
        dl.download_log(LOG_ID, tmp_path, session=FakeSession([FakeResponse(MAGIC[:-1])]))
    assert not (tmp_path / f"{LOG_ID}.ulg").exists()


def test_download_retries_transient_errors_with_exponential_backoff(tmp_path, sleeps):
    payload = MAGIC + b"ok"
    sess = FakeSession([requests.ConnectionError("boom"), requests.Timeout("slow"), FakeResponse(payload)])
    out = dl.download_log(LOG_ID, tmp_path, session=sess)
    assert out.read_bytes() == payload
    assert len(sess.calls) == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0] > 0  # back-off grows


def test_download_gives_up_after_at_most_three_retries(tmp_path, sleeps):
    sess = FakeSession([requests.ConnectionError("down")])
    with pytest.raises(Exception):
        dl.download_log(LOG_ID, tmp_path, session=sess)
    assert 2 <= len(sess.calls) <= 4  # 1 attempt + <= 3 retries
    assert all(b > a for a, b in zip(sleeps, sleeps[1:]))
    assert not (tmp_path / f"{LOG_ID}.ulg").exists()


def test_download_retries_server_errors_then_succeeds(tmp_path, sleeps):
    payload = MAGIC + b"z"
    sess = FakeSession([FakeResponse(b"", 503), FakeResponse(payload)])
    out = dl.download_log(LOG_ID, tmp_path, session=sess)
    assert out.read_bytes() == payload
    assert len(sess.calls) == 2


def test_download_http_404_raises_and_leaves_no_file(tmp_path, sleeps):
    sess = FakeSession([FakeResponse(b"", 404)])
    with pytest.raises(Exception):
        dl.download_log(str(uuid.uuid4()), tmp_path, session=sess)
    assert list(tmp_path.glob("*.ulg")) == []
