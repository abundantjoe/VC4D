"""The remote stores retry a transfer cut off mid-body (seen on dl.ash2txt.org through a draining relay)."""
import pytest

requests = pytest.importorskip("requests")

from unroller import volume


def _fake(fails, exc):
    calls = {"n": 0}

    def request(self, method, url, *a, **kw):
        calls["n"] += 1
        if calls["n"] <= fails:
            raise exc("Connection broken: connection reset by peer")
        r = requests.models.Response()
        r.status_code, r._content = 200, b"ok"
        return r

    return request, calls


def test_retry_on_chunked_encoding_error(monkeypatch):
    req, calls = _fake(2, requests.exceptions.ChunkedEncodingError)
    monkeypatch.setattr(requests.Session, "request", req)
    s = volume._RetrySession()
    s.backoff = 0.0
    assert s.get("https://example.invalid/x").content == b"ok"
    assert calls["n"] == 3


def test_retry_gives_up(monkeypatch):
    req, calls = _fake(99, requests.exceptions.ConnectionError)
    monkeypatch.setattr(requests.Session, "request", req)
    s = volume._RetrySession()
    s.backoff, s.retries = 0.0, 2
    with pytest.raises(requests.exceptions.ConnectionError):
        s.get("https://example.invalid/x")
    assert calls["n"] == 3


def test_store_uses_retry_session():
    st = volume._Store("https://example.invalid/vol.zarr")
    assert isinstance(st._session, volume._retry_session_cls())
