"""OpenAI-compatible VLM client hardening: max_tokens, JPEG part, retry/backoff, and
AirgapError surfacing. Uses httpx's MockTransport so no server is needed; skipped where
httpx (the `vlm` extra) isn't installed."""

from __future__ import annotations

import json

import pytest

httpx = pytest.importorskip("httpx")

from fusion_ocr.config import AirgapError
from fusion_ocr.vlm.openai_compat import OpenAICompatVLM


def _client(handler, **kw) -> OpenAICompatVLM:
    c = OpenAICompatVLM(base_url="http://localhost:8080/v1", model="m", **kw)
    c._http = httpx.Client(transport=httpx.MockTransport(handler))   # inject the mock
    return c


def _ok(content="hello"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_read_sends_jpeg_part_and_max_tokens():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok("read!")

    c = _client(handler, max_tokens=1234)
    assert c.read(b"\xff\xd8\xffJPEG", "transcribe") == "read!"
    body = seen["body"]
    assert body["max_tokens"] == 1234
    url = body["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")   # JPEG, not PNG


def test_max_tokens_zero_omits_the_cap():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok()

    c = _client(handler, max_tokens=0)
    c.read(b"x", "p")
    assert "max_tokens" not in seen["body"]


def test_retries_a_transient_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr("fusion_ocr.vlm.openai_compat.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else _ok("recovered")

    c = _client(handler, max_retries=2)
    assert c.read(b"x", "p") == "recovered"
    assert calls["n"] == 3   # two 503s retried, third OK


def test_5xx_past_the_retry_budget_raises(monkeypatch):
    monkeypatch.setattr("fusion_ocr.vlm.openai_compat.time.sleep", lambda _s: None)
    c = _client(lambda r: httpx.Response(503), max_retries=2)
    with pytest.raises(httpx.HTTPStatusError):
        c.read(b"x", "p")


def test_airgap_surfaces_and_is_never_retried(monkeypatch):
    monkeypatch.setattr("fusion_ocr.vlm.openai_compat.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        err = httpx.ConnectError("blocked")
        err.__cause__ = AirgapError("airgap: outbound connection refused")  # how httpx wraps it
        raise err

    c = _client(handler, max_retries=3)
    with pytest.raises(AirgapError):
        c.read(b"x", "p")
    assert calls["n"] == 1   # failed loud on the first attempt, no spinning


def test_keep_alive_client_is_reused_across_reads():
    c = _client(lambda r: _ok())
    same = c._http
    c.read(b"a", "p")
    c.read(b"b", "p")
    assert c._http is same   # one client, not one per call
