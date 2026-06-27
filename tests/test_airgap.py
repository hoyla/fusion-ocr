"""Airgap guard — the sealed (most-sensitive) tier. Closes the holes the review found:
full loopback range, DNS, connect_ex, AF_UNIX. The fixture restores the patched socket
functions so the guard never leaks into the rest of the test run."""

from __future__ import annotations

import socket

import pytest

from fusion_ocr import config as config_mod
from fusion_ocr.config import AirgapError, _is_loopback_host


@pytest.fixture
def airgapped():
    config_mod.enforce_airgap()
    try:
        yield
    finally:
        config_mod._disable_airgap()


def test_loopback_range_recognised():
    assert _is_loopback_host("127.0.0.1")
    assert _is_loopback_host("127.0.0.5")          # the rest of 127/8, not just .0.1
    assert _is_loopback_host("::1")
    assert _is_loopback_host("::ffff:127.0.0.1")   # ipv4-mapped ipv6
    assert _is_loopback_host("localhost")
    assert not _is_loopback_host("8.8.8.8")
    assert not _is_loopback_host("transcription-gpu.internal")


def test_connect_to_remote_raises_airgap_error(airgapped):
    s = socket.socket()
    with pytest.raises(AirgapError):
        s.connect(("8.8.8.8", 53))
    s.close()


def test_connect_ex_is_guarded(airgapped):
    # connect_ex previously bypassed the guard entirely
    s = socket.socket()
    with pytest.raises(AirgapError):
        s.connect_ex(("8.8.8.8", 53))
    s.close()


def test_dns_lookup_of_remote_name_refused(airgapped):
    # resolving a hostname egresses a DNS query before connect could refuse it
    with pytest.raises(AirgapError):
        socket.getaddrinfo("transcription-gpu.internal", 8000)


def test_loopback_lookups_allowed(airgapped):
    # IP literals + localhost resolve locally (no egress) -> must not be refused
    socket.getaddrinfo("127.0.0.1", 8080)
    socket.getaddrinfo("localhost", 8080)


def test_af_unix_not_blocked(airgapped):
    # AF_UNIX is local IPC, not egress -> the guard must let it reach the real connect
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("no AF_UNIX on this platform")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with pytest.raises(OSError) as excinfo:
        s.connect("/tmp/fusion-airgap-does-not-exist.sock")
    assert not isinstance(excinfo.value, AirgapError)   # failed on ENOENT, not the guard
    s.close()


def test_guard_restores_cleanly(airgapped):
    pass  # if teardown didn't restore, later tests resolving names would break


def test_socket_unpatched_after_fixture():
    # outside the airgapped fixture the guard is gone (no leak)
    assert not getattr(socket.socket, "_fusion_airgapped", False)


def test_language_probe_reraises_airgap_error():
    # The script probe hits the VLM endpoint. In a sealed tier misconfigured to a remote
    # endpoint it must fail loud (like vlm_read), not silently default routing.
    from fusion_ocr.stages.language import _probe_script

    class _Remote:
        def read(self, png, prompt):
            raise AirgapError("remote endpoint refused")

    with pytest.raises(AirgapError):
        _probe_script(_Remote(), b"png")


def test_language_probe_swallows_ordinary_errors():
    # A genuine probe hiccup still degrades to "" (unknown script) — only AirgapError
    # is escalated.
    from fusion_ocr.stages.language import _probe_script

    class _Broken:
        def read(self, png, prompt):
            raise RuntimeError("model hiccup")

    assert _probe_script(_Broken(), b"png") == ""
