"""API upload-path safety (extra: api). Tests the pure sanitizers — importing the
module is side-effect-free now (app is lazy), so no server/config/sqlite is built."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="needs the api extra")

from fusion_ocr.api import _is_sha256, _safe_name  # noqa: E402


def test_safe_name_strips_directory_traversal():
    assert _safe_name("../../etc/passwd") == "passwd"      # the path-traversal payload
    assert _safe_name("/abs/path/x.pdf") == "x.pdf"
    assert _safe_name("sub/dir/report.pdf") == "report.pdf"
    assert _safe_name("normal.pdf") == "normal.pdf"


def test_safe_name_handles_empty_and_dot_segments():
    assert _safe_name(None) == "upload.pdf"
    assert _safe_name("") == "upload.pdf"
    assert _safe_name("..") == "upload.pdf"
    assert _safe_name(".") == "upload.pdf"


def test_is_sha256_rejects_path_components():
    assert _is_sha256("a" * 64)
    assert _is_sha256("0123456789abcdef" * 4)
    assert not _is_sha256("../../etc/passwd")   # the GET /jobs/{sha256} traversal payload
    assert not _is_sha256("a" * 63)             # wrong length
    assert not _is_sha256("g" * 64)             # non-hex
    assert not _is_sha256("")
