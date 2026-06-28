"""Image -> PDF ingest adapter: magic sniffing, conversion (incl. multi-page TIFF),
identity passthrough for PDFs, and the watcher wiring. Image conversion needs PyMuPDF
(the ocr extra), so those tests skip where it's absent."""

from __future__ import annotations

import pytest

from fusion_ocr import ingest


# ---- sniffing: pure, no deps -------------------------------------------------

def test_sniff_format_by_magic_bytes():
    assert ingest.sniff_format(b"%PDF-1.7\n...") == "pdf"
    assert ingest.sniff_format(b"\x89PNG\r\n\x1a\n....") == "png"
    assert ingest.sniff_format(b"\xff\xd8\xff\xe0JFIF") == "jpeg"
    assert ingest.sniff_format(b"II*\x00....") == "tiff"      # little-endian
    assert ingest.sniff_format(b"MM\x00*....") == "tiff"      # big-endian
    assert ingest.sniff_format(b"just some text") is None
    assert ingest.sniff_format(b"") is None


def test_to_pdf_unsupported_raises(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_bytes(b"plain text, not a document")
    with pytest.raises(ValueError):
        ingest.to_pdf(bad, tmp_path / "derived")


def test_to_pdf_passes_a_pdf_through_unchanged(tmp_path):
    pdf = tmp_path / "real.pdf"
    pdf.write_bytes(b"%PDF-1.4\nhello\n%%EOF")
    out, converted = ingest.to_pdf(pdf, tmp_path / "derived")
    assert out == pdf and converted is False     # identity — no derived copy made


# ---- conversion: needs PyMuPDF ----------------------------------------------

fitz = pytest.importorskip("fitz", reason="image conversion needs PyMuPDF (ocr extra)")


def _image(path, size=(240, 320), fmt_ext="png"):
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, *size))
    pix.clear_with(255)
    p = path.with_suffix(f".{fmt_ext}")
    pix.save(str(p))
    return p


def test_image_to_pdf_png_is_a_scanned_one_page_pdf(tmp_path):
    png = _image(tmp_path / "scan")
    out, converted = ingest.to_pdf(png, tmp_path / "derived")
    assert converted is True and out.exists()
    with fitz.open(out) as d:
        assert d.page_count == 1
        assert d[0].get_text().strip() == ""     # pure image -> no text layer -> OCR path
        assert d[0].get_pixmap().width > 0


def test_image_to_pdf_jpeg(tmp_path):
    jpg = _image(tmp_path / "scan", fmt_ext="jpg")
    out = ingest.image_to_pdf(jpg, tmp_path / "out.pdf")
    with fitz.open(out) as d:
        assert d.page_count == 1


def test_multipage_tiff_becomes_a_multipage_pdf(tmp_path):
    Image = pytest.importorskip("PIL.Image", reason="building a multi-frame TIFF needs Pillow")
    frames = [Image.new("RGB", (240, 320), c) for c in ("white", "white", "white")]
    tif = tmp_path / "multi.tiff"
    frames[0].save(tif, save_all=True, append_images=frames[1:])
    out = ingest.image_to_pdf(tif, tmp_path / "multi.pdf")
    with fitz.open(out) as d:
        assert d.page_count == 3                  # one PDF page per TIFF frame


def test_watcher_ingests_an_image_keyed_by_the_originals_hash(tmp_path, monkeypatch):
    from fusion_ocr import config as config_mod
    from fusion_ocr import watcher as watcher_mod
    from fusion_ocr.jobs import JobStore
    from fusion_ocr.models import Document
    from fusion_ocr.pipeline import sha256_of

    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    cfg = config_mod.Config(in_dir=in_dir, out_dir=out_dir, airgap=False)
    png = _image(in_dir / "scan")
    original_digest = sha256_of(png)

    seen = {}

    def fake_process(pdf, c, **kw):
        seen["pdf"] = str(pdf)
        seen["digest"] = kw.get("digest")
        return Document(source_path=str(pdf), sha256=kw.get("digest", ""))

    monkeypatch.setattr(watcher_mod, "process", fake_process)
    jobs = JobStore(out_dir / "jobs.sqlite")
    assert watcher_mod.scan_once(cfg, jobs, min_settle=0.0) == 1

    # keyed by the ORIGINAL image's hash; processed a derived source.pdf under out/<digest>/
    assert seen["digest"] == original_digest
    assert seen["pdf"].endswith(f"{original_digest}/source.pdf".replace("/", __import__("os").sep))
    assert (out_dir / original_digest / "source.pdf").exists()
    assert jobs.get(original_digest)["status"] == "done"
