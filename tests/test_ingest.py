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


# ---- review 03 ingest robustness: WebP/HEIC sniffing, page-size cap, EXIF, encrypted PDFs

def test_sniff_webp_and_heic_by_magic():
    assert ingest.sniff_format(b"RIFF\x10\x00\x00\x00WEBPVP8 ") == "webp"
    assert ingest.sniff_format(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00") == "heic"
    assert ingest.sniff_format(b"\x00\x00\x00\x1cftypmif1\x00\x00\x00\x00") == "heic"   # generic HEIF brand
    assert ingest.sniff_format(b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00") is None     # AVIF: not accepted
    assert ingest.sniff_format(b"RIFF\x10\x00\x00\x00WAVEfmt ") is None                 # RIFF but not WebP


def test_page_size_honours_dpi_and_caps_the_long_edge():
    w, h = ingest.page_size_pt(2480, 3508, (300, 300))
    assert (round(w), round(h)) == (595, 842)                        # a 300-DPI scan is A4
    assert ingest.page_size_pt(1000, 1300, None) == (750.0, 975.0)   # no DPI -> 96, as MuPDF did
    assert ingest.page_size_pt(1000, 1300, (0, 0)) == (750.0, 975.0)  # placeholder DPI -> same
    w, h = ingest.page_size_pt(5477, 3651, None)                     # a 20-MP photo
    assert round(w) == 1224 and round(h) == 816                      # long edge capped at 17 in
    assert abs(w / h - 5477 / 3651) < 1e-6                           # aspect preserved


def test_huge_photo_becomes_a_bounded_page_with_full_pixels(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    big = tmp_path / "photo.png"
    Image.new("RGB", (3000, 2000), "white").save(big)                 # no DPI metadata
    out = ingest.image_to_pdf(big, tmp_path / "photo.pdf")
    with fitz.open(out) as d:
        assert round(d[0].rect.width) == 1224 and round(d[0].rect.height) == 816
        xref = d[0].get_images(full=True)[0][0]
        assert d.extract_image(xref)["width"] == 3000                # pixels embedded untouched


def test_webp_converts_to_a_one_page_pdf(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from PIL import features
    if not features.check("webp"):
        pytest.skip("Pillow built without WebP")
    webp = tmp_path / "scan.webp"
    Image.new("RGB", (240, 320), "white").save(webp)
    assert ingest.peek(webp) == "webp"
    out, converted = ingest.to_pdf(webp, tmp_path / "derived")
    with fitz.open(out) as d:
        assert converted and d.page_count == 1 and d[0].get_text().strip() == ""


def test_heic_without_the_codec_is_a_clear_job_error(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pillow_heif", None)            # `import pillow_heif` fails
    heic = tmp_path / "IMG_0001.heic"
    heic.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00" + b"\x00" * 64)
    with pytest.raises(ingest.IngestError, match="heic"):
        ingest.to_pdf(heic, tmp_path / "derived")


def test_exif_rotated_jpeg_lands_upright_and_plain_jpeg_is_embedded_verbatim(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    rotated = tmp_path / "rot.jpg"
    im = Image.new("RGB", (400, 200), "white")
    ex = im.getexif()
    ex[0x0112] = 6                                                   # camera held portrait
    im.save(rotated, exif=ex.tobytes())
    with fitz.open(ingest.image_to_pdf(rotated, tmp_path / "rot.pdf")) as d:
        assert d[0].rect.width < d[0].rect.height                    # page shows it upright
    plain = tmp_path / "plain.jpg"
    Image.new("RGB", (400, 200), "white").save(plain, quality=90)
    with fitz.open(ingest.image_to_pdf(plain, tmp_path / "plain.pdf")) as d:
        xref = d[0].get_images(full=True)[0][0]
        assert d.extract_image(xref)["ext"] == "jpeg"                # original bytes, no re-encode


def test_encrypted_pdf_fails_fast_with_a_clear_error(tmp_path):
    from fusion_ocr import config as config_mod
    from fusion_ocr.models import Document
    from fusion_ocr.stages.triage import Triage
    locked = tmp_path / "locked.pdf"
    d = fitz.open(); d.new_page().insert_text((72, 72), "secret"); d.save(
        str(locked), encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="hunter2", owner_pw="hunter2"); d.close()
    assert "password" in ingest.readability_problem(locked)
    assert ingest.to_pdf(locked, tmp_path / "derived") == (locked, False)   # identity still
    with pytest.raises(ingest.IngestError, match="password-protected"):        # the job fails in words
        Triage().run(Document(source_path=str(locked), sha256="x"), config_mod.Config())
    ok = tmp_path / "open.pdf"
    d = fitz.open(); d.new_page(); d.save(str(ok)); d.close()
    assert ingest.readability_problem(ok) is None
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4\nnot really\n%%EOF")
    assert "cannot be opened" in ingest.readability_problem(corrupt)
    with pytest.raises(ingest.IngestError, match="cannot be opened"):
        Triage().run(Document(source_path=str(corrupt), sha256="x"), config_mod.Config())
