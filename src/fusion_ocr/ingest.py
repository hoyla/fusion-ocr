"""Input-format ingest adapter — normalise a supported input to a PDF, after which the
existing PDF-centric pipeline runs unchanged (PDF is the identity case).

The ORIGINAL is the canonical source: it is never mutated, and for a non-PDF input the PDF
is a DERIVED, provenanced artifact (principle 3 — never mutate source material). The job is
keyed by the hash of the *original* input, so re-dropping the same image is idempotent and
the artifacts trace back to what the user actually supplied.

Scope: PDF (identity) + raster images — PNG / JPEG / TIFF (incl. multi-page) / WebP / HEIC
(what an iPhone photo of a document actually is; needs the optional `heic` extra). Each frame
becomes one PDF page carrying the image at full resolution: a JPEG that needs no rotation is
embedded byte-for-byte, anything else (multi-frame TIFF, WebP, HEIC, an EXIF-rotated photo)
is re-encoded losslessly as PNG with the camera orientation applied. The PAGE SIZE honours the
image's DPI metadata (a 300-DPI scan is an A4 page), defaults to 96 DPI without it (MuPDF's
own default), and caps the long edge at MAX_PAGE_EDGE_IN: a 20-MP phone photo used to become
a ~60-inch page whose every raster in the pipeline was hundreds of MB (review 03). Office
docs (.docx/.xlsx/.pptx via LibreOffice) are a separate, heavier `office` extra and are
intentionally not handled here yet (roadmap, Later → Input formats).

Unreadable-but-recognised inputs (an encrypted PDF) raise IngestError with a message written
for the operator; the job records it verbatim (`GET /jobs/{sha}` → error) instead of failing
confusingly deep inside a stage.
"""

from __future__ import annotations

from pathlib import Path


class IngestError(ValueError):
    """An input we can identify but cannot process. The message is the job's error text —
    say what the operator should do about it."""


# Accept by MAGIC BYTES, not by extension — the same posture as the API's PDF gate, so a
# mislabelled or extension-less file is still classified by what it actually is.
_MAGICS: dict[str, tuple[bytes, ...]] = {
    "pdf": (b"%PDF-",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpeg": (b"\xff\xd8\xff",),
    "tiff": (b"II*\x00", b"MM\x00*"),   # little-endian / big-endian TIFF
}
# HEIF family brands (ISO BMFF `ftyp` box at offset 4): HEIC stills and the generic image
# brands iOS/Android write. AVIF is deliberately not accepted (not a document format we see).
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}
IMAGE_FORMATS = ("png", "jpeg", "tiff", "webp", "heic")

# Page-size policy for image inputs (see module docstring).
MAX_PAGE_EDGE_IN = 17.0     # tabloid/A3-class long edge; beyond it the page is scaled down
_DEFAULT_DPI = 96.0         # MuPDF's own assumption for an image without resolution metadata


def sniff_format(head: bytes) -> str | None:
    """The format of a file from its first bytes, or None if it's not one we ingest."""
    for fmt, sigs in _MAGICS.items():
        if any(head.startswith(sig) for sig in sigs):
            return fmt
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp" and head[8:12] in _HEIF_BRANDS:
        return "heic"
    return None


def peek(path) -> str | None:
    """Sniff `path` from disk; None if unreadable or an unsupported format."""
    try:
        with open(path, "rb") as f:
            return sniff_format(f.read(16))
    except OSError:
        return None


def is_supported(path) -> bool:
    return peek(path) is not None


def readability_problem(path) -> str | None:
    """Why a recognised input still can't be processed, in words for the operator — or
    None. Today: a PDF that needs a password to open (an owner-password-only PDF opens fine
    and is not flagged), or one PyMuPDF cannot parse at all (truncated / corrupt). Needs
    PyMuPDF; without it, None (the stage reports)."""
    if peek(path) != "pdf":
        return None
    try:
        import fitz  # PyMuPDF (ocr extra)
    except ImportError:
        return None
    try:
        with fitz.open(str(path)) as d:
            if d.needs_pass:
                return ("PDF is password-protected (encrypted): the pipeline never guesses "
                        "passwords — supply an unencrypted copy (e.g. `qpdf --decrypt "
                        "--password=… in.pdf out.pdf`)")
    except Exception as exc:  # noqa: BLE001 — PyMuPDF's own error is the diagnosis
        return f"PDF cannot be opened ({exc}) — the file may be truncated or corrupt"
    return None


def page_size_pt(px_w: int, px_h: int, dpi) -> tuple[float, float]:
    """PDF page size (points) for an image of px_w × px_h pixels: physical size from its
    DPI metadata (default 96), then scaled down uniformly so the long edge is at most
    MAX_PAGE_EDGE_IN. The pixels are embedded untouched either way — the cap only bounds
    the raster the pipeline renders from the page."""
    dx, dy = _dpi_pair(dpi)
    w_in, h_in = px_w / dx, px_h / dy
    over = max(w_in, h_in) / MAX_PAGE_EDGE_IN
    if over > 1.0:
        w_in, h_in = w_in / over, h_in / over
    return 72.0 * w_in, 72.0 * h_in


def _dpi_pair(dpi) -> tuple[float, float]:
    try:
        dx, dy = float(dpi[0]), float(dpi[1])
    except (TypeError, IndexError, ValueError):
        return _DEFAULT_DPI, _DEFAULT_DPI
    if dx <= 1.0 or dy <= 1.0:          # missing / placeholder resolution
        return _DEFAULT_DPI, _DEFAULT_DPI
    return dx, dy


def _register_heif() -> None:
    try:
        import pillow_heif
    except ImportError:
        raise IngestError("HEIC/HEIF input needs the optional `heic` extra (pillow-heif): "
                          "`pip install -e '.[heic]'` — or convert the photo to JPEG/PNG "
                          "first") from None
    pillow_heif.register_heif_opener()


def _png_bytes(im) -> bytes:
    import io
    if im.mode not in ("1", "L", "LA", "RGB", "RGBA", "P"):   # e.g. CMYK JPEG, I;16 TIFF
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")          # lossless: the derived page keeps every pixel
    return buf.getvalue()


def image_to_pdf(src, dst) -> Path:
    """Convert a raster image (PNG/JPEG/TIFF incl. multi-page/WebP/HEIC) to a PDF at `dst`,
    one page per frame at full resolution, page size per page_size_pt. The source is left
    untouched. Returns `dst`."""
    import fitz  # PyMuPDF (ocr extra); deferred so importing this module stays dep-free
    from PIL import Image, ImageOps, ImageSequence

    src, dst = Path(src), Path(dst)
    fmt = peek(src)
    if fmt == "heic":
        _register_heif()
    pdf = fitz.open()
    with Image.open(src) as im:
        n_frames = getattr(im, "n_frames", 1)
        for frame in ImageSequence.Iterator(im):
            orientation = frame.getexif().get(0x0112, 1)   # EXIF Orientation; 1 = upright
            fr = ImageOps.exif_transpose(frame) if orientation != 1 else frame
            w_pt, h_pt = page_size_pt(fr.width, fr.height,
                                      fr.info.get("dpi") or im.info.get("dpi"))
            page = pdf.new_page(width=w_pt, height=h_pt)
            if fmt == "jpeg" and n_frames == 1 and orientation == 1:
                page.insert_image(page.rect, filename=str(src))   # the JPEG bytes, verbatim
            else:
                page.insert_image(page.rect, stream=_png_bytes(fr))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(str(dst))
    pdf.close()
    return dst


def to_pdf(src, derived_dir) -> tuple[Path, bool]:
    """Normalise `src` to a PDF the pipeline can process.

    Returns (pdf_path, converted): a PDF passes through unchanged (identity; converted=False);
    a supported image is converted to `<derived_dir>/<stem>.pdf` (converted=True). Raises
    ValueError on an unsupported format (IngestError for a recognised image we can't
    decode, e.g. HEIC without its codec). The original `src` is always left intact.
    """
    src = Path(src)
    fmt = peek(src)
    if fmt is None:
        raise ValueError(f"unsupported input format: {src.name}")
    if fmt == "pdf":
        return src, False   # identity — readability is the first stage's check (triage)
    dst = Path(derived_dir) / f"{src.stem}.pdf"
    return image_to_pdf(src, dst), True
