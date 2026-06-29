"""Input-format ingest adapter — normalise a supported input to a PDF, after which the
existing PDF-centric pipeline runs unchanged (PDF is the identity case).

The ORIGINAL is the canonical source: it is never mutated, and for a non-PDF input the PDF
is a DERIVED, provenanced artifact (principle 3 — never mutate source material). The job is
keyed by the hash of the *original* input, so re-dropping the same image is idempotent and
the artifacts trace back to what the user actually supplied.

Scope today: PDF (identity) + raster images (PNG / JPEG / TIFF, including multi-page TIFF).
PyMuPDF opens an image as an N-page document and `convert_to_pdf` embeds it at full source
resolution — multi-page TIFF becomes a multi-page PDF — so the image flows straight through
the scanned-page path. Office docs (.docx/.xlsx/.pptx via LibreOffice) are a separate, heavier
`office` extra and are intentionally not handled here yet (roadmap, Later → Input formats).
"""

from __future__ import annotations

from pathlib import Path

# Accept by MAGIC BYTES, not by extension — the same posture as the API's PDF gate, so a
# mislabelled or extension-less file is still classified by what it actually is.
_MAGICS: dict[str, tuple[bytes, ...]] = {
    "pdf": (b"%PDF-",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpeg": (b"\xff\xd8\xff",),
    "tiff": (b"II*\x00", b"MM\x00*"),   # little-endian / big-endian TIFF
}
IMAGE_FORMATS = ("png", "jpeg", "tiff")


def sniff_format(head: bytes) -> str | None:
    """The format of a file from its first bytes, or None if it's not one we ingest."""
    for fmt, sigs in _MAGICS.items():
        if any(head.startswith(sig) for sig in sigs):
            return fmt
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


def image_to_pdf(src, dst) -> Path:
    """Convert a raster image (PNG/JPEG/TIFF, incl. multi-page TIFF) to a PDF at `dst`,
    embedding each frame at full resolution — one PDF page per frame. The source is left
    untouched. Returns `dst`."""
    import fitz  # PyMuPDF (ocr extra); deferred so importing this module stays dep-free

    src, dst = Path(src), Path(dst)
    with fitz.open(src) as img:            # PyMuPDF reads an image as an N-page document
        pdf_bytes = img.convert_to_pdf()   # full-res embed; multi-frame TIFF -> multi-page PDF
    dst.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open("pdf", pdf_bytes) as pdf:
        pdf.save(str(dst))
    return dst


def to_pdf(src, derived_dir) -> tuple[Path, bool]:
    """Normalise `src` to a PDF the pipeline can process.

    Returns (pdf_path, converted): a PDF passes through unchanged (identity; converted=False);
    a supported image is converted to `<derived_dir>/<stem>.pdf` (converted=True). Raises
    ValueError on an unsupported format. The original `src` is always left intact.
    """
    src = Path(src)
    fmt = peek(src)
    if fmt is None:
        raise ValueError(f"unsupported input format: {src.name}")
    if fmt == "pdf":
        return src, False
    dst = Path(derived_dir) / f"{src.stem}.pdf"
    return image_to_pdf(src, dst), True
