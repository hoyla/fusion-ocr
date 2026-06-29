"""Per-process page-raster cache.

Several stages rasterise the same page at the same DPI independently — layout, table and
vlm_read all render full pages at 150 DPI — so a scanned page is rendered three times over.
This caches the rendered Pixmap keyed by (path, mtime, page_index, dpi, clip) and hands it
to every consumer: the ndarray path (OCR / layout / table) and the PNG path (VLM read /
script probe). It also centralises the pixmap->ndarray conversion that those three stages
had each copy-pasted.

A fitz Pixmap is self-contained — its `samples` survive the source document closing — so
it's safe to keep after the `with fitz.open(...)` block that produced it has exited.

Eviction is LRU under a total-bytes budget: bounded memory regardless of page count, and a
document too large to fit just re-renders the evicted pages (no worse than before). The lock
keeps it correct once process() runs off the event loop (Tier-3 #2)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

_BUDGET_BYTES = 256 * 1024 * 1024     # cap; pixmaps are large, so bound by bytes not count
_lock = threading.Lock()
_cache: OrderedDict = OrderedDict()   # key -> fitz.Pixmap, most-recently-used last
_bytes = 0


def _mtime_ns(path: str) -> int:
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return 0


def _pix_bytes(pix) -> int:
    return pix.width * pix.height * pix.n


def page_pixmap(pdf, page_index: int, dpi: int, clip=None):
    """Cached fitz Pixmap for (this document's path, page, dpi, clip). `pdf` is an open fitz
    document, used only to render on a miss — a cache hit ignores it. `clip` is a hashable
    bbox tuple (x0, y0, x1, y1) or None for the full page."""
    global _bytes
    key = (pdf.name, _mtime_ns(pdf.name), page_index, dpi, clip)
    with _lock:
        pix = _cache.get(key)
        if pix is not None:
            _cache.move_to_end(key)
            return pix

    import fitz  # render outside the lock — PDF rasterisation is the slow part
    kwargs = {"dpi": dpi}
    if clip is not None:
        kwargs["clip"] = fitz.Rect(*clip)
    pix = pdf[page_index].get_pixmap(**kwargs)
    size = _pix_bytes(pix)

    with _lock:
        if key not in _cache:                 # a racing miss may have inserted it already
            _cache[key] = pix
            _bytes += size
            while _bytes > _BUDGET_BYTES and len(_cache) > 1:
                _, evicted = _cache.popitem(last=False)
                _bytes -= _pix_bytes(evicted)
    return pix


def page_png(pdf, page_index: int, dpi: int, clip=None) -> bytes:
    """PNG bytes for the page (or a clipped region) — lossless; used where exactness matters."""
    return page_pixmap(pdf, page_index, dpi, clip=clip).tobytes("png")


def page_jpeg(pdf, page_index: int, dpi: int, clip=None, quality: int = 85) -> bytes:
    """JPEG bytes for the page (or a clipped region) — the shape the VLM client wants. JPEG
    over PNG keeps the base64 payload small (a 150-DPI page is multi-MB; base64 adds ~33%),
    which matters on the wire every page on the remote-reader / in-VPC path. JPEG can't carry
    an alpha channel, so an alpha pixmap is flattened to RGB first."""
    import fitz

    pix = page_pixmap(pdf, page_index, dpi, clip=clip)
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)   # drop alpha; JPEG is opaque
    return pix.tobytes("jpeg", jpg_quality=quality)


def page_ndarray(pdf, page_index: int, dpi: int):
    """RGB, 3-channel, C-contiguous uint8 — the shape PaddleOCR / layout / table expect."""
    import numpy as np

    pix = page_pixmap(pdf, page_index, dpi)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:        # drop alpha
        arr = arr[:, :, :3]
    elif pix.n == 1:      # grey -> 3-channel
        arr = np.repeat(arr, 3, axis=2)
    return np.ascontiguousarray(arr)


def clear() -> None:
    """Drop all cached rasters (between documents, to reclaim memory, or in tests)."""
    global _bytes
    with _lock:
        _cache.clear()
        _bytes = 0
