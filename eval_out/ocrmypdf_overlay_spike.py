"""Spike: can OCRmyPDF v17's engine-agnostic OcrElement API render fusion-ocr's
segment data (text + boxes, English AND Thai) as a searchable invisible text layer?

Findings + the migration guard: Docs/dev_notes/ocrmypdf_v17_overlay_spike.md. Verified
PASS against ocrmypdf 17.10.0 on 2026-08-19.

ocrmypdf is deliberately NOT a project dependency (yet) — run this in a scratch venv:
    python3 -m venv /tmp/spike && /tmp/spike/bin/pip install ocrmypdf pymupdf
    /tmp/spike/bin/python eval_out/ocrmypdf_overlay_spike.py

Method: draw a fake 'scan' (image-only PDF page with an English line and a Thai line at
KNOWN pixel positions), hand those (text, bbox) pairs to Fpdf2PdfRenderer as an OcrElement
tree — the same data fusion-ocr's segments carry — and verify with PyMuPDF that both
strings are searchable AND the hit boxes land where the ink was drawn.
"""
from pathlib import Path

import fitz

from ocrmypdf.font import MultiFontManager
from ocrmypdf.fpdf_renderer import Fpdf2PdfRenderer
from ocrmypdf.hocrtransform import BoundingBox, OcrClass, OcrElement

HERE = Path(__file__).parent
DPI = 150
PAGE_W_PX, PAGE_H_PX = int(8.27 * DPI), int(11.69 * DPI)   # A4 at 150 dpi

EN_TEXT, EN_BOX = "confidential memorandum", (150, 300, 900, 350)      # px
TH_TEXT, TH_BOX = "บริษัท", (150, 500, 500, 560)  # บริษัท


def make_scan(path: Path):
    """An image-only page with the two strings drawn as ink (rendered then rasterised)."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W_PX * 72 / DPI, height=PAGE_H_PX * 72 / DPI)
    s = 72 / DPI
    page.insert_text((EN_BOX[0] * s, EN_BOX[3] * s), EN_TEXT, fontsize=28)
    font = fitz.Font("notos") if "notos" in fitz.fitz_fontdescriptors else None
    try:
        page.insert_text((TH_BOX[0] * s, TH_BOX[3] * s), TH_TEXT, fontsize=28,
                         fontname="arialuni",
                         fontfile="/Library/Fonts/Arial Unicode.ttf")
    except Exception:
        page.insert_text((TH_BOX[0] * s, TH_BOX[3] * s), TH_TEXT, fontsize=28)
    pix = page.get_pixmap(dpi=DPI)
    img = HERE / "page.png"
    pix.save(img)
    scan = fitz.open()
    p2 = scan.new_page(width=page.rect.width, height=page.rect.height)
    p2.insert_image(p2.rect, filename=str(img))
    scan.save(path)
    return img


def word_elems(text, box, lang):
    x0, y0, x1, y1 = box
    words = text.split()
    out, step = [], (x1 - x0) / max(len(words), 1)
    for i, w in enumerate(words):
        out.append(OcrElement(
            ocr_class=OcrClass.WORD, text=w, confidence=95.0, language=lang,
            bbox=BoundingBox(x0 + i * step, y0, x0 + (i + 1) * step, y1)))
    return OcrElement(ocr_class=OcrClass.LINE, text="", language=lang,
                      bbox=BoundingBox(*box), children=out)


def main():
    scan = HERE / "scan.pdf"
    img = make_scan(scan)

    page_elem = OcrElement(
        ocr_class=OcrClass.PAGE, text="", dpi=DPI,
        bbox=BoundingBox(0, 0, PAGE_W_PX, PAGE_H_PX),
        children=[word_elems(EN_TEXT, EN_BOX, "eng"),
                  word_elems(TH_TEXT, TH_BOX, "tha")])

    mfm = MultiFontManager()
    out = HERE / "sandwich.pdf"
    Fpdf2PdfRenderer(page_elem, DPI, mfm, invisible_text=True, image=img).render(out)

    d = fitz.open(out)
    pg = d[0]
    print(f"page size: {pg.rect}")
    ok = True
    for label, text, box in (("EN", "confidential", EN_BOX), ("TH", TH_TEXT, TH_BOX)):
        hits = pg.search_for(text)
        exp_y = (box[1] * 72 / DPI, box[3] * 72 / DPI)
        if not hits:
            print(f"{label} '{text}': NOT SEARCHABLE"); ok = False; continue
        hit = hits[0]
        y_ok = abs(((hit.y0 + hit.y1) / 2) - (exp_y[0] + exp_y[1]) / 2) < 15
        print(f"{label} '{text}': {len(hits)} hit(s) at y {hit.y0:.0f}-{hit.y1:.0f} "
              f"(expected ~{exp_y[0]:.0f}-{exp_y[1]:.0f}) -> "
              f"{'PLACED OK' if y_ok else 'MISPLACED'}")
        ok &= y_ok
    ext = pg.get_text().strip().replace("\n", " | ")
    print(f"extracted text: {ext!r}")
    print("SPIKE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
