# Spike: OCRmyPDF v17's `OcrElement` API as the overlay-assembly target (2026-08-19)

**Question.** Review 03's standing criticism is wheel-reinvention at the *implementation*
layer, and the 2026-08 landscape re-check flagged a new adoption candidate: OCRmyPDF 17
(2026) added an **engine-agnostic text-layer API** — plugins hand it a typed `OcrElement`
tree (pages/lines/words + boxes + language + confidence) instead of fabricating hOCR XML.
Could that replace the bespoke `overlay/` writer (PyMuPDF invisible text), per
build-vs-adopt?

**Method** (`eval_out/ocrmypdf_overlay_spike.py`, run against ocrmypdf **17.10.0** in a
scratch venv — it is deliberately NOT a project dependency yet): draw a fake "scan" (an
image-only page with an English line and a Thai line — `บริษัท` — at known pixel
positions), feed exactly the data our fused segments carry (text + bbox + language) to
`Fpdf2PdfRenderer` as an `OcrElement` tree, and verify with PyMuPDF that both strings are
searchable and the hits land on the drawn ink.

## Findings — all four checks pass

1. **The API exists and fits our data shape exactly.** `ocrmypdf.hocrtransform.OcrElement`
   is a dataclass tree (`ocr_class`, `bbox`/`poly`, `text`, `confidence`, `language`,
   `direction`, `textangle`, `font`, `dpi`, children) — a superset of what fusion emits.
   No hOCR XML anywhere.
2. **Search + placement are correct.** `confidential` and `บริษัท` are each found once, hit
   boxes within ~1–2 pt of the drawn ink (word-level: their renderer places each word).
3. **Invisible mode verified at the content-stream level:** the output carries `3 Tr`
   (invisible render mode) text operators only, no visible-text ops.
4. **The Thai-font problem is handled *better* than our writer.**
   `ocrmypdf.font.MultiFontManager` does per-word font selection by **glyph-coverage
   analysis over installed system fonts** with graceful fallback: with no Thai-capable font
   in its chain it still emitted a *searchable, copyable* invisible layer and printed a
   loud, actionable warning naming the missing glyphs and the font to install. Our
   original failure mode (base-14 helv → Thai silently unsearchable, a real 2026-06-26
   bug) is structurally impossible here. This is exactly the "hardened against the long
   tail by a maintained library" argument from [principles.md](../principles.md).

## What a migration would (and would not) touch

- **Adopt:** `OcrElement` + `Fpdf2PdfRenderer` + `MultiFontManager` for text-layer
  *emission* — the hard part (fonts, CID mapping, per-word placement, RTL/rotation fields
  we currently don't attempt). These are the documented public plugin surface of v17, not
  internals.
- **Keep ours:** the graft onto the original PDF (PyMuPDF overlay merge) and everything
  upstream — fusion, provenance, the segment index. OCRmyPDF's graft machinery is a
  private module (`_graft`); principles.md says adopt the *public* surface.
- **New dependency:** `ocrmypdf` (plus its `fpdf2`/pikepdf chain) behind the existing
  overlay extra; pin in constraints. No Tesseract/Ghostscript needed for this path.

## Guards before merging any migration (the committed bar)

- The **searchability eval** must reproduce on the real corpus — especially the Thai terms
  (หนังสือบริคณห์สนธิ / บริษัท / กระทรวงพาณิชย์, the 2026-06-26 regression set) and the
  **rotated Goldfinch page** (searchable recall 0.65 — must not drop; `OcrElement` carries
  `textangle`, so rotation may even improve).
- **Band placement** (stream C metric) unchanged or better on FUNSD/SROIE.
- NFC normalisation parity (our writer NFC-normalises before insertion — verify theirs or
  keep normalising upstream).

*Blocked today on the samples corpus being on this machine (same blocker as stream E1);
the spike itself is complete and repeatable. Adoption decision = Luke's, on the PR that
carries the guard numbers.*
