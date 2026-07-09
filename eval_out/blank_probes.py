"""Evidence-plan stream D2 — blank / near-blank probes (the ink-gate's core claim).

Twelve synthetic single-page "scans" generated deterministically here with PyMuPDF drawing
primitives (the SCRIPT is committed, not the PDFs), each rasterised to an image-only page so the
pipeline must OCR it (no text layer to shortcut). Run through the FULL pipeline with the live MLX
reader, fresh out_dir, a distinct digest per probe (the 2026-07-06 guard-check method):

  3 blank    (white / off-white / light-grey)
  3 speckle  (faint grey dots, seeded RNG, increasing density)     <- the near-blanks: the real test
  2 stamp    (circular rubber-stamp outline; solid dark box) — text-free
  1 COPY     (circular stamp + the word "COPY")   ref = "COPY"
  3 ruled    (table grid / lined page / empty form boxes) — text-free

Ref = "" for all but the COPY probe.

Metric unit is WORDS (pinned): invented words = len(word_tokens(hyp)) - word_overlap, for BOTH
  - the UNGATED reading (`recovered_text`, == vlm_reading when VLM-read), and
  - the GATED text (non-superseded best_text in reading order — D1's definition = the overlay).
Char insertion_rate is logged too but degenerates to a raw count when ref is "".

PASS/FAIL (the ONLY hard bar in stream D): GATED invented words = 0 on EVERY probe. Any nonzero is
an ink-gate bug (tripwire a) — STOP and diagnose. Ungated counts are reported with no bar (they
quantify the reading view's exposure — the thing the gate protects the overlay from).

The four real OCR-Quality probes (blank 924/967/969, loop 654) are the companion set — already
verified in situ 2026-07-06 (`manifests/ocrq_full_2026-07-06.md`); cited there, NOT re-run.

Needs MLX on :8080.  Run:  .venv/bin/python eval_out/blank_probes.py
"""
import csv
import json
import random
import time
from datetime import date
from pathlib import Path

import fitz

from fusion_ocr import config as cm
from fusion_ocr.compose import reading_key
from fusion_ocr.eval.harness import recovered_text
from fusion_ocr.eval.metrics import score, word_tokens
from fusion_ocr.pipeline import process

RES = Path("eval_out/blank_probes"); RES.mkdir(parents=True, exist_ok=True)
PDFS = RES / "pdfs"; PDFS.mkdir(exist_ok=True)   # gitignored — regenerable from this script
W, H = 612.0, 792.0   # US Letter points
DPI = 150

WHITE = (1, 1, 1)


def _rasterise(page_doc, dst: Path):
    """Flatten a drawn page to an image-only PDF (no text layer) — a faithful 'scan'."""
    pix = page_doc[0].get_pixmap(dpi=DPI)
    out = fitz.open()
    w, h = pix.width * 72.0 / DPI, pix.height * 72.0 / DPI
    pg = out.new_page(width=w, height=h)
    pg.insert_image(pg.rect, pixmap=pix)
    out.save(str(dst)); out.close()


def _blank(dst, shade):
    d = fitz.open(); p = d.new_page(width=W, height=H)
    p.draw_rect(fitz.Rect(0, 0, W, H), color=shade, fill=shade)
    _rasterise(d, dst); d.close()


def _speckle(dst, n, seed):
    d = fitz.open(); p = d.new_page(width=W, height=H)
    p.draw_rect(fitz.Rect(0, 0, W, H), color=WHITE, fill=WHITE)
    rng = random.Random(seed)
    for _ in range(n):
        x, y = rng.uniform(20, W - 20), rng.uniform(20, H - 20)
        r = rng.uniform(0.6, 1.4)
        g = rng.uniform(0.70, 0.82)                 # faint grey
        p.draw_circle((x, y), r, color=(g, g, g), fill=(g, g, g))
    _rasterise(d, dst); d.close()


def _stamp_circle(dst):
    d = fitz.open(); p = d.new_page(width=W, height=H)
    p.draw_rect(fitz.Rect(0, 0, W, H), color=WHITE, fill=WHITE)
    c = (W / 2, H / 2)
    p.draw_circle(c, 90, color=(0.15, 0.15, 0.15), width=3)
    p.draw_circle(c, 78, color=(0.15, 0.15, 0.15), width=1.5)
    _rasterise(d, dst); d.close()


def _solid_box(dst):
    d = fitz.open(); p = d.new_page(width=W, height=H)
    p.draw_rect(fitz.Rect(0, 0, W, H), color=WHITE, fill=WHITE)
    p.draw_rect(fitz.Rect(W / 2 - 80, H / 2 - 40, W / 2 + 80, H / 2 + 40),
                color=(0.12, 0.12, 0.12), fill=(0.12, 0.12, 0.12))
    _rasterise(d, dst); d.close()


def _copy_stamp(dst):
    d = fitz.open(); p = d.new_page(width=W, height=H)
    p.draw_rect(fitz.Rect(0, 0, W, H), color=WHITE, fill=WHITE)
    c = (W / 2, H / 2)
    p.draw_circle(c, 95, color=(0.6, 0.1, 0.1), width=3)
    p.insert_text((W / 2 - 62, H / 2 + 12), "COPY", fontsize=44,
                  color=(0.6, 0.1, 0.1), fontname="helv")
    _rasterise(d, dst); d.close()


def _grid(dst):
    d = fitz.open(); p = d.new_page(width=W, height=H)
    p.draw_rect(fitz.Rect(0, 0, W, H), color=WHITE, fill=WHITE)
    for i in range(6):
        x = 80 + i * 90
        p.draw_line((x, 100), (x, 700), color=(0.5, 0.5, 0.5), width=1)
    for j in range(8):
        y = 100 + j * 85
        p.draw_line((80, y), (530, y), color=(0.5, 0.5, 0.5), width=1)
    _rasterise(d, dst); d.close()


def _lined(dst):
    d = fitz.open(); p = d.new_page(width=W, height=H)
    p.draw_rect(fitz.Rect(0, 0, W, H), color=WHITE, fill=WHITE)
    for j in range(22):
        y = 90 + j * 30
        p.draw_line((70, y), (542, y), color=(0.55, 0.55, 0.7), width=0.8)
    _rasterise(d, dst); d.close()


def _form_boxes(dst):
    d = fitz.open(); p = d.new_page(width=W, height=H)
    p.draw_rect(fitz.Rect(0, 0, W, H), color=WHITE, fill=WHITE)
    for j in range(6):
        y = 120 + j * 90
        p.draw_rect(fitz.Rect(80, y, 300, y + 40), color=(0.4, 0.4, 0.4), width=1)
        p.draw_line((330, y + 38), (540, y + 38), color=(0.4, 0.4, 0.4), width=1)
    _rasterise(d, dst); d.close()


# (name, ref, builder) — 12 probes, deterministic
PROBES = [
    ("blank_white", "", lambda p: _blank(p, (1, 1, 1))),
    ("blank_offwhite", "", lambda p: _blank(p, (0.98, 0.98, 0.96))),
    ("blank_lightgrey", "", lambda p: _blank(p, (0.92, 0.92, 0.92))),
    ("speckle_lo", "", lambda p: _speckle(p, 40, 101)),
    ("speckle_mid", "", lambda p: _speckle(p, 150, 202)),
    ("speckle_hi", "", lambda p: _speckle(p, 500, 303)),
    ("stamp_circle", "", _stamp_circle),
    ("stamp_solidbox", "", _solid_box),
    ("stamp_copy", "COPY", _copy_stamp),
    ("ruled_grid", "", _grid),
    ("ruled_lined", "", _lined),
    ("ruled_formboxes", "", _form_boxes),
]


def invented_words(ref, hyp):
    s = score(ref, hyp)
    return max(0, len(word_tokens(hyp)) - s["word_overlap"]), s


cfg = cm.load()
import dataclasses
cfg = dataclasses.replace(cfg, out_dir=RES / "out")

CSV = RES / "results.csv"
COLS = ["probe", "ref", "vlm_invoked", "n_segments", "t_vlm", "secs",
        "ungated_words", "ungated_invented", "ungated_ins_chars",
        "gated_words", "gated_invented", "gated_ins_chars",
        "ungated_text", "gated_text"]
fh = CSV.open("w", newline=""); w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()

results = []
for name, ref, build in PROBES:
    pdf = PDFS / f"{name}.pdf"
    build(pdf)
    t0 = time.time()
    doc = process(pdf, cfg, digest=f"probe_{name}")
    secs = time.time() - t0
    page = doc.pages[0]
    ung = "\n".join(recovered_text(p) for p in doc.pages)
    segs = [s for p in doc.pages for s in p.segments if s.best_text and not s.superseded]
    segs.sort(key=lambda s: reading_key(s, page.regions, page.rotation, page.width, page.height))
    gat = "\n".join(s.best_text for s in segs)

    u_inv, su = invented_words(ref, ung)
    g_inv, sg = invented_words(ref, gat)
    vlm_invoked = bool(page.read_model) or bool(page.vlm_reading.strip())
    t_vlm = round(doc.stage_seconds.get("vlm_read", 0.0), 2)
    n_seg = sum(len(p.segments) for p in doc.pages)

    row = dict(probe=name, ref=ref, vlm_invoked=int(vlm_invoked), n_segments=n_seg,
               t_vlm=t_vlm, secs=round(secs, 1),
               ungated_words=len(word_tokens(ung)), ungated_invented=u_inv,
               ungated_ins_chars=su["char_ins"],
               gated_words=len(word_tokens(gat)), gated_invented=g_inv,
               gated_ins_chars=sg["char_ins"],
               ungated_text=ung.replace("\n", " ⏎ ")[:300],
               gated_text=gat.replace("\n", " ⏎ ")[:300])
    w.writerow(row); fh.flush()
    results.append(row)
    flag = "" if g_inv == 0 else "  <<< GATED INVENTED != 0 (tripwire a)"
    print(f"{name:16} ref={ref!r:8} vlm={vlm_invoked} nseg={n_seg} t_vlm={t_vlm:5}s | "
          f"ungated: words={row['ungated_words']:3} inv={u_inv:3} | "
          f"gated: words={row['gated_words']:3} inv={g_inv:3}{flag}", flush=True)
fh.close()

failed = [r for r in results if r["gated_invented"] != 0]
copy_ok = next((r for r in results if r["probe"] == "stamp_copy"), None)
summary = {"date": str(date.today()), "n_probes": len(results),
           "gated_invented_total": sum(r["gated_invented"] for r in results),
           "ungated_invented_total": sum(r["ungated_invented"] for r in results),
           "failures": [r["probe"] for r in failed],
           "copy_recovered": bool(copy_ok and "copy" in (copy_ok["gated_text"].lower()))}
(RES / "summary.json").write_text(json.dumps(summary, indent=2))

print("\n=== D2 SUMMARY ===")
print(f"gated invented words TOTAL across 12 probes: {summary['gated_invented_total']} "
      f"(PASS bar = 0)")
print(f"ungated invented words TOTAL (reading-view exposure, no bar): "
      f"{summary['ungated_invented_total']}")
print(f"COPY recovered in gated overlay: {summary['copy_recovered']}")
if failed:
    print(f"\n!! TRIPWIRE (a) FIRED — gated invented words != 0 on: {summary['failures']}")
    print("   STOP and diagnose (ink-gate bug) before a manifest verdict.")
else:
    print("\nPASS: ink-gate held — 0 invented words in the gated overlay on every probe.")
print("DONE", flush=True)
