# Run manifest — Evidence-plan stream D2, blank / near-blank probes

*[evidence_plan.md](../../Docs/dev_notes/evidence_plan.md) stream D2 — the ink-gate's one hard
pass/fail: on blank/near-blank input the searchable overlay invents **zero** words. Twelve
synthetic single-page "scans" drawn deterministically with PyMuPDF primitives (the SCRIPT is
committed, not the PDFs), each rasterised to an image-only page and run through the FULL pipeline
with the live MLX reader (Qwen3.5-9B-MLX-4bit), fresh out_dir, distinct digest per probe. Runner
`eval_out/blank_probes.py`.*

## Result — PASS (gated invented words = 0 on every probe)

| probe | ref | VLM invoked | det segs | ungated invented | **gated invented** |
| --- | --- | --- | --- | --- | --- |
| blank_white / offwhite / lightgrey | "" | no | 0 | 0 | **0** |
| speckle_lo / mid / hi | "" | no | 0 | 0 | **0** |
| stamp_circle / stamp_solidbox | "" | no | 0 | 0 | **0** |
| **stamp_copy** | "COPY" | **yes** (12.7s) | 1 | 0 | **0** (recovered "COPY") |
| ruled_grid / lined / formboxes | "" | no | 0 | 0 | **0** |

- **Tripwire (a) does NOT fire:** 0 invented words in the gated overlay on all 12 probes. The
  ink-gate's core claim holds. `ungated` invention is **also 0** on all 12.
- **The COPY positive control passes:** a genuine inked word on an otherwise blank page is
  detected, the VLM is invoked, reads "COPY", and the overlay carries exactly that one word with
  nothing invented — recall without hallucination.

## Mechanism — and an honest limit of the synthetic set

**11 of 12 probes produced zero detector segments**, so they were gated *upstream* (the
detector/blank-gate found no ink) and the VLM was **never invoked** — neither the overlay nor the
reading view could receive anything. Only the COPY probe reached the VLM, and there the reader was
correct.

Two consequences worth stating plainly:

- **Post-#21 the blank-gate protects the *reading view* too.** On these blanks the ungated
  `document.md` is also clean (0 invented) — the VLM never sees a blank page, so it cannot invent
  the classic blank-page formula (the pre-guard OCR-Quality idx-924 mode). This confirms the
  2026-07-06 guard behaviour on fresh synthetic blanks.
- **The synthetics did NOT drive the VLM to hallucinate on a near-blank**, because the detector
  filtered the faint speckle, rule lines and text-free stamps (0 segments) and the one mark that
  tripped detection (COPY) was a real word. So the probes robustly demonstrate (i) the detector
  manufactures no text from blank/speckle/rule/stamp marks, and (ii) the blank-gate shields both
  views on blanks — but they do **not** independently exercise the ink-gate's distinctive
  "drop VLM text that has no ink under it" path, because nothing reached that path. The real-data
  companion for that path is the OCR-Quality blank/loop set (idx 924/967/969/654), verified in
  situ 2026-07-06 (`manifests/ocrq_full_2026-07-06.md`) — cited, not re-run.

## Ties to D1 (regime)

D1 pointed here for the ink-gate's *insertion benefit* (it doesn't appear on ink-full FUNSD/SROIE).
D2 refines that: in the **current** pipeline the blank regime is handled by the **blank-gate**, so
on blanks the ungated and gated views are *both* clean and the gated-vs-ungated benefit is ~0
(both 0), not "ungated dirty / gated clean". The ink-gate's marginal value *over* the blank-gate
needs a page that (a) trips detection yet (b) has the VLM invent beyond the ink — rare on clean or
blank inputs, and not produced by D1's corpora or D2's synthetics. Its clearest empirical
demonstration remains the pre-guard OCRQ-924 instance. **This is the same open interpretive thread
as D1's tripwire (b) and is flagged for Luke's certification**, not asserted here.

## What this settles / doesn't

- **Settles:** the searchable overlay does not invent text on blank/near-blank pages (0/12); the
  detector does not manufacture text from speckle/rules/stamps; the blank-gate protects the reading
  view as well; a real inked word is recovered cleanly (COPY).
- **Doesn't settle:** the ink-gate's benefit *net of* the blank-gate is not isolated by this set
  (nothing reached the VLM-invents-past-ink path). A harder probe — a degraded page that trips
  detection but carries no real text — would isolate it; noted as a follow-up, not a blocker.

## Artifacts

- `eval_out/blank_probes/results.csv` — per-probe VLM-invoked / segs / invented (ungated+gated) + texts
- `eval_out/blank_probes/summary.json` — totals + pass flag
- `eval_out/blank_probes.py` — the runner (deterministic PDF generation; regenerate the PDFs from it)
