# Output artifacts — what's in `out/<sha256>/`

Every job writes its results to a **content-addressed** folder under `out/`:

```
out/<sha256>/              # <sha256> = SHA-256 of the original input file
├── document.md            ┐
├── overlay.pdf            ├─ deliverables — what consumers use
├── segment_index.json     ┘
├── doc.json               — final machine-readable Document state
├── doc.00-triage.json     ┐
│   …                      ├─ per-stage resume snapshots (interim)
├── doc.09-render.json     ┘
└── source.pdf             — image inputs only (the derived PDF)
```

The folder name is the **content hash of the input**, so the same file always lands in the
same place and a re-run is idempotent. That makes the folder opaque to a human, though — to
tell which job is which, the original filename is recorded *inside*: `doc.json` →
`source_path` (and in `document.md`'s provenance header). *(A `sha → original filename`
manifest is a [roadmap](dev_notes/roadmap.md) item.)*

## Deliverables — what you consume

These three are the product. Most consumers (Giant, a reporter) only ever touch these.

| File | What it is |
| --- | --- |
| **`document.md`** | The **reading view** — the clean, structured transcript: the VLM reading with tables as markdown and a provenance header. *Ungated*: it carries the full reading even where a line couldn't be anchored to a box. This is what a human reads. |
| **`overlay.pdf`** | The page with an **invisible, searchable, highlightable text layer** placed on the detected boxes (PyMuPDF render mode 3). Open it in any PDF viewer: text is selectable, `find` works, and a hit highlights on the right line. *Gated* — only ink-backed text, so the deterministic side keeps the VLM from inventing geography. |
| **`segment_index.json`** | The **provenance / drill-back record** — every segment with its box, `det_text`, `vlm_text`, the chosen `best_text`, `source`, and `read_by`. Powers "show me this line *in situ*" and lets any claim be traced back to the exact box, page, and engine/model that read it. |

> Searchability note: a **born-digital** page whose own text layer already covers it gets
> **no** `overlay.pdf` of its own — that existing text layer *is* the searchable content
> (adding an overlay would double search hits). See [eval-labelling.md](eval-labelling.md).

## Internal — resumable state

| File | What it is |
| --- | --- |
| **`doc.json`** | The **final `Document` state** — the complete structured result: every page, segment and region, the recipe, plus page-level `vlm_reading` and box coordinates. Identical to the last stage snapshot (`doc.09-render.json`). Reach for this when you want the full machine-readable state rather than the three deliverables. |
| **`doc.NN-<stage>.json`** | The **per-stage resume cache** — the full `Document` *after* each pipeline stage (`00`–`09`). Keyed on the content hash **and** a recipe fingerprint (pipeline + models + routes + prompt text + output flags), so a re-run after changing a prompt or model **reprocesses** instead of silently returning a stale result; `--rerun-from <stage>` reuses the earlier stages (e.g. retune the VLM prompt without redoing OCR). They accumulate — nothing is cleaned. |
| **`source.pdf`** | **Image inputs only** (PNG / JPEG / TIFF). The derived, provenanced PDF the pipeline actually ran on — images are normalised to PDF on ingest (PDF is the identity case, so PDF inputs have no `source.pdf`). The folder is keyed by the *original image's* hash; the original stays the canonical reference. |

### Pipeline stage order (what builds on what)

Each snapshot is the `Document` state *after* that stage; every stage takes the previous
`Document` and returns an enriched one:

```
00 triage → 01 layout → 02 table → 03 language → 04 ocr_det
   → 05 vlm_read → 06 table_read → 07 fusion → 08 table_fill → 09 render
```

Geometry is established deterministically (`layout` / `table` / `ocr_det`), the VLM reads
(`vlm_read` / `table_read`), **`fusion`** distributes that reading onto the detected boxes,
and **`render`** emits the three deliverables above. The *why* is in
[motivation_and_strategy.md](dev_notes/motivation_and_strategy.md).

## The queue (a sibling, not a per-job artifact)

`out/jobs.sqlite` is the shared job queue / status store (`JobStore`) the API and the worker
both use — `POST /jobs` enqueues into it, `GET /jobs/{sha256}` polls it. It lives beside the
job folders, not inside one. See [configuration.md](configuration.md).
