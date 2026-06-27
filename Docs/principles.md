# Build vs adopt — orchestrate trusted tools, build only the connective tissue

OCR, layout analysis, table recognition and reading order are exhaustively researched
and maturely implemented. Any document-processing problem we hit has almost certainly
been solved — and hardened against the long tail of real documents — by experts. So the
**default is to adopt**, not to reinvent. This file is the standing rule, learned the
hard way (we hand-rolled an XY-cut reading order that PaddleOCR already shipped, and
coasted on an old layout-model default that cost us reading-order quality).

## The rule

> Orchestrate every *solved component*. Build *only* the connective tissue and the
> journalism-specific guarantees no component provides — and keep that bespoke surface as
> small as possible.

**Build only if** one of these holds (and say which, in the commit):

1. **No component provides it.** The VLM↔box fusion, the anti-hallucination ink-gate,
   per-region routing, mixed-content composition, provenance, the airgap guard — there is
   no off-the-shelf "OCR pipeline a newsroom lawyer can defend." That glue is ours.
2. **A hard constraint rules out the mature option.** On-estate / airgap rules out the
   most mature OCR (AWS Textract, Google Document AI, Azure). That constraint is *why*
   this project exists; it doesn't license reinventing the on-prem components.
3. **The integration genuinely costs more than a tiny, better-tested-for-our-case piece.**
   Rare. Must be justified, kept small, and **measured** against the eval.

Otherwise: adopt.

## Adopt on the *public* surface

"Maintained by many, used by millions" describes a library's **public API**, not its
internal helpers. A private symbol (`pkg.internal.impl._helper`) carries the *fragility*
of a dependency without the *stability* of one — it can move or rename on any release with
no deprecation. Prefer the public predictor / pipeline / model API. If only an internal
exists and it's equivalent to what we have, that's the weakest case for swapping; reach
instead for the public, more-capable surface (e.g. a maintained *model*, not a buried
function). Updating a dependency when it improves is routine; a silent bespoke bug on a
layout we never tested is not.

## Determinism is for the defensibility backbone, not every step

We value deterministic, explainable behaviour — but spend it where it matters. **Geometry,
the ink-gate, and provenance stay deterministic** (every claim must back to a box and a
source). **Reading order, recognition, table structure** can come from learned models:
they are *sequencing and recognition*, not provenance claims — a segment still backs to
its deterministic box regardless of what ordered it. A model that's right beats an
explainable rule that's wrong on an untested page.

## Audit the defaults, periodically

Calling `LayoutDetection()` / `TableStructureRecognition()` with no model name gives you
whatever was the default when the wrapper was written — not the latest. Periodically check
each model we instantiate against what's available and what we're actually downloading
(`~/.paddlex/official_models/`), and move to the current generation deliberately. (This is
how we found we were on PP-DocLayout_plus-L instead of PP-DocLayoutV2, and SLANet instead
of SLANeXt.)

## Measure before committing either way

We have an eval (`fusion_ocr.eval`) precisely so build-vs-adopt is decided by numbers, not
by hunch or by attachment to code we wrote. Where a clean metric isn't available (e.g.
table-structure quality has no ground truth in our corpus), say so, and adopt the
maintained current tool on the strength of its provenance rather than pretending to a
measurement we don't have.
