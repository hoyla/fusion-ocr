# Roadmap

Forward-looking work, roughly by priority. Shipped capabilities are logged in
[done.md](done.md); the rationale behind the architecture is in
[motivation_and_strategy.md](motivation_and_strategy.md).

Ordering principle (principle 6, *look before infra*): the scale-triggered items below are
deliberately **not** built yet — the interfaces are shaped so they drop in when the load is
real, not before.

## Now / near-term

- **Hand-labelled eval set** for degraded scans + handwriting. The born-digital eval
  (~95% recall) is a *difficulty floor* — rendered-clean pages flatter the system; the hard
  cases need real labels to measure.
- **The "Giant rejects" eval** — old (tesseract / OCRmyPDF) vs new on the real reject corpus.
  This is the headline value claim, measured rather than asserted.
- **Word-level overlay subdivision** for precise click-to-highlight. Honest word boxes only
  from the Apple Vision per-word API / PyMuPDF `words` — *not* proportional splitting (that
  manufactures precision; principle: calibrate, don't manufacture).
- **Thai overlay search reliability** (combining vowels / tone marks, NFC vs NFD). Reading is
  solid; reliable highlight is the remaining gap.

## Next

- **Tables:** multi-level-header semantics for `find_tables`; cross-validate `find_tables` vs
  the vision grid; cleaner per-cell content on scanned tables.
- **Reading order:** a hand-labelled set to actually measure order (CER is reading-order-noisy
  on multi-column, so it isn't a reliable oracle today).
- **Qwen3.5-VL re-test** when its MLX build lands (was a statistical tie with Qwen3-VL-8B;
  revisit then).
- **Result push for non-airgap tiers:** an optional webhook / callback on completion. The
  sealed (airgap) tier stays poll-only by construction — the process can't dial out — so this
  is a tier-gated enhancement, never the default.

## Later — scale-triggered (don't build until load is real)

- **Distributed queue adapter** (ElasticMQ / SQS, on-estate) implementing the `JobStore`
  method surface — when workers span machines or need shared durable state.
- **Object-store adapter** (Garage on-estate, or S3-in-VPC for a less-sensitive tier) behind
  `storage.py` — when the local filesystem stops sufficing.
- **Worker options:** an in-process worker for single-box convenience; a multi-worker pool
  (the atomic claim already makes this safe).
- **API at scale:** rate limits beyond the upload cap; richer observability / metrics.

_Trigger conditions are stated so we don't over-build: the seams (`JobStore`, `storage.py`,
the OpenAI-compatible reader endpoint) keep these swaps config-deep, not rewrites._
