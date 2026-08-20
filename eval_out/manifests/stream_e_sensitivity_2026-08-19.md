# Stream E — threshold sensitivity + escalate_below decision — 2026-08-19

**Runners:** `eval_out/stream_e_sensitivity.py` (E1/E2), `eval_out/stream_e_escalation.py`
(E3) — implementing the 2026-07-12 pins in `evidence_plan.md §E` verbatim. **Machine:**
Luke's laptop (noise floor zero, stream G — single runs bind; per-item rows in
`eval_out/stream_e_sensitivity/results.csv`). **Data:** all 349 archived stream-A
pre-fusion snapshots, copied from the CORSAIR archive to the repo 2026-08-19 (path
relocation only). **Integrity check (tripwire a): PASS** — on 16 probed items (8 per
corpus) the re-scored ungated reading reproduces the committed `stream_a_vlm/results.csv`
to 4dp AND baseline re-fusion reproduces the archived strict placement counts exactly. The
copy is the committed run; fusion.py is drift-free against it. **Projection:** 1.6s/item →
full n=349, no fallback sampling. **Scope deviation (pre-registered):** `_MR_COVERAGE` /
`_LARGE_IMAGE_FRAC` not swept — flat by construction on these corpora (see the pins,
"Deviation 1"); re-trigger = first mixed-content gold corpus.

## E1 — the sweep (gated product text + placement, micro-avg; `*` = shared baseline)

### `fuse_min_sim` → **mild** (max |Δ| 0.0060; bars: <0.005 benign, ≥0.02 load-bearing)

| value | ds | n | recall | precision | band | strict |
| --- | --- | --- | --- | --- | --- | --- |
| 0.238 | funsd | 199 | 0.8163 | 0.8067 | 0.5576 | 0.4266 |
| 0.238 | sroie | 150 | 0.9432 | 0.8807 | 0.9186 | 0.5804 |
| 0.289 | funsd | 199 | 0.8140 | 0.8116 | 0.5621 | 0.4295 |
| 0.289 | sroie | 150 | 0.9430 | 0.8874 | 0.9218 | 0.5839 |
| 0.34* | funsd | 199 | 0.8137 | 0.8162 | 0.5637 | 0.4312 |
| 0.34* | sroie | 150 | 0.9419 | 0.8996 | 0.9241 | 0.5856 |
| 0.391 | funsd | 199 | 0.8139 | 0.8197 | 0.5657 | 0.4330 |
| 0.391 | sroie | 150 | 0.9411 | 0.9027 | 0.9254 | 0.5864 |
| 0.442 | funsd | 199 | 0.8129 | 0.8231 | 0.5679 | 0.4343 |
| 0.442 | sroie | 150 | 0.9397 | 0.9093 | 0.9252 | 0.5864 |

Monotone, tiny: raising it trades ≤0.4 recall points for ≤0.7 precision/placement points
across the whole ±30% range. 0.34 is not knife-edge. **Verdict: document, stop.**

### `fuse_det_conf_trust` → pinned-rule label **load-bearing** (max |Δ| 0.0264) — but read the shape

| value | ds | n | recall | precision | band | strict |
| --- | --- | --- | --- | --- | --- | --- |
| 0.56 | funsd | 199 | 0.8134 | 0.8166 | 0.5637 | 0.4313 |
| 0.56 | sroie | 150 | 0.9418 | 0.8982 | 0.9241 | 0.5856 |
| 0.68 | funsd | 199 | 0.8134 | 0.8163 | 0.5637 | 0.4313 |
| 0.68 | sroie | 150 | 0.9419 | 0.8982 | 0.9241 | 0.5856 |
| 0.80* | funsd | 199 | 0.8137 | 0.8162 | 0.5637 | 0.4312 |
| 0.80* | sroie | 150 | 0.9419 | 0.8996 | 0.9241 | 0.5856 |
| 0.92 | funsd | 199 | 0.8138 | 0.8163 | 0.5635 | 0.4310 |
| 0.92 | sroie | 150 | 0.9425 | 0.8976 | 0.9241 | 0.5856 |
| 1.04 (guard-OFF) | funsd | 199 | 0.8230 | 0.7990 | 0.5476 | 0.4188 |
| 1.04 (guard-OFF) | sroie | 150 | 0.9479 | 0.8542 | 0.8977 | 0.5620 |

**Tripwire (b) FIRED — at the guard-off endpoint only** (recall +0.009/+0.006 vs band
−0.016/−0.026, opposite directions at material magnitude, both corpora). Per the pins the
verdict below went to senior eyes as a proposed diagnosis (the D1 precedent) and is
**CERTIFIED (Luke, 2026-08-20): the diagnosis is accepted — keep 0.34/0.80, no
calibration pass.**

> **Diagnosis:** the entire "load-bearing" signal is the 1.04 endpoint, which the pins
> pre-labelled guard-off (no real confidence reaches 1.04 → the misalignment refusal never
> fires). In the operating range 0.56–0.92 every metric is flat to ≤0.0014 — sweeping the
> *value* does nothing, because pages where the aligned VLM line is dissimilar AND det_conf
> lands between 0.56 and 0.92 are rare. What the endpoint measures is the **gate's
> existence**: with it off, dissimilar VLM lines get stamped onto confident boxes — a few
> more true words survive into the gated text (recall up ~0.6–0.9pt) sitting on ink that
> doesn't match them (band placement down 1.6–2.6pt, precision down 1.7–4.5pt). That is
> the exact regression signature stream C pre-registered ("improves recall/CER but drops
> placement = regression"), produced deliberately. **First direct measurement of the
> anti-misalignment gate's value.** Proposed disposition: keep 0.80; classify the constant
> "benign in range — its value is existence, not tuning"; no calibration pass.

Both constants were already config-exposed, settings-registered, and in
`recipe_fingerprint` (the pins' correction) — nothing to fold in.

## E2 — fresh-default confirmation (Qwen3.6, seeded n=20 FUNSD, full pipeline + sweep)

20 fresh full-pipeline reads under the current default (`Qwen3.6-35B-A3B-4bit`, MLX,
laptop), then the fusion-only sweep of the same 8 non-baseline configs on the fresh
`doc.06` snapshots:

| constant | value | recall | precision | band | strict |
| --- | --- | --- | --- | --- | --- |
| fuse_min_sim | 0.238 | 0.8112 | 0.8072 | 0.5099 | 0.3881 |
| fuse_min_sim | 0.289 | 0.8071 | 0.8086 | 0.5107 | 0.3875 |
| (baseline) | 0.34* | 0.8071 | 0.8086 | 0.5107 | 0.3875 |
| fuse_min_sim | 0.391 | 0.8051 | 0.8080 | 0.5151 | 0.3883 |
| fuse_min_sim | 0.442 | 0.7967 | 0.8091 | 0.5167 | 0.3881 |
| fuse_det_conf_trust | 0.56–0.92 | 0.8065–0.8071 | 0.8085–0.8086 | 0.5101–0.5107 | 0.3870–0.3875 |
| fuse_det_conf_trust | 1.04 (guard-OFF) | 0.8322 | 0.7884 | 0.5011 | 0.3812 |

**The pinned check PASSES: direction of effect agrees with E1 for both constants on every
metric** (min_sim up → recall down / precision up / band up; conf_trust flat in range;
guard-off → recall up / precision down / band down). Tripwire (c) clear — the E1
conclusions attach to the current default reader.

Tripwire (b) also fired twice at n=20 (min_sim=0.442: recall −0.0104 vs band +0.0060;
guard-off: recall +0.0252 vs band −0.0096). **Diagnosis — CERTIFIED with E1's (Luke,
2026-08-20):** these are the same phenomenon seen from both ends — the fusion gate is a
recall↔placement **dial**, so *any* material movement along it shows the opposite-direction
pattern by construction. The C-registered "regression" framing targets unintended code
changes, not a deliberate sweep of the gate itself. In E1 (n=349) the 0.442 trade sat
below the materiality bar (−0.0008 / +0.0042); n=20 magnitudes are noisier, as the pins
anticipated ("magnitude may differ"). No action proposed; baseline 0.34/0.80 holds.

## E3 — `escalate_below` rescue analysis → **KEEP-DISABLED** (unconditional under the pinned rule)

From stream F's committed CSV (both models × the same 55 items; rescue = other model
≥ +0.05 word_recall on a worst-decile item): **q36's worst decile rescued by q35 = 0;
q35's worst decile rescued by q36 = 3** (92039708_9710 0.44→0.58, 0001463282 0.47→0.66,
87672097 0.63→0.68). Full table: `eval_out/stream_e_sensitivity/e3_rescue_table.md`.

Decile condition (<2 in both directions) NOT met → the pinned outcome is
**keep-disabled + document**; refusal-rescue reads are moot for the rule. Reading: model
diversity demonstrably rescues (a stronger model recovers half of a weaker model's worst
decile), so escalation pays **once a stronger-than-default tier exists** — today none does
locally (Qwen3.6 is the top; stream F). The re-add trigger remains the in-VPC vLLM tier.
Threshold calibration = follow-up with senior eyes, per the rule. The keep/delete lands on
its own PR — this manifest is the evidence table; no config was changed.

## Evidence-plan bookkeeping

With E executed, every stream of the campaign (A, B, C, D, E, F, G) has run. End-state
items from `evidence_plan.md`: the four top thresholds have sensitivity curves (two swept
here; two documented-not-sweepable with a re-trigger); `escalate_below` is
evaluated-with-verdict (keep-disabled, evidence above). Both tripwire-(b) diagnoses
(E1 guard-off endpoint; E2 dial framing) are **CERTIFIED (Luke, 2026-08-20)** — nothing
outstanding; the campaign's end state is met in full.
