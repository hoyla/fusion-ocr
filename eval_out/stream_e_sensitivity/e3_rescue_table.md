# Stream E3 — escalate_below rescue analysis

Universe: the 55 items scored by BOTH `q36_35ba3b_4bit` and `q35_9b_4bit` in stream F
(`eval_out/stream_f_model_ab/results.csv`). Rescue = the other model scores >= +0.05 word_recall on an
item in this model's worst decile. Rule + interpretation are pinned in
evidence_plan.md §E3 (2026-07-12) — registered before these numbers were computed.

## Worst decile of `q36_35ba3b_4bit` — rescues by `q35_9b_4bit`: **0**

| item | q36_35ba3b_4bit | q35_9b_4bit | rescued |
| --- | --- | --- | --- |
| funsd/0060270727 | 0.5581 | 0.5581 | no |
| funsd/86328049_8050 | 0.5628 | 0.5578 | no |
| funsd/92039708_9710 | 0.5844 | 0.4416 | no |
| funsd/0012529284 | 0.6239 | 0.6239 | no |
| funsd/0060214859 | 0.6391 | 0.6391 | no |
| funsd/0001463282 | 0.6560 | 0.4725 | no |

## Worst decile of `q35_9b_4bit` — rescues by `q36_35ba3b_4bit`: **3**

| item | q35_9b_4bit | q36_35ba3b_4bit | rescued |
| --- | --- | --- | --- |
| funsd/92039708_9710 | 0.4416 | 0.5844 | YES |
| funsd/0001463282 | 0.4725 | 0.6560 | YES |
| funsd/86328049_8050 | 0.5578 | 0.5628 | no |
| funsd/0060270727 | 0.5581 | 0.5581 | no |
| funsd/0012529284 | 0.6239 | 0.6239 | no |
| funsd/87672097 | 0.6262 | 0.6845 | YES |

## Rule evaluation

- decile rescues: q36_35ba3b_4bit worst-decile rescued-by-q35_9b_4bit = 0; q35_9b_4bit worst-decile rescued-by-q36_35ba3b_4bit = 3
- decile condition (`< 2 in BOTH directions`): NOT met

**Outcome (unconditional): KEEP-DISABLED + document** — decile rescues >= 2 in at least one direction; threshold calibration becomes a follow-up (senior eyes). Refusal reads are moot for the rule.
