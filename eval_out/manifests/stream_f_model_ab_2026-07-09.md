# Run manifest — Evidence-plan stream F, model & quant A/Bs

*[evidence_plan.md](../../Docs/dev_notes/evidence_plan.md) stream F — the model/runtime deltas we
shipped without measuring, now measured at n≥50 against the **zero** noise floor (stream G). Runner
`eval_out/stream_f_model_ab.py`; both eval sets are the hand-labelled 5 hard pages (incl. the
Mandelson handwriting) + a seeded **FUNSD n=50** (seed=7, the campaign's FUNSD convention). Every
model sees the identical items; scores the ungated reading (`recovered_text`). **The keep/switch
calls are surfaced for Luke — this run replaces the anecdotes, it does not flip any default.***

## Results (micro-avg; medCER + runaway = outlier-robust; mean t_vlm)

| model | set | n | recall | prec | CER | **medCER** | ins | **run** | t_vlm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B-4bit *(default)* | all | 55 | 0.8457 | 0.8813 | 0.1397 | 0.1097 | 0.0541 | 0 | 19.2s |
| Qwen3-VL-8B-4bit *(rollback)* | all | 55 | 0.8439 | 0.8859 | ~~4.887~~ | 0.1260 | ~~4.789~~ | **1** | 23.4s |
| Qwen3.5-9B-8bit *(quant)* | all | 55 | 0.8515 | 0.8810 | 0.1484 | 0.1085 | 0.0617 | 0 | 23.6s |
| **Qwen3.6-35B-A3B-4bit** *(new-gen)* | all | 55 | **0.8641** | **0.8932** | **0.1246** | **0.0974** | 0.0557 | 0 | **13.8s** |

(FUNSD-50 / label-5 breakdowns are in `results.csv`. Qwen3-VL-8B's micro CER/ins of ~4.9 is a
**single-outlier artifact** — see below; its median is 0.126, in line with the others.)

## Comparison 1 — model generation (default vs rollback): keep Qwen3.5-9B

At n=55 the recall gap that motivated the 2026-06-30 switch (Δ0.005/n=4) is a **genuine tie**
(recall 0.846 vs 0.844; medCER actually favours Qwen3.5-9B by 0.016). The defensible reason to keep
Qwen3.5-9B is **robustness + speed, not accuracy**: Qwen3-VL-8B fell into a **bare-`.` repetition
loop on 1/50 FUNSD items** (`01191071_1072` → 262,144 chars, recall 0, 114s) that the current
`[illegible]`-tuned repetition guard did not catch; Qwen3.5-9B had zero such failures and is ~4s
faster. *Recommendation: keep Qwen3.5-9B over Qwen3-VL-8B.*

## Comparison 2 — quantisation (4-bit vs 8-bit Qwen3.5-9B): keep 4-bit

8-bit buys a **real but marginal** quality edge — recall +0.006, medCER a tie (+0.001), a little
better on the hard labelled pages — at a **~23% latency cost (23.6s vs 19.2s) and 2× memory**
(10.5 GB vs ~5 GB resident). Per the pre-registration the delta is "real" (any delta beats the zero
floor), but it is not worth the speed/memory cost for the dozen-docs/day on-device MVP.
*Recommendation: keep 4-bit.*

## Comparison 3 — new generation (Qwen3.6-35B-A3B): a compelling upgrade candidate

**Qwen3.6-35B-A3B wins on every axis at once** — the first model in this project to do so:

| vs the default Qwen3.5-9B-4bit | recall | medCER | t_vlm | runaway |
| --- | --- | --- | --- | --- |
| Δ (all) | **+0.018** (0.864 vs 0.846) | **−0.012 better** (0.097 vs 0.110) | **−5.4s faster** (13.8 vs 19.2) | 0 vs 0 |
| Δ (FUNSD) | +0.019 | −0.009 better | −5.3s faster | 0 |
| Δ (labelled) | +0.016 (0.980 vs 0.964) | −0.020 better | −6.3s faster | 0 |

It is a **256-expert / 8-active MoE** (`model_type: qwen3_5_moe`; ~3B active of 35B total), which is
why a nominally larger model is **~28% faster** than the 9B dense default while reading **more**
accurately (recall 0.980 on the Mandelson handwriting). Runs on mlx_vlm 0.6.3 unchanged. The cost is
**~20 GB resident memory** (vs ~5 GB) — comfortable on the 64 GB M1 Max, but material for a smaller
deployment target. *Recommendation: strong candidate for the generalist default — but validate
broader before switching in production (n=55 on 2 corpora here; the Thai/Typhoon routing, table
reads, and the full document mix are untested with it), and confirm the 20 GB footprint fits the
deployment. This is Luke's call.*

## Product finding — generalise the repetition guard

The Qwen3-VL-8B `.`-loop slipped through because the degenerate-repetition guard keys on
`[illegible]`-style loops, not arbitrary low-entropy floods. Any reader can hit this; the guard
should reject **any** run whose output is a single character / short cycle repeated past a
threshold (and a `max_tokens` cap would bound the 114s cost). Suggested as a roadmap hardening item.

## Operational note (data hygiene — the honest record)

The Qwen3.6 arm was re-run three times before landing clean, and the manifest numbers are from the
**final clean single run** (55 rows, 0 duplicates, all 55 verified real reads). What went wrong and
was corrected: (a) the MLX server OOM-crashed when a standalone 20 GB model-load smoke-test was run
**concurrently** with the live server + an active arm (operator error — the fix is to smoke-test via
the server, never a concurrent standalone load); (b) a runner launched with `python … &` (no
`nohup`) survived as an **orphan** and ran concurrently with the relaunch, producing duplicate rows
and concurrent-write-corrupted `doc.json` files. Both were caught by the first-item reality checks
(empty reading / `t_vlm ≈ 2s` = dead-server fallback) and the duplicate/real-read integrity checks,
then fully reset. The `medCER` + `runaway` columns added to the runner this session are the standing
guard so no single outlier (or bad row) can silently drive a conclusion.

## What this settles / doesn't

- **Settles:** the two shipped-without-measuring deltas — keep Qwen3.5-9B over Qwen3-VL-8B (tie
  quality, but a rare catastrophic loop + slower), keep 4-bit over 8-bit (marginal gain, real cost).
  Both replace anecdotes with n=55 numbers on a zero noise floor.
- **Opens:** Qwen3.6-35B-A3B is measurably better (quality **and** speed) — a real default-upgrade
  candidate pending broader validation. The quant arm's original blocker (no 8-bit cached) is
  resolved (pulled).
- **Doesn't settle:** n=55 on FUNSD+labelled only; single machine / MLX build / temp 0.0. No
  production switch is made here.

## Artifacts

- `eval_out/stream_f_model_ab/results.csv` — per-item scores for all 4 models (recall/prec/CER/WER/
  insertion + raw counts + t_vlm)
- `eval_out/stream_f_model_ab.py` — the runner (4 arms; resumable; medCER + runaway summary)
