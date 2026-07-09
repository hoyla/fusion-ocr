"""Evidence-plan stream D3 — divergence triage (the qualitative anchor for the D1 numbers).

Universe = the same 349 archived stream-A VLM docs (NO VLM compute). Per item, disagreement is
the word-multiset F1 between the normalized `vlm_reading` and the concatenation of non-superseded
segments' `det_text`:  F1 = 2·overlap / (|vlm_words| + |det_words|)  (SROIE casefolded both sides,
`datasets._CASELESS_REF`), overlap as in `metrics.score`.

Candidates (pinned — "both confident, but they disagree"):
  F1 < 0.5  AND  mean det_conf over non-superseded segments >= 0.80  (the codebase's own det-trust
  bar, fuse_det_conf_trust)  AND  both sides non-trivial (normalized vlm_reading >= 20 chars AND
  >= 3 non-superseded segments carrying det_text).

Sample: random.seed(1); sample(candidates, min(20, len(candidates))). If fewer than 20 qualify,
take all and record the count — NO threshold-relaxing top-up (that would unpin the rule).

These corpora have real gold, so verdicts are gold-anchored: per item this runner emits each
side's score() vs gold (recall/CER), so "which side is right" starts from the objective number;
Claude-Vision inspects the image to classify VLM-wrong / det-wrong / both-wrong / reference-fault
(keep the 4th bucket). Luke certifies the final table. Verdicts land in
`eval_out/divergence_triage/verdicts.md`; this runner writes the machine table + a scaffold.

D3 needs NO reader.  Run:  .venv/bin/python eval_out/divergence_triage.py
"""
import csv
import json
import random
from collections import Counter
from pathlib import Path

from fusion_ocr.eval.datasets import _CASELESS_REF, iter_pairs
from fusion_ocr.eval.metrics import normalize, score, word_tokens
from fusion_ocr.models import Document

ARCH = Path("/Volumes/CORSAIR/Work_Corsair/fusion-ocr/eval_out/stream_a_vlm")
OUT = ARCH / "out"
SRC_CSV = ARCH / "results.csv"
RES = Path("eval_out/divergence_triage"); RES.mkdir(parents=True, exist_ok=True)

DET_TRUST = 0.80   # fuse_det_conf_trust
F1_MAX = 0.5
MIN_VLM_CHARS = 20
MIN_DET_SEGS = 3

# ref + image maps by (dataset, stem)
refmap, imgmap = {}, {}
for ds in ("funsd", "sroie"):
    for split in ("train", "test", "val"):
        for img, ref in iter_pairs(ds, split=split):
            refmap[(ds, img.stem)] = ref
            imgmap[(ds, img.stem)] = img

items = [(r["dataset"], r["id"]) for r in csv.DictReader(SRC_CSV.open())]


def det_concat(page_segs):
    return " ".join((s.det_text or "") for s in page_segs)


rows = []
for ds, rid in items:
    doc = Document.from_json((OUT / f"{ds}_{rid}" / "doc.json").read_text())
    caseless = ds in _CASELESS_REF
    fold = str.casefold if caseless else (lambda x: x)

    vlm = "\n".join(p.vlm_reading for p in doc.pages)
    live = [s for p in doc.pages for s in p.segments if not s.superseded]
    det_segs = [s for s in live if (s.det_text or "").strip()]
    det = det_concat(det_segs)

    vlm_w = [fold(w) for w in word_tokens(vlm)]
    det_w = [fold(w) for w in word_tokens(det)]
    overlap = sum((Counter(vlm_w) & Counter(det_w)).values())
    denom = len(vlm_w) + len(det_w)
    f1 = (2 * overlap / denom) if denom else 0.0

    confs = [s.det_conf for s in live if s.det_conf is not None]
    mean_conf = sum(confs) / len(confs) if confs else 0.0

    vlm_norm = normalize(vlm)
    candidate = (f1 < F1_MAX and mean_conf >= DET_TRUST
                 and len(vlm_norm) >= MIN_VLM_CHARS and len(det_segs) >= MIN_DET_SEGS)

    rows.append(dict(dataset=ds, id=rid, f1=round(f1, 4), mean_det_conf=round(mean_conf, 4),
                     vlm_chars=len(vlm_norm), n_det_segs=len(det_segs),
                     candidate=int(candidate)))

# durable full table (every item's F1, so the selection is reproducible/auditable)
with (RES / "f1_all.csv").open("w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

candidates = [r for r in rows if r["candidate"]]
random.seed(1)
sample = random.sample(candidates, min(20, len(candidates)))
sample.sort(key=lambda r: r["f1"])   # most-divergent first for inspection

print(f"universe={len(rows)}  candidates(F1<{F1_MAX} & conf>={DET_TRUST} & non-trivial)="
      f"{len(candidates)}  sampled={len(sample)}  (no top-up; seed=1)")
from collections import Counter as C
print("candidates by dataset:", dict(C(r["dataset"] for r in candidates)))

# per-sample gold anchoring: each side's score vs gold
detail = []
for r in sample:
    ds, rid = r["dataset"], r["id"]
    doc = Document.from_json((OUT / f"{ds}_{rid}" / "doc.json").read_text())
    caseless = ds in _CASELESS_REF
    ref = refmap[(ds, rid)]
    vlm = "\n".join(p.vlm_reading for p in doc.pages)
    live = [s for p in doc.pages for s in p.segments if not s.superseded]
    det = " ".join((s.det_text or "") for s in live if (s.det_text or "").strip())
    sv = score(ref, vlm, caseless=caseless)
    sd = score(ref, det, caseless=caseless)
    detail.append(dict(
        dataset=ds, id=rid, f1=r["f1"], mean_det_conf=r["mean_det_conf"],
        img=str(imgmap[(ds, rid)]),
        vlm_recall=round(sv["word_recall"], 4), vlm_cer=round(sv["cer"], 4),
        det_recall=round(sd["word_recall"], 4), det_cer=round(sd["cer"], 4),
        gold_favors=("vlm" if sv["word_recall"] > sd["word_recall"] else
                     "det" if sd["word_recall"] > sv["word_recall"] else "tie"),
        vlm_reading=vlm, det_text=det, gold=ref))

(RES / "sample.json").write_text(json.dumps(detail, indent=2, ensure_ascii=False))

with (RES / "triage_table.csv").open("w", newline="") as fh:
    cols = ["dataset", "id", "f1", "mean_det_conf", "vlm_recall", "vlm_cer",
            "det_recall", "det_cer", "gold_favors", "img"]
    wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    wr.writeheader(); wr.writerows(detail)

print("\ngold_favors tally (objective anchor, pre-Vision):",
      dict(C(d["gold_favors"] for d in detail)))
print("per-item (most divergent first):")
for d in detail:
    print(f"  {d['dataset']}/{d['id']} F1={d['f1']} conf={d['mean_det_conf']} | "
          f"VLM r={d['vlm_recall']}/cer={d['vlm_cer']}  DET r={d['det_recall']}/cer={d['det_cer']}"
          f"  -> gold favors {d['gold_favors'].upper()}")
print("\nwrote sample.json (full text for Vision) + triage_table.csv + f1_all.csv")
