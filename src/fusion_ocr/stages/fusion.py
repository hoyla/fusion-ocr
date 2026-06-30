"""Stage 6 — Fusion + alignment.

Brings the deterministic geometry and the VLM reading together:

1. SPATIAL OVERLAP between exact text-layer segments and OCR segments —
   a clean text-layer segment (exact glyphs) beats an overlapping OCR box; an OCR
   box beats an overlapping CONTAMINATED text-layer span (the Thai PUA case).

2. LINE FUSION for OCR pages with a VLM reading:
   * cluster PaddleOCR's (often over-segmented) boxes into visual lines and merge
     each cluster to one box — this is the geometry;
   * distribute the VLM reading onto those line-boxes at the WORD level
     (`_word_distribute`) — this is the semantics. Even garbled handwriting keeps roughly
     the right words in roughly the right places, so fuzzy word alignment spreads a long
     prose line across the several visual lines it spans — which line-level alignment
     could not (the VLM returns prose; the detector finds many short lines, so one VLM
     line can't cover them all). It degrades honestly: too few anchors (a reading-order
     mismatch / wholesale recognition failure) falls back to the line-level Needleman-
     Wunsch baseline (`_nw_align`); unanchored edge clusters get nothing rather than
     smeared words; and a confidence gate keeps confident printed det_text when the VLM
     share barely resembles it.

`det_text` / `vlm_text` are always kept beside best_text for provenance. The clean, ungated
reading is `document.md` (the page's vlm_reading); the overlay + segment_index are the
box-anchored, gated record. Region-aware clustering keeps columns apart using the layout
stage's PP-DocLayoutV2 reading order; the VLM reading is already column-aware too.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from ..compose import classify_regions, in_machine_readable_region, reading_key
from ..config import Config
from ..models import Box, Document, Page, Segment

_IOU_OVERLAP = 0.5
_GAP = -0.2  # alignment gap penalty
_OCR_SOURCES = {"paddle", "vision"}  # deterministic OCR engines (geometry + det_text)
_MAX_DP_CELLS = 1_500_000  # word-level DP guard; above this fall back to the cheaper line-level
_ANCHOR_SIM = 0.5          # a det<->VLM word match at least this similar counts as an anchor
_MIN_ANCHOR_FRAC = 0.3     # too few anchored clusters -> alignment untrustworthy, fall back
# The anti-misalignment gate thresholds (fuse_min_sim / fuse_det_conf_trust) live on
# Config — surfaceable and tunable via the settings registry / API.


def _iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _overlaps(seg: Segment, others: list[Segment]) -> bool:
    return any(_iou(seg.box, o.box) >= _IOU_OVERLAP for o in others)


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _assign_region(seg: Segment, regions) -> int:
    """Index of the region whose box contains the segment's centre, else -1."""
    x0, y0, x1, y1 = seg.box.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for i, r in enumerate(regions):
        rx0, ry0, rx1, ry1 = r.box.bbox
        if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
            return i
    return -1


def _cluster_within_regions(segs: list[Segment], regions) -> list[list[Segment]]:
    """Cluster OCR boxes into lines WITHIN each region (so side-by-side columns don't
    merge), emitting clusters in region reading order. Boxes outside every region are
    clustered together at the end."""
    order = sorted(range(len(regions)), key=lambda i: regions[i].reading_order)
    buckets: dict[int, list[Segment]] = {i: [] for i in range(len(regions))}
    outside: list[Segment] = []
    for s in segs:
        ri = _assign_region(s, regions)
        (buckets[ri] if ri >= 0 else outside).append(s)
    clusters: list[list[Segment]] = []
    for i in order:
        if buckets[i]:
            clusters.extend(_cluster_lines(buckets[i]))
    if outside:
        clusters.extend(_cluster_lines(outside))
    return clusters


def _cluster_lines(segs: list[Segment]) -> list[list[Segment]]:
    """Group boxes into visual lines by vertical-centre proximity; sort lines
    top-to-bottom and boxes left-to-right within each."""
    ordered = sorted(segs, key=lambda s: (s.box.bbox[1] + s.box.bbox[3]) / 2)
    clusters: list[dict] = []
    for s in ordered:
        y0, y1 = s.box.bbox[1], s.box.bbox[3]
        cy, h = (y0 + y1) / 2, max(y1 - y0, 1.0)
        for cl in clusters:
            if abs(cy - cl["cy"]) < 0.6 * h:
                cl["segs"].append(s)
                cl["cy"] = sum((x.box.bbox[1] + x.box.bbox[3]) / 2 for x in cl["segs"]) / len(cl["segs"])
                break
        else:
            clusters.append({"cy": cy, "segs": [s]})
    clusters.sort(key=lambda cl: cl["cy"])
    for cl in clusters:
        cl["segs"].sort(key=lambda s: s.box.bbox[0])
    return [cl["segs"] for cl in clusters]


def _merge_box(segs: list[Segment]) -> Box:
    xs0 = min(s.box.bbox[0] for s in segs)
    ys0 = min(s.box.bbox[1] for s in segs)
    xs1 = max(s.box.bbox[2] for s in segs)
    ys1 = max(s.box.bbox[3] for s in segs)
    return Box(points=[(xs0, ys0), (xs1, ys0), (xs1, ys1), (xs0, ys1)])


def _nw_align(cluster_texts: list[str], vlm_lines: list[str]) -> dict[int, int]:
    """Needleman-Wunsch over the two ordered sequences, scored by fuzzy similarity.
    Returns cluster_idx -> vlm_line_idx for matched pairs (monotonic, gaps allowed)."""
    n, m = len(cluster_texts), len(vlm_lines)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _GAP
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(
                dp[i - 1][j - 1] + _sim(cluster_texts[i - 1], vlm_lines[j - 1]),
                dp[i - 1][j] + _GAP,
                dp[i][j - 1] + _GAP,
            )
    mapping: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        diag = dp[i - 1][j - 1] + _sim(cluster_texts[i - 1], vlm_lines[j - 1])
        if dp[i][j] == diag:
            mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + _GAP:
            i -= 1
        else:
            j -= 1
    return mapping


def _word_distribute(cluster_texts: list[str], vlm_reading: str) -> list[str] | None:
    """Spread the VLM reading across the in-order line clusters at the WORD level — one text
    per cluster, or None if the alignment is too weak to trust (caller falls back to line-level).

    Line-level alignment ([_nw_align]) breaks on handwriting: the VLM returns prose (a few long
    lines) while the detector finds many short visual lines, so one VLM line can't cover the
    several clusters it actually spans — the others get gapped and keep their garbled det_text.
    But even garbled handwriting keeps roughly the right WORDS in roughly the right places
    (`rturaing`≈`returning`), so we fuzzy-align det words to VLM words and project each VLM word
    back onto the cluster its aligned det word belongs to. A long sentence then flows across its
    visual lines instead of collapsing onto the first box.

    Trust guards (so it degrades honestly, never confidently-wrong):
    - **Anchors.** A det<->VLM word match >= _ANCHOR_SIM is an anchor. If too few clusters are
      anchored (a different reading order broke the monotonic chain, or recognition failed
      wholesale), return None — line-level NW is the safer baseline.
    - **No edge-smearing.** Distribute only WITHIN the outermost anchored clusters; VLM words
      past them (an entirely undetected region) are dropped here, not smeared onto a boundary
      box — they remain in the reading view (document.md), just not pinned to a coordinate."""
    det_tagged = [(ci, w) for ci, t in enumerate(cluster_texts) for w in t.split()]
    vlm_words = vlm_reading.split()
    if not det_tagged or not vlm_words:
        return None

    A = [w for _, w in det_tagged]
    n, m = len(A), len(vlm_words)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _GAP
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _GAP
    for i in range(1, n + 1):
        ai, prev, row = A[i - 1], dp[i - 1], dp[i]
        for j in range(1, m + 1):
            row[j] = max(prev[j - 1] + _sim(ai, vlm_words[j - 1]),
                         prev[j] + _GAP, row[j - 1] + _GAP)

    ops: list[tuple] = []   # (cluster_idx | None, vlm_word | None, is_anchor)
    i, j = n, m
    while i > 0 or j > 0:
        s = _sim(A[i - 1], vlm_words[j - 1]) if i > 0 and j > 0 else 0.0
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + s:
            ops.append((det_tagged[i - 1][0], vlm_words[j - 1], s >= _ANCHOR_SIM)); i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + _GAP:
            ops.append((det_tagged[i - 1][0], None, False)); i -= 1
        else:
            ops.append((None, vlm_words[j - 1], False)); j -= 1
    ops.reverse()

    per: list[list[str]] = [[] for _ in cluster_texts]
    anchored = [False] * len(cluster_texts)
    cur = 0
    for ci, w, is_anchor in ops:
        if ci is not None:
            cur = ci
            if is_anchor:
                anchored[ci] = True
        if w is not None:
            per[cur].append(w)

    idx = [k for k, a in enumerate(anchored) if a]
    if len(idx) < max(1, _MIN_ANCHOR_FRAC * len(cluster_texts)):
        return None   # untrustworthy (reorder / wholesale failure) -> let the caller fall back
    # Keep only clusters within the anchored span: an unanchored cluster OUTSIDE it (e.g. an
    # undetected edge line) gets nothing rather than smeared words. (A trailing word the
    # detector merely missed at the end of an anchored line still lands on that line.)
    lo, hi = idx[0], idx[-1]
    return [" ".join(per[ci]) if lo <= ci <= hi else "" for ci in range(len(cluster_texts))]


class Fusion:
    name = "fusion"

    def run(self, doc: Document, cfg: Config) -> Document:
        for page in doc.pages:
            self._compose(page)
            if page.vlm_reading.strip():
                self._fuse_lines(page, cfg)
            for seg in page.segments:
                if not seg.superseded and not seg.best_text:
                    seg.best_text = seg.vlm_text or seg.det_text or ""
                    if seg.source in _OCR_SOURCES and seg.vlm_text:
                        seg.source = "fused"
            # combine both sets in reading order (machine-readable + OCR)
            page.segments.sort(key=lambda s: reading_key(
                s, page.regions, page.rotation, page.width, page.height))
        return doc

    def _compose(self, page: Page) -> None:
        """Combine the machine-readable text layer with OCR of the image areas. The
        decision is per-REGION where layout gave us regions (a region covered by clean
        text is machine-readable -> its OCR is redundant), else per-box overlap.
        Nothing is dropped — the weaker source is marked `superseded` for provenance."""
        clean_tl = [s for s in page.segments if s.source == "textlayer" and s.best_text]
        contaminated_tl = [s for s in page.segments
                           if s.source == "textlayer" and not s.best_text]
        ocr = [s for s in page.segments if s.source in _OCR_SOURCES]

        if page.regions:
            classify_regions(page.regions, clean_tl)
            for s in ocr:
                # superseded if the enclosing region is machine-readable, OR the OCR box
                # directly overlaps a clean text-layer line. The overlap check catches a
                # machine-readable header/footer band on a scanned form: the region may
                # not cross the coverage threshold, but where exact text sits on top of
                # the OCR the OCR is still redundant (else it renders twice).
                if in_machine_readable_region(s, page.regions) or _overlaps(s, clean_tl):
                    s.superseded = True   # exact text layer covers this box
        else:
            for s in ocr:
                if _overlaps(s, clean_tl):
                    s.superseded = True   # exact text layer covers this box

        kept_ocr = [s for s in ocr if not s.superseded]
        for s in contaminated_tl:
            if _overlaps(s, kept_ocr):
                s.superseded = True       # contaminated layer -> OCR repairs it

    def _fuse_lines(self, page: Page, cfg: Config) -> None:
        ocr = [s for s in page.segments
               if s.source in _OCR_SOURCES and not s.superseded]
        ocr_ids = {id(s) for s in ocr}
        others = [s for s in page.segments if id(s) not in ocr_ids]
        if not ocr:
            return
        vlm_lines = [ln.strip() for ln in page.vlm_reading.splitlines() if ln.strip()]
        # region-aware where layout gave us regions; else global y-band clustering
        clusters = (_cluster_within_regions(ocr, page.regions)
                    if page.regions else _cluster_lines(ocr))
        cluster_text = [" ".join(s.det_text or "" for s in cl) for cl in clusters]

        # Marry the VLM reading to the line boxes. Word-level distribution spreads a long prose
        # line across the several visual lines it spans (handwriting — the headline case). It
        # declines (returns None) when the page is huge, the alignment is untrustworthy (a
        # reading-order mismatch), or there's no reading — then fall back to the line-level NW
        # baseline. Either path yields one text per cluster.
        n_words = sum(len(t.split()) for t in cluster_text) * len(page.vlm_reading.split())
        assigned = None
        if vlm_lines and n_words <= _MAX_DP_CELLS:
            assigned = _word_distribute(cluster_text, page.vlm_reading)
        if assigned is None:
            mapping = _nw_align(cluster_text, vlm_lines) if vlm_lines else {}
            assigned = [vlm_lines[mapping[ci]] if ci in mapping else "" for ci in range(len(clusters))]

        fused: list[Segment] = []
        for ci, cl in enumerate(clusters):
            line = assigned[ci].strip()
            # Anti-misalignment gate: NW always pairs a cluster with *some* line rather
            # than gapping both, so a confident OCR cluster can be handed a dissimilar VLM
            # line (an off-by-one in the reading). Where the detector is sure it read real
            # text yet the aligned line barely resembles it, that's misalignment, not a
            # correction — keep the trusted det_text. Gated on det_conf so the handwriting
            # path (garbled det_text, low conf, VLM is the real reading) is never penalised.
            if line and _sim(cluster_text[ci], line) < cfg.fuse_min_sim \
                    and max((s.det_conf or 0.0) for s in cl) >= cfg.fuse_det_conf_trust:
                line = ""
            # a cluster with no aligned VLM line keeps its real engine's source
            # (vision/paddle) — NOT a hard-coded "paddle" (that mis-credited the engine)
            base_source = cl[0].source if cl else "paddle"
            fused.append(Segment(
                id=f"p{page.index}-f{ci}",
                page=page.index,
                box=_merge_box(cl),
                det_text=cluster_text[ci],
                det_conf=max((s.det_conf or 0.0) for s in cl),
                vlm_text=line or None,
                best_text=line,
                source="fused" if line else base_source,
                read_by=page.read_model if line else "",
            ))
        page.segments = others + fused
