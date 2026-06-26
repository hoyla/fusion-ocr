"""The data model that flows through the pipeline.

Design rule: every stage *enriches* the Document in place and returns it. Nothing
overwrites a prior stage's raw output — `det_text` and `vlm_text` are both kept
alongside the chosen `best_text`, so the record is its own provenance trail and can
be serialised between stages for cheap resume / re-run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

Point = tuple[float, float]

SegmentSource = Literal["textlayer", "paddle", "vision", "vlm", "fused"]
RegionKind = Literal["paragraph", "table", "figure", "header", "footer", "other"]


@dataclass
class Box:
    """A quadrilateral (Paddle returns 4 points; tracks skew/rotation).

    Axis-aligned bbox is derived on demand for renderers that want a rectangle.
    """

    points: list[Point]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class Segment:
    """One unit of text + its geometry. Geometry always comes from the
    deterministic side (text layer or PaddleOCR); semantics may be refined by the
    VLM. `best_text` is what the overlay carries."""

    id: str
    page: int
    box: Box
    best_text: str = ""
    source: SegmentSource = "paddle"
    det_text: str | None = None
    det_conf: float | None = None
    vlm_text: str | None = None
    read_by: str = ""  # provenance: which VLM read this region (model name)
    # True when a better source covers this area (e.g. OCR superseding a contaminated
    # text layer, or the text layer superseding redundant OCR). Retained for provenance
    # (principle: never mutate/discard source), excluded from the primary output.
    superseded: bool = False
    translations: dict[str, str] = field(default_factory=dict)


@dataclass
class Region:
    """A layout region from PP-DocLayout. Reading order is what lets the VLM's linear
    transcription be aligned back onto boxes. For `table` regions, `table_html` holds
    the deterministic cell structure and `cells` the per-cell boxes (page coords)."""

    box: Box
    kind: RegionKind = "paragraph"
    reading_order: int = 0
    source: str = ""  # "textlayer" (covered by clean machine-readable text) | "ocr"
    table_html: str = ""
    cells: list[Box] = field(default_factory=list)


@dataclass
class Page:
    index: int
    width: float = 0.0
    height: float = 0.0
    has_text_layer: bool = False
    needs_ocr: bool = True
    rotation: int = 0
    script: str = ""        # detected script -> routing (e.g. "thai", "latin")
    read_model: str = ""    # VLM model used to read this page (provenance)
    image_ref: str | None = None
    vlm_reading: str = ""  # raw VLM transcription of the page (clean reading view)
    regions: list[Region] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)


@dataclass
class Document:
    source_path: str
    sha256: str
    languages: list[str] = field(default_factory=list)
    pages: list[Page] = field(default_factory=list)
    # artifact name -> path, e.g. "overlay_pdf", "markdown.en", "segment_index"
    artifacts: dict[str, str] = field(default_factory=dict)
    # name of the last stage that completed, for resume
    stage_completed: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Document":
        raw = json.loads(text)
        pages = []
        for p in raw.get("pages", []):
            regions = [
                Region(box=Box(points=[tuple(pt) for pt in r["box"]["points"]]),
                       kind=r.get("kind", "paragraph"),
                       reading_order=r.get("reading_order", 0),
                       source=r.get("source", ""),
                       table_html=r.get("table_html", ""),
                       cells=[Box(points=[tuple(pt) for pt in cb["points"]])
                              for cb in r.get("cells", [])])
                for r in p.get("regions", [])
            ]
            segments = [
                Segment(
                    id=s["id"], page=s["page"],
                    box=Box(points=[tuple(pt) for pt in s["box"]["points"]]),
                    best_text=s.get("best_text", ""), source=s.get("source", "paddle"),
                    det_text=s.get("det_text"), det_conf=s.get("det_conf"),
                    vlm_text=s.get("vlm_text"), read_by=s.get("read_by", ""),
                    superseded=s.get("superseded", False),
                    translations=s.get("translations", {}),
                )
                for s in p.get("segments", [])
            ]
            pages.append(Page(index=p["index"], width=p.get("width", 0.0),
                              height=p.get("height", 0.0),
                              has_text_layer=p.get("has_text_layer", False),
                              needs_ocr=p.get("needs_ocr", True),
                              rotation=p.get("rotation", 0),
                              script=p.get("script", ""),
                              read_model=p.get("read_model", ""),
                              image_ref=p.get("image_ref"),
                              vlm_reading=p.get("vlm_reading", ""),
                              regions=regions, segments=segments))
        return cls(source_path=raw["source_path"], sha256=raw["sha256"],
                   languages=raw.get("languages", []), pages=pages,
                   artifacts=raw.get("artifacts", {}),
                   stage_completed=raw.get("stage_completed"))
