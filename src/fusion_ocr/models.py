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

SegmentSource = Literal["textlayer", "paddle", "vlm", "fused"]
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
    translations: dict[str, str] = field(default_factory=dict)


@dataclass
class Region:
    """A layout region from PP-StructureV3. Reading order is what lets the VLM's
    linear transcription be aligned back onto boxes later."""

    box: Box
    kind: RegionKind = "paragraph"
    reading_order: int = 0


@dataclass
class Page:
    index: int
    width: float = 0.0
    height: float = 0.0
    has_text_layer: bool = False
    needs_ocr: bool = True
    rotation: int = 0
    image_ref: str | None = None
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
                       reading_order=r.get("reading_order", 0))
                for r in p.get("regions", [])
            ]
            segments = [
                Segment(
                    id=s["id"], page=s["page"],
                    box=Box(points=[tuple(pt) for pt in s["box"]["points"]]),
                    best_text=s.get("best_text", ""), source=s.get("source", "paddle"),
                    det_text=s.get("det_text"), det_conf=s.get("det_conf"),
                    vlm_text=s.get("vlm_text"), translations=s.get("translations", {}),
                )
                for s in p.get("segments", [])
            ]
            pages.append(Page(index=p["index"], width=p.get("width", 0.0),
                              height=p.get("height", 0.0),
                              has_text_layer=p.get("has_text_layer", False),
                              needs_ocr=p.get("needs_ocr", True),
                              rotation=p.get("rotation", 0),
                              image_ref=p.get("image_ref"),
                              regions=regions, segments=segments))
        return cls(source_path=raw["source_path"], sha256=raw["sha256"],
                   languages=raw.get("languages", []), pages=pages,
                   artifacts=raw.get("artifacts", {}),
                   stage_completed=raw.get("stage_completed"))
