"""The data model that flows through the pipeline.

Design rule: every stage *enriches* the Document in place and returns it. Nothing
overwrites a prior stage's raw output — `det_text` and `vlm_text` are both kept
alongside the chosen `best_text`, so the record is its own provenance trail and can
be serialised between stages for cheap resume / re-run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from types import UnionType
from typing import Literal, Union, get_args, get_origin, get_type_hints

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
    the deterministic cell structure and `cells` the per-cell boxes (page coords);
    `table_vlm` holds the focused VLM reading of the table (clean content), with
    `table_read_by` naming the model. Both representations are kept — geometry from the
    deterministic grid, content from the VLM — never one overwriting the other."""

    box: Box
    kind: RegionKind = "paragraph"
    reading_order: int = 0
    source: str = ""  # "textlayer" (covered by clean machine-readable text) | "ocr"
    table_html: str = ""
    cells: list[Box] = field(default_factory=list)
    table_vlm: str = ""      # focused VLM read of a table region (markdown/HTML)
    table_read_by: str = ""  # provenance: model that produced table_vlm
    # provenance: how table_html was produced — "" (none), "find_tables" (PyMuPDF, exact
    # text-layer extraction for born-digital), or "table_structure" (PaddleOCR vision).
    table_engine: str = ""


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
    # fingerprint of the processing recipe (pipeline + model + prompts + config) that
    # produced this doc. Resume only reuses a snapshot whose recipe matches the current
    # one, so changing a prompt/model/route reprocesses instead of silently reusing.
    recipe: str = ""
    # wall-clock seconds spent in each stage that actually ran this invocation (stage name
    # -> seconds). Populated by the pipeline; lets "where did the time go?" be answered from
    # the output, not just ad-hoc profiling. Stages reused from a resume snapshot aren't here.
    stage_seconds: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "Document":
        return _from_dict(cls, json.loads(text))


def _build(typ, value):
    """Reconstruct a value of annotated type ``typ`` from JSON-decoded ``value``,
    recursing into dataclasses and list/tuple element types. Plain/unknown types
    (str, int, dict, Literal, ...) pass through unchanged."""
    if value is None:
        return None
    if is_dataclass(typ):
        return _from_dict(typ, value)
    origin = get_origin(typ)
    if origin in (Union, UnionType):                       # e.g. str | None
        args = [a for a in get_args(typ) if a is not type(None)]
        return _build(args[0], value) if len(args) == 1 else value
    if origin in (list, tuple) and get_args(typ):          # list[Box], list[Point], ...
        elem = get_args(typ)[0]
        built = [_build(elem, v) for v in value]
        return tuple(built) if origin is tuple else built
    return value


def _from_dict(cls, data: dict):
    """Schema-driven dataclass deserializer. Walks ``cls``'s own fields and their
    resolved types, so a newly-added field (with a default) round-trips automatically
    — there's no parallel hand-written mapping to forget to update, which used to drop
    fields silently on resume. Unknown keys in ``data`` are ignored (forward-compat)."""
    hints = get_type_hints(cls)
    return cls(**{f.name: _build(hints[f.name], data[f.name])
                  for f in fields(cls) if f.name in data})
