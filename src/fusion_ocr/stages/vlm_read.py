"""Stage 5 — VLM read (the semantics track).

REAL IMPL (extra: vlm): for each region, send the crop to the VLM via the swappable
OpenAI-compatible client. Prose -> verbatim transcription; tables -> markdown/HTML;
plus translation into each target language. The VLM provides *reading*, not
geometry — its output is aligned onto the deterministic boxes in fusion.

WALKING SKELETON: passthrough. The VLM client is constructed but not called, so the
plumbing stays model-free until the `vlm` extra + a running endpoint are present.
"""

from __future__ import annotations

from ..config import Config
from ..models import Document
from ..vlm.client import get_client


class VlmRead:
    name = "vlm_read"

    def run(self, doc: Document, cfg: Config) -> Document:
        client = get_client(cfg)  # noqa: F841 — wired, not yet called
        # TODO: per-region client.read(crop, prompt) -> Segment.vlm_text / translations
        return doc
