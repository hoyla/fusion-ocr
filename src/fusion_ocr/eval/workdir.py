"""Run-scoped working directories for the eval — under eval_out/, never /tmp.

Every eval runner (born-digital harness, hand-labelled set, 3rd-party datasets) renders pages
to images, writes image-only PDFs and runs the full pipeline for each item it scores — so its
working directory ends up holding every recovered word of every page, in the per-stage
snapshots. That used to go to `tempfile.mkdtemp()`, i.e. `/tmp`, and was never cleaned up:
harmless for public benchmark data, but the day someone ran `--labels` on a confidential
document its full text would sit in a world-readable temp dir indefinitely (review 03).

Work now lives in a run-scoped directory under `eval_out/_work/` — gitignored, inside the
repo tree the operator already treats as sensitive — and is removed when the run finishes.
`keep=True` (CLI `--keep-work`) retains it for inspection; a caller-supplied directory is
always the caller's to manage and is never removed here.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

_log = logging.getLogger(__name__)

# Relative to the cwd the eval is run from — the same convention as the eval's other paths
# (`samples/...` in a labelset, `eval_out/<run>/` for the committed run scripts).
WORK_ROOT = Path("eval_out") / "_work"


def new_workdir(prefix: str, root: str | Path | None = None) -> Path:
    """Create and return a fresh, uniquely named run directory under `root` (WORK_ROOT)."""
    root = Path(root) if root is not None else WORK_ROOT
    root.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    path, n = root / f"{prefix}-{stamp}", 0
    while path.exists():                       # two runs in the same second, same process
        n += 1
        path = root / f"{prefix}-{stamp}-{n}"
    path.mkdir()
    return path


@contextmanager
def workdir(prefix: str, given: str | Path | None = None, keep: bool = False):
    """Yield the directory an eval run should work in.

    `given` (a caller-owned path, e.g. a test's tmp_path) is yielded as-is and never
    removed. Otherwise a fresh run directory is created under WORK_ROOT and removed on
    exit — including on error — unless `keep` is set, in which case its path is logged.
    """
    if given is not None:
        yield Path(given)
        return
    path = new_workdir(prefix)
    try:
        yield path
    finally:
        if keep:
            _log.info("eval work kept at %s", path)
        else:
            shutil.rmtree(path, ignore_errors=True)
