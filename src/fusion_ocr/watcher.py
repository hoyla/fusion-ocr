"""Drop-folder entrypoint — and the QUEUE WORKER.

  $ python -m fusion_ocr.watcher          # watch in/, drain the queue
  $ python -m fusion_ocr.watcher --once    # process whatever's queued now, then exit

Drop a PDF into in/, artifacts appear in out/<sha256>/. Idempotent by content hash.

This loop is the worker that drains the JobStore queue: it claims QUEUED jobs (atomically)
and processes them. Jobs reach the queue two ways — a file dropped in in/, or POST /jobs on
the API (which writes the file here and registers it queued without processing). So in a
deployment you run this alongside `fusion-ocr-serve`: the API enqueues, this worker drains.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import config as config_mod
from . import ingest, storage
from .jobs import JobStore
from .pipeline import process, sha256_of
from .vlm.openai_compat import preflight_reader


def _move_out(src: Path, in_dir: Path, subdir: str, digest: str) -> None:
    """Move a handled file into in_dir/<subdir>/<digest><suffix> (collision-free, ties it to
    its job; keeps the original extension so an image input stays recognisable). The scan
    skips subdirectories, so files here aren't re-processed. Best-effort: a failed move is
    logged, not fatal."""
    target_dir = in_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        src.replace(target_dir / f"{digest}{src.suffix.lower()}")   # atomic within one fs
    except OSError as exc:
        print(f"[warn] could not move {src.name} -> {subdir}/: {exc}", file=sys.stderr)


def scan_once(cfg: config_mod.Config, jobs: JobStore,
              force: bool = False, rerun_from: str | None = None,
              min_settle: float = 2.0, move_processed: bool = False) -> int:
    in_dir = Path(cfg.in_dir)
    in_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    reprocess = force or rerun_from is not None
    now = time.time()
    # Ingest boundary: PDF (identity) + raster images (PNG/JPEG/TIFF) normalised to PDF here,
    # before the queue (see ingest.py). Sniff every loose file by magic bytes; unsupported
    # drops are ignored. iterdir (not glob) is non-recursive, so processed/ and failed/ are
    # skipped and the moved files aren't re-processed.
    for src in sorted(f for f in in_dir.iterdir() if f.is_file()):
        fmt = ingest.peek(src)
        if fmt is None:
            continue   # not a supported input
        # Settle gate: skip a file still being written (mtime within min_settle of now).
        # Hashing a half-copied drop would key the job under a digest that changes once
        # the copy finishes — process it on a later scan instead.
        if now - src.stat().st_mtime < min_settle:
            continue
        # Key the job by the ORIGINAL input's hash (stable identity + provenance) even when a
        # derived PDF is what gets processed — re-dropping the same image is then idempotent.
        digest = sha256_of(src)
        jobs.upsert_queued(digest, str(src))      # ensure registered (no-op if already)
        # Status-driven worker: atomically claim a QUEUED job. This drains both folder drops
        # and API-enqueued uploads (POST /jobs registers them queued + leaves the file here),
        # and the atomic claim makes running several workers safe. Skip if not claimable
        # (already running/done/error, unless an explicit reprocess is asked).
        if not jobs.claim(digest, reprocess=reprocess):
            continue
        try:
            if fmt == "pdf":
                pdf = src
            else:
                # Derived, provenanced PDF beside the artifacts (out/<digest>/source.pdf) —
                # the exact thing that was OCR'd; the original image stays the canonical source.
                job_dir = storage.job_dir(cfg, digest)
                job_dir.mkdir(parents=True, exist_ok=True)
                pdf = ingest.image_to_pdf(src, job_dir / "source.pdf")
            doc = process(pdf, cfg, force=force, rerun_from=rerun_from, digest=digest)
            jobs.set_status(digest, "done")
            print(f"[done] {src.name} -> out/{digest}/  "
                  f"({len(doc.artifacts)} artifacts)")
            processed += 1
            if move_processed:
                _move_out(src, in_dir, "processed", digest)
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the loop
            jobs.set_status(digest, "error", str(exc))
            print(f"[error] {src.name}: {exc}", file=sys.stderr)
            if move_processed:
                _move_out(src, in_dir, "failed", digest)
    return processed


def main() -> None:
    # Line-buffer stdout so the worker's [watch]/[done] progress shows up live in a redirected
    # log (a file / systemd journal), where stdout is otherwise block-buffered and the status
    # lines wouldn't appear until the buffer filled — making a busy or stuck worker look idle.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(prog="fusion-ocr")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--once", action="store_true", help="process then exit")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--force", action="store_true",
                        help="reprocess from scratch even if already done (with --once)")
    parser.add_argument("--rerun-from", metavar="STAGE", default=None,
                        help="re-run from STAGE onward, reusing earlier cached stages "
                             "(e.g. --rerun-from vlm_read to retune the prompt; --once)")
    args = parser.parse_args()

    cfg = config_mod.load(args.config)
    if cfg.airgap:
        config_mod.enforce_airgap()
        print("[airgap] outbound connections refused (loopback only)")

    jobs = JobStore(Path(cfg.out_dir) / "jobs.sqlite")
    print(f"[watch] {cfg.in_dir}/  ->  {cfg.out_dir}/   (vlm: {cfg.vlm.base_url})")

    # Preflight the reader once at startup: a tiny inference proving it can actually READ (a
    # wedged server answers /v1/models 200 while failing generation, so a plain ping wouldn't
    # catch it). Non-fatal — a dead reader only means pages fall back to det_text, now VISIBLY
    # (page.read_failed + a logged warning) — but surfacing it here beats discovering it after a
    # corpus silently degraded. Also warms the model so the first real document isn't slow.
    ok, detail = preflight_reader(cfg)
    if ok:
        print(f"[reader] {detail}")
    else:
        print(f"[warn] READER PREFLIGHT FAILED — VLM pages will fall back to det_text until the "
              f"reader is up: {detail}", file=sys.stderr)

    if args.once:
        # --once never moves: a manual re-run shouldn't disturb the drop folder.
        scan_once(cfg, jobs, force=args.force, rerun_from=args.rerun_from)
        return
    # Loop mode watches for NEW files; --force/--rerun-from are for one-shot reprocessing.
    try:
        while True:
            scan_once(cfg, jobs, move_processed=cfg.move_processed)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[stop]")


if __name__ == "__main__":
    main()
