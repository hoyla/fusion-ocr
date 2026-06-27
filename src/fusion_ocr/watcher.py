"""Drop-folder entrypoint — the simplest way to run the MVP.

  $ python -m fusion_ocr.watcher          # watch in/, process new PDFs
  $ python -m fusion_ocr.watcher --once    # process whatever's there now, then exit

Drop a PDF into in/, artifacts appear in out/<sha256>/. Idempotent by content hash.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import config as config_mod
from .jobs import JobStore
from .pipeline import process, sha256_of


def scan_once(cfg: config_mod.Config, jobs: JobStore,
              force: bool = False, rerun_from: str | None = None,
              min_settle: float = 2.0) -> int:
    in_dir = Path(cfg.in_dir)
    in_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    reprocess = force or rerun_from is not None
    now = time.time()
    for pdf in sorted(in_dir.glob("*.pdf")):
        # Settle gate: skip a file still being written (mtime within min_settle of now).
        # Hashing a half-copied drop would key the job under a digest that changes once
        # the copy finishes — process it on a later scan instead.
        if now - pdf.stat().st_mtime < min_settle:
            continue
        digest = sha256_of(pdf)
        newly = jobs.upsert_queued(digest, str(pdf))
        if not newly and not reprocess:
            continue  # already seen — idempotent (unless an explicit reprocess is asked)
        jobs.set_status(digest, "running")
        try:
            doc = process(pdf, cfg, force=force, rerun_from=rerun_from, digest=digest)
            jobs.set_status(digest, "done")
            print(f"[done] {pdf.name} -> out/{digest}/  "
                  f"({len(doc.artifacts)} artifacts)")
            processed += 1
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the loop
            jobs.set_status(digest, "error", str(exc))
            print(f"[error] {pdf.name}: {exc}", file=sys.stderr)
    return processed


def main() -> None:
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

    if args.once:
        scan_once(cfg, jobs, force=args.force, rerun_from=args.rerun_from)
        return
    # Loop mode watches for NEW files; --force/--rerun-from are for one-shot reprocessing.
    try:
        while True:
            scan_once(cfg, jobs)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[stop]")


if __name__ == "__main__":
    main()
