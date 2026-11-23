#!/usr/bin/env python
"""Run the download → process → QC → pack stages in order (02c, 03, 04, 05, verify).

  smoke  ~20 clips end to end in an isolated $NG_DATA_ROOT/smoke/ (own manifest copy, processed dir, shards),
         then bundles the debug grids/videos + inspect report into smoke/smoke_results.tgz.
  bulk   02c layout detection for all videos (resumable), then repeated cycles of
             03 download/process (--cycle-clips)  →  04 VLM caption+QC  →  05 pack (--append --delete-processed)
         until the manifest is done or the disk floor is reached. Safe to stop and re-run at any time.

Examples:
  python run_pipeline.py smoke --proxy http://127.0.0.1:8899
  python run_pipeline.py bulk  --proxy http://127.0.0.1:8899 --workers 8 --min-free-gb 60
  python run_pipeline.py bulk  --dry-run                     # print the commands only

Stages 01, 02b and 02 (metadata, LLM labels, manifest) are run separately; pass --resample to re-run stage 02
(re-sample everything, balanced-first order) before starting.
"""
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ng_common import DATA_ROOT, setup_logging

HERE = Path(__file__).resolve().parent
PY = sys.executable


class StepFailed(RuntimeError):
    pass


def run(cmd: list[str], log, dry: bool, stdout_path: Path | None = None) -> None:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    log.info("▶ %s", pretty)
    if dry:
        return
    t0 = time.time()
    if stdout_path is None:  # stream to the console AND keep a copy (incl. tracebacks) for the bundle
        (DATA_ROOT / "logs").mkdir(parents=True, exist_ok=True)
        name = Path(cmd[1]).stem if len(cmd) > 1 else "step"
        with open(DATA_ROOT / "logs" / f"console_{name}.log", "a") as tee:
            tee.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {pretty}\n")
            p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            for chunk in iter(lambda: p.stdout.read(4096), b""):
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                tee.write(chunk.decode(errors="replace"))
            rc = p.wait()
    else:
        with open(stdout_path, "w") as fh:
            rc = subprocess.run(cmd, cwd=HERE, stdout=fh).returncode
    log.info("  finished in %.0fs (exit %d)", time.time() - t0, rc)
    if rc != 0:
        raise StepFailed(f"step failed (exit {rc}): {pretty}")


def net_args(a) -> list[str]:
    out = []
    if a.proxy:
        out += ["--proxy", a.proxy]
    if a.cookies:
        out += ["--cookies", a.cookies]
    return out


def layout_args(a) -> list[str]:
    return ["--model", a.layout_model] if a.layout_model else []


def proxy_ok(proxy: str | None, timeout: float = 15) -> bool:
    """Can we reach YouTube through the proxy? (no proxy configured = nothing to check)"""
    if not proxy:
        return True
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    try:
        with opener.open("https://www.youtube.com/robots.txt", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def require_proxy(a, log, wait: bool) -> None:
    if a.dry_run or proxy_ok(a.proxy):
        return
    msg = (f"Proxy {a.proxy} is not reachable. On the Mac: keep `proxy --hostname 127.0.0.1 --port 8899` and the "
           f"`ssh -N -R 8899:127.0.0.1:8899 ...` tunnel running (and the Mac awake: caffeinate -dimsu).")
    if not wait:
        raise StepFailed(msg)
    log.error("%s Waiting for it to come back (checking every 30 s)...", msg)
    while not proxy_ok(a.proxy):
        time.sleep(30)
    log.info("Proxy is back.")


def count_clips(processed: Path) -> int:
    return sum(1 for _ in processed.glob("*/*/meta.json")) if processed.exists() else 0


def resample(a, log):
    run([PY, "02_profile_and_sample.py", "--enrich-youtube", "--target-hours", "1e9", "--skip-profile"], log, a.dry_run)


# ============================================================================ smoke
def smoke(a, log):
    root = DATA_ROOT / "smoke"
    man_main = DATA_ROOT / "manifest_2d.parquet"
    man = root / "manifest_smoke.parquet"
    processed, wds, debug, tmp = root / "processed", root / "webdataset", root / "debug", root / "tmp_03"
    if a.resample:
        resample(a, log)
    if not a.dry_run:
        for p in (processed, wds, debug, tmp):
            shutil.rmtree(p, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy(man_main, man)  # isolated copy: 04/05 write into it, the real manifest stays untouched
        debug.mkdir(parents=True, exist_ok=True)
    failed = None
    try:
        require_proxy(a, log, wait=False)
        if not a.skip_layout:  # 02c results are shared with bulk (same videos, same answers)
            run([PY, "02c_detect_layout.py", "--manifest", str(man), "--limit", str(max(40, 2 * a.clips)),
                 "--workers", str(a.workers), *layout_args(a), *net_args(a)], log, a.dry_run)
        run([PY, "03_download_and_process.py", "--manifest", str(man), "--processed-dir", str(processed),
             "--tmp-dir", str(tmp), "--limit", str(a.clips), "--workers", str(a.workers),
             "--require-layout-check", "--min-video-gameplay-frac", str(a.min_video_gameplay_frac), *net_args(a)],
            log, a.dry_run)
        if not a.dry_run and count_clips(processed) == 0:
            raise StepFailed("stage 03 produced no clips (see logs/03_download_and_process.log; proxy / YouTube access?)")
        run([PY, "04_caption_frames.py", "--manifest", str(man), "--processed-dir", str(processed)], log, a.dry_run)
        run([PY, "05_pack_webdataset.py", "--manifest", str(man), "--processed-dir", str(processed),
             "--out", str(wds), "--overwrite"], log, a.dry_run)
        try:
            run([PY, "verify_dataset.py", "--root", str(wds), "--out-dir", str(debug), "--debug-videos", "2",
                 "--num-shards", "50"], log, a.dry_run, stdout_path=debug / "verify.txt")
        except StepFailed as e:  # keep going: the bundle is most useful exactly when something failed
            log.warning("%s (see debug/verify.txt)", e)
    except StepFailed as e:
        failed = e
        log.error("%s -- bundling what exists so far", e)
    finally:
        if not a.dry_run:
            bundle_smoke(root, man, processed, wds, debug, log)
    if failed:
        raise failed


def bundle_smoke(root: Path, man: Path, processed: Path, wds: Path, debug: Path, log) -> None:
    """Always produce smoke_results.tgz: debug grids/videos, verify/inspect reports and every stage's log."""
    debug.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([PY, "inspect_clips.py", "--processed-dir", str(processed), "--manifest", str(man)],
                       cwd=HERE, stdout=open(debug / "inspect.txt", "w"), stderr=subprocess.STDOUT, timeout=600)
    except Exception as e:
        log.warning("inspect_clips failed: %s", e)
    if (wds / "name_mismatch_report.csv").exists():
        shutil.copy(wds / "name_mismatch_report.csv", debug / "name_mismatch_report.csv")
    logs_dst = debug / "logs"
    logs_dst.mkdir(exist_ok=True)
    for f in (DATA_ROOT / "logs").glob("*.log"):
        try:  # last ~2000 lines of each stage log
            lines = f.read_text(errors="replace").splitlines()[-2000:]
            (logs_dst / f.name).write_text("\n".join(lines))
        except OSError:
            pass
    bundle = root / "smoke_results.tgz"
    try:
        subprocess.run(["tar", "czf", str(bundle), "-C", str(root), "debug"], check=True)
        log.info("Smoke bundle: %s  (download this)", bundle)
    except Exception as e:
        log.error("could not create %s: %s", bundle, e)


# ============================================================================ bulk
def bulk(a, log):
    processed = DATA_ROOT / "processed"
    if a.resample:
        resample(a, log)
    if not a.skip_layout:
        require_proxy(a, log, wait=True)
        run([PY, "02c_detect_layout.py", "--workers", str(a.workers), *layout_args(a), *net_args(a)], log, a.dry_run)
    cycle = 0
    while True:
        cycle += 1
        log.info("════════ cycle %d ════════", cycle)
        before = count_clips(processed)
        require_proxy(a, log, wait=True)
        proxy_dropped = False
        try:
            run([PY, "03_download_and_process.py", "--limit", str(a.cycle_clips), "--workers", str(a.workers),
                 "--min-free-gb", str(a.min_free_gb), "--require-layout-check",
                 "--min-video-gameplay-frac", str(a.min_video_gameplay_frac), *net_args(a)], log, a.dry_run)
        except StepFailed as e:
            if "exit 3" not in str(e):
                raise
            log.warning("stage 03 stopped because the proxy went down; packing what we have, then waiting")
            proxy_dropped = True
        new = count_clips(processed) - before
        log.info("cycle %d: %d new clips processed", cycle, new)
        if count_clips(processed) or a.dry_run:
            run([PY, "04_caption_frames.py"], log, a.dry_run)
            run([PY, "05_pack_webdataset.py", "--append", "--delete-processed"], log, a.dry_run)
        run([PY, "03_download_and_process.py", "--status"], log, a.dry_run)
        if a.dry_run or (a.max_cycles and cycle >= a.max_cycles):
            break
        if new == 0 and not proxy_dropped:
            log.info("No new clips this cycle: either everything is done, the disk floor (--min-free-gb %s) is "
                     "reached even after packing, or downloads are blocked (see 03's log). Stopping.", a.min_free_gb)
            break
    run([PY, "verify_dataset.py", "--debug-videos", "2"], log, a.dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["smoke", "bulk"])
    ap.add_argument("--proxy", default=None, help="e.g. http://127.0.0.1:8899 (passed to 02c and 03)")
    ap.add_argument("--cookies", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--clips", type=int, default=20, help="smoke: clips to process")
    ap.add_argument("--cycle-clips", type=int, default=20000, help="bulk: clips per download/pack cycle (~110 h)")
    ap.add_argument("--max-cycles", type=int, default=0, help="bulk: stop after N cycles (0 = until done)")
    ap.add_argument("--min-free-gb", type=float, default=60.0)
    ap.add_argument("--min-video-gameplay-frac", type=float, default=0.5)
    ap.add_argument("--skip-layout", action="store_true", help="don't run 02c (already done)")
    ap.add_argument("--layout-model", default=None,
                    help="VLM for 02c, e.g. Qwen/Qwen2.5-VL-32B-Instruct-AWQ (better layouts than the 7B default)")
    ap.add_argument("--resample", action="store_true", help="re-run stage 02 first")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    a = ap.parse_args()
    log = setup_logging("run_pipeline")
    log.info("NG_DATA_ROOT=%s mode=%s", DATA_ROOT, a.mode)
    try:
        smoke(a, log) if a.mode == "smoke" else bulk(a, log)
    except StepFailed as e:
        log.error("%s\nEverything is resumable: fix the issue and re-run the same command.", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
