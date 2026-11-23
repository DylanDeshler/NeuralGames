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

import atexit
import json
import urllib.request

from ng_common import DATA_ROOT, DEFAULT_VLM, setup_logging

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


def bucket_args(a) -> list[str]:
    return ["--buckets", a.buckets] if a.buckets else []


def qc_args(a) -> list[str]:
    return ["--qc-frames", str(a.qc_frames)] if a.qc_frames else []


def pack_args(a) -> list[str]:
    return ["--min-gameplay-frac", str(a.min_gameplay_frac)] if a.min_gameplay_frac is not None else []


def vlm_args(a) -> list[str]:
    """Route 02c / 04 to the shared model server when one is configured."""
    if a.vlm_server:
        return ["--backend", "openai", "--api-base", a.vlm_server, "--model", a.vlm_model]
    return ["--model", a.vlm_model] if a.vlm_model != DEFAULT_VLM else []


def layout_args(a) -> list[str]:
    if a.vlm_server:
        return vlm_args(a)
    return ["--model", a.layout_model] if a.layout_model else vlm_args(a)


def server_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def start_vlm_server(a, log) -> None:
    """One `vllm serve` for the whole run: 02c and 04 send requests instead of each loading the model."""
    port = 8000
    a.vlm_server = f"http://localhost:{port}/v1"
    if server_ready(a.vlm_server):
        log.info("Using the model server already running at %s", a.vlm_server)
        return
    (DATA_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    out = open(DATA_ROOT / "logs" / "vllm_server.log", "a")
    cmd = ["vllm", "serve", a.vlm_model, "--port", str(port), "--max-model-len", "8192",
           "--limit-mm-per-prompt", json.dumps({"image": max(6, a.qc_frames or 4)}),
           "--gpu-memory-utilization", "0.90", *(["--enforce-eager"] if a.server_eager else [])]
    log.info("▶ starting model server: %s (log: logs/vllm_server.log)", " ".join(cmd))
    if a.dry_run:
        return
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    t0 = time.time()
    while not server_ready(a.vlm_server):
        if proc.poll() is not None:
            raise StepFailed("model server exited during startup; see logs/vllm_server.log")
        if time.time() - t0 > 1800:
            raise StepFailed("model server not ready after 30 min; see logs/vllm_server.log")
        time.sleep(10)
    log.info("Model server ready after %.0fs", time.time() - t0)


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


def require_js_runtime(a, log) -> None:
    """Without a JS runtime yt-dlp can't solve YouTube's challenge and gets real-time-throttled stream URLs
    (a 20 s clip takes ~20 s to fetch). Refuse to start downloading in that state."""
    if a.dry_run or a.allow_no_js:
        return
    if shutil.which("deno") is None:
        raise StepFailed("deno (JavaScript runtime for yt-dlp) not found on PATH: YouTube downloads would be throttled "
                         "to real time. Install: curl -fsSL https://deno.land/install.sh | sh ; export "
                         "PATH=$HOME/.deno/bin:$PATH ; pip install -U 'yt-dlp[default]'  (or pass --allow-no-js)")


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
        wipe = (wds, debug, tmp) if (a.resume and man.exists()) else (processed, wds, debug, tmp)
        for p in wipe:  # --resume keeps already-downloaded smoke clips (03 skips them; 04/05 redo the rest)
            shutil.rmtree(p, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        if not (a.resume and man.exists()):
            shutil.copy(man_main, man)  # isolated copy: 04/05 write into it, the real manifest stays untouched
        debug.mkdir(parents=True, exist_ok=True)
    failed = None
    try:
        require_js_runtime(a, log)
        require_proxy(a, log, wait=False)
        if a.start_vlm_server:
            start_vlm_server(a, log)
        if not a.skip_layout:  # 02c results are shared with bulk (same videos, same answers)
            run([PY, "02c_detect_layout.py", "--manifest", str(man), "--limit", str(max(40, 2 * a.clips)),
                 "--workers", str(a.workers), *layout_args(a), *net_args(a)], log, a.dry_run)
        run([PY, "03_download_and_process.py", "--manifest", str(man), "--processed-dir", str(processed),
             "--tmp-dir", str(tmp), "--limit", str(a.clips), "--workers", str(a.workers),
             "--require-layout-check", "--min-video-gameplay-frac", str(a.min_video_gameplay_frac), *bucket_args(a), *net_args(a)],
            log, a.dry_run)
        if not a.dry_run and count_clips(processed) == 0:
            raise StepFailed("stage 03 produced no clips (see logs/03_download_and_process.log; proxy / YouTube access?)")
        run([PY, "04_caption_frames.py", "--manifest", str(man), "--processed-dir", str(processed), *qc_args(a), *vlm_args(a)], log, a.dry_run)
        run([PY, "05_pack_webdataset.py", "--manifest", str(man), "--processed-dir", str(processed),
             "--out", str(wds), "--overwrite", *pack_args(a)], log, a.dry_run)
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
    require_js_runtime(a, log)
    if a.start_vlm_server:
        start_vlm_server(a, log)
    bg02c = None
    if not a.skip_layout:
        require_proxy(a, log, wait=True)
        cmd = [PY, "02c_detect_layout.py", "--workers", str(a.workers), *layout_args(a), *net_args(a)]
        if a.vlm_server and not a.dry_run:  # shared server: check layouts in the background while downloading
            log.info("▶ (background) %s  -> logs/console_02c_detect_layout.log", " ".join(cmd))
            bg02c = subprocess.Popen(cmd, cwd=HERE, stdout=open(DATA_ROOT / "logs" / "console_02c_detect_layout.log", "a"),
                                     stderr=subprocess.STDOUT)
            atexit.register(lambda: bg02c.poll() is None and bg02c.terminate())
        else:
            run(cmd, log, a.dry_run)
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
                 "--min-video-gameplay-frac", str(a.min_video_gameplay_frac), *bucket_args(a), *net_args(a)], log, a.dry_run)
        except StepFailed as e:
            if "exit 3" not in str(e):
                raise
            log.warning("stage 03 stopped because the proxy went down; packing what we have, then waiting")
            proxy_dropped = True
        new = count_clips(processed) - before
        log.info("cycle %d: %d new clips processed", cycle, new)
        if count_clips(processed) or a.dry_run:
            run([PY, "04_caption_frames.py", *qc_args(a), *vlm_args(a)], log, a.dry_run)
            run([PY, "05_pack_webdataset.py", "--append", "--delete-processed", *pack_args(a)], log, a.dry_run)
            if a.push_to_hub:  # check this cycle's shards before they leave the machine, then upload + free disk
                run([PY, "verify_dataset.py", "--num-shards", "5", "--debug-videos", "0",
                     "--out-dir", str(DATA_ROOT / "debug" / f"cycle_{cycle:03d}")], log, a.dry_run)
                run([PY, "06_push_to_hub.py", "--repo-id", a.push_to_hub, "--delete-local"], log, a.dry_run)
        run([PY, "03_download_and_process.py", "--status"], log, a.dry_run)
        if a.dry_run or (a.max_cycles and cycle >= a.max_cycles):
            break
        if new == 0 and bg02c is not None and bg02c.poll() is None:
            log.info("No new clips yet: waiting for the background layout check (02c) to cover more videos")
            time.sleep(120)
            continue
        if new == 0 and not proxy_dropped:
            log.info("No new clips this cycle: either everything is done, the disk floor (--min-free-gb %s) is "
                     "reached even after packing, or downloads are blocked (see 03's log). Stopping.", a.min_free_gb)
            break
    if not a.push_to_hub:  # with --push-to-hub every cycle was verified before upload
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
    ap.add_argument("--buckets", default=None,
                    help="passed to stage 03, e.g. '21x12' (single 16:9 canvas, others padded) or '21x12,19x14'")
    ap.add_argument("--qc-frames", type=int, default=None, help="passed to stage 04 (frames per clip for the QC)")
    ap.add_argument("--min-gameplay-frac", type=float, default=None, help="passed to stage 05")
    ap.add_argument("--start-vlm-server", action="store_true",
                    help="start one `vllm serve` for the run (02c and 04 share it; bulk then runs 02c in the background)")
    ap.add_argument("--vlm-server", default=None, help="use an already running OpenAI-compatible server, e.g. http://localhost:8000/v1")
    ap.add_argument("--vlm-model", default=DEFAULT_VLM)
    ap.add_argument("--server-eager", action=argparse.BooleanOptionalAction, default=True,
                    help="start the server with --enforce-eager (robust startup on this machine)")
    ap.add_argument("--layout-model", default=None,
                    help="VLM for 02c, e.g. Qwen/Qwen2.5-VL-32B-Instruct-AWQ (better layouts than the 7B default)")
    ap.add_argument("--resample", action="store_true", help="re-run stage 02 first")
    ap.add_argument("--resume", action="store_true", help="smoke: keep already-processed smoke clips instead of starting over")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--allow-no-js", action="store_true", help="run even without deno (downloads will be throttled)")
    ap.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                    help="bulk: after each cycle verify + upload new shards to this PRIVATE HF dataset repo and delete "
                         "them locally (disk then only holds one cycle)")
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
