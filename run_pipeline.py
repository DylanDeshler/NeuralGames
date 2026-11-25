#!/usr/bin/env python
"""Run the download → process → QC → pack stages in order (02c, 03, 04, 05, verify).

  smoke  ~20 clips end to end in an isolated <data-root>/smoke/ (own manifest copy, processed dir, shards),
         then bundles the debug grids/videos + inspect report into smoke/smoke_results.tgz.
  bulk   02c layout detection for all videos (resumable), then repeated cycles of
             03 download/process (--cycle-clips)  →  04 VLM caption+QC  →  05 pack (--append --delete-processed)
         until the manifest is done or the disk floor is reached. Safe to stop and re-run at any time.

  build  CPU machine, no GPU/VLM: repeated cycles of 03 → 05 (no captions) → verify → 06 push. Pulls the
         layout file (02c) and manifest from the Hub repo's meta/ folder each cycle. Several machines:
         --machine-shard i/N (disjoint videos, tagged shard names).
  gpu    GPU machine: model server + 02c for all videos (pushes meta/layout_overrides.parquet every few
         minutes for the build machines) + 07 caption pass over shards as they arrive on the Hub.

Examples:
  python run_pipeline.py --data-root /data/games gpu   --start-vlm-server --push-to-hub <user>/nitrogen-2d-wm
  python run_pipeline.py --data-root /data/games build --push-to-hub <user>/nitrogen-2d-wm --workers 32 [--machine-shard 0/2]
  python run_pipeline.py --data-root /data/games smoke --proxy http://127.0.0.1:8899
  python run_pipeline.py --data-root /data/games bulk  --proxy http://127.0.0.1:8899 --workers 8 --min-free-gb 60
  python run_pipeline.py --data-root /data/games bulk  --dry-run                     # print the commands only

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
import os
import urllib.request

from ng_common import add_data_root_arg, DATA_ROOT, DEFAULT_VLM, REPO_COOKIES, setup_logging

HERE = Path(__file__).resolve().parent
PY = sys.executable


class StepFailed(RuntimeError):
    pass


def with_root(cmd: list[str]) -> list[str]:
    """Every pipeline script requires --data-root: pass this run's root explicitly (never via the environment)."""
    if len(cmd) > 1 and cmd[0] == PY and cmd[1].endswith(".py") and "--data-root" not in cmd:
        return [cmd[0], cmd[1], "--data-root", str(DATA_ROOT), *cmd[2:]]
    return cmd


def run(cmd: list[str], log, dry: bool, stdout_path: Path | None = None) -> None:
    cmd = with_root(cmd)
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
    """Output-geometry/encode options shared by every stage-03 call."""
    out = ["--buckets", a.buckets] if a.buckets else []
    if a.crf is not None:
        out += ["--enc-args", f"-preset medium -crf {a.crf} -pix_fmt yuv444p"]
    return out


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
    vllm_bin = Path(PY).parent / "vllm"  # same env as this Python: no reliance on PATH after a reboot
    cmd = [str(vllm_bin) if vllm_bin.exists() else "vllm", "serve", a.vlm_model, "--port", str(port), "--max-model-len", "8192",
           "--limit-mm-per-prompt", json.dumps({"image": max(6, a.qc_frames or 4)}),
           "--gpu-memory-utilization", "0.90", *(["--enforce-eager"] if a.server_eager else [])]
    log.info("▶ starting model server: %s (log: logs/vllm_server.log)", " ".join(cmd))
    if a.dry_run:
        return
    out.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} starting: {' '.join(cmd)}\n")
    out.flush()
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    a._vlm_proc = proc
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


def check_direct_access(a, log) -> None:
    """Direct (no-proxy) downloads from a datacenter IP need cookies + the bgutil PO-token server."""
    if a.dry_run or a.proxy:
        return
    if a.cookies and not os.path.isfile(a.cookies):
        raise StepFailed(f"--cookies {a.cookies} does not exist")
    if not a.cookies:
        if REPO_COOKIES.is_file():
            log.info("Using repo cookies %s (stages copy it to <data-root>/cookies_runtime.txt)", REPO_COOKIES.name)
        else:
            log.warning("No --proxy and no cookies (--cookies or %s): YouTube usually bot-blocks datacenter IPs. "
                        "See README 'Cloud access with cookies' / tools/setup_youtube_access.sh.", REPO_COOKIES.name)
    import socket
    try:
        socket.create_connection(("127.0.0.1", 4416), timeout=3).close()
    except OSError:
        log.warning("bgutil PO-token server not reachable on 127.0.0.1:4416; downloads may be blocked or throttled. "
                    "Start it: bash tools/setup_youtube_access.sh  (or docker start bgutil-provider)")


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


# ============================================================================ Hub meta sync (gpu <-> build)
META_FILES = {"layout_overrides.parquet": DATA_ROOT / "layout_overrides.parquet",
              "manifest_2d.parquet": DATA_ROOT / "manifest_2d.parquet"}


def require_hub_write(repo: str, log) -> None:
    """Fail in seconds (not after a 5-min model-server start) if we can't write to the Hub repo."""
    from huggingface_hub import HfApi
    try:
        who = HfApi(token=os.environ.get("HF_TOKEN")).whoami()
    except Exception as e:
        raise StepFailed(f"Not logged in to Hugging Face ({type(e).__name__}). Run `hf auth login` with a WRITE token "
                         f"(or check that a stale HF_TOKEN variable isn't set).")
    name = who.get("name")
    orgs = {o.get("name") for o in who.get("orgs", [])}
    if repo.split("/")[0] not in {name} | orgs:
        raise StepFailed(f"Logged in to Hugging Face as {name!r}, which can't create {repo!r}")
    log.info("Hugging Face: logged in as %s", name)


def hub_push_meta(repo: str, names: list[str], log) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    ops = [CommitOperationAdd(f"meta/{n}", str(META_FILES[n])) for n in names if META_FILES[n].exists()]
    if not ops:
        return
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.create_commit(repo, repo_type="dataset", operations=ops, commit_message="update " + ", ".join(names))
    log.info("pushed %s to %s/meta", ", ".join(n for n in names if META_FILES[n].exists()), repo)


def hub_pull_packed(repo: str, log) -> None:
    """Every machine's packed-uuid lists (meta/packed/*.txt) -> webdataset/_hub_packed/, which stage 03 treats as
    done, so no machine re-downloads a clip that is already in a shard on the Hub."""
    from huggingface_hub import HfApi, hf_hub_download
    try:
        files = [f for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(repo, repo_type="dataset")
                 if f.startswith("meta/packed/") and f.endswith(".txt")]
    except Exception as e:
        log.warning("could not list %s (%s)", repo, type(e).__name__)
        return
    dst = DATA_ROOT / "webdataset" / "_hub_packed"
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=DATA_ROOT / "hub_meta")
        shutil.copy(p, dst / Path(f).name)
    if files:
        log.info("pulled %d packed-uuid lists from the Hub", len(files))


def hub_pull_meta(repo: str, names: list[str], log) -> None:
    from huggingface_hub import hf_hub_download
    for n in names:
        try:
            p = hf_hub_download(repo, f"meta/{n}", repo_type="dataset", local_dir=DATA_ROOT / "hub_meta")
        except Exception as e:  # not pushed yet: keep whatever is local
            log.warning("could not pull meta/%s from %s (%s)", n, repo, type(e).__name__)
            continue
        dst = META_FILES[n]
        if not dst.exists() or Path(p).read_bytes() != dst.read_bytes():
            shutil.copy(p, dst)
            log.info("updated %s from the Hub", dst.name)


# ============================================================================ build (CPU) / gpu modes
def build(a, log):
    """CPU-only: download/process → pack without captions → push. Captions come later from 07 on a GPU box."""
    if not a.push_to_hub:
        raise StepFailed("build mode needs --push-to-hub REPO_ID (layout file comes from, and shards go to, the Hub)")
    if not a.dry_run:
        require_hub_write(a.push_to_hub, log)
    processed = DATA_ROOT / "processed"
    require_js_runtime(a, log)
    check_direct_access(a, log)
    tag = a.machine_tag or (f"m{a.machine_shard.split('/')[0]}" if a.machine_shard else "")
    shard_args = ["--shard", a.machine_shard] if a.machine_shard else []
    idle, cycle = 0, 0
    while True:
        cycle += 1
        log.info("════════ build cycle %d ════════", cycle)
        if not a.dry_run:
            hub_pull_meta(a.push_to_hub, ["manifest_2d.parquet", "layout_overrides.parquet"], log)
            hub_pull_packed(a.push_to_hub, log)
        before = count_clips(processed)
        todo = max(0, a.cycle_clips - before)
        if todo:
            run([PY, "03_download_and_process.py", "--limit", str(todo), "--workers", str(a.workers),
                 "--min-free-gb", str(a.min_free_gb), "--require-layout-check", "--hwaccel", "none",
                 "--video-group-window", str(a.video_group_window),
                 "--min-video-gameplay-frac", str(a.min_video_gameplay_frac), *shard_args, *bucket_args(a), *net_args(a)],
                log, a.dry_run)
        new = count_clips(processed)
        if new or a.dry_run:
            run([PY, "05_pack_webdataset.py", "--append", "--delete-processed", "--allow-missing-caption",
                 "--min-gameplay-frac", "0", "--no-require-frame0-gameplay", *(["--tag", tag] if tag else [])],
                log, a.dry_run)
            run([PY, "verify_dataset.py", "--num-shards", "5", "--debug-videos", "0", "--allow-missing-caption", "--allow-missing-split",
                 "--out-dir", str(DATA_ROOT / "debug" / f"build_{tag or 'm'}_{cycle:03d}")], log, a.dry_run)
            run([PY, "06_push_to_hub.py", "--repo-id", a.push_to_hub, "--delete-local", *(["--tag", tag] if tag else [])],
                log, a.dry_run)
        run([PY, "03_download_and_process.py", "--status", *shard_args], log, a.dry_run)
        if a.dry_run or (a.max_cycles and cycle >= a.max_cycles):
            break
        idle = idle + 1 if new == 0 else 0
        if idle >= a.idle_cycles:
            log.info("Nothing new for %d cycles: the manifest (with the current layout file) is done.", idle)
            break
        if new == 0:  # 02c may still be adding videos on the GPU box
            log.info("No clips this cycle: waiting 10 min for more layout checks from the GPU box")
            time.sleep(600)


def pd_read(path, columns=None):
    import pandas as pd
    return pd.read_parquet(path, columns=columns)


def layout_count(lo: Path) -> int:
    try:
        return len(pd_read(lo, ["video_id"])) if lo.exists() else 0
    except Exception:  # being rewritten right now
        return -1


def caption_progress() -> tuple[int, int, str]:
    """(shards, clips, last rate) from 07's log."""
    p = DATA_ROOT / "logs" / "07_caption_shards.log"
    shards = clips = 0
    rate = "-"
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            if " clips in " in line and "clips/s" in line:
                shards += 1
                try:
                    clips += int(line.split(": ")[-1].split(" clips in ")[0].split()[-1])
                    rate = line.split("(")[1].split(")")[0]
                except (IndexError, ValueError):
                    pass
    return shards, clips, rate


def gpu_status(log, procs: dict, lo: Path, n_videos, n0: int, t_start: float, server: str | None) -> None:
    n = layout_count(lo)
    parts = []
    if "02c" in procs:
        if n >= 0 and n_videos:
            rate = (n - n0) / max(1e-6, (time.time() - t_start) / 3600)
            eta = (n_videos - n) / rate if rate > 0 else float("inf")
            parts.append(f"02c {n:,}/{n_videos:,} videos ({100 * n / n_videos:.0f}%), {rate:.0f}/h, ETA "
                         + (f"{eta:.1f} h" if eta != float("inf") else "?")
                         + ("" if procs["02c"].poll() is None else f" [exited {procs['02c'].returncode}]"))
        else:
            parts.append(f"02c {max(n, 0):,} videos")
    sh, cl, r = caption_progress()
    parts.append(f"captions {sh} shards / {cl:,} clips (last {r})"
                 + ("" if procs["07"].poll() is None else f" [exited {procs['07'].returncode}]"))
    parts.append("server " + ("up" if server and server_ready(server) else "DOWN"))
    log.info(" | ".join(parts))


def gpu(a, log):
    """Model server + 02c (layouts, pushed for the build machines) + 07 captions over shards on the Hub."""
    if not a.push_to_hub:
        raise StepFailed("gpu mode needs --push-to-hub REPO_ID")
    if not a.dry_run:
        require_hub_write(a.push_to_hub, log)
    if a.start_vlm_server:
        start_vlm_server(a, log)
    if not a.vlm_server and not a.dry_run:
        raise StepFailed("gpu mode needs a model server: --start-vlm-server or --vlm-server URL")
    if not a.dry_run:
        hub_push_meta(a.push_to_hub, ["manifest_2d.parquet", "layout_overrides.parquet"], log)
    procs = {}
    if not a.skip_layout:
        cmd = [PY, "02c_detect_layout.py", "--workers", str(a.workers), *layout_args(a), *net_args(a)]
        log.info("▶ (background) %s", " ".join(with_root(cmd)))
        if not a.dry_run:
            procs["02c"] = subprocess.Popen(with_root(cmd), cwd=HERE, stderr=subprocess.STDOUT,
                                            stdout=open(DATA_ROOT / "logs" / "console_02c_detect_layout.log", "a"))
    cmd = [PY, "07_caption_shards.py", "--repo-id", a.push_to_hub, "--follow", "--idle-exit-min", "1440", "--backend", "openai",
           "--api-base", a.vlm_server or "http://localhost:8000/v1", "--model", a.vlm_model, *qc_args(a)]
    log.info("▶ (background) %s", " ".join(with_root(cmd)))
    if a.dry_run:
        return
    procs["07"] = subprocess.Popen(with_root(cmd), cwd=HERE, stderr=subprocess.STDOUT,
                                   stdout=open(DATA_ROOT / "logs" / "console_07_caption_shards.log", "a"))
    for p in procs.values():
        atexit.register(lambda p=p: p.poll() is None and p.terminate())
    lo = META_FILES["layout_overrides.parquet"]
    last = lo.stat().st_mtime if lo.exists() else 0
    n_videos = None
    try:
        n_videos = pd_read(META_FILES["manifest_2d.parquet"], ["video_id"]).video_id.nunique()
    except Exception:
        pass
    t_start, n0, last_push = time.time(), layout_count(lo), time.time()
    log.info("Status every %d s (details: tail -f %s/logs/02c_detect_layout.log  and  .../07_caption_shards.log)",
             a.status_every, DATA_ROOT)
    launch = {}  # name -> command, for restarts
    if "02c" in procs:
        launch["02c"] = (procs["02c"].args, "console_02c_detect_layout.log")
    launch["07"] = (procs["07"].args, "console_07_caption_shards.log")
    restarts = {"server": 0, "02c": 0, "07": 0}
    down_checks = 0
    while True:
        time.sleep(a.status_every)
        gpu_status(log, procs, lo, n_videos, n0, t_start, a.vlm_server)
        # watchdog: model server (02c / 07 pause on their own while it is down, nothing gets recorded)
        if a.start_vlm_server:
            vp = getattr(a, "_vlm_proc", None)
            down_checks = 0 if server_ready(a.vlm_server) else down_checks + 1
            if (vp is not None and vp.poll() is not None) or down_checks >= 3:
                restarts["server"] += 1
                log.error("model server is down (exit %s): restarting it (restart #%d; crash log: logs/vllm_server.log)",
                          vp.poll() if vp else "?", restarts["server"])
                if vp is not None and vp.poll() is None:
                    vp.terminate()
                    try:
                        vp.wait(60)
                    except subprocess.TimeoutExpired:
                        vp.kill()
                try:
                    start_vlm_server(a, log)
                    down_checks = 0
                except StepFailed as e:
                    log.error("server restart failed: %s (retrying next check)", e)
        # watchdog: background stages that died with an error are restarted (both resume from their outputs)
        for name, (cmd, logname) in launch.items():
            p = procs.get(name)
            if p is not None and p.poll() is not None:
                if p.returncode == 0:  # finished normally (02c: all videos checked; 07: idle limit reached)
                    continue
                restarts[name] += 1
                if restarts[name] > 20:
                    log.error("%s keeps failing (exit %s); giving up on it. See logs/%s", name, p.returncode, logname)
                    continue
                log.error("%s exited with %s: restarting it (#%d). See logs/%s", name, p.returncode, restarts[name], logname)
                procs[name] = subprocess.Popen(cmd, cwd=HERE, stderr=subprocess.STDOUT,
                                               stdout=open(DATA_ROOT / "logs" / logname, "a"))
        if all(procs[n].poll() is not None and (procs[n].returncode == 0 or restarts[n] > 20) for n in launch):
            break
        if time.time() - last_push >= 600 and lo.exists() and lo.stat().st_mtime > last:
            last, last_push = lo.stat().st_mtime, time.time()  # share new layout checks with the build machines
            try:
                hub_push_meta(a.push_to_hub, ["layout_overrides.parquet"], log)
            except Exception as e:
                log.warning("layout push failed (will retry): %s", e)
        if "02c" in procs and procs["02c"].poll() is not None and not getattr(procs["02c"], "_reported", False):
            procs["02c"]._reported = True
            log.info("02c finished (exit %s): layout file pushed within 10 min; the CPU build can start after that",
                     procs["02c"].returncode)
    if lo.exists() and lo.stat().st_mtime > last:
        hub_push_meta(a.push_to_hub, ["layout_overrides.parquet"], log)
    for name, p in procs.items():
        log.info("%s exited with %s", name, p.returncode)


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
        check_direct_access(a, log)
        require_proxy(a, log, wait=False)
        if a.start_vlm_server:
            start_vlm_server(a, log)
        if not a.skip_layout:  # 02c results are shared with bulk (same videos, same answers)
            run([PY, "02c_detect_layout.py", "--manifest", str(man), "--limit", str(max(40, 2 * a.clips)),
                 "--workers", str(a.workers), *layout_args(a), *net_args(a)], log, a.dry_run)
        run([PY, "03_download_and_process.py", "--manifest", str(man), "--processed-dir", str(processed),
             "--tmp-dir", str(tmp), "--webdataset-dir", str(wds), "--limit", str(a.clips), "--workers", str(a.workers),
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
        subprocess.run(with_root([PY, "inspect_clips.py", "--processed-dir", str(processed), "--manifest", str(man)]),
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
    if a.push_to_hub and not a.dry_run:
        require_hub_write(a.push_to_hub, log)
    if a.resample:
        resample(a, log)
    require_js_runtime(a, log)
    check_direct_access(a, log)
    if a.start_vlm_server:
        start_vlm_server(a, log)
    bg02c = None
    if not a.skip_layout:
        require_proxy(a, log, wait=True)
        cmd = [PY, "02c_detect_layout.py", "--workers", str(a.workers), *layout_args(a), *net_args(a)]
        if a.vlm_server and not a.dry_run:  # shared server: check layouts in the background while downloading
            log.info("▶ (background) %s  -> logs/console_02c_detect_layout.log", " ".join(with_root(cmd)))
            bg02c = subprocess.Popen(with_root(cmd), cwd=HERE, stdout=open(DATA_ROOT / "logs" / "console_02c_detect_layout.log", "a"),
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
        # clips already on disk (downloaded before a restart, not packed yet) count toward this cycle, so an
        # interrupted cycle is finished and packed instead of growing by another --cycle-clips every restart
        todo = max(0, a.cycle_clips - before)
        if before:
            log.info("%d unpacked clips from an earlier run on disk: downloading %d more this cycle", before, todo)
        try:
            if todo:
                run([PY, "03_download_and_process.py", "--limit", str(todo), "--workers", str(a.workers),
                     "--min-free-gb", str(a.min_free_gb), "--require-layout-check",
                     "--min-video-gameplay-frac", str(a.min_video_gameplay_frac), *bucket_args(a), *net_args(a)], log, a.dry_run)
        except StepFailed as e:
            if "exit 3" not in str(e):
                raise
            log.warning("stage 03 stopped because the proxy went down; packing what we have, then waiting")
            proxy_dropped = True
        new = count_clips(processed) - before + (before if not todo else 0)  # resumed full cycle: pack it now
        log.info("cycle %d: %d new clips processed", cycle, new)
        if count_clips(processed) or a.dry_run:
            run([PY, "04_caption_frames.py", *qc_args(a), *vlm_args(a)], log, a.dry_run)
            run([PY, "05_pack_webdataset.py", "--append", "--delete-processed", *pack_args(a)], log, a.dry_run)
            if a.push_to_hub:  # check this cycle's shards before they leave the machine, then upload + free disk
                run([PY, "verify_dataset.py", "--num-shards", "5", "--debug-videos", "0", "--allow-missing-split",
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
    add_data_root_arg(ap)
    ap.add_argument("mode", choices=["smoke", "bulk", "build", "gpu"])
    ap.add_argument("--proxy", default=None, help="e.g. http://127.0.0.1:8899 (passed to 02c and 03)")
    ap.add_argument("--cookies", default=None, help="default: repo www.youtube.com_cookies.txt if present")
    ap.add_argument("--workers", type=int, default=8, help="download/process threads (build: ~1 per vCPU)")
    ap.add_argument("--machine-shard", default=None, help="build: i/N, this machine's disjoint share of the videos")
    ap.add_argument("--machine-tag", default=None, help="build: tag in shard names (default m<i> with --machine-shard)")
    ap.add_argument("--video-group-window", type=int, default=200000,
                    help="build: process a video's chunks back to back within windows of N clips (fewer URL resolves)")
    ap.add_argument("--status-every", type=int, default=120, help="gpu: seconds between status lines")
    ap.add_argument("--idle-cycles", type=int, default=6, help="build: stop after N cycles in a row with nothing new")
    ap.add_argument("--clips", type=int, default=20, help="smoke: clips to process")
    ap.add_argument("--cycle-clips", type=int, default=8000,
                    help="bulk: clips per download/caption/pack/push cycle (~44 h of video, ~3 h of downloading at ~45 clips/min)")
    ap.add_argument("--max-cycles", type=int, default=0, help="bulk: stop after N cycles (0 = until done)")
    ap.add_argument("--min-free-gb", type=float, default=60.0)
    ap.add_argument("--min-video-gameplay-frac", type=float, default=0.5)
    ap.add_argument("--skip-layout", action="store_true", help="don't run 02c (already done)")
    ap.add_argument("--buckets", default=None,
                    help="passed to stage 03, e.g. '21x12' (single 16:9 canvas, others padded) or '21x12,19x14'")
    ap.add_argument("--crf", type=int, default=None, help="x264 CRF for stage 03 (default 18; 20 is ~20-25%% smaller)")
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
    log.info("data root=%s mode=%s", DATA_ROOT, a.mode)
    try:
        {"smoke": smoke, "bulk": bulk, "build": build, "gpu": gpu}[a.mode](a, log)
    except StepFailed as e:
        log.error("%s\nEverything is resumable: fix the issue and re-run the same command.", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
