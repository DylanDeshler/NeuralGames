#!/usr/bin/env python
"""Stage 02c (GPU + network): find the real game screen inside stream layouts, once per video.

Why: many NitroGen videos are stream captures whose frame also contains Twitch chat, a webcam, alert
banners or decorative panels, and NitroGen's metadata has no `bbox_game_area` for them. Without this
step, stage 03 would encode chat text and the streamer's face into the training frames.

A stream's layout is fixed for the whole video, so per video we:
  1. grab --frames-per-video frames (spread over its manifest chunks) through the same yt-dlp/ffmpeg
     path as stage 03 (respects --proxy / --cookies);
  2. ask Qwen2.5-VL (grounding) for {game_box, overlay boxes inside the game, screen type};
  3. take the per-coordinate median over frames and write data/layout_overrides.parquet.

Stage 03 then uses `game_area` for videos whose metadata has none, and adds `overlays` to the
inpainting boxes (like bbox_others). Boxes covering ~the whole frame are stored as "full frame" (no crop).

Resumable: videos already in the output are skipped.

Example:
  python 02c_detect_layout.py --proxy http://127.0.0.1:8899 --workers 8
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ng_common import DATA_ROOT, atomic_write_parquet, setup_logging

_spec = importlib.util.spec_from_file_location("stage03", Path(__file__).with_name("03_download_and_process.py"))
S3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S3)

PROMPT = """This is a frame from a gameplay video. It may be a stream layout: besides the game itself the frame can
contain a webcam, a chat panel, follower/donation alerts, a controller/input display, logos, text panels,
decorative borders or a blurred/zoomed copy of the game used as a background.

Return JSON only:
{"game_box": [x1, y1, x2, y2],
 "overlays_in_game": [[x1, y1, x2, y2], ...],
 "screen": "gameplay|menu|title|loading|cutscene|map_or_inventory|black|not_a_game",
 "is_stream_layout": true|false}

- game_box: the rectangle of the actual game screen only (exclude webcam, chat, panels, borders, background blur).
  If the game fills the whole frame, return the full frame.
- overlays_in_game: webcam / chat / logo / input-display boxes that sit ON TOP of the game screen (may be empty).
- Coordinates are pixels in this image ({w}x{h})."""


def fit28(w: int, h: int, long_side: int = 560) -> tuple[int, int]:
    """Resize target whose sides are multiples of 28, so Qwen2.5-VL sees the image unresized and its
    pixel coordinates map back exactly."""
    s = long_side / max(w, h)
    return max(28, int(round(w * s / 28)) * 28), max(28, int(round(h * s / 28)) * 28)


def to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse(text: str, w: int, h: int) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

    def box(b):
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            return None
        x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
        y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
        if x2 - x1 < 0.05 * w or y2 - y1 < 0.05 * h:
            return None
        return [x1 / w, y1 / h, x2 / w, y2 / h]  # relative

    g = box(d.get("game_box"))
    if g is None:
        return None
    ovs = [b for b in (box(o) for o in (d.get("overlays_in_game") or [])) if b is not None]
    return {"game": g, "overlays": ovs, "screen": str(d.get("screen", "")).lower(),
            "stream": bool(d.get("is_stream_layout", False))}


# ============================================================================ frame grabbing
def grab_frames(vid: str, url: str, times: list[float], tmp: Path, args, resolver, limiter) -> tuple[list[Image.Image], str]:
    """Single frames at `times`. Direct stream URL with hard network timeouts; if that fails, yt-dlp
    section downloads (tried once per video, not per frame). Returns (frames, method)."""
    frames: list[Image.Image] = []
    fmt = S3.format_selector(15.0, args.max_src_height)
    info = None
    try:
        info = resolver.resolve(vid, url, fmt, args.max_src_height, limiter)
    except S3.PermanentError:
        raise
    except Exception:
        info = None
    method = "direct"
    use_direct = bool(info and info.get("url"))
    for i, t in enumerate(times):
        out = tmp / f"f{i}.png"
        ok = False
        if use_direct:
            hdr = "".join(f"{k}: {v}\r\n" for k, v in (info.get("headers") or {}).items())
            cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                   "-rw_timeout", str(int(args.net_timeout * 1e6)),
                   *(["-headers", hdr] if hdr else []),
                   *(["-http_proxy", args.proxy] if args.proxy and args.proxy.startswith("http") else []),
                   "-ss", f"{t:.2f}", "-i", info["url"], "-frames:v", "1", str(out)]
            try:
                ok = subprocess.run(cmd, capture_output=True, timeout=args.net_timeout * 2).returncode == 0 and out.exists()
            except subprocess.TimeoutExpired:
                ok = False
            if not ok:
                use_direct = False  # don't keep hammering a URL that failed; switch this video to sections
        if not ok:
            method = "sections"
            try:
                raw = S3.download_section(url, fmt, t, t + 1.5, tmp, args)
                ok = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(raw),
                                     "-frames:v", "1", str(out)], capture_output=True, timeout=30).returncode == 0
                raw.unlink(missing_ok=True)
            except S3.PermanentError:
                raise
            except Exception:
                ok = False
            if not ok:
                break  # both paths failed for this video: give up on it instead of retrying every frame
        if ok and out.exists():
            frames.append(Image.open(out).convert("RGB"))
    return frames, method


def aggregate(results: list[dict], full_thresh: float) -> dict:
    gs = np.array([r["game"] for r in results])
    g = np.median(gs, axis=0)
    spread = float(np.abs(gs - g).max()) if len(gs) > 1 else 0.0
    area = (g[2] - g[0]) * (g[3] - g[1])
    full = (g[0] <= 1 - full_thresh and g[1] <= 1 - full_thresh and g[2] >= full_thresh and g[3] >= full_thresh)
    ovs = [o for r in results for o in r["overlays"]]
    # keep overlays reported in >= half the frames (greedy clustering by IoU)
    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0
    clusters: list[list[list[float]]] = []
    for o in ovs:
        for cl in clusters:
            if iou(o, cl[0]) > 0.5:
                cl.append(o)
                break
        else:
            clusters.append([o])
    need = int(np.ceil(len(results) / 2))
    keep = [np.median(np.array(cl), axis=0).tolist() for cl in clusters if len(cl) >= need]
    screens = [r["screen"] for r in results]
    return {"game_area": None if full else json.dumps({"xtl": g[0], "ytl": g[1], "xbr": g[2], "ybr": g[3]}),
            "game_area_frac": float(area), "box_spread": spread, "overlays": json.dumps(keep),
            "frac_gameplay": float(np.mean([s == "gameplay" for s in screens])),
            "stream_layout": bool(np.mean([r["stream"] for r in results]) >= 0.5), "n_frames": len(results)}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DATA_ROOT / "manifest_2d.parquet"))
    ap.add_argument("--out", default=str(DATA_ROOT / "layout_overrides.parquet"))
    ap.add_argument("--tmp-dir", default=str(DATA_ROOT / "tmp_02c"))
    ap.add_argument("--scope", choices=["missing", "all"], default="missing",
                    help="'missing' = only videos without bbox_game_area in the metadata")
    ap.add_argument("--frames-per-video", type=int, default=3)
    ap.add_argument("--full-thresh", type=float, default=0.97, help="box edges beyond this => full frame, no crop")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32, help="videos per VLM batch")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    # network (same semantics as stage 03)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--cookies", default=None)
    ap.add_argument("--cookies-from-browser", default=None)
    ap.add_argument("--max-starts-per-min", type=float, default=60.0, help="URL resolutions per minute (1 per video)")
    ap.add_argument("--rate-penalty", type=float, default=120.0)
    ap.add_argument("--url-ttl", type=float, default=3 * 3600)
    ap.add_argument("--download-timeout", type=float, default=60)
    ap.add_argument("--net-timeout", type=float, default=20, help="seconds without network progress before a fetch fails")
    ap.add_argument("--max-src-height", type=int, default=480, help="probe-frame resolution (layout needs little detail)")
    args = ap.parse_args()

    log = setup_logging("02c_detect_layout")
    man = pd.read_parquet(args.manifest, columns=["video_id", "url", "start_time", "duration", "chunk_id",
                                                  "bbox_game_area", "download_priority"])
    has_ga = man["bbox_game_area"].map(lambda s: isinstance(S3._jload(s), dict))
    vids = man[~has_ga] if args.scope == "missing" else man
    # most valuable videos first (earliest in the balanced download order)
    order = vids.groupby("video_id")["download_priority"].min().sort_values().index.tolist()
    out_path = Path(args.out)
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["video_id"])
    if len(done):  # retry transient failures; keep successes and permanent failures
        final = done["status"].isin(["ok", "unparsed"]) | done["status"].astype(str).str.startswith("permanent")
        done = done[final]
    todo = [v for v in order if v not in set(done["video_id"])]
    if args.limit:
        todo = todo[: args.limit]
    log.info("Videos: %d in scope (%s), %d done, %d to check", len(order), args.scope, len(done), len(todo))
    if not todo:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=4096, limit_mm_per_prompt={"image": 1}, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem, enforce_eager=args.enforce_eager, seed=0,
              mm_processor_kwargs={"min_pixels": 64 * 28 * 28, "max_pixels": 640 * 28 * 28})
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    resolver, limiter = S3.StreamResolver(args, log), S3.RateLimiter(args.max_starts_per_min)
    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    rows = []

    def fetch(v):
        g = vids[vids.video_id == v].sort_values("chunk_id")
        pick = g.iloc[np.linspace(0, len(g) - 1, min(args.frames_per_video, len(g))).astype(int)]
        times = [float(s) + min(10.0, float(d) / 2) for s, d in zip(pick.start_time, pick.duration)]
        url = g.url.iloc[0] if isinstance(g.url.iloc[0], str) else f"https://www.youtube.com/watch?v={v}"
        tmp = Path(args.tmp_dir) / v
        tmp.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            frames, method = grab_frames(v, url, times, tmp, args, resolver, limiter)
            return v, frames, None, method, time.time() - t0
        except S3.PermanentError as e:
            return v, [], f"permanent: {str(e)[:100]}", "-", time.time() - t0
        except Exception as e:
            return v, [], f"error: {str(e)[:100]}", "-", time.time() - t0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def flush():
        nonlocal done
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            atomic_write_parquet(done, out_path)
            rows.clear()

    def run_vlm(got):
        convs, owners = [], []
        for v, frames, err, _, _ in got:
            if not frames:
                rows.append({"video_id": v, "status": err or "no_frames"})
                continue
            for f in frames:
                w, h = fit28(*f.size)
                convs.append([{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": to_data_uri(f.resize((w, h), Image.Resampling.BICUBIC))}},
                    {"type": "text", "text": PROMPT.replace("{w}", str(w)).replace("{h}", str(h))}]}])
                owners.append((v, w, h))
        outs = llm.chat(convs, sp, use_tqdm=False) if convs else []
        per_vid: dict[str, list[dict]] = {}
        for (v, w, h), o in zip(owners, outs):
            r = parse(o.outputs[0].text, w, h)
            if r:
                per_vid.setdefault(v, []).append(r)
        for v, frames, err, method, secs in got:
            if frames:
                res = per_vid.get(v, [])
                rows.append({"video_id": v, "status": "ok" if res else "unparsed", "fetch_method": method,
                             "fetch_s": round(secs, 1), **(aggregate(res, args.full_thresh) if res else {})})
        flush()

    stats = {"ok": 0, "fail": 0, "direct": 0, "sections": 0, "secs": 0.0}
    ready = []
    pbar = tqdm(total=len(todo), desc="videos", unit="vid")
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, v) for v in todo]
        for f in as_completed(futs):
            r = f.result()
            ready.append(r)
            stats["ok" if r[1] else "fail"] += 1
            if r[3] in ("direct", "sections"):
                stats[r[3]] += 1
            stats["secs"] += r[4]
            n = stats["ok"] + stats["fail"]
            pbar.update(1)
            pbar.set_postfix(ok=stats["ok"], fail=stats["fail"], direct=stats["direct"], sections=stats["sections"],
                             s_per_vid=f"{stats['secs'] / n:.0f}")
            if n <= 3 or (r[2] and stats["fail"] <= 5):
                log.info("%s: %d frames via %s in %.0fs %s", r[0], len(r[1]), r[3], r[4], r[2] or "")
            if len(ready) >= args.batch_size:
                run_vlm(ready)
                ready = []
    if ready:
        run_vlm(ready)
    pbar.close()
    flush()
    ok = done[done["status"] == "ok"]
    log.info("Done: %d videos ok | cropped (stream layout / sub-area): %d | with in-game overlays: %d | "
             "mostly non-gameplay frames: %d | failed: %d", len(ok), ok["game_area"].notna().sum(),
             (ok["overlays"] != "[]").sum(), (ok["frac_gameplay"] < 0.5).sum(), (done["status"] != "ok").sum())


if __name__ == "__main__":
    main()
