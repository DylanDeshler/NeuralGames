#!/usr/bin/env python
"""Stage 03 — fetch exactly the 20 s window of each manifest chunk, resize it into an aspect-ratio
bucket (no letterboxing), inpaint the controller overlay, and save frames + aligned (T, 21) float32 actions.

Geometry: crop bbox_game_area -> decode at analysis res -> detect static black pillar/letterbox bars
(e.g. a 4:3 game inside a 16:9 video) -> pick the bucket (in --patch units, ~256 tokens by default) with
the closest aspect -> resize to cover it and centre-crop the small excess (<= --max-crop, typically 1-2%),
so there's no distortion and no bars. Clips with unusual aspect ratios fall back to fit+pad (recorded in pad/content_wh).

Per chunk output (data/processed/<video_id>/<uuid>/):
    frames.mp4    W x H (bucket, e.g. 336x192 for 16:9) H.264 yuv444p CRF 10, exactly T frames at --target-fps
    actions.npy   (T, 21) float32 [j_left_x, j_left_y, j_right_x, j_right_y, 17 buttons], pooled to the
                  output fps (sticks: mean over the stride window, buttons: OR) so short taps survive
    frame0.png    first (inpainted) frame, used by the captioner
    meta.json     original metadata + processing parameters  (written last = "done" marker)

Video fetch methods (--method):
  direct    yt-dlp resolves the stream URL; ffmpeg seeks with HTTP range requests and decodes only
            [start, end) — frame-accurate, downloads only that window.
  sections  yt-dlp --download-sections "*start-end" --force-keyframes-at-cuts into a temp file,
            then the same ffmpeg processing locally.
  auto      direct, falling back to sections on failure (default).

Resumable: chunks with meta.json are skipped; permanently unavailable videos are recorded in
data/processed/_failed.jsonl and skipped on re-runs (use --retry-failed to try again).

Example:
  python 03_download_and_process.py --workers 8 --max-starts-per-min 30
  python 03_download_and_process.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from tqdm import tqdm

from ng_common import DATA_ROOT, ACTION_DIM, ACTION_NAMES, atomic_write_json, free_gb, processed_dir_for, setup_logging

PERMANENT_ERR = re.compile(
    r"(private video|video unavailable|video is unavailable|does not exist|has been removed|no longer available|account associated|"
    r"copyright|members-only|join this channel|confirm your age|is not available|terminated|"
    r"This live event|premieres in)", re.I)
RATE_ERR = re.compile(r"(HTTP Error 429|too many requests|not a bot|sign in to confirm|rate.?limit)", re.I)


class PermanentError(Exception):
    pass


class ShortClipError(Exception):
    pass


# ============================================================================ utilities
class RateLimiter:
    """Token bucket shared by all worker threads: at most `per_min` downloads started per minute."""

    def __init__(self, per_min: float):
        self.interval = 60.0 / per_min if per_min > 0 else 0.0
        self.lock = threading.Lock()
        self.next_t = time.monotonic()
        self.penalty_until = 0.0

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_t, self.penalty_until)
            self.next_t = start + self.interval
        time.sleep(max(0.0, start - time.monotonic()))

    def penalize(self, seconds: float):
        with self.lock:
            self.penalty_until = max(self.penalty_until, time.monotonic() + seconds)


class StreamResolver:
    """Caches yt-dlp format resolution per (video, height cap); remembers dead videos."""

    def __init__(self, args, log):
        self.args, self.log = args, log
        self.cache: dict[tuple, tuple[float, dict]] = {}
        self.locks: dict[str, threading.Lock] = {}
        self.glock = threading.Lock()
        self.dead: dict[str, str] = {}
        self.bot_streak = 0  # consecutive bot-check / 429 failures across all workers

    def ydl_opts(self, fmt: str) -> dict:
        o = {"quiet": True, "no_warnings": True, "noplaylist": True, "format": fmt, "socket_timeout": 30}
        if self.args.cookies:
            o["cookiefile"] = self.args.cookies
        if self.args.cookies_from_browser:
            o["cookiesfrombrowser"] = (self.args.cookies_from_browser,)
        if self.args.proxy:
            o["proxy"] = self.args.proxy
        return o

    def resolve(self, video_id: str, url: str, fmt: str, cap: int, limiter: RateLimiter) -> dict:
        if video_id in self.dead:
            raise PermanentError(self.dead[video_id])
        with self.glock:
            lock = self.locks.setdefault(video_id, threading.Lock())
        with lock:
            key = (video_id, cap)
            hit = self.cache.get(key)
            if hit and time.time() - hit[0] < self.args.url_ttl:
                return hit[1]
            import yt_dlp
            limiter.acquire()
            try:
                with yt_dlp.YoutubeDL(self.ydl_opts(fmt)) as ydl:
                    info = ydl.extract_info(url, download=False)
            except Exception as e:
                msg = str(e)
                if PERMANENT_ERR.search(msg) and not RATE_ERR.search(msg):
                    self.dead[video_id] = msg[:300]
                    raise PermanentError(msg[:300]) from e
                if RATE_ERR.search(msg):
                    limiter.penalize(self.args.rate_penalty)
                raise
            f = (info.get("requested_formats") or [info])[0]
            res = {"url": f["url"], "headers": f.get("http_headers") or info.get("http_headers") or {},
                   "width": f.get("width"), "height": f.get("height"), "fps": f.get("fps"),
                   "format_id": f.get("format_id"), "vcodec": f.get("vcodec"), "title": info.get("title")}
            self.cache[key] = (time.time(), res)
            return res

    def invalidate(self, video_id: str):
        for k in [k for k in self.cache if k[0] == video_id]:
            self.cache.pop(k, None)


def format_selector(fps: float, cap: int) -> str:
    f = max(1, int(fps) - 1)
    return (f"bv*[height<={cap}][fps>={f}][vcodec^=avc1]/bv*[height<={cap}][fps>={f}]/"
            f"bv*[height<={max(cap, 1080)}][fps>={f}]/bv*[height<={cap}]/b[height<={cap}]/bv*/b")


def probe_dims(path: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split("\n")[0].split("x")[:2]
    return int(w), int(h)


def _jload(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        return json.loads(s) if isinstance(s, str) else s
    except json.JSONDecodeError:
        return None


# ============================================================================ geometry
# Aspect-ratio buckets in PATCH units (W x H). "auto" = canonical buckets fitted to common game aspect
# ratios (~256 tokens at --patch 16, JiT-style), plus filler buckets so that EVERY aspect in
# [0.5, 2.45] is within --max-crop of some bucket (no clip ever needs padding):
#   21x12 16:9 (crop 1.6%) | 19x14 4:3 (1.8%) | 17x15 8:7 SNES/NES (0.8%) | 18x12 3:2 GBA (0%)
#   19x12 16:10 (1.0%) | 16x16 1:1 | 24x10 21:9 (1.2%) | 14x19 3:4 | 12x21 9:16
CANONICAL_BUCKETS = "21x12,19x14,17x15,18x12,19x12,16x16,24x10,14x19,12x21"
DEFAULT_BUCKETS = "auto"


def parse_buckets(spec: str) -> list[tuple[int, int]]:
    out = []
    for b in spec.split(","):
        w, h = b.lower().strip().split("x")
        out.append((int(w), int(h)))
    return out


def auto_buckets(canonical: str, max_crop: float, tokens: int = 256, lo: float = 0.5, hi: float = 2.45,
                 tol: float = 0.12) -> list[tuple[int, int]]:
    """Canonical buckets + greedy gap filling until every aspect in [lo, hi] is within max_crop."""
    buckets = parse_buckets(canonical)
    cands = [(w, h) for w in range(4, 64) for h in range(4, 64) if tokens * (1 - tol) <= w * h <= tokens * (1 + tol / 2)]
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), 4000))
    for _ in range(64):
        t = np.array([w / h for w, h in buckets])
        worst = (1 - np.minimum(grid[:, None] / t, t / grid[:, None])).min(1)
        if worst.max() <= max_crop:
            break
        a = grid[int(np.argmax(worst))]  # least-covered aspect -> add the candidate closest to it
        buckets.append(min(cands, key=lambda c: (abs(np.log(c[0] / c[1] / a)), abs(c[0] * c[1] - tokens))))
    return sorted(set(buckets), key=lambda b: b[0] / b[1])


def source_crop(row, src_w: int, src_h: int) -> tuple[int, int, int, int]:
    """bbox_game_area (relative) -> (x0, y0, w, h) in source pixels, even dims; full frame if absent."""
    ga = _jload(row.bbox_game_area)
    if isinstance(ga, dict) and all(k in ga for k in ("xtl", "ytl", "xbr", "ybr")):
        x0, y0 = max(0, int(round(ga["xtl"] * src_w))), max(0, int(round(ga["ytl"] * src_h)))
        x1 = min(src_w, max(int(round(ga["xbr"] * src_w)), x0 + 16))
        y1 = min(src_h, max(int(round(ga["ybr"] * src_h)), y0 + 16))
    else:
        x0, y0, x1, y1 = 0, 0, src_w, src_h
    return x0, y0, (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2


def analysis_dims(cw: int, ch: int, max_dim: int) -> tuple[int, int]:
    s = min(1.0, max_dim / max(cw, ch))
    return max(2, int(round(cw * s / 2)) * 2), max(2, int(round(ch * s / 2)) * 2)


def _run_len(mask: np.ndarray) -> int:
    """Number of leading True values."""
    idx = np.flatnonzero(~mask)
    return int(idx[0]) if len(idx) else len(mask)


def detect_borders(frames: list[np.ndarray], thresh: float, min_frac: float = 0.02,
                   sym_tol: float = 0.03) -> tuple[int, int, int, int]:
    """Static black bars (pillar/letterbox inside the video) -> (left, top, right, bottom) in px.

    A column/row is border if its 99th-percentile brightness, maxed over time, stays below `thresh`.
    Bars must be roughly symmetric (they're centred) and leave >= 50% of the axis, otherwise that axis
    is left alone; this protects dark game scenes from being cropped.
    """
    sample = frames[:: max(1, len(frames) // 32)]
    m = np.max(np.stack([f.max(axis=2) for f in sample]), axis=0).astype(np.float32)  # (h, w) max over time
    h, w = m.shape
    col_dark = np.percentile(m, 99, axis=0) < thresh
    row_dark = np.percentile(m, 99, axis=1) < thresh
    L, R = _run_len(col_dark), _run_len(col_dark[::-1])
    T, B = _run_len(row_dark), _run_len(row_dark[::-1])
    if not (min(L, R) >= min_frac * w and abs(L - R) <= sym_tol * w + 2 and w - L - R >= 0.5 * w):
        L = R = 0
    if not (min(T, B) >= min_frac * h and abs(T - B) <= sym_tol * h + 2 and h - T - B >= 0.5 * h):
        T = B = 0
    return L, T, R, B


def choose_bucket(cw: int, ch: int, buckets: list[tuple[int, int]], patch: int, max_crop: float) -> dict:
    """Pick the bucket with the least aspect mismatch (ties -> more tokens).

    cover: scale to cover the bucket and centre-crop the (<= max_crop) excess, so there's no distortion and no bars
    fit:   if no bucket is within max_crop, scale to fit and pad (black bars, recorded as pad/content_wh)
    """
    a = cw / ch
    best = min(buckets, key=lambda b: (1 - min(a / (b[0] / b[1]), (b[0] / b[1]) / a), -b[0] * b[1]))
    W, H = best[0] * patch, best[1] * patch
    t = W / H
    crop = 1 - min(a / t, t / a)
    if crop <= max_crop:
        s = max(W / cw, H / ch)
        rw, rh = max(W, int(round(cw * s))), max(H, int(round(ch * s)))
        return {"mode": "cover", "W": W, "H": H, "bucket": f"{best[0]}x{best[1]}", "rw": rw, "rh": rh,
                "ox": -((rw - W) // 2), "oy": -((rh - H) // 2), "pad": [0, 0], "content_wh": [W, H],
                "crop_frac": crop}
    s = min(W / cw, H / ch)
    rw, rh = min(W, int(round(cw * s))), min(H, int(round(ch * s)))
    px, py = (W - rw) // 2, (H - rh) // 2
    return {"mode": "fit", "W": W, "H": H, "bucket": f"{best[0]}x{best[1]}", "rw": rw, "rh": rh,
            "ox": px, "oy": py, "pad": [px, py], "content_wh": [rw, rh], "crop_frac": 0.0}


def overlay_boxes_src(row, src_w: int, src_h: int, mask_others: bool) -> list[tuple[float, float, float, float]]:
    """Controller overlay (+ bbox_others) as (x, y, w, h) in source pixels."""
    meta_h, meta_w = int(row.res_h or src_h), int(row.res_w or src_w)
    sx, sy = src_w / meta_w, src_h / meta_h
    boxes = []
    ov = _jload(row.bbox_controller_overlay)
    if isinstance(ov, (list, tuple)) and len(ov) == 4 and all(v is not None for v in ov):
        boxes.append((ov[0] * sx, ov[1] * sy, ov[2] * sx, ov[3] * sy))
    if mask_others:
        for b in _jload(row.bbox_others) or []:
            if isinstance(b, dict) and all(k in b for k in ("xtl", "ytl", "xbr", "ybr")):
                boxes.append((b["xtl"] * src_w, b["ytl"] * src_h,
                              (b["xbr"] - b["xtl"]) * src_w, (b["ybr"] - b["ytl"]) * src_h))
    return boxes


def build_mask(boxes_src, crop_src, ana_wh, borders, geo, dilate: int):
    """Map source-px boxes through crop -> analysis scale -> border crop -> bucket resize/crop/pad."""
    x0, y0, cw, ch = crop_src
    aw, ah = ana_wh
    L, T, R, B = borders
    kx, ky = aw / cw, ah / ch                          # source -> analysis
    fx, fy = geo["rw"] / (aw - L - R), geo["rh"] / (ah - T - B)  # analysis content -> resized
    W, H = geo["W"], geo["H"]
    px, py = geo["pad"]
    nw, nh = geo["content_wh"]
    mask = np.zeros((H, W), np.uint8)
    out = []
    for bx, by, bw, bh in boxes_src:
        def tx(x):
            return ((x - x0) * kx - L) * fx + geo["ox"]

        def ty(y):
            return ((y - y0) * ky - T) * fy + geo["oy"]
        X0, Y0 = max(px, int(np.floor(tx(bx) - dilate))), max(py, int(np.floor(ty(by) - dilate)))
        X1, Y1 = min(px + nw, int(np.ceil(tx(bx + bw) + dilate))), min(py + nh, int(np.ceil(ty(by + bh) + dilate)))
        if X1 > X0 and Y1 > Y0:
            mask[Y0:Y1, X0:X1] = 255
            out.append([X0, Y0, X1 - X0, Y1 - Y0])
    return mask, out


def warp(frame: np.ndarray, borders, geo) -> np.ndarray:
    L, T, R, B = borders
    h, w = frame.shape[:2]
    content = frame[T:h - B, L:w - R]
    r = cv2.resize(content, (geo["rw"], geo["rh"]), interpolation=cv2.INTER_AREA)
    if geo["mode"] == "cover":
        return np.ascontiguousarray(r[-geo["oy"]:-geo["oy"] + geo["H"], -geo["ox"]:-geo["ox"] + geo["W"]])
    canvas = np.zeros((geo["H"], geo["W"], 3), np.uint8)
    canvas[geo["oy"]:geo["oy"] + geo["rh"], geo["ox"]:geo["ox"] + geo["rw"]] = r
    return canvas


# ============================================================================ core processing
def decode_process_encode(input_args: list[str], row, src_wh: tuple[int, int], fps_out: float, n_out: int,
                          out_mp4: Path, frame0_png: Path, args, tmp: Path) -> tuple[int, dict]:
    """ffmpeg decode (crop game area, analysis-res) -> buffer -> detect bars -> choose bucket ->
    resize/crop + inpaint in python -> ffmpeg encode. Returns (decoded_frames, geometry)."""
    src_w, src_h = src_wh
    crop_src = source_crop(row, src_w, src_h)
    x0, y0, cw, ch = crop_src
    aw, ah = analysis_dims(cw, ch, args.analysis_max_dim)
    dec_cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *input_args, "-an", "-sn",
               "-vf", f"fps={fps_out:.6f},crop={cw}:{ch}:{x0}:{y0},scale={aw}:{ah}:flags=area,format=rgb24",
               "-frames:v", str(n_out), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    with open(tmp / "dec.log", "wb") as dec_err:
        dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=dec_err)
        fbytes, frames = aw * ah * 3, []
        try:
            while len(frames) < n_out:
                buf = dec.stdout.read(fbytes)
                if len(buf) < fbytes:
                    break
                frames.append(np.frombuffer(buf, np.uint8).reshape(ah, aw, 3))
        finally:
            dec.stdout.close()
            dec.kill()
            dec.wait()
    decoded = len(frames)
    if decoded == 0 or n_out - decoded > args.max_pad_frames:
        raise ShortClipError(f"only {decoded}/{n_out} frames decoded: {(tmp / 'dec.log').read_text()[-400:]}")
    frames += [frames[-1]] * (n_out - decoded)  # tiny tail deficit (rounding at clip end): repeat last frame

    borders = detect_borders(frames, args.border_thresh) if args.detect_borders else (0, 0, 0, 0)
    L, T, R, B = borders
    geo = choose_bucket(aw - L - R, ah - T - B, args.bucket_list, args.patch, args.max_crop)
    mask, boxes = build_mask(overlay_boxes_src(row, src_w, src_h, args.mask_others), crop_src, (aw, ah),
                             borders, geo, args.mask_dilate)
    W, H = geo["W"], geo["H"]
    enc_cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps_out:.6f}", "-i", "pipe:0",
               "-c:v", args.vcodec, *args.enc_args.split(), "-g", str(args.gop), "-movflags", "+faststart",
               str(out_mp4)]
    with open(tmp / "enc.log", "wb") as enc_err:
        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=enc_err)
        try:
            for i, f in enumerate(frames):
                out = warp(f, borders, geo)
                if boxes:
                    out = cv2.inpaint(out, mask, args.inpaint_radius, cv2.INPAINT_TELEA)
                if i == 0:
                    cv2.imwrite(str(frame0_png), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                enc.stdin.write(out.tobytes())
        finally:
            try:
                enc.stdin.close()
            except BrokenPipeError:
                pass
            enc.wait()
    if enc.returncode != 0:
        raise RuntimeError(f"encoder failed: {(tmp / 'enc.log').read_text()[-500:]}")
    geo.update(crop_src=list(crop_src), src_wh=[src_w, src_h], analysis_wh=[aw, ah], borders=list(borders),
               inpaint_boxes=boxes, inpaint_frac=float((mask > 0).sum()) / max(1, W * H))
    return decoded, geo


def fps_stride(fps_ann: float, target: float) -> int | None:
    """Integer frame stride mapping annotation fps -> target fps (None if incompatible, e.g. 25 -> 30)."""
    if not target or target <= 0:
        return 1
    k = int(round(fps_ann / target))
    return k if k >= 1 and abs(fps_ann / k - target) < 0.05 else None


def pool_actions(a: np.ndarray, stride: int, n_out: int) -> np.ndarray:
    if stride == 1:
        return a[:n_out]
    a = a[: n_out * stride].reshape(n_out, stride, ACTION_DIM)
    return np.concatenate([a[:, :, :4].mean(1), a[:, :, 4:].max(1)], axis=1).astype(np.float32)


def process_chunk(row, args, resolver: StreamResolver, limiter: RateLimiter, log) -> dict:
    out_dir = processed_dir_for(args.processed_dir, row.video_id, row.uuid)
    if (out_dir / "meta.json").exists():
        return {"status": "skipped"}
    src_actions = out_dir / "_actions_src.npy"
    if not src_actions.exists():
        raise RuntimeError("staged actions missing")
    actions = np.load(src_actions)
    fps_ann = float(row.fps or 30.0)
    n_ann = int(actions.shape[0])
    stride = fps_stride(fps_ann, args.target_fps)
    if stride is None:
        raise PermanentError(f"annotation fps {fps_ann} is not an integer multiple of --target-fps {args.target_fps}")
    fps_out, n_out = fps_ann / stride, n_ann // stride
    start = (row.start_frame / fps_ann) if pd.notna(row.start_frame) and row.start_frame is not None else float(row.start_time)
    dur = n_ann / fps_ann
    url = row.url if isinstance(row.url, str) and row.url else f"https://www.youtube.com/watch?v={row.video_id}"
    # small embedded game areas need a higher-res source to survive the crop
    ga = _jload(row.bbox_game_area)
    # need the game content to be >= ~1.5x the output bucket height after cropping bbox_game_area
    frac_h = (ga.get("ybr", 1) - ga.get("ytl", 0)) if isinstance(ga, dict) else 1.0
    frac_w = (ga.get("xbr", 1) - ga.get("xtl", 0)) if isinstance(ga, dict) else 1.0
    rw, rh = float(row.res_w or 1920) * frac_w, float(row.res_h or 1080) * frac_h  # content size (metadata)
    out_h = choose_bucket(max(rw, 1), max(rh, 1), args.bucket_list, args.patch, args.max_crop)["H"]
    need = 1.5 * out_h / max(frac_h, 0.1)  # source height so the cropped content is >= 1.5x the output
    cap = int(min(1080, max(args.max_src_height, next((r for r in (360, 480, 720, 1080) if r >= need), 1080))))

    tmp = Path(args.tmp_dir) / row.uuid
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    work = tmp / "out"
    work.mkdir()
    method_used, info, decoded, geo = None, {}, 0, None
    try:
        methods = ["direct", "sections"] if args.method == "auto" else [args.method]
        last_err = None
        for method in methods:
            try:
                fmt = format_selector(fps_out, cap)  # only need >= output fps (e.g. 30), not the 60 fps annotation
                if method == "direct":
                    info = resolver.resolve(row.video_id, url, fmt, cap, limiter)
                    if not info.get("width") or not info.get("height"):
                        raise RuntimeError("stream dims unknown")
                    src_wh = (info["width"], info["height"])
                    hdr = "".join(f"{k}: {v}\r\n" for k, v in (info.get("headers") or {}).items())
                    inp = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                           "-rw_timeout", "30000000", *(["-headers", hdr] if hdr else []),
                           # googlevideo URLs are bound to the resolving IP: fetch through the same proxy
                           *(["-http_proxy", args.proxy] if args.proxy and args.proxy.startswith("http") else []),
                           "-ss", f"{start:.3f}", "-t", f"{dur + 1.0:.3f}", "-i", info["url"]]
                else:
                    if row.video_id in resolver.dead:
                        raise PermanentError(resolver.dead[row.video_id])
                    limiter.acquire()
                    raw = download_section(url, fmt, start, start + dur + 0.5, tmp, args)
                    w, h = probe_dims(raw)
                    info = {"width": w, "height": h, "format_id": "sections"}
                    src_wh = (w, h)
                    inp = ["-i", str(raw)]
                decoded, geo = decode_process_encode(inp, row, src_wh, fps_out, n_out, work / "frames.mp4",
                                                     work / "frame0.png", args, tmp)
                method_used = method
                break
            except PermanentError:
                raise
            except Exception as e:
                last_err = e
                if method == "direct":
                    resolver.invalidate(row.video_id)
                if RATE_ERR.search(str(e)):
                    limiter.penalize(args.rate_penalty)
                    resolver.bot_streak += 1
                log.debug("%s via %s failed: %s", row.uuid, method, e)
        if method_used is None:
            raise last_err or RuntimeError("no method succeeded")

        acts = pool_actions(actions, stride, n_out)
        assert acts.shape == (n_out, ACTION_DIM), acts.shape
        np.save(work / "actions.npy", acts.astype(np.float32))
        orig = _jload(row.meta_json) or {}
        meta = {
            "uuid": row.uuid, "video_id": row.video_id, "chunk_id": int(row.chunk_id), "split": row.split,
            "game": row.game_name, "game_key": row.game_key, "genre": row.genre, "label_source": row.label_source,
            "n_frames": int(n_out), "fps": fps_out, "frame_stride": stride, "annotation_fps": fps_ann,
            "annotation_frames": n_ann,
            "start_time": start, "duration": dur, "action_names": ACTION_NAMES, "action_source": row.action_source,
            "width": geo["W"], "height": geo["H"], "bucket": geo["bucket"], "patch": args.patch,
            "fit_mode": geo["mode"], "aspect_crop_frac": geo["crop_frac"], "pad": geo["pad"],
            "content_wh": geo["content_wh"], "crop_src": geo["crop_src"], "src_wh": geo["src_wh"],
            "analysis_wh": geo["analysis_wh"], "border_crop_ltrb": geo["borders"],
            "inpaint_boxes": geo["inpaint_boxes"], "inpaint_frac": geo["inpaint_frac"], "decoded_frames": int(decoded),
            "padded_frames": int(n_out - decoded), "method": method_used, "format_id": info.get("format_id"),
            "vcodec": args.vcodec, "enc_args": args.enc_args, "original_metadata": orig,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        atomic_write_json(work / "meta.json", meta)
        # move results into place; meta.json last so its presence means "complete"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("frames.mp4", "actions.npy", "frame0.png"):
            os.replace(work / name, out_dir / name)
        os.replace(work / "meta.json", out_dir / "meta.json")
        src_actions.unlink(missing_ok=True)
        resolver.bot_streak = 0
        return {"status": "ok", "method": method_used, "padded": n_out - decoded, "bucket": geo["bucket"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def download_section(url: str, fmt: str, t0: float, t1: float, tmp: Path, args) -> Path:
    cmd = ["yt-dlp", "--quiet", "--no-warnings", "--no-playlist", "--no-part", "-f", fmt,
           "--download-sections", f"*{t0:.3f}-{t1:.3f}", "--force-keyframes-at-cuts",
           "-N", "4", "--socket-timeout", "30", "-o", str(tmp / "raw.%(ext)s"), url]
    if args.cookies:
        cmd[1:1] = ["--cookies", args.cookies]
    if args.cookies_from_browser:
        cmd[1:1] = ["--cookies-from-browser", args.cookies_from_browser]
    if args.proxy:
        cmd[1:1] = ["--proxy", args.proxy]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.download_timeout)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout)[-600:]
        if PERMANENT_ERR.search(msg) and not RATE_ERR.search(msg):
            raise PermanentError(msg)
        raise RuntimeError(f"yt-dlp failed: {msg}")
    files = [f for f in tmp.glob("raw.*") if f.suffix not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("yt-dlp produced no file")
    return files[0]


# ============================================================================ staging + bookkeeping
def stage_actions(man: pd.DataFrame, args, log) -> None:
    """Write each pending chunk's native-rate actions to <out>/_actions_src.npy (grouped by shard)."""
    meta_dir = Path(args.meta_dir)
    todo = man[[not (processed_dir_for(args.processed_dir, v, u) / "_actions_src.npy").exists()
                for v, u in zip(man.video_id, man.uuid)]]
    if todo.empty:
        return
    log.info("Staging actions for %s chunks from %d shard files", f"{len(todo):,}", todo.shard.nunique())
    for shard, sdf in tqdm(todo.groupby("shard"), desc="stage actions"):
        t = ds.dataset(meta_dir / "actions" / f"SHARD_{int(shard):04d}.parquet").to_table(
            columns=["uuid", "n_frames", "actions"], filter=pc.field("uuid").isin(sdf["uuid"].tolist()))
        vid_of = dict(zip(sdf.uuid, sdf.video_id))
        for u, n, b in zip(t.column("uuid").to_pylist(), t.column("n_frames").to_pylist(), t.column("actions").to_pylist()):
            a = np.frombuffer(b, np.float32).reshape(n, ACTION_DIM)
            d = processed_dir_for(args.processed_dir, vid_of[u], u)
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "_actions_src.npy", a)


def load_failed(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                out[r["uuid"]] = r
            except Exception:
                pass
    return out


def print_status(man: pd.DataFrame, args) -> None:
    done = np.array([(processed_dir_for(args.processed_dir, v, u) / "meta.json").exists()
                     for v, u in zip(man.video_id, man.uuid)])
    failed = load_failed(Path(args.processed_dir) / "_failed.jsonl")
    perm = man.uuid.map(lambda u: failed.get(u, {}).get("kind") == "permanent").to_numpy() & ~done
    print(f"total {len(man):,} | done {done.sum():,} ({done.mean():.1%}, {man.duration[done].sum() / 3600:.1f} h) | "
          f"permanently failed {perm.sum():,} | pending {(~done & ~perm).sum():,}")
    for sp, sdf in man.assign(done=done).groupby("split"):
        print(f"  {sp:<18} done {sdf.done.sum():>7,} / {len(sdf):>7,}")
    if done.any():
        h = man[done].groupby("game_key")["duration"].sum() / 3600
        p = (h / h.sum()).to_numpy()
        cap = man.loc[done, "download_priority"].max() if "download_priority" in man.columns else float("nan")
        print(f"  downloaded subset: {len(h)} games, eff_games {np.exp(-(p * np.log(p)).sum()):.1f}, "
              f"top game {p.max():.1%}, per-game cap reached ≈ {cap:.1f} h")
    print(f"  disk used: {shutil.disk_usage(args.processed_dir).used / 1e9:.0f} GB (volume), "
          f"free {free_gb(args.processed_dir):.0f} GB")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DATA_ROOT / "manifest_2d.parquet"))
    ap.add_argument("--meta-dir", default=str(DATA_ROOT / "nitrogen_meta"))
    ap.add_argument("--processed-dir", default=str(DATA_ROOT / "processed"))
    ap.add_argument("--tmp-dir", default=str(DATA_ROOT / "tmp_03"))
    ap.add_argument("--split", choices=["train", "val_unseen_games", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="process at most N chunks (testing)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-starts-per-min", type=float, default=30.0, help="rate limit for YouTube requests")
    ap.add_argument("--rate-penalty", type=float, default=120.0, help="global pause (s) after a 429/bot check")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-failed", action="store_true", help="retry chunks marked permanently failed")
    ap.add_argument("--method", choices=["auto", "direct", "sections"], default="auto")
    ap.add_argument("--max-src-height", type=int, default=480,
                    help="source resolution cap. 480p is >= 2x the default buckets and ~1/3 the bytes of 720p60; raised "
                         "automatically (up to 1080) when bbox_game_area is a small part of the frame")
    ap.add_argument("--patch", type=int, default=16, help="model patch size in px; buckets are in patch units")
    ap.add_argument("--buckets", default=DEFAULT_BUCKETS,
                    help="'auto' (canonical game aspects + gap fillers, ~256 tokens each) or an explicit "
                         "list 'WxH,...' in patches, e.g. '21x12,19x14,16x16'")
    ap.add_argument("--max-crop", type=float, default=0.05,
                    help="max fraction of the content cropped to fill a bucket; beyond this, pad instead")
    ap.add_argument("--detect-borders", action=argparse.BooleanOptionalAction, default=True,
                    help="detect static black pillar/letterbox bars inside the video (4:3 game in a 16:9 video)")
    ap.add_argument("--border-thresh", type=float, default=24.0, help="max brightness (0-255) of a bar")
    ap.add_argument("--analysis-max-dim", type=int, default=0,
                    help="intermediate decode size (long side); 0 = auto (>= 1.5x the largest bucket side)")
    ap.add_argument("--target-fps", type=float, default=30.0,
                    help="output fps (0 = native). Per-chunk integer stride: 60->30 keeps every 2nd frame, 30->30 "
                         "keeps all; actions pooled over each stride window (sticks mean, buttons OR)"),
    ap.add_argument("--mask-others", action=argparse.BooleanOptionalAction, default=True,
                    help="also inpaint bbox_others regions (webcams, banners)")
    ap.add_argument("--mask-dilate", type=int, default=2, help="mask dilation in output pixels")
    ap.add_argument("--inpaint-radius", type=float, default=3.0)
    ap.add_argument("--max-pad-frames", type=int, default=6, help="max tail frames to pad by repetition")
    ap.add_argument("--vcodec", default="libx264", help="libx264 | h264_nvenc | libx265 ...")
    ap.add_argument("--enc-args", default="-preset medium -crf 10 -pix_fmt yuv444p",
                    help="encoder args; lossless: '-preset veryslow -crf 0 -pix_fmt yuv444p'; "
                         "NVENC: '-preset p7 -tune hq -rc constqp -qp 10 -pix_fmt yuv444p'")
    ap.add_argument("--gop", type=int, default=60)
    ap.add_argument("--url-ttl", type=float, default=3 * 3600)
    ap.add_argument("--download-timeout", type=float, default=300)
    ap.add_argument("--min-free-gb", type=float, default=50.0)
    ap.add_argument("--cookies", default=None)
    ap.add_argument("--cookies-from-browser", default=None)
    ap.add_argument("--proxy", default=None,
                    help="proxy for yt-dlp AND the ffmpeg fetch (googlevideo URLs are IP-bound), e.g. "
                         "http://user:pass@host:port (ffmpeg needs http; socks works only with --method sections)")
    ap.add_argument("--max-bot-streak", type=int, default=30,
                    help="stop submitting after this many consecutive bot-check/429 failures (IP is blocked)")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    args = ap.parse_args()

    args.bucket_list = (auto_buckets(CANONICAL_BUCKETS, args.max_crop) if args.buckets == "auto"
                        else parse_buckets(args.buckets))
    if not args.analysis_max_dim:
        args.analysis_max_dim = max(512, int(1.5 * args.patch * max(max(b) for b in args.bucket_list)))
    log = setup_logging("03_download_and_process")
    log.info("Buckets (patch %d): %s", args.patch,
             ", ".join(f"{w}x{h}={w * args.patch}x{h * args.patch}" for w, h in args.bucket_list))
    for tool in ("ffmpeg", "ffprobe", "yt-dlp"):
        if shutil.which(tool) is None:
            raise SystemExit(f"'{tool}' not found on PATH")
    man = pd.read_parquet(args.manifest)
    if args.split != "all":
        man = man[man.split == args.split]
    # interleave games so early progress covers many titles (and spreads load across videos)
    if "download_priority" in man.columns:
        # balanced-first order: any prefix of the run is a per-game-capped ("water-filling") subset
        man = man.sort_values("download_priority").reset_index(drop=True)
    else:
        man = man.sample(frac=1.0, random_state=0).reset_index(drop=True)
    Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
    if args.status:
        print_status(man, args)
        return

    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    Path(args.tmp_dir).mkdir(parents=True)
    failed_path = Path(args.processed_dir) / "_failed.jsonl"
    failed = load_failed(failed_path)
    done_mask = np.array([(processed_dir_for(args.processed_dir, v, u) / "meta.json").exists()
                          for v, u in zip(man.video_id, man.uuid)])
    todo = man[~done_mask]
    if not args.retry_failed:
        todo = todo[todo.uuid.map(lambda u: failed.get(u, {}).get("kind") != "permanent")]
    ok_fps = todo.fps.map(lambda f: fps_stride(float(f), args.target_fps) is not None)
    if (~ok_fps).any():
        log.warning("Skipping %d chunks whose fps can't reach --target-fps %.0f with an integer stride: %s",
                    (~ok_fps).sum(), args.target_fps, todo.fps[~ok_fps].round().value_counts().to_dict())
        todo = todo[ok_fps]
    if args.limit:
        todo = todo.head(args.limit)
    log.info("Manifest %s chunks | already done %s | to process %s", f"{len(man):,}", f"{done_mask.sum():,}",
             f"{len(todo):,}")
    if todo.empty:
        print_status(man, args)
        return
    stage_actions(todo, args, log)

    resolver, limiter = StreamResolver(args, log), RateLimiter(args.max_starts_per_min)
    flock = threading.Lock()
    stats = {"ok": 0, "permanent": 0, "transient": 0, "padded": 0}
    buckets: dict[str, int] = {}

    def record_failure(row, kind, err):
        with flock, open(failed_path, "a") as fh:
            fh.write(json.dumps({"uuid": row.uuid, "video_id": row.video_id, "kind": kind,
                                 "error": str(err)[:500], "t": time.time()}) + "\n")

    def job(row):
        for attempt in range(args.retries):
            try:
                return process_chunk(row, args, resolver, limiter, log)
            except PermanentError as e:
                record_failure(row, "permanent", e)
                return {"status": "permanent"}
            except Exception as e:
                if attempt == args.retries - 1:
                    record_failure(row, "transient", e)
                    log.warning("%s failed after %d attempts: %s", row.uuid, args.retries, str(e)[:300])
                    log.debug(traceback.format_exc())
                    return {"status": "transient"}
                time.sleep(5 * 2 ** attempt)

    rows = list(todo.itertuples(index=False))
    pbar = tqdm(total=len(rows), desc="clips", unit="clip")
    with ThreadPoolExecutor(args.workers) as ex:
        pending, it = set(), iter(rows)
        stop = False
        while True:
            if not stop and resolver.bot_streak >= args.max_bot_streak:
                log.error("YouTube is bot-blocking this IP (%d consecutive bot-check/429 failures). Stopping; progress "
                          "is saved. Use --cookies (logged-in, ideally with a PO-token provider plugin) or --proxy, "
                          "or run stage 03 from another network.", resolver.bot_streak)
                stop = True
            while not stop and len(pending) < args.workers * 2:
                if free_gb(args.processed_dir) < args.min_free_gb:
                    log.error("Free disk < %.0f GB — stopping submission (re-run to resume).", args.min_free_gb)
                    stop = True
                    break
                r = next(it, None)
                if r is None:
                    stop = True
                    break
                pending.add(ex.submit(job, r))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                res = f.result() or {}
                s = res.get("status")
                if s in stats:
                    stats[s] += 1
                stats["padded"] += int(res.get("padded", 0) > 0)
                if res.get("bucket"):
                    buckets[res["bucket"]] = buckets.get(res["bucket"], 0) + 1
                pbar.update(1)
                pbar.set_postfix(ok=stats["ok"], perm=stats["permanent"], fail=stats["transient"])
    pbar.close()
    log.info("Done: %s | buckets: %s", stats, dict(sorted(buckets.items(), key=lambda kv: -kv[1])))
    print_status(man, args)


if __name__ == "__main__":
    main()
