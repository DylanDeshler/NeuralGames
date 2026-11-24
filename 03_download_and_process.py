#!/usr/bin/env python
"""Stage 03 — fetch exactly the 20 s window of each manifest chunk, resize it into an aspect-ratio
bucket (no letterboxing), inpaint the controller overlay, and save frames + aligned (T, 21) float32 actions.

Geometry: crop bbox_game_area -> decode at analysis res -> detect static black pillar/letterbox bars
(e.g. a 4:3 game inside a 16:9 video) -> pick the bucket (in --patch units, ~256 tokens by default) with
the closest aspect -> resize to cover it and centre-crop the small excess (<= --max-crop, typically 1-2%),
so there's no distortion and no bars. Clips with unusual aspect ratios fall back to fit+pad (recorded in pad/content_wh).

Per chunk output (data/processed/<video_id>/<uuid>/):
    frames.mp4    W x H (bucket, e.g. 336x192 for 16:9) H.264 yuv444p CRF 18 (~0.12 GB per hour of data, measured;
                  41.6 dB PSNR vs CRF 10 on a real clip), exactly T frames at --target-fps
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
import logging
import shutil
import zlib
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

from ng_common import default_cookies, DATA_ROOT, ACTION_DIM, ACTION_NAMES, atomic_write_json, free_gb, processed_dir_for, setup_logging

PERMANENT_ERR = re.compile(
    r"(private video|video unavailable|video is unavailable|does not exist|has been removed|no longer available|account associated|"
    r"copyright|members-only|join this channel|confirm your age|is not available|terminated|"
    r"This live event|premieres in)", re.I)
PROXY_ERR = re.compile(r"(Unable to connect to proxy|ProxyError|proxy.*(refused|unreachable)|Connection refused.*8899)", re.I)
RATE_ERR = re.compile(r"(HTTP Error 429|too many requests|not a bot|sign in to confirm|rate.?limit)", re.I)


class _NullLogger:
    def debug(self, msg):
        pass

    warning = info = error = debug


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


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
        self.geo_locks: dict[str, threading.Lock] = {}
        self.proxy_streak = 0  # consecutive "cannot connect to proxy" failures

    def ydl_opts(self, fmt: str) -> dict:
        o = {"quiet": True, "no_warnings": True, "noplaylist": True, "format": fmt, "socket_timeout": 30,
             "logger": _NullLogger()}  # errors are captured from exceptions; keep the console clean
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

    def geo_lock(self, video_id: str) -> threading.Lock:
        with self.glock:
            return self.geo_locks.setdefault(video_id, threading.Lock())

    def invalidate(self, video_id: str):
        for k in [k for k in self.cache if k[0] == video_id]:
            self.cache.pop(k, None)


def format_selector(fps: float, cap: int, prefer_min: int = 400) -> str:
    """Smallest decent stream: prefer ~480p, else the next tier up to `cap` (often 720p: many videos have
    no 480p and YouTube's 360p is ~0.4 Mbit/s mush), and only fall back to lower resolutions if nothing else."""
    f = max(1, int(fps) - 1)
    mid = max(prefer_min, min(cap, 540))
    tiers = [f"[height>={prefer_min}][height<={mid}]", f"[height>{mid}][height<={cap}]", f"[height<={cap}]"]
    sel = []
    for t in tiers:
        sel += [f"bv*{t}[fps>={f}][vcodec^=avc1]", f"bv*{t}[fps>={f}]"]
    return "/".join(sel + [f"bv*[height<={cap}]", f"b[height<={cap}]", "bv*", "b"])


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


def static_bottom_band(frames: list[np.ndarray], borders, min_frac: float = 0.02, max_frac: float = 0.12,
                       static_std: float = 4.0, static_share: float = 0.90) -> int:
    """Rows to trim at the bottom: a stream status bar is (almost) unchanged for the whole clip while the game
    above it moves. Returns the band height in analysis px (0 if none). Only used for stream-layout videos."""
    L, T, R, B = borders
    sample = frames[:: max(1, len(frames) // 60)]
    g = np.stack([f[T:f.shape[0] - B, L:f.shape[1] - R].mean(axis=2) for f in sample]).astype(np.float32)
    if g.shape[0] < 10:
        return 0
    static_rows = (g.std(axis=0) < static_std).mean(axis=1) >= static_share  # per row: share of static pixels
    h = len(static_rows)
    n = 0
    while n < h and static_rows[h - 1 - n]:
        n += 1
    if not (min_frac * h <= n <= max_frac * h):
        return 0
    share = (g.std(axis=0) < static_std).mean(axis=1)
    band, above = share[h - n:].mean(), share[max(0, h - n - int(0.10 * h)): h - n].mean()
    return n if band - above >= 0.10 else 0  # bar must be clearly more static than the game just above it


def choose_bucket(cw: int, ch: int, buckets: list[tuple[int, int]], patch: int, max_crop: float,
                  force: tuple[int, int] | None = None) -> dict:
    """Pick the bucket with the least aspect mismatch (ties -> more tokens).

    cover: scale to cover the bucket and centre-crop the (<= max_crop) excess, so there's no distortion and no bars
    fit:   if no bucket is within max_crop, scale to fit and pad (black bars, recorded as pad/content_wh)
    """
    a = cw / ch
    best = force or min(buckets, key=lambda b: (1 - min(a / (b[0] / b[1]), (b[0] / b[1]) / a), -b[0] * b[1]))
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
        if bw * bh > 0.35 * cw * ch:  # backstop: never inpaint most of the picture (bad box upstream)
            continue

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
_HWACCEL = {"on": False}  # set in main(); switched off for good after a failed GPU decode


def hwaccel_available() -> bool:
    """ffmpeg built with CUDA hwaccel and an NVIDIA GPU visible."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "cuda" in out.split()


def decode_process_encode(input_args: list[str], row, src_wh: tuple[int, int], fps_out: float, n_out: int,
                          out_mp4: Path, frame0_png: Path, args, tmp: Path, lock: dict | None = None) -> tuple[int, dict]:
    """ffmpeg decode (crop game area, analysis-res) -> buffer -> detect bars -> choose bucket ->
    resize/crop + inpaint in python -> ffmpeg encode. Returns (decoded_frames, geometry).

    `lock` (per-video geometry from an earlier chunk): reuse its content rectangle and bucket instead of
    re-detecting, so every chunk of a video is framed identically."""
    src_w, src_h = src_wh
    if lock:
        rx, ry, rw, rh = lock["content_rel"]
        x0, y0 = int(round(rx * src_w)), int(round(ry * src_h))
        crop_src = (x0, y0, max(16, int(round(rw * src_w)) // 2 * 2), max(16, int(round(rh * src_h)) // 2 * 2))
    else:
        crop_src = source_crop(row, src_w, src_h)
    x0, y0, cw, ch = crop_src
    aw, ah = analysis_dims(cw, ch, args.analysis_max_dim)
    def run_decoder(hw: bool) -> list[np.ndarray]:
        # NVDEC decodes on the GPU (frames come back to RAM for the cheap fps/crop/scale filters); the 4-core
        # box is CPU-bound on 720p60 software decoding otherwise
        dec_cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *(["-hwaccel", "cuda"] if hw else []),
                   *input_args, "-an", "-sn",
                   "-vf", f"fps={fps_out:.6f},crop={cw}:{ch}:{x0}:{y0},scale={aw}:{ah}:flags=area,format=rgb24",
                   "-frames:v", str(n_out), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        with open(tmp / "dec.log", "wb") as dec_err:
            dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=dec_err)
            fbytes, out = aw * ah * 3, []
            try:
                while len(out) < n_out:
                    buf = dec.stdout.read(fbytes)
                    if len(buf) < fbytes:
                        break
                    out.append(np.frombuffer(buf, np.uint8).reshape(ah, aw, 3))
            finally:
                dec.stdout.close()
                dec.kill()
                dec.wait()
        return out

    t_dec0 = time.time()
    hw = _HWACCEL["on"]
    frames = run_decoder(hw)
    if hw and not frames:  # GPU decode failed (unsupported codec, out of NVDEC sessions...): retry on CPU
        err = (tmp / "dec.log").read_text(errors="replace")[-300:]
        hw_err = any(k in err.lower() for k in ("cuda", "nvdec", "cuvid", "hardware", "device", "hwaccel"))
        if hw_err and _HWACCEL["on"]:  # a GPU problem (not a network error): CPU for the rest of the run
            _HWACCEL["on"] = False
            logging.getLogger("03_download_and_process").warning(
                "GPU decoding failed, switching to CPU decoding: %s", err.strip().replace("\n", " | "))
        hw = False
        frames = run_decoder(False)
    decoded = len(frames)
    t_fetch = time.time() - t_dec0  # network download + decode of the clip window
    t_proc0 = time.time()
    if decoded == 0 or n_out - decoded > args.max_pad_frames:
        raise ShortClipError(f"only {decoded}/{n_out} frames decoded: {(tmp / 'dec.log').read_text()[-400:]}")
    frames += [frames[-1]] * (n_out - decoded)  # tiny tail deficit (rounding at clip end): repeat last frame

    borders = detect_borders(frames, args.border_thresh) if (args.detect_borders and not lock) else (0, 0, 0, 0)
    if not lock and args.trim_static_bars and bool(getattr(row, "stream_layout", False)):
        band = static_bottom_band(frames, borders)
        if band:
            borders = (borders[0], borders[1], borders[2], borders[3] + band)  # trimmed like a bottom bar
    L, T, R, B = borders
    force = tuple(int(v) for v in lock["bucket"].split("x")) if (lock and lock.get("bucket")) else None
    geo = choose_bucket(aw - L - R, ah - T - B, args.bucket_list, args.patch, args.max_crop, force=force)
    # final content rectangle in source-relative coords (what a per-video geometry lock stores)
    geo["content_rel"] = [(x0 + L * cw / aw) / src_w, (y0 + T * ch / ah) / src_h,
                          (aw - L - R) * cw / aw / src_w, (ah - T - B) * ch / ah / src_h]
    geo["geometry_locked"] = bool(lock)
    mask, boxes = build_mask(overlay_boxes_src(row, src_w, src_h, args.mask_others), crop_src, (aw, ah),
                             borders, geo, args.mask_dilate)
    W, H = geo["W"], geo["H"]
    enc_cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps_out:.6f}", "-i", "pipe:0",
               "-c:v", args.vcodec, *args.enc_args.split(), "-g", str(args.gop or max(1, round(2 * fps_out))),
               "-movflags", "+faststart",
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
    geo["frac_black"] = float(np.mean([f.mean() < 8 for f in frames]))  # fades / loading / signal loss
    geo["timing"] = {"fetch_decode_s": round(t_fetch, 2), "process_encode_s": round(time.time() - t_proc0, 2),
                     "gpu_decode": hw}
    geo.update(crop_src=list(crop_src), src_wh=[src_w, src_h], analysis_wh=[aw, ah], borders=list(borders),
               inpaint_boxes=boxes, inpaint_frac=float((mask > 0).sum()) / max(1, W * H))
    return decoded, geo


NOMINAL_FPS = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0, 120.0)


def snap_fps(f: float) -> float:
    """n_frames/duration is noisy for irregular chunks (e.g. 60.24); snap to the nominal rate within 2.5%."""
    if not f or not np.isfinite(f):
        return 30.0
    best = min(NOMINAL_FPS, key=lambda n: abs(f / n - 1))
    return best if abs(f / best - 1) < 0.025 else float(f)


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
    fps_ann = snap_fps(float(row.fps or 30.0))
    n_ann = int(actions.shape[0])
    stride = fps_stride(fps_ann, args.target_fps)
    if stride is None:
        raise PermanentError(f"annotation fps {fps_ann} is not an integer multiple of --target-fps {args.target_fps}")
    fps_out, n_out = fps_ann / stride, n_ann // stride
    # metadata start_time is authoritative; start_frame/fps would drift if fps is even slightly off
    start = float(row.start_time) if pd.notna(row.start_time) else float(row.start_frame) / fps_ann
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
    t_resolve, t_chunk0 = 0.0, time.time()
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
                    t_res0 = time.time()
                    info = resolver.resolve(row.video_id, url, fmt, cap, limiter)
                    t_resolve = time.time() - t_res0
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
                gpath = Path(args.processed_dir) / row.video_id / "_geometry.json"
                c_rel = getattr(row, "content_rel", None)
                if isinstance(c_rel, str) and c_rel:  # 02c decided this video's framing from several probes
                    preset = {"content_rel": json.loads(c_rel), "bucket": None, "source": "02c"}
                else:
                    preset = None
                need_lock = preset is None and args.lock_geometry and not gpath.exists()
                with (resolver.geo_lock(row.video_id) if need_lock else _nullcontext()):  # fallback: first chunk decides
                    lock = preset or (json.loads(gpath.read_text()) if (args.lock_geometry and gpath.exists()) else None)
                    decoded, geo = decode_process_encode(inp, row, src_wh, fps_out, n_out, work / "frames.mp4",
                                                         work / "frame0.png", args, tmp, lock=lock)
                    if preset is None and args.lock_geometry and lock is None and geo.get("frac_black", 1.0) < 0.3:
                        gpath.parent.mkdir(parents=True, exist_ok=True)  # don't let a fade/black chunk set the framing
                        atomic_write_json(gpath, {"content_rel": geo["content_rel"], "bucket": geo["bucket"],
                                                  "from_uuid": row.uuid, "src_wh": list(src_wh)})
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
                if PROXY_ERR.search(str(e)):
                    resolver.proxy_streak += 1
                    raise  # no point trying other methods or retrying: the proxy itself is down
                log.debug("%s via %s failed: %s", row.uuid, method, e)
        if method_used is None:
            raise last_err or RuntimeError("no method succeeded")

        acts = pool_actions(actions, stride, n_out)
        assert acts.shape == (n_out, ACTION_DIM), acts.shape
        np.save(work / "actions.npy", acts.astype(np.float32))
        # native-rate actions too (float16, n_ann x 21): overlay-vs-frame lag is often 1-3 native frames (capture /
        # OBS latency), below one output frame, so pooled actions alone can't be re-aligned at training time
        np.save(work / "actions_native.npy", np.asarray(actions[:n_out * stride], np.float16))
        orig = _jload(row.meta_json) or {}
        meta = {
            "uuid": row.uuid, "video_id": row.video_id, "chunk_id": int(row.chunk_id), "split": row.split,
            "game": row.game_name, "game_key": row.game_key, "genre": row.genre, "label_source": row.label_source,
            "n_frames": int(n_out), "fps": fps_out, "frame_stride": stride, "annotation_fps": fps_ann,
            "annotation_frames": n_ann,
            "start_time": start, "duration": dur, "action_names": ACTION_NAMES, "action_source": row.action_source,
            "width": geo["W"], "height": geo["H"], "bucket": geo["bucket"], "patch": args.patch,
            "fit_mode": geo["mode"], "aspect_crop_frac": geo["crop_frac"], "pad": geo["pad"],
            "geometry_locked": geo.get("geometry_locked", False), "content_rel": geo.get("content_rel"),
            "geometry_source": ("02c" if isinstance(getattr(row, "content_rel", None), str) and row.content_rel else
                                "lock" if geo.get("geometry_locked") else "per_chunk"),
            "content_wh": geo["content_wh"], "crop_src": geo["crop_src"], "src_wh": geo["src_wh"],
            "analysis_wh": geo["analysis_wh"], "border_crop_ltrb": geo["borders"],
            "inpaint_boxes": geo["inpaint_boxes"], "inpaint_frac": geo["inpaint_frac"], "decoded_frames": int(decoded),
            "frac_black": geo["frac_black"], "layout_override": bool(getattr(row, "layout_override", False)),
            "timing": {**geo.get("timing", {}), "resolve_s": round(t_resolve, 2),
                       "total_s": round(time.time() - t_chunk0, 2)},
            "padded_frames": int(n_out - decoded), "method": method_used, "format_id": info.get("format_id"),
            "vcodec": args.vcodec, "enc_args": args.enc_args, "original_metadata": orig,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        atomic_write_json(work / "meta.json", meta)
        # move results into place; meta.json last so its presence means "complete"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("frames.mp4", "actions.npy", "actions_native.npy", "frame0.png"):
            os.replace(work / name, out_dir / name)
        os.replace(work / "meta.json", out_dir / "meta.json")
        src_actions.unlink(missing_ok=True)
        resolver.bot_streak = 0
        resolver.proxy_streak = 0
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


def backfill_native_actions(man: pd.DataFrame, args, log) -> None:
    """Processed clips from before actions_native.npy existed: rebuild it from the stage-01 action files
    (same chunk, first n_frames * frame_stride native frames), so they can be packed with it."""
    need = []
    for mp in Path(args.processed_dir).glob("*/*/meta.json"):
        if not (mp.parent / "actions_native.npy").exists():
            need.append(json.loads(mp.read_text()))
    if not need:
        log.info("backfill: every processed clip already has actions_native.npy")
        return
    by_uuid = {m["uuid"]: m for m in need}
    sub = man[man.uuid.isin(by_uuid)]
    n_ok = 0
    for shard, sdf in tqdm(sub.groupby("shard"), desc="backfill native actions"):
        t = ds.dataset(Path(args.meta_dir) / "actions" / f"SHARD_{int(shard):04d}.parquet").to_table(
            columns=["uuid", "n_frames", "actions"], filter=pc.field("uuid").isin(sdf["uuid"].tolist()))
        for u, n, b in zip(t.column("uuid").to_pylist(), t.column("n_frames").to_pylist(), t.column("actions").to_pylist()):
            m = by_uuid[u]
            a = np.frombuffer(b, np.float32).reshape(n, ACTION_DIM)[: int(m["n_frames"]) * int(m["frame_stride"])]
            pooled = np.load(processed_dir_for(args.processed_dir, m["video_id"], u) / "actions.npy")
            if np.abs(pool_actions(a, int(m["frame_stride"]), int(m["n_frames"])) - pooled).max() > 1e-4:
                log.warning("backfill: %s native actions don't pool to the stored ones, skipped", u)
                continue
            np.save(processed_dir_for(args.processed_dir, m["video_id"], u) / "actions_native.npy", a.astype(np.float16))
            n_ok += 1
    log.info("backfill: wrote actions_native.npy for %d / %d clips", n_ok, len(need))


def apply_layout_overrides(man: pd.DataFrame, args, log) -> pd.DataFrame:
    """Fill bbox_game_area / extend bbox_others from 02c's per-video layout detection."""
    p = Path(args.layout_overrides)
    has_ga = man["bbox_game_area"].map(lambda s: isinstance(_jload(s), dict))
    if not p.exists():
        if args.require_layout_check:
            log.warning("--require-layout-check but %s missing: skipping %d chunks without bbox_game_area", p, (~has_ga).sum())
            return man[has_ga]
        log.warning("No %s: stream layouts (chat/webcam panels) in videos without bbox_game_area will NOT be "
                    "cropped. Run 02c_detect_layout.py first.", p.name)
        return man
    lo = pd.read_parquet(p)
    ok = lo[lo["status"] == "ok"].set_index("video_id")
    vid = man["video_id"]
    # final game area = metadata area (or full frame) ∩ 02c area (or full frame); skip suspicious shrinkage
    new_ga, changed, suspicious = [], [], 0
    for meta_s, v in zip(man["bbox_game_area"], vid):
        meta_ga = _jload(meta_s)
        det = _jload(ok.at[v, "game_area"]) if v in ok.index and isinstance(ok.at[v, "game_area"], str) else None
        if not isinstance(det, dict):
            new_ga.append(meta_s)
            changed.append(False)
            continue
        base = meta_ga if isinstance(meta_ga, dict) else {"xtl": 0.0, "ytl": 0.0, "xbr": 1.0, "ybr": 1.0}
        inter = {"xtl": max(base["xtl"], det["xtl"]), "ytl": max(base["ytl"], det["ytl"]),
                 "xbr": min(base["xbr"], det["xbr"]), "ybr": min(base["ybr"], det["ybr"])}
        a_base = (base["xbr"] - base["xtl"]) * (base["ybr"] - base["ytl"])
        a_int = max(0.0, inter["xbr"] - inter["xtl"]) * max(0.0, inter["ybr"] - inter["ytl"])
        if a_int < 0.3 * a_base:  # boxes barely overlap: don't trust the combination
            suspicious += 1
            new_ga.append(meta_s)
            changed.append(False)
        else:
            new_ga.append(json.dumps(inter))
            changed.append(True)
    man["bbox_game_area"] = new_ga
    fill = pd.Series(changed, index=man.index)
    man["layout_override"] = fill
    # per-video final content rectangle from 02c's multi-probe consensus (preferred over per-chunk detection)
    man["content_rel"] = vid.map(ok["content_rel"]) if "content_rel" in ok else None
    man["stream_layout"] = vid.map(ok["stream_layout"]).fillna(False).astype(bool) if "stream_layout" in ok else False
    if suspicious:
        log.warning("Layout: %d chunks where the 02c box barely overlaps the metadata box -> kept metadata", suspicious)
    with_ov = vid.isin(ok.index[ok["overlays"].fillna("[]") != "[]"])

    def merge_others(row_others, video):
        cur = _jload(row_others) or []
        extra = [{"xtl": b[0], "ytl": b[1], "xbr": b[2], "ybr": b[3]} for b in json.loads(ok.at[video, "overlays"])]
        return json.dumps(cur + extra)
    man.loc[with_ov, "bbox_others"] = [merge_others(o, v) for o, v in zip(man.loc[with_ov, "bbox_others"], vid[with_ov])]
    keep = pd.Series(True, index=man.index)
    if args.require_layout_check:  # 02c now checks every video (it also refines metadata boxes)
        keep &= vid.isin(lo.set_index("video_id").index)
    if args.min_video_gameplay_frac > 0:
        keep &= ~vid.map(ok["frac_gameplay"]).fillna(1.0).lt(args.min_video_gameplay_frac)
    if args.max_video_3d_frac < 1 and "frac_3d" in ok.columns:  # older 02c rows have no answer: kept, 05 filters per clip
        n3d = vid.map(ok["frac_3d"]).gt(args.max_video_3d_frac)
        keep &= ~n3d
        log.info("02c judged %d videos 3D (%s chunks skipped)", vid[n3d].nunique(), f"{n3d.sum():,}")
    log.info("Layout overrides: game area filled for %s chunks (%d videos), overlays added for %s chunks; "
             "%s chunks skipped by layout filters", f"{fill.sum():,}", vid[fill].nunique(), f"{with_ov.sum():,}",
             f"{(~keep).sum():,}")
    return man[keep]


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


def finished_uuids(args) -> set[str]:
    """uuids stage 05 packed into shards or dropped (their processed dirs may be gone after --delete-processed)."""
    out: set[str] = set()
    root = Path(args.webdataset_dir)
    for f in [*root.glob("*/_packed_uuids.txt"), *root.glob("*/_dropped_uuids.txt"), *root.glob("_hub_packed/*.txt")]:
        out.update(f.read_text().split())
    return out


def print_status(man: pd.DataFrame, args) -> None:
    finished = finished_uuids(args)
    done = np.array([u in finished or (processed_dir_for(args.processed_dir, v, u) / "meta.json").exists()
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
    metas = sorted(Path(args.processed_dir).glob("*/*/meta.json"), key=lambda p: p.stat().st_mtime)[-300:]
    tims = [json.loads(p.read_text()).get("timing") for p in metas]
    tims = [t for t in tims if t]
    if tims:
        med = {k: float(np.median([t.get(k, 0) for t in tims])) for k in ("resolve_s", "fetch_decode_s", "process_encode_s", "total_s")}
        print(f"  per-clip median over last {len(tims)} clips: resolve {med['resolve_s']:.1f}s | download+decode "
              f"{med['fetch_decode_s']:.1f}s | process+encode {med['process_encode_s']:.1f}s | total {med['total_s']:.1f}s"
              f" | GPU-decoded {np.mean([bool(t.get('gpu_decode')) for t in tims]):.0%}")
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
    ap.add_argument("--max-starts-per-min", type=float, default=60.0,
                    help="max yt-dlp URL resolves per minute (one per video while cached; ffmpeg range fetches are not limited)")
    ap.add_argument("--shard", default=None, help="i/N: process only videos with crc32(video_id) %% N == i (multi-machine)")
    ap.add_argument("--video-group-window", type=int, default=20000,
                    help="within each window of N clips (priority order), process a video's chunks back to back so one "
                         "URL resolve serves them all (0 = strict priority order)")
    ap.add_argument("--rate-penalty", type=float, default=120.0, help="global pause (s) after a 429/bot check")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-failed", action="store_true", help="retry chunks marked permanently failed")
    ap.add_argument("--method", choices=["auto", "direct", "sections"], default="auto")
    ap.add_argument("--max-src-height", type=int, default=720,
                    help="source resolution cap. The selector prefers ~480p and uses up to this when no 480p exists "
                         "(YouTube's 360p is too compressed); raised automatically (up to 1080) for small game areas")
    ap.add_argument("--patch", type=int, default=16, help="model patch size in px; buckets are in patch units")
    ap.add_argument("--buckets", default=DEFAULT_BUCKETS,
                    help="'auto' (canonical game aspects + gap fillers, ~256 tokens each) or an explicit "
                         "list 'WxH,...' in patches, e.g. '21x12,19x14,16x16'")
    ap.add_argument("--max-crop", type=float, default=0.05,
                    help="max fraction of the content cropped to fill a bucket; beyond this, pad instead")
    ap.add_argument("--lock-geometry", action=argparse.BooleanOptionalAction, default=True,
                    help="decide crop + bucket once per video (first good chunk) and reuse it for all its chunks")
    ap.add_argument("--trim-static-bars", action=argparse.BooleanOptionalAction, default=True,
                    help="for videos 02c flagged as stream layouts, trim a bottom strip that stays static all clip "
                         "(stream status bars the VLM missed)")
    ap.add_argument("--detect-borders", action=argparse.BooleanOptionalAction, default=True,
                    help="detect static black pillar/letterbox bars inside the video (4:3 game in a 16:9 video)")
    ap.add_argument("--border-thresh", type=float, default=24.0, help="max brightness (0-255) of a bar")
    ap.add_argument("--analysis-max-dim", type=int, default=0,
                    help="intermediate decode size (long side); 0 = auto (>= 1.5x the largest bucket side)")
    ap.add_argument("--target-fps", type=float, default=15.0,
                    help="output fps (0 = native). Per-chunk integer stride: 60->15 keeps every 4th frame, 30->15 "
                         "every 2nd; actions pooled over each stride window (sticks mean, buttons OR, so taps survive)"),
    ap.add_argument("--mask-others", action=argparse.BooleanOptionalAction, default=True,
                    help="also inpaint bbox_others regions (webcams, banners)")
    ap.add_argument("--mask-dilate", type=int, default=2, help="mask dilation in output pixels")
    ap.add_argument("--inpaint-radius", type=float, default=3.0)
    ap.add_argument("--max-pad-frames", type=int, default=6, help="max tail frames to pad by repetition")
    ap.add_argument("--webdataset-dir", default=str(DATA_ROOT / "webdataset"),
                    help="stage 05 output: clips listed in its _packed_uuids.txt / _dropped_uuids.txt count as done")
    ap.add_argument("--backfill-native", action="store_true",
                    help="add actions_native.npy to already-processed clips from the stage-01 files, then exit")
    ap.add_argument("--hwaccel", choices=["auto", "cuda", "none"], default="auto",
                    help="decode on the GPU with NVDEC (auto = if ffmpeg has CUDA and nvidia-smi exists); falls back "
                         "to CPU after a failure")
    ap.add_argument("--vcodec", default="libx264", help="libx264 | h264_nvenc | libx265 ...")
    ap.add_argument("--enc-args", default="-preset medium -crf 18 -pix_fmt yuv444p",
                    help="encoder args; lossless: '-preset veryslow -crf 0 -pix_fmt yuv444p'; "
                         "NVENC: '-preset p7 -tune hq -rc constqp -qp 10 -pix_fmt yuv444p'")
    ap.add_argument("--gop", type=int, default=0, help="keyframe interval in frames; 0 = 2 s at the output fps")
    ap.add_argument("--url-ttl", type=float, default=3 * 3600)
    ap.add_argument("--download-timeout", type=float, default=300)
    ap.add_argument("--min-free-gb", type=float, default=50.0)
    ap.add_argument("--cookies", default=None, help="Netscape cookies.txt (default: repo www.youtube.com_cookies.txt if present)")
    ap.add_argument("--cookies-from-browser", default=None)
    ap.add_argument("--proxy", default=None,
                    help="proxy for yt-dlp AND the ffmpeg fetch (googlevideo URLs are IP-bound), e.g. "
                         "http://user:pass@host:port (ffmpeg needs http; socks works only with --method sections)")
    ap.add_argument("--max-bot-streak", type=int, default=30,
                    help="stop submitting after this many consecutive bot-check/429 failures (IP is blocked)")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    ap.add_argument("--layout-overrides", default=str(DATA_ROOT / "layout_overrides.parquet"),
                    help="per-video game areas / overlay boxes from 02c_detect_layout.py (used if the file exists)")
    ap.add_argument("--require-layout-check", action="store_true",
                    help="skip videos without bbox_game_area that 02c hasn't checked yet (avoids encoding stream layouts)")
    ap.add_argument("--max-video-3d-frac", type=float, default=0.5,
                    help="skip videos where more than this share of 02c's probe frames looked 3D (1 = off)")
    ap.add_argument("--min-video-gameplay-frac", type=float, default=0.0,
                    help="skip videos where 02c saw gameplay in fewer than this fraction of its probe frames")
    args = ap.parse_args()
    args.cookies = default_cookies(args.cookies)

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
    _HWACCEL["on"] = args.hwaccel == "cuda" or (args.hwaccel == "auto" and hwaccel_available())
    log.info("Decoding on %s", "GPU (NVDEC, -hwaccel cuda)" if _HWACCEL["on"] else "CPU")
    man = pd.read_parquet(args.manifest)
    man = apply_layout_overrides(man, args, log)
    if args.split != "all":
        man = man[man.split == args.split]
    if args.shard:  # several machines: each takes a disjoint, stable set of videos
        i, n = (int(x) for x in args.shard.split("/"))
        man = man[man.video_id.map(lambda v: zlib.crc32(str(v).encode()) % n == i)]
        log.info("Machine shard %d/%d: %s chunks from %s videos", i, n, f"{len(man):,}", f"{man.video_id.nunique():,}")
    # interleave games so early progress covers many titles (and spreads load across videos)
    if "download_priority" in man.columns:
        # balanced-first order: any prefix of the run is a per-game-capped ("water-filling") subset
        man = man.sort_values("download_priority").reset_index(drop=True)
    else:
        man = man.sample(frac=1.0, random_state=0).reset_index(drop=True)
    Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
    if args.backfill_native:
        backfill_native_actions(pd.read_parquet(args.manifest), args, log)
        return
    if args.status:
        print_status(man, args)
        return

    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    Path(args.tmp_dir).mkdir(parents=True)
    failed_path = Path(args.processed_dir) / "_failed.jsonl"
    failed = load_failed(failed_path)
    # done = processed on disk, or already packed into shards / dropped by stage 05 (whose --delete-processed
    # removed the clip dir): without the latter, every bulk cycle would re-download the previous cycles' clips
    finished = finished_uuids(args)
    done_mask = np.array([u in finished or (processed_dir_for(args.processed_dir, v, u) / "meta.json").exists()
                          for v, u in zip(man.video_id, man.uuid)])
    if finished:
        log.info("%s clips already packed or dropped by stage 05 (%s)", f"{len(finished):,}", args.webdataset_dir)
    todo = man[~done_mask]
    if not args.retry_failed:
        todo = todo[todo.uuid.map(lambda u: failed.get(u, {}).get("kind") != "permanent")]
    ok_fps = todo.fps.map(lambda f: fps_stride(snap_fps(float(f)), args.target_fps) is not None)
    if (~ok_fps).any():
        log.warning("Skipping %d chunks whose fps can't reach --target-fps %.0f with an integer stride: %s",
                    (~ok_fps).sum(), args.target_fps, todo.fps[~ok_fps].round().value_counts().to_dict())
        todo = todo[ok_fps]
    if args.video_group_window > 0 and len(todo):
        # Balanced-first order puts a different video on almost every clip, so every clip paid a fresh yt-dlp
        # URL resolve (rate-limited, seconds each). Within each window of the priority order, process a video's
        # chunks back to back: one resolve serves all of them; balance across windows is unchanged.
        todo = todo.reset_index(drop=True)
        todo["_win"] = np.arange(len(todo)) // args.video_group_window
        todo = todo.sort_values(["_win", "video_id", "chunk_id"]).drop(columns="_win").reset_index(drop=True)
    if args.limit:  # after grouping: a cycle takes whole runs of a video's chunks, not one chunk from each of
        todo = todo.head(args.limit)  # thousands of videos (every new video costs a rate-limited URL resolve)
    if args.video_group_window > 0 and len(todo):
        log.info("Grouped by video within windows of %d: %s clips from %s videos (%.1f clips per URL resolve)",
                 args.video_group_window, f"{len(todo):,}", f"{todo.video_id.nunique():,}",
                 len(todo) / max(1, todo.video_id.nunique()))
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
                if PROXY_ERR.search(str(e)):
                    return {"status": "transient"}  # proxy down: don't burn retries, main loop stops
                if attempt == args.retries - 1:
                    record_failure(row, "transient", e)
                    log.warning("%s failed after %d attempts: %s", row.uuid, args.retries, str(e)[:300])
                    log.debug(traceback.format_exc())
                    return {"status": "transient"}
                time.sleep(5 * 2 ** attempt)

    rows = list(todo.itertuples(index=False))
    proxy_down = False
    pbar = tqdm(total=len(rows), desc="clips", unit="clip")
    with ThreadPoolExecutor(args.workers) as ex:
        pending, it = set(), iter(rows)
        stop = False
        while True:
            if not stop and resolver.proxy_streak >= 3:
                log.error("Cannot connect to the proxy %s (%d failures in a row). Is the SSH tunnel / proxy running? "
                          "Stopping; nothing was marked failed, re-run when it's back.", args.proxy, resolver.proxy_streak)
                stop = True
                proxy_down = True
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
    if proxy_down:
        raise SystemExit(3)  # distinct exit code: run_pipeline waits for the proxy and retries


if __name__ == "__main__":
    main()
