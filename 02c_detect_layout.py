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
import logging
import base64
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ng_common import (ServerDown, wait_for_server, add_data_root_arg, default_cookies, DATA_ROOT, DEFAULT_VLM, atomic_write_parquet, openai_chat_many, setup_logging, strip_thinking,
                       vllm_chat_kwargs, vllm_mm_kwargs)

_spec = importlib.util.spec_from_file_location("stage03", Path(__file__).with_name("03_download_and_process.py"))
S3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S3)

# Native grounding-style prompts (the format Qwen-VL models are trained on: "Locate ..., output bbox coordinates in
# JSON format" -> [{"bbox_2d": [x1, y1, x2, y2], "label": ...}]). The coordinate convention is whatever the model
# natively uses; calibrate_coords() detects 0-1000 vs pixels at startup.
PROMPT_GAME = ("Locate the game screen in this image: the rectangle where the video game itself is rendered (its world, "
               "characters and the game's own HUD). Exclude livestream elements around it such as chat, webcam, status or "
               "ticker bars, decorative borders and blurred backgrounds. If the game fills the whole image, the game "
               "screen is the whole image. Output its bbox coordinates in JSON format with the label \"game screen\".")

PROMPT_LAYOUT = ("Locate every livestream or recording element in this image that is not part of the video game: chat panel, "
                 "webcam, status bar (streamer name, donation or follower goals, counters, clock, social handles), "
                 "decorative border, blurred background, alert, input display (on-screen controller). Do not include the "
                 "game's own HUD (health bars, score, lives, minimap). Output bbox coordinates in JSON format, with the label "
                 "set to one of: chat, webcam, status_bar, border, background, alert, input_display. Output [] if there are none.")

PROMPT_SCREEN = ("What is shown in this frame of a gameplay video? Answer with exactly two words. First word, one of: gameplay, "
                 "menu, title, loading, cutscene, map_or_inventory, black, not_a_game. Second word, the game's view: 2d if "
                 "the player moves on a flat plane seen from a fixed side, overhead or isometric camera (side-scrollers, "
                 "top-down, fighting games, and 2.5D games drawn with 3D models but played on a 2D plane), 3d if the player "
                 "moves freely in depth with a camera behind or around them, none if no game world is visible.")
SCREENS = ["gameplay", "menu", "title", "loading", "cutscene", "map_or_inventory", "black", "not_a_game"]


def fit28(w: int, h: int, long_side: int = 560) -> tuple[int, int]:
    """Resize target whose sides are multiples of 28, so Qwen2.5-VL sees the image unresized and its
    pixel coordinates map back exactly."""
    s = long_side / max(w, h)
    return max(28, int(round(w * s / 28)) * 28), max(28, int(round(h * s / 28)) * 28)


def to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _json(text: str):
    m = re.search(r"\{.*\}", strip_thinking(text), re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _box(b, w: int, h: int):
    """Box -> relative [0,1] coords. The prompt asks for 0-1000 normalized coordinates; if a model answers in
    pixels anyway (values beyond 1000), interpret them in the image's pixel space (w, h)."""
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in b)
    except (TypeError, ValueError):
        return None
    if max(x1, y1, x2, y2) <= 1000.5 and NORMALIZED:
        w = h = 1000
    x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
    y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
    if x2 - x1 < 0.03 * w or y2 - y1 < 0.02 * h:
        return None
    return [x1 / w, y1 / h, x2 / w, y2 / h]  # relative


BUCKETS: list = []
NORMALIZED = True  # prompts request 0-1000 coordinates (set False for models that only speak pixels)
PANEL_TYPES = {"chat", "status_bar", "border", "background", "alert", "other", "logo"}
OVERLAY_TYPES = {"webcam", "input_display", "logo", "alert"}


def _grounding_items(text: str) -> list[dict] | None:
    """Parse native grounding output: a JSON list of {"bbox_2d"|"bbox"|"box": [...], "label": ...} (possibly in a
    ```json fence, or wrapped in an object). Returns None if nothing parseable."""
    t = strip_thinking(text)
    m = re.search(r"\[.*\]", t, re.S)
    cand = None
    if m:
        try:
            cand = json.loads(m.group(0))
        except json.JSONDecodeError:
            cand = None
    if cand is None:
        d = _json(t)
        if isinstance(d, dict):
            cand = d.get("elements") or d.get("objects") or ([d] if any(k in d for k in ("bbox_2d", "bbox", "box", "game_box")) else [])
    if not isinstance(cand, list):
        return None
    if cand and all(isinstance(v, (int, float)) for v in cand):  # a bare [x1, y1, x2, y2]
        cand = [{"bbox_2d": cand}]
    out = []
    for it in cand:
        if isinstance(it, dict):
            b = it.get("bbox_2d") or it.get("bbox") or it.get("box") or it.get("game_box")
            if b is not None:
                out.append({"box": b, "label": str(it.get("label", "")).lower().strip()})
    return out


def parse_game(text: str, w: int, h: int) -> dict | None:
    items = _grounding_items(text)
    if not items:
        return None
    boxes = [b for b in (_box(i["box"], w, h) for i in items) if b is not None]
    if not boxes:
        return None
    g = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))  # if several, the largest is the game screen
    return {"game": g, "screen": ""}


def parse_layout(text: str, w: int, h: int) -> list[tuple[str, list[float]]] | None:
    items = _grounding_items(text)
    if items is None:
        return None
    out = []
    for i in items:
        b = _box(i["box"], w, h)
        if b and (b[2] - b[0]) * (b[3] - b[1]) <= 0.6:  # an "element" covering most of the frame = the game itself
            out.append((i["label"] or "other", b))
    return out


def parse_screen(text: str) -> str:
    t = strip_thinking(text).lower()
    hits = [(t.find(s_), s_) for s_ in SCREENS if s_ in t]
    return min(hits)[1] if hits else ""


def parse_view(text: str) -> str:
    """'2d' | '3d' | '' (none / unparsed) from the second word of the screen answer."""
    m = re.search(r"\b(2\.5|2|3)\s*-?d\b", strip_thinking(text).lower())
    return ("3d" if m.group(1) == "3" else "2d") if m else ""  # 2.5D counts as 2D


def parse(text: str, w: int, h: int) -> dict | None:
    """Back-compat single-prompt parser (game_box + overlays_in_game/layout_panels)."""
    d = _json(text)
    if not isinstance(d, dict):
        return None
    g = _box(d.get("game_box"), w, h)
    if g is None:
        return None
    els = [("other", b) for b in (_box(o, w, h) for o in list(d.get("overlays_in_game") or []) + list(d.get("layout_panels") or []))
           if b is not None and (b[2] - b[0]) * (b[3] - b[1]) <= 0.6]
    return {"game": g, "overlays": [b for _, b in els], "elements": els, "screen": str(d.get("screen", "")).lower(),
            "stream": bool(d.get("is_stream_layout", False))}


class LocalVLM:
    """vLLM in-process (loads the model; ~minutes)."""
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=args.model, max_model_len=8192, limit_mm_per_prompt={"image": 1},
                       gpu_memory_utilization=args.gpu_mem, enforce_eager=args.enforce_eager, seed=0,
                       **vllm_mm_kwargs(args.model, 640 * 28 * 28))
        self.sp = SamplingParams(temperature=0.0, max_tokens=400)
        self.kw = vllm_chat_kwargs(args.model)

    def chat(self, convs):
        return [o.outputs[0].text for o in self.llm.chat(convs, self.sp, use_tqdm=False, **self.kw)] if convs else []


class ServerVLM:
    """A running OpenAI-compatible server (`vllm serve <model>`): no model load, shareable with stage 04."""
    def __init__(self, args):
        from openai import OpenAI
        self.client = OpenAI(base_url=args.api_base, api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.model, self.conc, self.api_base = args.model, args.concurrency, args.api_base
        self.log = logging.getLogger("02c_detect_layout")

    def chat(self, convs):
        while convs:  # a crashed/restarting server pauses the batch instead of recording empty answers
            try:
                return openai_chat_many(self.client, self.model, convs, 400, self.conc)
            except ServerDown as e:
                self.log.warning("%s: waiting for it, then retrying this batch", e)
                wait_for_server(self.api_base, self.log)
        return []


def calibrate_coords(vlm, log) -> bool:
    """Ask the model for the box of a white rectangle at a known spot in a 576x320 image and decide whether it
    answers in 0-1000 normalized coordinates (as instructed) or in pixels. Returns NORMALIZED."""
    w, h = 576, 320
    img = Image.new("RGB", (w, h), (0, 0, 0))
    img.paste((255, 255, 255), (288, 40, 560, 280))  # normalized ~ [500, 125, 972, 875]; pixels = [288, 40, 560, 280]
    q = 'Locate the white rectangle in this image, output its bbox coordinates in JSON format.'
    out = vlm.chat([[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": to_data_uri(img)}},
                                                  {"type": "text", "text": q}]}]])
    items = _grounding_items(out[0]) or []
    b = items[0]["box"] if items else None
    try:
        x1, y1, x2, y2 = (float(v) for v in b)
    except (TypeError, ValueError):
        log.warning("Coordinate calibration unparsable (%r); assuming 0-1000 normalized", out[0][:120])
        return True
    err_norm = abs(x1 - 500) + abs(y1 - 125) + abs(x2 - 972) + abs(y2 - 875)
    err_px = abs(x1 - 288) + abs(y1 - 40) + abs(x2 - 560) + abs(y2 - 280)
    normalized = err_norm <= err_px
    log.info("Coordinate calibration: model answered %s -> using %s coordinates", [x1, y1, x2, y2],
             "0-1000 normalized" if normalized else "PIXEL")
    return normalized


def combine(game: dict | None, elements: list | None, screen: str = "", view: str = "") -> dict | None:
    """One frame: game box from the game question, refined by the elements from the layout question."""
    if game is None and elements is None:
        return None
    g = game["game"] if game else [0.0, 0.0, 1.0, 1.0]
    els = elements or []
    return {"game": g, "overlays": [b for _, b in els], "elements": els,
            "screen": screen or (game["screen"] if game else ""), "view": view,
            "stream": bool(els) or g != [0.0, 0.0, 1.0, 1.0]}


# ============================================================================ frame grabbing
def _decode_raw(input_args: list[str], src_wh: tuple[int, int], seconds: float, args) -> np.ndarray | None:
    """Decode `seconds` of video at 15 fps, long side --probe-size, to an (T, h, w, 3) uint8 array."""
    sw, sh = src_wh
    s_ = args.probe_size / max(sw, sh)
    w, h = max(2, int(round(sw * s_ / 2)) * 2), max(2, int(round(sh * s_ / 2)) * 2)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *input_args, "-an", "-sn",
           "-vf", f"fps=15,scale={w}:{h}:flags=area,format=rgb24", "-frames:v", str(int(seconds * 15)),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=args.net_timeout * 3).stdout
    except subprocess.TimeoutExpired:
        return None
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3) if n >= 5 else None


def grab_probes(vid: str, url: str, times: list[float], tmp: Path, args, resolver, limiter) -> tuple[list[np.ndarray], str]:
    """Short probe clips (--probe-seconds at 15 fps) at `times`: the VLM looks at each probe's middle frame,
    the pixel checks (black bars, static status bar) use all of its frames. Direct stream URL with hard network
    timeouts; yt-dlp sections as a fallback (switched to once per video). Returns (probes, method)."""
    probes: list[np.ndarray] = []
    fmt = S3.format_selector(15.0, args.max_src_height)
    try:
        info = resolver.resolve(vid, url, fmt, args.max_src_height, limiter)
    except S3.PermanentError:
        raise
    except Exception:
        info = None
    method = "direct"
    use_direct = bool(info and info.get("url") and info.get("width") and info.get("height"))
    for t in times:
        arr = None
        if use_direct:
            hdr = "".join(f"{k}: {v}\r\n" for k, v in (info.get("headers") or {}).items())
            inp = ["-rw_timeout", str(int(args.net_timeout * 1e6)), *(["-headers", hdr] if hdr else []),
                   *(["-http_proxy", args.proxy] if args.proxy and args.proxy.startswith("http") else []),
                   "-ss", f"{t:.2f}", "-t", f"{args.probe_seconds + 0.5:.2f}", "-i", info["url"]]
            arr = _decode_raw(inp, (info["width"], info["height"]), args.probe_seconds, args)
            if arr is None:
                use_direct = False
        if arr is None:
            method = "sections"
            try:
                raw = S3.download_section(url, fmt, t, t + args.probe_seconds + 0.5, tmp, args)
                arr = _decode_raw(["-i", str(raw)], S3.probe_dims(raw), args.probe_seconds, args)
                raw.unlink(missing_ok=True)
            except S3.PermanentError:
                raise
            except Exception:
                arr = None
            if arr is None:
                break
        probes.append(arr)
    return probes, method


def video_metadata_area(g: pd.DataFrame) -> dict | None:
    """Most common NitroGen bbox_game_area of a video's chunks (None = full frame)."""
    vals = [s_ for s_ in g["bbox_game_area"] if isinstance(S3._jload(s_), dict)]
    return S3._jload(pd.Series(vals).mode().iat[0]) if vals else None


def pixel_trims(probes: list[np.ndarray], game_rel: list[float], stream: bool) -> dict:
    """Black-bar / static-status-bar trims inside the game box, decided by CONSENSUS over the probes
    (bars: median per side; bottom status bar: must be found in at least half the probes)."""
    x0, y0, x1, y1 = game_rel
    sides, bands, used = [], [], 0
    for arr in probes:
        h, w = arr.shape[1:3]
        sub = arr[:, int(round(y0 * h)):max(int(round(y0 * h)) + 2, int(round(y1 * h))),
                  int(round(x0 * w)):max(int(round(x0 * w)) + 2, int(round(x1 * w)))]
        if sub.mean() < 8:  # mostly black probe (fade / loading): no evidence
            continue
        used += 1
        frames = list(sub)
        sh, sw = sub.shape[1:3]
        L, T, R, B = S3.detect_borders(frames, 24.0)
        sides.append((L / sw, T / sh, R / sw, B / sh))
        band = S3.static_bottom_band(frames, (L, T, R, B)) if stream else 0
        bands.append(band / sh)
    if not used:
        return {"l": 0.0, "t": 0.0, "r": 0.0, "b": 0.0, "band": 0.0, "probes_used": 0}
    med = np.median(np.array(sides), axis=0)
    pos = [b for b in bands if b > 0]
    band = float(np.median(pos)) if len(pos) >= max(1, int(np.ceil(used / 2))) else 0.0
    return {"l": float(med[0]), "t": float(med[1]), "r": float(med[2]), "b": float(med[3]), "band": band,
            "band_votes": f"{len(pos)}/{used}", "probes_used": used}


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
    # Panels the VLM reports as "overlays" that touch an edge of the game box and span most of it (chat
    # columns, status bars, decorative bezels) are layout, not in-game overlays: crop them off instead of
    # inpainting a huge region. Remaining overlays (webcams etc. inside the picture) stay for inpainting.
    g = g.copy()
    inner = []
    for o in keep:
        h_span = (min(o[3], g[3]) - max(o[1], g[1])) / max(g[3] - g[1], 1e-6)
        w_span = (min(o[2], g[2]) - max(o[0], g[0])) / max(g[2] - g[0], 1e-6)
        if o[0] <= g[0] + 0.03 and h_span >= 0.6 and o[2] < g[0] + 0.5 * (g[2] - g[0]):
            g[0] = max(g[0], o[2])
        elif o[2] >= g[2] - 0.03 and h_span >= 0.6 and o[0] > g[0] + 0.5 * (g[2] - g[0]):
            g[2] = min(g[2], o[0])
        elif o[1] <= g[1] + 0.03 and w_span >= 0.6 and o[3] < g[1] + 0.5 * (g[3] - g[1]):
            g[1] = max(g[1], o[3])
        elif o[3] >= g[3] - 0.03 and w_span >= 0.6 and o[1] > g[1] + 0.5 * (g[3] - g[1]):
            g[3] = min(g[3], o[1])
        else:
            gw, gh = max(g[2] - g[0], 1e-6), max(g[3] - g[1], 1e-6)
            frac = (o[2] - o[0]) * (o[3] - o[1]) / (gw * gh)
            inside = (min(o[2], g[2]) - max(o[0], g[0])) > 0 and (min(o[3], g[3]) - max(o[1], g[1])) > 0
            if inside and 0.015 <= frac <= 0.30:  # webcam / input display on top of the picture: inpaint
                inner.append(o)
    keep = inner
    area = (g[2] - g[0]) * (g[3] - g[1])
    full = (g[0] <= 1 - full_thresh and g[1] <= 1 - full_thresh and g[2] >= full_thresh and g[3] >= full_thresh)
    screens = [r["screen"] for r in results]
    return {"game_area": None if full else json.dumps({"xtl": g[0], "ytl": g[1], "xbr": g[2], "ybr": g[3]}),
            "game_area_frac": float(area), "box_spread": spread, "overlays": json.dumps(keep),
            "frac_gameplay": float(np.mean([s == "gameplay" for s in screens])),
            # share of probes judged 3D among those with a 2d/3d answer (NaN = no answer: old cache or no gameplay)
            "frac_3d": (float(np.mean([r.get("view") == "3d" for r in results if r.get("view")]))
                        if any(r.get("view") for r in results) else float("nan")),
            "stream_layout": bool(np.mean([r["stream"] for r in results]) >= 0.5), "n_frames": len(results)}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_root_arg(ap)
    ap.add_argument("--manifest", default=str(DATA_ROOT / "manifest_2d.parquet"))
    ap.add_argument("--out", default=str(DATA_ROOT / "layout_overrides.parquet"))
    ap.add_argument("--tmp-dir", default=str(DATA_ROOT / "tmp_02c"))
    ap.add_argument("--scope", choices=["missing", "all"], default="all",
                    help="'all' (default): NitroGen's own game areas often still include stream bars, so check every "
                         "video; 'missing' = only videos without bbox_game_area")
    ap.add_argument("--frames-per-video", type=int, default=3, help="probe clips per video (spread over its chunks)")
    ap.add_argument("--full-thresh", type=float, default=0.97, help="box edges beyond this => full frame, no crop")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32, help="videos per VLM batch")
    ap.add_argument("--model", default=DEFAULT_VLM,
                    help="any vLLM-supported image model (default Qwen3.8-27B FP8, fits a 48 GB GPU)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--backend", choices=["vllm", "openai"], default="vllm",
                    help="'openai' = use a running `vllm serve` / OpenAI-compatible server (no model load per run)")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--concurrency", type=int, default=32, help="parallel requests to the server")
    ap.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    # network (same semantics as stage 03)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--cookies", default=None, help="Netscape cookies.txt (default: repo www.youtube.com_cookies.txt if present)")
    ap.add_argument("--cookies-from-browser", default=None)
    ap.add_argument("--max-starts-per-min", type=float, default=60.0, help="URL resolutions per minute (1 per video)")
    ap.add_argument("--rate-penalty", type=float, default=120.0)
    ap.add_argument("--url-ttl", type=float, default=3 * 3600)
    ap.add_argument("--download-timeout", type=float, default=60)
    ap.add_argument("--net-timeout", type=float, default=20, help="seconds without network progress before a fetch fails")
    ap.add_argument("--probe-seconds", type=float, default=4.0, help="length of each probe clip (pixel checks)")
    ap.add_argument("--probe-size", type=int, default=576, help="probe decode size (long side)")
    ap.add_argument("--patch", type=int, default=16, help="only for the predicted-bucket report")
    ap.add_argument("--max-src-height", type=int, default=480, help="probe-frame resolution (layout needs little detail)")
    args = ap.parse_args()
    args.cookies = default_cookies(args.cookies)

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
        # "unparsed" is retried too: before ServerDown existed, a crashed model server produced empty answers
        # that were recorded as unparsed (and those videos then skipped the crop fix)
        final = done["status"].isin(["ok"]) | done["status"].astype(str).str.startswith("permanent")
        done = done[final]
    todo = [v for v in order if v not in set(done["video_id"])]
    if args.limit:
        todo = todo[: args.limit]
    log.info("Videos: %d in scope (%s), %d done, %d to check", len(order), args.scope, len(done), len(todo))
    if not todo:
        return

    vlm = ServerVLM(args) if args.backend == "openai" else LocalVLM(args)
    global NORMALIZED
    NORMALIZED = calibrate_coords(vlm, log)
    resolver, limiter = S3.StreamResolver(args, log), S3.RateLimiter(args.max_starts_per_min)
    global BUCKETS
    BUCKETS = S3.auto_buckets(S3.CANONICAL_BUCKETS, 0.05)  # only for the predicted-bucket report; 03 decides
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
            frames, method = grab_probes(v, url, times, tmp, args, resolver, limiter)
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
            for fi, arr in enumerate(frames):
                f = Image.fromarray(arr[len(arr) // 2])
                w, h = fit28(*f.size)
                uri = to_data_uri(f.resize((w, h), Image.Resampling.BICUBIC))
                for kind, prompt in (("game", PROMPT_GAME), ("layout", PROMPT_LAYOUT), ("screen", PROMPT_SCREEN)):
                    convs.append([{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": prompt}]}])
                    owners.append((v, fi, kind, w, h))
        outs = vlm.chat(convs)
        answers: dict[tuple[str, int], dict] = {}
        empty: dict[str, int] = {}  # video -> number of questions with NO answer at all (request failed)
        for (v, fi, kind, w, h), txt in zip(owners, outs):
            if not (txt or "").strip():
                empty[v] = empty.get(v, 0) + 1
            answers.setdefault((v, fi), {})[kind] = (parse_game(txt, w, h) if kind == "game" else
                                                     parse_layout(txt, w, h) if kind == "layout" else parse_screen(txt))
            if kind == "screen":
                answers[(v, fi)]["view"] = parse_view(txt)
        per_vid: dict[str, list[dict]] = {}
        for (v, fi), a in sorted(answers.items()):
            r = combine(a.get("game"), a.get("layout"), a.get("screen") or "", a.get("view") or "")
            if r:
                per_vid.setdefault(v, []).append(r)
        for v, frames, err, method, secs in got:
            if frames:
                res = per_vid.get(v, [])
                # a video is "ok" only if EVERY question got an answer: a missing game-box answer would silently
                # mean "full frame, no crop". Incomplete videos are not used by stage 03 and are re-checked next run.
                status = "incomplete" if empty.get(v) else ("ok" if res else "unparsed")
                row = {"video_id": v, "status": status, "fetch_method": method, "fetch_s": round(secs, 1),
                       "n_empty_answers": int(empty.get(v, 0)), "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "vlm_model": args.model, **(aggregate(res, args.full_thresh) if res and status == "ok" else {})}
                if res and status == "ok":
                    row.update(final_geometry(row, frames, vids[vids.video_id == v]))
                rows.append(row)
        flush()

    def final_geometry(row: dict, probes: list[np.ndarray], g: pd.DataFrame) -> dict:
        """Per-video content rectangle = (metadata area ∩ VLM game box) minus consensus pixel trims."""
        det = json.loads(row["game_area"]) if isinstance(row.get("game_area"), str) else None
        meta = video_metadata_area(g)
        full = {"xtl": 0.0, "ytl": 0.0, "xbr": 1.0, "ybr": 1.0}
        a_, b_ = meta or full, det or full
        box = [max(a_["xtl"], b_["xtl"]), max(a_["ytl"], b_["ytl"]), min(a_["xbr"], b_["xbr"]), min(a_["ybr"], b_["ybr"])]
        if (box[2] - box[0]) * (box[3] - box[1]) < 0.3 * (a_["xbr"] - a_["xtl"]) * (a_["ybr"] - a_["ytl"]):
            box = [a_["xtl"], a_["ytl"], a_["xbr"], a_["ybr"]]  # barely overlapping boxes: trust metadata
        tr = pixel_trims(probes, box, bool(row.get("stream_layout")))
        gw, gh = box[2] - box[0], box[3] - box[1]
        x = box[0] + tr["l"] * gw
        y = box[1] + tr["t"] * gh
        w = gw * max(0.2, 1 - tr["l"] - tr["r"])
        h = gh * max(0.2, 1 - tr["t"] - tr["b"] - tr["band"])
        sh, sw = probes[0].shape[1:3]
        bucket = S3.choose_bucket(w * sw, h * sh, BUCKETS, args.patch, 0.05)["bucket"]
        return {"content_rel": json.dumps([x, y, w, h]), "pred_bucket": bucket, "trims": json.dumps(tr),
                "content_aspect": round(w * sw / max(h * sh, 1e-6), 4)}

    stats = {"ok": 0, "fail": 0, "direct": 0, "sections": 0, "secs": 0.0}
    ready = []
    pbar = tqdm(total=len(todo), desc="videos", unit="vid")
    # Bounded in-flight work: at most ~2x workers fetches pending, and each finished future is dropped right after
    # use. (Submitting every video up front kept ALL finished futures - and their ~34 MB of probe frames each -
    # alive for the whole run: ~34 GB per 1,000 videos, which froze the GPU box when RAM ran out.)
    def results():
        it = iter(todo)
        pending = set()
        with ThreadPoolExecutor(args.workers) as ex:
            for v in it:
                pending.add(ex.submit(fetch, v))
                if len(pending) >= 2 * args.workers:
                    break
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in finished:
                    yield f.result()
                    nxt = next(it, None)
                    if nxt is not None:
                        pending.add(ex.submit(fetch, nxt))
                del finished

    for r in results():
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
