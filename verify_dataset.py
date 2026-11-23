#!/usr/bin/env python
"""Sanity-check packed WebDataset shards and export visual debug grids.

Checks per sample:
  * required members present (video, actions.npy, txt, json, cls)
  * len(frames) == len(actions) == json.n_frames  (exact temporal alignment)
  * actions shape (T, 21) float32, sticks in [-1, 1], buttons in {0, 1}, no NaN/Inf
  * frames uint8 in [0, 255], shape (T, H, W, 3) matching json width/height (patch-divisible), not all
    black, pad bars black (fit-mode clips only)
  * caption non-empty; game_id consistent with game_ids.json; inpaint_boxes present and inside the frame
Global checks: no game appears in both train and val_unseen_games.

Outputs (in --out-dir):
  grid_<split>.png          rows = samples, cols = evenly spaced frames with stick vectors and
                            pressed buttons overlaid, inpainted (loss-masked) regions outlined in red,
                            caption under each row
  debug_<split>_<key>.mp4   (with --debug-videos N) full clip with live action overlay — the best way
                            to eyeball whether stick motion matches on-screen motion

Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np

from ng_common import DATA_ROOT, ACTION_DIM, BUTTONS

BTN_ABBR = ["Dd", "Dl", "Dr", "Du", "LB", "LS", "LT", "RB", "RS", "RT", "A", "X", "B", "Y", "Bk", "St", "G"]


def read_tar(path: Path) -> dict[str, dict[str, bytes]]:
    samples: dict[str, dict[str, bytes]] = defaultdict(dict)
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = Path(m.name).name
            key, _, ext = name.partition(".")
            samples[key][ext] = tf.extractfile(m).read()
    return samples


def decode_video(blob: bytes) -> np.ndarray:
    import av
    with av.open(io.BytesIO(blob)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def check_sample(key: str, s: dict, game_ids: dict, allow_missing_caption: bool = False) -> tuple[list[str], dict]:
    errs = []
    for req in ("actions.npy", "txt", "json", "cls"):
        if req not in s:
            errs.append(f"missing {req}")
    if "mp4" not in s and "frames.npy" not in s:
        errs.append("missing video (mp4/frames.npy)")
    if errs:
        return errs, {}
    meta = json.loads(s["json"])
    acts = np.load(io.BytesIO(s["actions.npy"]))
    frames = decode_video(s["mp4"]) if "mp4" in s else np.load(io.BytesIO(s["frames.npy"]))
    T = len(frames)
    if len(acts) != T:
        errs.append(f"len(frames)={T} != len(actions)={len(acts)}")
    if meta.get("n_frames") != T:
        errs.append(f"json n_frames={meta.get('n_frames')} != len(frames)={T}")
    if acts.dtype != np.float32 or acts.ndim != 2 or acts.shape[1] != ACTION_DIM:
        errs.append(f"actions dtype/shape {acts.dtype} {acts.shape}")
    else:
        if not np.isfinite(acts).all():
            errs.append("actions contain NaN/Inf")
        if acts[:, :4].min() < -1 - 1e-6 or acts[:, :4].max() > 1 + 1e-6:
            errs.append(f"sticks out of [-1,1]: [{acts[:, :4].min():.3f}, {acts[:, :4].max():.3f}]")
        if not np.isin(acts[:, 4:], (0.0, 1.0)).all():
            errs.append("buttons not binary")
    if frames.dtype != np.uint8:
        errs.append(f"frames dtype {frames.dtype}")
    W, H = meta.get("width"), meta.get("height")
    if frames.ndim != 4 or frames.shape[1:] != (H, W, 3):
        errs.append(f"frames shape {frames.shape} != (T, {H}, {W}, 3) from json")
    if meta.get("patch") and (W % meta["patch"] or H % meta["patch"]):
        errs.append(f"{W}x{H} not divisible by patch {meta['patch']}")
    if frames.min() < 0 or frames.max() > 255:
        errs.append("frames out of [0,255]")
    if frames.max() < 10:
        errs.append("clip is entirely black")
    px, py = meta.get("pad", [0, 0])
    if py > 2 and frames[:, : py - 2].mean() > 8:  # padding (fit mode only) should be black (allow ringing)
        errs.append(f"pad rows not black (mean {frames[:, :py - 2].mean():.1f})")
    if px > 2 and frames[:, :, : px - 2].mean() > 8:
        errs.append(f"pad columns not black (mean {frames[:, :, :px - 2].mean():.1f})")
    if "inpaint_boxes" not in meta:
        errs.append("json missing inpaint_boxes (repack with current 05_pack_webdataset.py)")
    for b in meta.get("inpaint_boxes") or []:
        if len(b) != 4 or b[2] <= 0 or b[3] <= 0 or b[0] < 0 or b[1] < 0 or b[0] + b[2] > W or b[1] + b[3] > H:
            errs.append(f"bad inpaint box {b}")
    if not s["txt"].decode().strip() and not allow_missing_caption:
        errs.append("empty caption (run stage 04, or pass --allow-missing-caption)")
    gk = meta.get("game_key")
    if game_ids and gk in game_ids and int(s["cls"]) != game_ids[gk]:
        errs.append(f"cls {s['cls']} != game_ids[{gk}]={game_ids[gk]}")
    return errs, {"frames": frames, "acts": acts, "meta": meta, "caption": s["txt"].decode()}


# ============================================================================ drawing
def draw_overlay(frame: np.ndarray, a: np.ndarray, scale: int = 2, boxes=None) -> np.ndarray:
    import cv2
    img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).copy()
    H, W = img.shape[:2]
    for x, y, w, h in boxes or []:  # loss-masked (inpainted) regions, dashed-looking red outline
        cv2.rectangle(img, (x * scale, y * scale), ((x + w) * scale - 1, (y + h) * scale - 1), (255, 0, 0), 1)
    r = int(22 * scale / 2)
    for k, cx in enumerate((r + 6, W - r - 6)):  # left stick bottom-left, right stick bottom-right
        cy = H - r - 6
        cv2.circle(img, (cx, cy), r, (255, 255, 255), 1, cv2.LINE_AA)
        x, y = a[2 * k], a[2 * k + 1]  # (-1,-1) = top-left, matches image coordinates
        col = (0, 255, 0) if np.hypot(x, y) > 0.2 else (160, 160, 160)
        cv2.arrowedLine(img, (cx, cy), (int(cx + x * r), int(cy + y * r)), col, 2, cv2.LINE_AA, tipLength=0.3)
    bw = (W - 4) // len(BUTTONS)
    for i, ab in enumerate(BTN_ABBR):
        x0 = 2 + i * bw
        on = a[4 + i] > 0.5
        cv2.rectangle(img, (x0, 2), (x0 + bw - 2, 16), (0, 200, 255) if on else (60, 60, 60), -1)
        cv2.putText(img, ab, (x0 + 1, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0) if on else (200, 200, 200), 1)
    return img


def make_grid(items: list[dict], n_cols: int, path: Path) -> None:
    import cv2
    rows = []
    for it in items:
        T = len(it["frames"])
        idx = np.linspace(0, T - 1, n_cols).astype(int)
        tiles = []
        for i in idx:
            t = draw_overlay(it["frames"][i], it["acts"][i], boxes=it["meta"].get("inpaint_boxes"))
            cv2.putText(t, f"t={i}", (4, t.shape[0] - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            tiles.append(t)
        row = np.concatenate(tiles, 1)
        bar = np.zeros((44, row.shape[1], 3), np.uint8)
        m = it["meta"]
        head = (f"{m.get('game')} [{m.get('split')}] {m.get('uuid')}  fps={m.get('fps')}  T={T}  "
                f"gameplay={m.get('vlm_frac_gameplay')} screens={m.get('vlm_screens')} vlm_title={m.get('vlm_game_title')!r}")
        cv2.putText(bar, head, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cap = textwrap.shorten(it["caption"], width=int(row.shape[1] / 6.2))
        cv2.putText(bar, cap, (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        rows.append(np.concatenate([bar, row], 0))
    wmax = max(r.shape[1] for r in rows)  # buckets have different widths: right-pad rows
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    grid = np.concatenate(rows, 0)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def make_debug_video(it: dict, path: Path) -> None:
    import av
    fps = float(it["meta"].get("fps", 30))
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=round(fps))
        boxes = it["meta"].get("inpaint_boxes")
        f0 = draw_overlay(it["frames"][0], it["acts"][0], boxes=boxes)
        st.width, st.height, st.pix_fmt = f0.shape[1], f0.shape[0], "yuv420p"
        for fr, a in zip(it["frames"], it["acts"]):
            for pkt in st.encode(av.VideoFrame.from_ndarray(draw_overlay(fr, a, boxes=boxes), format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DATA_ROOT / "webdataset"))
    ap.add_argument("--splits", nargs="+", default=["train", "val_unseen_games"])
    ap.add_argument("--num-shards", type=int, default=3, help="random shards per split to check")
    ap.add_argument("--max-samples", type=int, default=64, help="max samples checked per split")
    ap.add_argument("--grid-rows", type=int, default=8)
    ap.add_argument("--grid-cols", type=int, default=6)
    ap.add_argument("--debug-videos", type=int, default=2)
    ap.add_argument("--out-dir", default=str(DATA_ROOT / "debug"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-missing-caption", action="store_true",
                    help="don't fail samples without a caption (e.g. smoke tests before stage 04)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root, out = Path(args.root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gid_path = root / "game_ids.json"
    game_ids = json.loads(gid_path.read_text()) if gid_path.exists() else {}
    total_fail, games_by_split = 0, {}
    for split in args.splits:
        shards = sorted((root / split).rglob("*.tar"))
        if not shards:
            print(f"[{split}] no shards found")
            continue
        idx = json.loads((root / split / "index.json").read_text()) if (root / split / "index.json").exists() else {}
        print(f"\n[{split}] {len(shards)} shards, index: samples={idx.get('samples')} hours={idx.get('hours')} "
              f"drops={idx.get('drops')}")
        for b, st in sorted((idx.get("buckets") or {}).items(), key=lambda kv: -kv[1]["hours"]):
            print(f"    bucket {b:>6} {st['width']}x{st['height']}: {st['samples']:,} samples, {st['hours']} h")
        picked = rng.sample(shards, min(args.num_shards, len(shards)))
        checked, failed, grid_items = 0, 0, []
        games = set()
        lens, stick_act, btn_act = [], [], []
        for sh in picked:
            samples = read_tar(sh)
            keys = list(samples)
            rng.shuffle(keys)
            for key in keys:
                if checked >= args.max_samples:
                    break
                errs, it = check_sample(key, samples[key], game_ids, args.allow_missing_caption)
                checked += 1
                if errs:
                    failed += 1
                    print(f"  FAIL {sh.name}:{key}: {'; '.join(errs)}")
                    continue
                games.add(it["meta"].get("game_key"))
                lens.append(len(it["acts"]))
                stick_act.append(float((np.linalg.norm(it["acts"][:, :2], axis=1) > 0.2).mean()))
                btn_act.append(float((it["acts"][:, 4:].max(1) > 0).mean()))
                if len(grid_items) < max(args.grid_rows, args.debug_videos):
                    grid_items.append(it)
        # also collect game keys from all json members of all shards for the leakage check
        all_games = set()
        for sh in shards:
            with tarfile.open(sh) as tf:
                for m in tf:
                    if m.name.endswith(".json"):
                        all_games.add(json.loads(tf.extractfile(m).read()).get("game_key"))
        games_by_split[split] = all_games
        total_fail += failed
        print(f"  checked {checked} samples from {len(picked)} shard(s): {checked - failed} OK, {failed} FAILED")
        if lens:
            print(f"  T: min={min(lens)} max={max(lens)} | movement-active frac mean={np.mean(stick_act):.2f} "
                  f"| button-active frac mean={np.mean(btn_act):.2f} | games in sample={len(games)} "
                  f"| games in split={len(all_games)}")
        if grid_items:
            g = out / f"grid_{split}.png"
            make_grid(grid_items[: args.grid_rows], args.grid_cols, g)
            print(f"  wrote {g}")
            for it in grid_items[: args.debug_videos]:
                p = out / f"debug_{split}_{it['meta']['uuid']}.mp4"
                make_debug_video(it, p)
                print(f"  wrote {p}")
    if "train" in games_by_split and "val_unseen_games" in games_by_split:
        leak = games_by_split["train"] & games_by_split["val_unseen_games"]
        if leak:
            total_fail += 1
            print(f"\nFAIL: {len(leak)} game(s) appear in both train and val_unseen_games: {sorted(leak)[:10]}")
        else:
            print(f"\nOK: train ({len(games_by_split['train'])} games) and val_unseen_games "
                  f"({len(games_by_split['val_unseen_games'])} games) are disjoint")
    print("\nRESULT:", "PASS" if total_fail == 0 else f"FAIL ({total_fail} problems)")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
