#!/usr/bin/env python
"""Stage 05 — pack processed clips into sharded WebDataset tars (~1 GB each).

Layout:
  data/webdataset/
    train/<bucket>/train-<bucket>-000000.tar ...      one shard series per aspect bucket (e.g. 21x12 = 336x192
    val_unseen_games/<bucket>/...                     at patch 16), so the trainer can draw same-shape batches
    game_ids.json            game_key -> int id (shared across splits)
    <split>/index.json       per-bucket shards/samples/hours/width/height (written last = split "done" marker)

Each sample (key = chunk uuid):
  <key>.mp4           W x H video (bucket resolution), T frames  (or <key>.frames.npy uint8 (T,256,256,3) with --frames-format npy)
  <key>.actions.npy   (T, 21) float32  [j_left(2), j_right(2), buttons(17)]
  <key>.txt           caption
  <key>.cls           integer game_id
  <key>.json          metadata (game_id, game, genre, fps, n_frames, video_id, caption, action_names,
                      pad/content_wh (letterbox) and inpaint_boxes (overlay regions) for loss masks — ng_masks.py)

Training-side decoding example (webdataset):
  ds = wds.WebDataset(urls).decode().to_tuple("mp4", "actions.npy", "txt", "cls")   # mp4 bytes -> decode with av/decord

Re-running skips splits whose index.json exists (use --overwrite to rebuild).
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ng_common import DATA_ROOT, ACTION_DIM, ACTION_NAMES, atomic_write_json, processed_dir_for, setup_logging


def decode_mp4(path: Path) -> np.ndarray:
    import av
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DATA_ROOT / "manifest_2d.parquet"))
    ap.add_argument("--processed-dir", default=str(DATA_ROOT / "processed"))
    ap.add_argument("--out", default=str(DATA_ROOT / "webdataset"))
    ap.add_argument("--shard-size-gb", type=float, default=1.0)
    ap.add_argument("--max-samples-per-shard", type=int, default=100000)
    ap.add_argument("--frames-format", choices=["mp4", "npy"], default="mp4",
                    help="npy stores raw uint8 frames (~120 MB per 600-frame 336x192 clip!)")
    ap.add_argument("--splits", nargs="+", default=["train", "val_unseen_games"])
    ap.add_argument("--allow-missing-caption", action="store_true")
    ap.add_argument("--drop-vlm-3d", action="store_true", help="drop chunks the VLM judged not 2D")
    ap.add_argument("--drop-dark-frame0", action="store_true")
    ap.add_argument("--max-black-frac", type=float, default=0.25,
                    help="drop clips where more than this fraction of frames is near-black (fades, loading)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--delete-processed", action="store_true",
                    help="delete each processed clip dir once the shard containing it is closed (halves peak disk; "
                         "processed/ can't be re-packed afterwards)")
    args = ap.parse_args()

    import webdataset as wds

    log = setup_logging("05_pack_webdataset")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(args.manifest)

    # stable game ids across splits (sorted by key so ids don't depend on sampling order)
    gid_path = out / "game_ids.json"
    game_ids = json.loads(gid_path.read_text()) if gid_path.exists() else {}
    for k in sorted(man["game_key"].unique()):
        game_ids.setdefault(k, len(game_ids))
    atomic_write_json(gid_path, game_ids)

    for split in args.splits:
        sdir = out / split
        if (sdir / "index.json").exists() and not args.overwrite:
            log.info("[%s] already packed (index.json exists) — skipping", split)
            continue
        shutil.rmtree(sdir, ignore_errors=True)
        sdir.mkdir(parents=True)
        sm = man[man.split == split]
        # shuffle so each shard mixes games (important for streaming training)
        sm = sm.sample(frac=1.0, random_state=args.seed)
        drops = {"not_processed": 0, "no_caption": 0, "vlm_3d": 0, "dark": 0, "black_frames": 0, "bad_shape": 0}
        stats: dict[str, dict] = {}   # bucket -> {samples, frames, hours, width, height}
        writers: dict[str, "wds.ShardWriter"] = {}
        pending_del: dict[str, list[Path]] = {}  # bucket -> clip dirs whose shard is still open
        n_deleted = 0

        def flush_deletes(dirs: list[Path]):
            nonlocal n_deleted
            for dd in dirs:
                shutil.rmtree(dd, ignore_errors=True)
                n_deleted += 1
            dirs.clear()
        try:
            for row in tqdm(sm.itertuples(index=False), total=len(sm), desc=f"pack {split}"):
                d = processed_dir_for(args.processed_dir, row.video_id, row.uuid)
                mp = d / "meta.json"
                if not mp.exists():
                    drops["not_processed"] += 1
                    continue
                meta = json.loads(mp.read_text())
                cap = meta.get("caption")
                if not cap and not args.allow_missing_caption:
                    drops["no_caption"] += 1
                    continue
                if args.drop_vlm_3d and meta.get("vlm_is_2d") is False:
                    drops["vlm_3d"] += 1
                    continue
                if args.drop_dark_frame0 and meta.get("frame0_dark"):
                    drops["dark"] += 1
                    continue
                if meta.get("frac_black", 0.0) > args.max_black_frac:
                    drops["black_frames"] += 1
                    continue
                acts = np.load(d / "actions.npy")
                if acts.ndim != 2 or acts.shape[1] != ACTION_DIM or acts.shape[0] != meta["n_frames"]:
                    drops["bad_shape"] += 1
                    continue
                bucket = meta.get("bucket", f"{meta['width']}x{meta['height']}px")
                if bucket not in writers:  # one shard series per aspect bucket -> same-shape batches
                    (sdir / bucket).mkdir(parents=True, exist_ok=True)
                    writers[bucket] = wds.ShardWriter(str(sdir / bucket / f"{split}-{bucket}-%06d.tar"),
                                                      maxsize=int(args.shard_size_gb * 1e9),
                                                      maxcount=args.max_samples_per_shard, verbose=0)
                    stats[bucket] = {"samples": 0, "frames": 0, "hours": 0.0, "width": meta["width"],
                                     "height": meta["height"], "patch": meta.get("patch")}
                gid = int(game_ids[row.game_key])
                info = {
                    "uuid": row.uuid, "game_id": gid, "game": meta.get("game"), "game_key": row.game_key,
                    "genre": meta.get("genre"), "split": split, "video_id": row.video_id,
                    "chunk_id": meta.get("chunk_id"), "fps": meta["fps"], "n_frames": meta["n_frames"],
                    "width": meta["width"], "height": meta["height"], "bucket": bucket, "patch": meta.get("patch"),
                    "fit_mode": meta.get("fit_mode"), "aspect_crop_frac": meta.get("aspect_crop_frac"),
                    "pad": meta["pad"], "content_wh": meta["content_wh"],
                    # inpainted regions (output px, already dilated): exclude from the loss, see ng_masks.py
                    "inpaint_boxes": meta.get("inpaint_boxes", []), "inpaint_frac": meta.get("inpaint_frac", 0.0),
                    "caption": cap or "", "vlm_perspective": meta.get("vlm_perspective"),
                    "vlm_is_2d": meta.get("vlm_is_2d"), "action_names": ACTION_NAMES,
                    "action_source": meta.get("action_source"), "start_time": meta.get("start_time"),
                    "source_url": f"https://www.youtube.com/watch?v={row.video_id}",
                }
                abuf = io.BytesIO()
                np.save(abuf, acts.astype(np.float32))
                sample = {"__key__": row.uuid, "actions.npy": abuf.getvalue(), "txt": cap or "",
                          "cls": str(gid), "json": json.dumps(info).encode()}
                if args.frames_format == "mp4":
                    sample["mp4"] = (d / "frames.mp4").read_bytes()
                else:
                    fbuf = io.BytesIO()
                    np.save(fbuf, decode_mp4(d / "frames.mp4"))
                    sample["frames.npy"] = fbuf.getvalue()
                w = writers[bucket]
                shard_before = getattr(w, "shard", None)
                w.write(sample)
                if args.delete_processed:
                    if getattr(w, "shard", None) != shard_before and shard_before is not None:
                        flush_deletes(pending_del.setdefault(bucket, []))  # previous shard is closed on disk
                    pending_del.setdefault(bucket, []).append(d)
                st = stats[bucket]
                st["samples"] += 1
                st["frames"] += int(meta["n_frames"])
                st["hours"] += meta["n_frames"] / meta["fps"] / 3600
        finally:
            for w in writers.values():
                w.close()
        if args.delete_processed:  # all shards closed successfully
            for dirs in pending_del.values():
                flush_deletes(dirs)
            log.info("[%s] deleted %s processed clip dirs after packing", split, f"{n_deleted:,}")
        for b, st in stats.items():
            st["shards"] = sorted(f"{b}/{p.name}" for p in (sdir / b).glob("*.tar"))
            st["hours"] = round(st["hours"], 2)
        n = sum(st["samples"] for st in stats.values())
        hours = sum(st["hours"] for st in stats.values())
        atomic_write_json(sdir / "index.json", {
            "split": split, "samples": n, "frames": sum(st["frames"] for st in stats.values()),
            "hours": round(hours, 2), "buckets": stats, "frames_format": args.frames_format,
            "action_names": ACTION_NAMES, "drops": drops, "games": int(sm.game_key.nunique())})
        log.info("[%s] %s samples, %.1f h | buckets: %s | dropped %s", split, f"{n:,}", hours,
                 {b: f"{st['width']}x{st['height']} {st['hours']:.1f}h" for b, st in
                  sorted(stats.items(), key=lambda kv: -kv[1]["hours"])}, drops)


if __name__ == "__main__":
    main()
