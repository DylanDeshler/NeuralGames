#!/usr/bin/env python
"""Stage 05 — pack processed clips into sharded WebDataset tars (~1 GB each).

Layout:
  data/webdataset/
    train/train-000000.tar ...
    val_unseen_games/val_unseen_games-000000.tar ...
    game_ids.json            game_key -> int id (shared across splits)
    <split>/index.json       shards, sample counts, hours (written last = split "done" marker)

Each sample (key = chunk uuid):
  <key>.mp4           256x256 video, T frames  (or <key>.frames.npy uint8 (T,256,256,3) with --frames-format npy)
  <key>.actions.npy   (T, 21) float32  [j_left(2), j_right(2), buttons(17)]
  <key>.txt           caption
  <key>.cls           integer game_id
  <key>.json          metadata (game_id, game, genre, fps, n_frames, video_id, caption, action_names, ...)

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
                    help="npy stores raw uint8 frames (~235 MB per 1200-frame clip!)")
    ap.add_argument("--splits", nargs="+", default=["train", "val_unseen_games"])
    ap.add_argument("--allow-missing-caption", action="store_true")
    ap.add_argument("--drop-vlm-3d", action="store_true", help="drop chunks the VLM judged not 2D")
    ap.add_argument("--drop-dark-frame0", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
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
        drops = {"not_processed": 0, "no_caption": 0, "vlm_3d": 0, "dark": 0, "bad_shape": 0}
        n, frames_total, hours = 0, 0, 0.0
        pattern = str(sdir / f"{split}-%06d.tar")
        with wds.ShardWriter(pattern, maxsize=int(args.shard_size_gb * 1e9),
                             maxcount=args.max_samples_per_shard, verbose=0) as sink:
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
                acts = np.load(d / "actions.npy")
                if acts.ndim != 2 or acts.shape[1] != ACTION_DIM or acts.shape[0] != meta["n_frames"]:
                    drops["bad_shape"] += 1
                    continue
                gid = int(game_ids[row.game_key])
                info = {
                    "uuid": row.uuid, "game_id": gid, "game": meta.get("game"), "game_key": row.game_key,
                    "genre": meta.get("genre"), "split": split, "video_id": row.video_id,
                    "chunk_id": meta.get("chunk_id"), "fps": meta["fps"], "n_frames": meta["n_frames"],
                    "size": meta["size"], "pad": meta["pad"], "content_wh": meta["content_wh"],
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
                sink.write(sample)
                n += 1
                frames_total += int(meta["n_frames"])
                hours += meta["n_frames"] / meta["fps"] / 3600
        shards = sorted(p.name for p in sdir.glob(f"{split}-*.tar"))
        atomic_write_json(sdir / "index.json", {
            "split": split, "samples": n, "frames": frames_total, "hours": round(hours, 2), "shards": shards,
            "frames_format": args.frames_format, "action_names": ACTION_NAMES, "drops": drops,
            "games": int(sm.game_key.nunique())})
        log.info("[%s] %s samples, %.1f h, %d shards -> %s | dropped %s", split, f"{n:,}", hours, len(shards),
                 sdir, drops)


if __name__ == "__main__":
    main()
