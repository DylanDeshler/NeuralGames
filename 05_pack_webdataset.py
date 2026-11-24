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

from ng_common import add_data_root_arg, DATA_ROOT, ACTION_DIM, ACTION_NAMES, atomic_write_json, processed_dir_for, setup_logging
from ng_labels import GameLabeler


def decode_mp4(path: Path) -> np.ndarray:
    import av
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def name_check(man: pd.DataFrame, args, out: Path, log) -> set[str]:
    """Compare each video's label with the game stage 04's VLM recognizes in its frames (majority over the
    video's clips). Writes name_mismatch_report.csv (informational: VLM titles are unreliable); returns val videos
    that look like a TRAIN game only when --val-leak-check is on."""
    if "vlm_game_title" not in man.columns or man["vlm_game_title"].notna().sum() == 0:
        log.info("Name check skipped (no vlm_game_title in the manifest; run stage 04 first)")
        return set()
    lab = GameLabeler(Path(args.genres), Path(args.labels_csv) if args.labels_csv else None)
    t = man[man["vlm_game_title"].notna()]
    title = t.groupby("video_id")["vlm_game_title"].agg(lambda s: s.mode().iat[0])
    support = t.groupby("video_id")["vlm_game_title"].agg(lambda s: float((s == s.mode().iat[0]).mean()))
    key = title.map(lambda x: lab.group_of.get(lab.canonical_key(x), lab.canonical_key(x)))
    per_vid = man.drop_duplicates("video_id").set_index("video_id")[["game_key", "game_name", "split"]]
    rep = per_vid.join(pd.DataFrame({"vlm_title": title, "vlm_key": key, "vlm_support": support}), how="inner")
    rep = rep[rep["vlm_key"] != rep["game_key"]]
    train_keys = set(man.loc[man["split"] == "train", "game_key"])
    rep["vlm_key_in_train"] = rep["vlm_key"].isin(train_keys)
    hours = man.groupby("video_id")["duration"].sum() / 3600
    rep["hours"] = hours.reindex(rep.index).round(2)
    rep.sort_values("hours", ascending=False).to_csv(out / "name_mismatch_report.csv")
    leak = set(rep.index[(rep["split"] == "val_unseen_games") & rep["vlm_key_in_train"]]) if args.val_leak_check else set()
    log.info("Name check: %d of %d checked videos disagree with their label (%.0f h, see name_mismatch_report.csv); "
             "%d val videos look like a train game -> dropped from val", len(rep), len(title), rep["hours"].sum(), len(leak))
    return leak


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_root_arg(ap)
    ap.add_argument("--manifest", default=str(DATA_ROOT / "manifest_2d.parquet"))
    ap.add_argument("--processed-dir", default=str(DATA_ROOT / "processed"))
    ap.add_argument("--out", default=str(DATA_ROOT / "webdataset"))
    ap.add_argument("--shard-size-gb", type=float, default=1.0)
    ap.add_argument("--max-samples-per-shard", type=int, default=100000)
    ap.add_argument("--frames-format", choices=["mp4", "npy"], default="mp4",
                    help="npy stores raw uint8 frames (~120 MB per 600-frame 336x192 clip!)")
    ap.add_argument("--splits", nargs="+", default=["train", "val_unseen_games"])
    ap.add_argument("--allow-missing-caption", action="store_true")
    ap.add_argument("--drop-vlm-3d", action=argparse.BooleanOptionalAction, default=True,
                    help="drop chunks the stage-04 VLM judged 3D (default on; --no-drop-vlm-3d to keep them)")
    ap.add_argument("--drop-dark-frame0", action="store_true")
    ap.add_argument("--min-gameplay-frac", type=float, default=0.75,
                    help="drop clips where stage 04's VLM saw gameplay in fewer than this fraction of sampled frames "
                         "(menus, title/option screens, loading, emulator UIs). 0 disables")
    ap.add_argument("--require-frame0-gameplay", action=argparse.BooleanOptionalAction, default=True,
                    help="drop clips whose first frame (the caption / text-to-world conditioning frame) isn't gameplay")
    ap.add_argument("--val-leak-check", action=argparse.BooleanOptionalAction, default=False,
                    help="drop val_unseen_games videos whose frames the VLM 'recognizes' as a different train game. OFF by "
                         "default: in smoke tests the VLM's title guesses on small frames were mostly wrong (3/3 false drops). "
                         "The mismatch report is still written for manual review.")
    ap.add_argument("--genres", default="./config/genres_2d.yaml")
    ap.add_argument("--labels-csv", default="./config/game_labels.csv")
    ap.add_argument("--max-black-frac", type=float, default=0.25,
                    help="drop clips where more than this fraction of frames is near-black (fades, loading)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--append", action="store_true",
                    help="add newly processed clips to existing shards (new shard files; already-packed uuids skipped). "
                         "Use with --delete-processed for repeated download/pack cycles")
    ap.add_argument("--tag", default="", help="machine tag in shard names (train-21x12-<tag>-000000.tar) for multi-machine builds")
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

    leak_videos = name_check(man, args, out, log)

    def prefix(split: str, bucket: str) -> str:
        """Shard file prefix; --tag keeps several build machines' shards apart on the Hub."""
        return f"{split}-{bucket}-{args.tag}-" if args.tag else f"{split}-{bucket}-"

    for split in args.splits:
        sdir = out / split
        prev_index = json.loads((sdir / "index.json").read_text()) if (sdir / "index.json").exists() else None
        packed_path = sdir / "_packed_uuids.txt"
        if args.append:
            sdir.mkdir(parents=True, exist_ok=True)
            already = set(packed_path.read_text().split()) if packed_path.exists() else set()
        else:
            if prev_index and not args.overwrite:
                log.info("[%s] already packed (index.json exists) — skipping (use --append or --overwrite)", split)
                continue
            shutil.rmtree(sdir, ignore_errors=True)
            sdir.mkdir(parents=True)
            already, prev_index = set(), None
        sm = man[man.split == split]
        if already:
            sm = sm[~sm["uuid"].isin(already)]
        # shuffle so each shard mixes games (important for streaming training)
        sm = sm.sample(frac=1.0, random_state=args.seed)
        drops = {"not_processed": 0, "no_caption": 0, "vlm_3d": 0, "dark": 0, "black_frames": 0,
                 "non_gameplay": 0, "qc_unparsed": 0, "frame0_not_gameplay": 0, "val_leak": 0, "bad_shape": 0}
        stats: dict[str, dict] = {}   # bucket -> {samples, frames, hours, width, height}
        newly_packed: list[str] = []
        state_path = sdir / "_next_shard.json"
        next_shard: dict[str, int] = json.loads(state_path.read_text()) if (args.append and state_path.exists()) else {}
        writers: dict[str, "wds.ShardWriter"] = {}
        pending_del: dict[str, list[tuple[Path, str]]] = {}  # bucket -> (clip dir, uuid) whose shard is still open
        n_deleted = 0
        recorded: set[str] = set()  # uuids already appended to _packed_uuids.txt during this run
        dropped_path = sdir / "_dropped_uuids.txt"

        def flush_deletes(entries: list[tuple[Path, str]], bucket: str | None = None):
            """The shard holding these clips is closed: record them as packed FIRST (so a crash can't lose them:
            stage 03 skips packed/dropped uuids), persist the shard counter, then delete the clip dirs."""
            nonlocal n_deleted
            if not entries:
                return
            uu = [u for _, u in entries]
            with open(packed_path, "a") as fh:
                fh.write("\n".join(uu) + "\n")
            recorded.update(uu)
            if bucket is not None and bucket in writers:  # the open shard (w.shard - 1) gets rewritten on restart
                next_shard[bucket] = max(next_shard.get(bucket, 0), int(getattr(writers[bucket], "shard", 1)) - 1)
                atomic_write_json(state_path, next_shard)
            for dd, _ in entries:
                shutil.rmtree(dd, ignore_errors=True)
                n_deleted += 1
            entries.clear()

        def drop_dir(d: Path, uuid: str):
            """Dropped for quality in --delete-processed mode: remember it (03 must not re-download it), free disk."""
            with open(dropped_path, "a") as fh:
                fh.write(uuid + "\n")
            shutil.rmtree(d, ignore_errors=True)
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
                    if args.delete_processed:
                        drop_dir(d, row.uuid)
                    continue
                if args.drop_dark_frame0 and meta.get("frame0_dark"):
                    drops["dark"] += 1
                    continue
                fg, scr = meta.get("vlm_frac_gameplay"), meta.get("vlm_screens") or []
                if args.min_gameplay_frac > 0 and "caption" in meta and fg is None and not args.allow_missing_caption:
                    drops["qc_unparsed"] += 1
                    if args.delete_processed:
                        drop_dir(d, row.uuid)  # stage 04 ran but its QC output didn't parse: don't let it pass silently
                    continue
                if args.min_gameplay_frac > 0 and fg is not None and fg < args.min_gameplay_frac:
                    drops["non_gameplay"] += 1
                    if args.delete_processed:
                        drop_dir(d, row.uuid)
                    continue
                if args.require_frame0_gameplay and scr and scr[0] != "gameplay":
                    drops["frame0_not_gameplay"] += 1
                    if args.delete_processed:
                        drop_dir(d, row.uuid)  # frame 0 is the caption/conditioning frame
                    continue
                if split == "val_unseen_games" and row.video_id in leak_videos:
                    drops["val_leak"] += 1
                    continue
                if meta.get("frac_black", 0.0) > args.max_black_frac:
                    drops["black_frames"] += 1
                    if args.delete_processed:
                        drop_dir(d, row.uuid)
                    continue
                acts = np.load(d / "actions.npy")
                if acts.ndim != 2 or acts.shape[1] != ACTION_DIM or acts.shape[0] != meta["n_frames"]:
                    drops["bad_shape"] += 1
                    continue
                bucket = meta.get("bucket", f"{meta['width']}x{meta['height']}px")
                if bucket not in writers:  # one shard series per aspect bucket -> same-shape batches
                    (sdir / bucket).mkdir(parents=True, exist_ok=True)
                    # continue numbering from persistent state (local shards may have been uploaded and deleted)
                    if bucket in next_shard:
                        start = next_shard[bucket]
                        for stale in (sdir / bucket).glob(f"{prefix(split, bucket)}*.tar"):  # shard that was open when a run was killed:
                            if int(stale.stem.rsplit("-", 1)[-1]) >= start:  # its clips were never recorded, repack
                                log.warning("[%s] removing unfinished shard %s from an interrupted run", split, stale.name)
                                stale.unlink()
                    else:
                        start = len(list((sdir / bucket).glob(f"{prefix(split, bucket)}*.tar")))
                    writers[bucket] = wds.ShardWriter(str(sdir / bucket / f"{prefix(split, bucket)}%06d.tar"),
                                                      maxsize=int(args.shard_size_gb * 1e9),
                                                      maxcount=args.max_samples_per_shard, verbose=0,
                                                      start_shard=start)
                    stats[bucket] = {"samples": 0, "frames": 0, "hours": 0.0, "width": meta["width"],
                                     "height": meta["height"], "patch": meta.get("patch")}
                gid = int(game_ids[row.game_key])
                info = {
                    "uuid": row.uuid, "game_id": gid, "game": meta.get("game"), "game_key": row.game_key,
                    "genre": meta.get("genre"), "split": split, "video_id": row.video_id,
                    "chunk_id": meta.get("chunk_id"), "fps": meta["fps"], "n_frames": meta["n_frames"],
                    "frame_stride": meta.get("frame_stride"), "annotation_fps": meta.get("annotation_fps"),
                    "width": meta["width"], "height": meta["height"], "bucket": bucket, "patch": meta.get("patch"),
                    "fit_mode": meta.get("fit_mode"), "aspect_crop_frac": meta.get("aspect_crop_frac"),
                    "pad": meta["pad"], "content_wh": meta["content_wh"],
                    # inpainted regions (output px, already dilated): exclude from the loss, see ng_masks.py
                    "inpaint_boxes": meta.get("inpaint_boxes", []), "inpaint_frac": meta.get("inpaint_frac", 0.0),
                    "caption": cap or "", "vlm_perspective": meta.get("vlm_perspective"),
                    "vlm_is_2d": meta.get("vlm_is_2d"), "action_names": ACTION_NAMES,
                    "vlm_frac_gameplay": meta.get("vlm_frac_gameplay"), "vlm_screens": meta.get("vlm_screens"),
                    "vlm_game_title": meta.get("vlm_game_title"),
                    "action_source": meta.get("action_source"), "start_time": meta.get("start_time"),
                    "source_url": f"https://www.youtube.com/watch?v={row.video_id}",
                }
                abuf = io.BytesIO()
                np.save(abuf, acts.astype(np.float32))
                sample = {"__key__": row.uuid, "actions.npy": abuf.getvalue(), "txt": cap or "",
                          "cls": str(gid), "json": json.dumps(info).encode()}
                if (d / "actions_native.npy").exists():  # (T*stride, 21) float16 at annotation fps, see README
                    sample["actions_native.npy"] = (d / "actions_native.npy").read_bytes()
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
                        flush_deletes(pending_del.setdefault(bucket, []), bucket)  # previous shard is closed on disk
                    pending_del.setdefault(bucket, []).append((d, row.uuid))
                newly_packed.append(row.uuid)
                st = stats[bucket]
                st["samples"] += 1
                st["frames"] += int(meta["n_frames"])
                st["hours"] += meta["n_frames"] / meta["fps"] / 3600
        finally:
            for b, w in writers.items():  # read the counter BEFORE close(): ShardWriter.close() deletes .shard
                next_shard[b] = max(next_shard.get(b, 0), int(getattr(w, "shard", 0)))
                w.close()
            atomic_write_json(state_path, next_shard)
        if args.delete_processed:  # all shards closed successfully (counter already saved above)
            for dirs in pending_del.values():
                flush_deletes(dirs)
            log.info("[%s] deleted %s processed clip dirs after packing", split, f"{n_deleted:,}")
        rest = [u for u in newly_packed if u not in recorded]
        if rest:  # record what's in the shards (append mode and stage 03 skip these next time)
            with open(packed_path, "a") as fh:
                fh.write("\n".join(rest) + "\n")
        if prev_index and args.append:  # merge with what earlier runs packed
            for b, old in (prev_index.get("buckets") or {}).items():
                st = stats.setdefault(b, {"samples": 0, "frames": 0, "hours": 0.0, "width": old["width"],
                                          "height": old["height"], "patch": old.get("patch")})
                for k in ("samples", "frames", "hours"):
                    st[k] += old.get(k, 0)
            for k, v in (prev_index.get("drops") or {}).items():
                drops[k] = drops.get(k, 0) + (v if k != "not_processed" else 0)
        for b, st in stats.items():
            old_shards = set(((prev_index or {}).get("buckets") or {}).get(b, {}).get("shards", [])) if args.append else set()
            st["shards"] = sorted(old_shards | {f"{b}/{p.name}" for p in (sdir / b).glob("*.tar")})
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
