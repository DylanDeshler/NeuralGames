#!/usr/bin/env python
"""Stage 01 — metadata/action sync from `nvidia/NitroGen` (no videos, no model weights).

Why this isn't a plain `snapshot_download(allow_patterns=["**/metadata.json", ...])`:
the HF repo stores each shard as a single `actions/SHARD_XXXX.tar.gz` (100 x ~1.6 GB), so those
patterns would match nothing. Extracting them naively yields ~25M tiny files (~450 GB on disk).

Instead, for each shard we:
  1. download the tarball with `huggingface_hub` (only `actions/SHARD_*.tar.gz` — never `ng.pt`),
  2. stream through it once, inspecting the real JSON/Parquet schema on the first chunk,
  3. write two compact zstd parquet files:
       index/SHARD_XXXX.parquet    one row per chunk: flattened metadata + activity stats
       actions/SHARD_XXXX.parquet  one row per chunk: uuid + (T, 21) float32 action bytes
  4. delete the tarball (unless --keep-tar) and write a `.done` marker -> fully resumable.

Use --extract-loose to additionally materialize the original per-chunk files for inspection.

Example:
  python 01_fetch_metadata.py --shards 0-1          # quick look at two shards
  python 01_fetch_metadata.py --shards all --workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ng_common import (add_data_root_arg, 
    DATA_ROOT,
    ACTION_DIM, ACTION_NAMES, actions_to_array, activity_stats, atomic_write_json, detect_action_schema,
    dir_size_bytes, free_gb, human_bytes, parse_shard_spec, setup_logging,
)

REPO_ID = "nvidia/NitroGen"
N_SHARDS = 100

INDEX_SCHEMA = pa.schema([
    ("uuid", pa.string()), ("chunk_dir", pa.string()), ("shard", pa.int16()),
    ("video_id", pa.string()), ("chunk_id", pa.int32()), ("chunk_size", pa.int32()),
    ("n_frames", pa.int32()), ("fps", pa.float32()),
    ("start_time", pa.float64()), ("end_time", pa.float64()), ("duration", pa.float32()),
    ("start_frame", pa.int64()), ("end_frame", pa.int64()),
    ("res_h", pa.int32()), ("res_w", pa.int32()), ("source", pa.string()), ("url", pa.string()),
    ("game", pa.string()), ("controller_type", pa.string()),
    ("bbox_controller_overlay", pa.string()), ("bbox_game_area", pa.string()), ("bbox_others", pa.string()),
    ("has_processed", pa.bool_()), ("action_source", pa.string()),
    ("frac_lstick_01", pa.float32()), ("frac_lstick_02", pa.float32()), ("frac_lstick_03", pa.float32()),
    ("frac_move_01", pa.float32()), ("frac_move_02", pa.float32()), ("frac_move_03", pa.float32()),  # stick OR d-pad
    ("frac_rstick_02", pa.float32()), ("frac_btn_gameplay", pa.float32()), ("frac_btn_any", pa.float32()),
    ("mean_btn_per_frame", pa.float32()), ("frac_any_input", pa.float32()),
    ("meta_json", pa.string()),
])
ACTIONS_SCHEMA = pa.schema([
    ("uuid", pa.string()), ("n_frames", pa.int32()), ("action_source", pa.string()),
    ("actions", pa.binary()),  # np.float32 (n_frames, 21) row-major bytes
])


# ----------------------------------------------------------------------------- per-chunk
def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _i(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def build_rows(chunk_dir: str, shard: int, files: dict[str, bytes], schema_cache: dict):
    """Turn one chunk's raw files into (index_row, actions_row). Returns None if unusable."""
    if "metadata.json" not in files:
        return None, "no_metadata"
    src = "processed" if "actions_processed.parquet" in files else ("raw" if "actions_raw.parquet" in files else None)
    if src is None:
        return None, "no_actions"
    md = json.loads(files["metadata.json"])
    # ParquetFile.read is ~2.5x cheaper than read_table for tiny files; no threads (we're already multi-process)
    table = pq.ParquetFile(pa.BufferReader(files[f"actions_{src}.parquet"])).read(use_threads=False)
    if "schema" not in schema_cache:
        schema_cache["schema"] = detect_action_schema(
            table.column_names, {f.name: str(f.type) for f in table.schema})
        schema_cache["arrow_schema"] = str(table.schema).split("-- schema metadata --")[0].strip()
        schema_cache["metadata_example"] = md
    arr = actions_to_array(table, schema_cache["schema"])
    ov = md.get("original_video", {}) or {}
    res = ov.get("resolution") or [None, None]
    duration = _f(ov.get("duration")) or ((_f(ov.get("end_time"), 0) - _f(ov.get("start_time"), 0)) or None)
    n = int(arr.shape[0])
    stats = activity_stats(arr)
    uuid = md.get("uuid") or chunk_dir
    known = {"uuid", "chunk_id", "chunk_size", "original_video", "game", "controller_type",
             "bbox_controller_overlay", "bbox_game_area", "bbox_others"}
    if set(md) - known:
        schema_cache.setdefault("unknown_meta_keys", set()).update(set(md) - known)
    idx = {
        "uuid": uuid, "chunk_dir": chunk_dir, "shard": shard,
        "video_id": ov.get("video_id") or chunk_dir.split("_chunk_")[0],
        "chunk_id": _i(md.get("chunk_id"), _i(chunk_dir.rsplit("_", 1)[-1], -1)),
        "chunk_size": _i(md.get("chunk_size"), n), "n_frames": n,
        "fps": (n / duration) if duration else None,
        "start_time": _f(ov.get("start_time")), "end_time": _f(ov.get("end_time")), "duration": duration,
        "start_frame": _i(ov.get("start_frame")), "end_frame": _i(ov.get("end_frame")),
        "res_h": _i(res[0]) if len(res) > 0 else None, "res_w": _i(res[1]) if len(res) > 1 else None,
        "source": ov.get("source"), "url": ov.get("url"),
        "game": (md.get("game") or "other").strip(), "controller_type": md.get("controller_type"),
        "bbox_controller_overlay": json.dumps(md.get("bbox_controller_overlay")),
        "bbox_game_area": json.dumps(md.get("bbox_game_area")),
        "bbox_others": json.dumps(md.get("bbox_others")),
        "has_processed": "actions_processed.parquet" in files, "action_source": src,
        "meta_json": json.dumps(md, separators=(",", ":")),
        **stats,
    }
    act = {"uuid": uuid, "n_frames": n, "action_source": src,
           "actions": np.ascontiguousarray(arr, dtype=np.float32).tobytes()}
    return (idx, act), None


# ----------------------------------------------------------------------------- per-shard
def _worker_logger() -> logging.Logger:
    """Stderr logger usable inside pool workers (handlers aren't inherited under spawn/forkserver)."""
    lg = logging.getLogger("01_fetch_metadata.worker")
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | worker | %(message)s", "%Y-%m-%d %H:%M:%S"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    return lg


def process_shard(shard: int, args) -> dict:
    out = Path(args.out)
    name = f"SHARD_{shard:04d}"
    idx_path, act_path = out / "index" / f"{name}.parquet", out / "actions" / f"{name}.parquet"
    done = out / "_done" / f"{name}.json"
    if done.exists() and idx_path.exists() and act_path.exists():
        return json.loads(done.read_text()) | {"skipped": True}

    t0 = time.time()
    # ---- 1. obtain tarball
    if args.from_local_tars:
        tar_path = Path(args.from_local_tars) / f"{name}.tar.gz"
        if not tar_path.exists():
            raise FileNotFoundError(tar_path)
    else:
        from huggingface_hub import hf_hub_download
        if free_gb(out) < args.min_free_gb:
            raise RuntimeError(f"Free disk below --min-free-gb={args.min_free_gb}")
        for attempt in range(args.retries):
            try:
                tar_path = Path(hf_hub_download(
                    repo_id=args.repo_id, repo_type="dataset", filename=f"actions/{name}.tar.gz",
                    revision=args.revision, local_dir=str(out / "_tars")))
                break
            except Exception:
                if attempt == args.retries - 1:
                    raise
                time.sleep(10 * 2 ** attempt)
    t_dl = time.time() - t0

    # ---- 2. stream tar -> rows
    idx_tmp, act_tmp = idx_path.with_suffix(".parquet.partial"), act_path.with_suffix(".parquet.partial")
    iw = pq.ParquetWriter(idx_tmp, INDEX_SCHEMA, compression="zstd")
    aw = pq.ParquetWriter(act_tmp, ACTIONS_SCHEMA, compression="zstd", compression_level=6)
    schema_cache: dict = {}
    counts = {"chunks": 0, "frames": 0, "skipped_no_metadata": 0, "skipped_no_actions": 0, "errors": 0}
    seen: set[str] = set()
    buf_idx, buf_act = [], []
    pending: dict[str, dict[str, bytes]] = {}
    cur_video = None

    def flush_pending():
        for cdir, files in pending.items():
            try:
                rows, why = build_rows(cdir, shard, files, schema_cache)
            except Exception:
                counts["errors"] += 1
                continue
            if rows is None:
                counts[f"skipped_{why}"] += 1
                continue
            if rows[0]["uuid"] in seen:  # duplicate dir appearing twice in the tar: keep first
                continue
            seen.add(rows[0]["uuid"])
            buf_idx.append(rows[0])
            buf_act.append(rows[1])
            counts["chunks"] += 1
            if counts["chunks"] % args.progress_every == 0:
                el = time.time() - t_parse
                wlog.info("%s: %s chunks parsed (%.0f s, %.0f chunks/s)", name, f"{counts['chunks']:,}", el,
                          counts["chunks"] / max(el, 1e-6))
            counts["frames"] += rows[0]["n_frames"]
        pending.clear()
        if len(buf_idx) >= 2000:
            write_buffers()

    def write_buffers():
        if buf_idx:
            iw.write_table(pa.Table.from_pylist(buf_idx, schema=INDEX_SCHEMA))
            aw.write_table(pa.Table.from_pylist(buf_act, schema=ACTIONS_SCHEMA))
            buf_idx.clear()
            buf_act.clear()

    loose_root = out / "loose" if args.extract_loose else None
    wlog = _worker_logger()
    wlog.info("%s: downloaded in %.0f s, parsing ...", name, t_dl)
    t_parse = time.time()
    # decompress in a separate process (pigz > gzip) so gzip and parquet parsing use different cores
    gz = shutil.which("pigz") or shutil.which("gzip")
    proc = subprocess.Popen([gz, "-dc", str(tar_path)], stdout=subprocess.PIPE, bufsize=1 << 20) if gz else None
    try:
        with (tarfile.open(fileobj=proc.stdout, mode="r|") if proc else tarfile.open(tar_path, mode="r|gz")) as tf:
            for m in tf:
                if not m.isfile():
                    continue
                parts = PurePosixPath(m.name).parts
                if len(parts) < 3:
                    continue
                video_dir, chunk_dir, fname = parts[-3], parts[-2], parts[-1]
                if fname not in ("metadata.json", "actions_raw.parquet", "actions_processed.parquet"):
                    continue
                data = tf.extractfile(m).read()
                if video_dir != cur_video:  # chunks are grouped per video in the tar
                    flush_pending()
                    cur_video = video_dir
                pending.setdefault(chunk_dir, {})[fname] = data
                if loose_root is not None:
                    p = loose_root / name / video_dir / chunk_dir / fname
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(data)
        flush_pending()
        write_buffers()
        if proc is not None and proc.wait() != 0:
            raise RuntimeError(f"{gz} failed on {tar_path} (exit {proc.returncode}) — corrupt download? delete it and re-run")
    finally:
        iw.close()
        aw.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
    os.replace(idx_tmp, idx_path)
    os.replace(act_tmp, act_path)

    if not args.keep_tar and not args.from_local_tars:
        try:
            tar_path.unlink()
        except OSError:
            pass

    info = {
        "shard": shard, **counts,
        "download_s": round(t_dl, 1), "total_s": round(time.time() - t0, 1),
        "index_bytes": idx_path.stat().st_size, "actions_bytes": act_path.stat().st_size,
        "schema": schema_cache.get("schema"), "arrow_schema": schema_cache.get("arrow_schema"),
        "metadata_example": schema_cache.get("metadata_example"),
        "unknown_meta_keys": sorted(schema_cache.get("unknown_meta_keys", [])),
    }
    atomic_write_json(done, info)
    return info


# ----------------------------------------------------------------------------- summary
def print_summary(out: Path, log) -> None:
    import pandas as pd

    idx_files = sorted((out / "index").glob("SHARD_*.parquet"))
    if not idx_files:
        log.warning("No index shards found yet.")
        return
    df = pd.concat([pd.read_parquet(f, columns=["uuid", "video_id", "game", "duration", "n_frames", "fps",
                                                "has_processed", "frac_lstick_02", "frac_btn_gameplay"])
                    for f in idx_files], ignore_index=True)
    hours = df["duration"].fillna(20).sum() / 3600
    print("\n" + "=" * 72)
    print(f" NitroGen metadata sync summary  ({out})")
    print("=" * 72)
    print(f" shards synced        : {len(idx_files)} / {N_SHARDS}")
    print(f" chunks               : {len(df):,}")
    print(f" videos               : {df.video_id.nunique():,}")
    print(f" frames               : {int(df.n_frames.sum()):,}")
    print(f" hours                : {hours:,.1f}")
    print(f" distinct game slugs  : {df.game.nunique():,}  ('other' = {(df.game == 'other').mean():.1%} of chunks)")
    print(f" has processed actions: {df.has_processed.mean():.1%}")
    print(f" fps distribution     : {df.fps.round().value_counts().head(5).to_dict()}")
    print(f" idle-filter pass rate: {((df.frac_lstick_02 >= .15) & (df.frac_btn_gameplay >= .08)).mean():.1%} "
          "(default thresholds)")
    print(f" disk usage           : index {human_bytes(dir_size_bytes(out / 'index'))}, "
          f"actions {human_bytes(dir_size_bytes(out / 'actions'))}, total {human_bytes(dir_size_bytes(out))}")
    top = df[df.game != "other"].groupby("game")["duration"].sum().div(3600).sort_values(ascending=False).head(10)
    print(" top named games (h)  : " + ", ".join(f"{g} {h:.0f}" for g, h in top.items()))
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_root_arg(ap)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", default=str(DATA_ROOT / "nitrogen_meta"))
    ap.add_argument("--shards", default="all", help="'all' or e.g. '0-9,42'")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)),
                    help="shards processed in parallel; parsing is CPU-bound, so ~#cores-1 "
                         "(each worker also needs ~2 GB scratch disk for its tarball)")
    ap.add_argument("--progress-every", type=int, default=10000, help="log every N parsed chunks per shard")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--keep-tar", action="store_true", help="keep downloaded tarballs")
    ap.add_argument("--extract-loose", action="store_true",
                    help="also write the original per-chunk files under <out>/loose (large!)")
    ap.add_argument("--from-local-tars", default=None, help="dir with pre-downloaded SHARD_XXXX.tar.gz")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    log = setup_logging("01_fetch_metadata")
    out = Path(args.out)
    for d in ("index", "actions", "_done", "_tars"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for stale in out.rglob("*.partial"):
        stale.unlink()

    if not args.summary_only:
        shards = parse_shard_spec(args.shards, N_SHARDS)
        log.info("Syncing %d shard(s) from %s with %d worker(s)", len(shards), args.repo_id, args.workers)
        schema_printed = False
        failures = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_shard, s, args): s for s in shards}
            pbar = tqdm(total=len(futs), desc="shards", unit="shard")
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    info = fut.result()
                except Exception as e:
                    failures.append(s)
                    log.error("SHARD_%04d failed: %s\n%s", s, e, traceback.format_exc())
                    pbar.update(1)
                    continue
                if not schema_printed and info.get("schema"):
                    schema_printed = True
                    log.info("Detected parquet schema (first shard):\n%s", info.get("arrow_schema"))
                    log.info("Action layout -> %s", json.dumps(info["schema"]["sticks"]))
                    if info["schema"]["buttons_missing"]:
                        log.warning("Missing button columns (filled with 0): %s", info["schema"]["buttons_missing"])
                    if info.get("unknown_meta_keys"):
                        log.warning("Unknown metadata keys: %s", info["unknown_meta_keys"])
                    log.info("Example metadata.json:\n%s", json.dumps(info.get("metadata_example"), indent=2)[:1500])
                    log.info("Output action vector (%d-d): %s", ACTION_DIM, ACTION_NAMES)
                log.info("SHARD_%04d %s: %s chunks, errors=%s, %.0fs", s,
                         "skipped (done)" if info.get("skipped") else "ok", f"{info.get('chunks', 0):,}",
                         info.get("errors"), info.get("total_s", 0))
                pbar.update(1)
            pbar.close()
        if failures:
            log.error("Failed shards (re-run to retry): %s", failures)
    print_summary(out, log)


if __name__ == "__main__":
    main()
