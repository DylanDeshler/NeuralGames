#!/usr/bin/env python
"""Stage 02 — schema profile, 2D-game filter, idle filter, distribution-aware sampling, game holdout.

Pipeline:
  1. Profile: print the index schema, the raw parquet/metadata schema detected in stage 01, and
     decode a few action rows to confirm layout/value ranges.
  2. Label games as 2D / 3D / unknown:
        config/game_labels.csv (manual)  >  exclude_3d  >  games_2d  >  YouTube title/tags match
        (for game == "other" or unknown slugs; --enrich-youtube)  >  keyword vote (--keyword-fallback)
  3. Idle filter: keep chunks with frac(||j_left|| > --move-thresh) >= --min-move-frac AND
     frac(any gameplay button) >= --min-btn-frac. (start/back/guide don't count as gameplay.)
  4. Print the empirical distribution of the 2D pool (games, hours, P50/P90/P99, top 15).
  5. --plan: print idle-filter sensitivity + an allocation table (power vs cap) over --plan-targets, then exit.
     Otherwise, solve alpha in [0, 1] with scipy.optimize.brentq so that
        sum_g floor(n_g ** alpha) * mean_chunk_hours_g  ==  --target-hours
     i.e. a sublinear per-game quota: big multi-biome games keep more chunks than small ones,
     but the top few titles can't dominate. Within each game, chunks are spread across videos
     (proportional) and evenly across each video's timeline (biome coverage, no adjacent dupes).
  6. Hold out ~--val-frac of complete games as `val_unseen_games` (never in train).
  7. Write ./data/manifest_2d.parquet (+ reports in ./data/reports/).

Example:
  python 02_profile_and_sample.py --enrich-youtube --plan                 # measure: pool + allocation table
  python 02_profile_and_sample.py --enrich-youtube --target-hours 5000 --allocation power   # decide
"""
from __future__ import annotations

import argparse
import json
import os
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from scipy.optimize import brentq
from tqdm import tqdm

from ng_common import add_data_root_arg, DATA_ROOT, ACTION_DIM, STICK_THRESHOLDS, atomic_write_parquet, setup_logging
from ng_labels import BotBlockedError, GameLabeler, YouTubeMeta, video_urls

LIGHT_COLS = ["uuid", "shard", "video_id", "chunk_id", "game", "duration", "n_frames", "fps",
              "res_h", "res_w", "action_source", "frac_lstick_01", "frac_lstick_02", "frac_lstick_03",
              "frac_rstick_02", "frac_btn_gameplay", "frac_btn_any", "frac_any_input"]
HEAVY_COLS = ["uuid", "chunk_dir", "start_time", "end_time", "start_frame", "end_frame", "source", "url",
              "controller_type", "bbox_controller_overlay", "bbox_game_area", "bbox_others", "meta_json"]

# ============================================================================ profiling
def profile(meta_dir: Path, log) -> None:
    done = sorted((meta_dir / "_done").glob("SHARD_*.json"))
    if done:
        info = json.loads(done[0].read_text())
        log.info("Raw parquet schema detected in stage 01 (%s):\n%s", done[0].stem, info.get("arrow_schema"))
        log.info("Stick layout: %s | missing buttons: %s | extra cols: %s",
                 json.dumps(info.get("schema", {}).get("sticks")),
                 info.get("schema", {}).get("buttons_missing"), info.get("schema", {}).get("extra_columns"))
        ex = info.get("metadata_example") or {}
        log.info("metadata.json keys: %s | original_video keys: %s", sorted(ex), sorted(ex.get("original_video", {})))
    idx_files = sorted((meta_dir / "index").glob("*.parquet"))
    if not idx_files:
        raise SystemExit(f"No index files in {meta_dir}/index — run 01_fetch_metadata.py first.")
    t = ds.dataset(idx_files[0]).head(3)
    log.info("Index schema:\n%s", t.schema.to_string(show_schema_metadata=False))
    log.info("Sample index rows:\n%s", t.drop_columns(["meta_json"]).to_pandas().T.to_string()[:3000])
    act = ds.dataset(meta_dir / "actions" / idx_files[0].name).head(3).to_pylist()
    for r in act:
        a = np.frombuffer(r["actions"], np.float32).reshape(-1, ACTION_DIM)
        log.info("actions[%s]: shape=%s sticks∈[%.2f, %.2f] buttons∈%s source=%s", r["uuid"], a.shape,
                 a[:, :4].min(), a[:, :4].max(), np.unique(a[:, 4:]).tolist(), r["action_source"])


def load_index(meta_dir: Path, log) -> pd.DataFrame:
    dset = ds.dataset(sorted((meta_dir / "index").glob("*.parquet")))
    cols = [c for c in LIGHT_COLS if c in dset.schema.names]
    log.info("Loading index (%d columns) ...", len(cols))
    df = dset.to_table(columns=cols).to_pandas()
    df["duration"] = df["duration"].fillna(df["n_frames"] / df["fps"].replace(0, np.nan)).fillna(20.0)
    log.info("Loaded %s chunks from %s videos", f"{len(df):,}", f"{df.video_id.nunique():,}")
    return df


# ============================================================================ helpers
def per_game_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("game_key").agg(game=("game_name", "first"), genre=("genre", "first"),
                                   chunks=("uuid", "size"), videos=("video_id", "nunique"),
                                   hours=("duration", "sum"))
    g["hours"] /= 3600
    return g.sort_values("hours", ascending=False)


def print_distribution(title: str, df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    g = per_game_table(df)
    h = g["hours"].to_numpy()
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)
    if len(g) == 0:
        print(" (empty)")
        return g
    print(f" games: {len(g):,}   videos: {df.video_id.nunique():,}   chunks: {len(df):,}   "
          f"hours: {h.sum():,.1f}")
    print(f" hours/game  P50={np.percentile(h, 50):.2f}  P90={np.percentile(h, 90):.2f}  "
          f"P99={np.percentile(h, 99):.2f}  max={h.max():.1f}")
    print(f" top-1 share {h[:1].sum() / h.sum():.1%}   top-5 share {h[:5].sum() / h.sum():.1%}   "
          f"top-15 share {h[:15].sum() / h.sum():.1%}")
    print(f" genres: {df.groupby('genre')['duration'].sum().div(3600).round(1).sort_values(ascending=False).to_dict()}")
    print(f" {'#':>3} {'game':<42} {'genre':<14} {'videos':>6} {'chunks':>8} {'hours':>8}")
    for i, (k, r) in enumerate(g.head(top).iterrows(), 1):
        print(f" {i:>3} {str(r.game)[:42]:<42} {str(r.genre)[:14]:<14} {r.videos:>6} {r.chunks:>8} {r.hours:>8.1f}")
    print("=" * 78)
    return g


def quota_hours(n_chunks: np.ndarray, mean_hours: np.ndarray, a: float) -> np.ndarray:
    """Per-game hours under quota_g = floor(n_g ** a) (capped at n_g)."""
    return np.minimum(n_chunks, np.floor(n_chunks.astype(np.float64) ** a)) * mean_hours


def solve_alpha(n_chunks: np.ndarray, mean_hours: np.ndarray, target: float, log, quiet: bool = False) -> float:
    def hours(a):
        return float(quota_hours(n_chunks, mean_hours, a).sum())

    say = (lambda *a: None) if quiet else log.info
    lo, hi = hours(0.0), hours(1.0)
    say("Sampling bounds: alpha=0 -> %.1f h (1 chunk/game), alpha=1 -> %.1f h (everything)", lo, hi)
    if target >= hi:
        if not quiet:
            log.warning("target %.1f h >= available %.1f h; taking everything (alpha=1).", target, hi)
        return 1.0
    if target <= lo:
        if not quiet:
            log.warning("target %.1f h <= %.1f h (one chunk per game); using alpha=0.", target, lo)
        return 0.0
    a = brentq(lambda x: hours(x) - target, 0.0, 1.0, xtol=1e-9, maxiter=300)
    # hours(alpha) is a monotone step function: return the smallest alpha that reaches the target
    lo_a, hi_a = (a, 1.0) if hours(a) < target else (0.0, a)
    for _ in range(60):
        mid = (lo_a + hi_a) / 2
        lo_a, hi_a = (mid, hi_a) if hours(mid) < target else (lo_a, mid)
    say("Solved alpha=%.5f -> %.1f h (target %.1f h)", hi_a, hours(hi_a), target)
    return hi_a


def solve_cap(n_chunks: np.ndarray, mean_hours: np.ndarray, target: float) -> float:
    """Per-game hour cap c such that sum_g min(avail_g, c) == target (water-filling; max-balance allocation)."""
    avail = n_chunks * mean_hours
    if target >= avail.sum():
        return float(avail.max())
    return float(brentq(lambda c: np.minimum(avail, c).sum() - target, 0.0, float(avail.max()), xtol=1e-6))


def cap_quota(n_chunks: np.ndarray, mean_hours: np.ndarray, cap_h: float) -> np.ndarray:
    return np.minimum(n_chunks, np.maximum(1, np.round(cap_h / mean_hours))).astype(int)


def allocation_metrics(h: np.ndarray) -> dict:
    """Concentration metrics for a per-game hours vector."""
    h = h[h > 0]
    tot = h.sum()
    p = h / tot
    hs = np.sort(h)[::-1]
    return {"hours": tot, "games": len(h), "eff_games": float(np.exp(-(p * np.log(p)).sum())),
            "top1_share": hs[:1].sum() / tot, "top5_share": hs[:5].sum() / tot, "top15_share": hs[:15].sum() / tot,
            "median_game_h": float(np.median(h)), "max_game_h": float(hs[0]),
            "games_ge_10h": int((h >= 10).sum()), "games_ge_100h": int((h >= 100).sum())}


def activity_sensitivity(d2: pd.DataFrame, move: pd.Series, out_csv: Path) -> None:
    """Pool hours/games under a grid of idle-filter thresholds (before the filter is applied)."""
    rows = []
    for mm in (0.0, 0.05, 0.10, 0.15, 0.25, 0.35):
        for mb in (0.0, 0.04, 0.08, 0.15, 0.25):
            k = (move >= mm) & (d2["frac_btn_gameplay"] >= mb)
            rows.append({"min_move_frac": mm, "min_btn_frac": mb, "hours": d2.loc[k, "duration"].sum() / 3600,
                         "games": d2.loc[k, "game_key"].nunique(), "kept_frac": float(k.mean()) if len(k) else 0})
    t = pd.DataFrame(rows)
    t.to_csv(out_csv, index=False)
    piv = t.pivot(index="min_move_frac", columns="min_btn_frac", values="hours")
    print("\n Idle-filter sensitivity — 2D pool hours (rows: min frac ||j_left||>thresh, cols: min frac buttons):")
    print(piv.round(0).to_string())


ASPECTS = [("21:9", 2.37), ("16:9", 16 / 9), ("16:10", 1.6), ("3:2", 1.5), ("4:3", 4 / 3), ("8:7", 8 / 7),
           ("1:1", 1.0), ("3:4", 0.75), ("9:16", 9 / 16)]


def aspect_report(d2: pd.DataFrame, meta_dir: Path, reports: Path) -> None:
    """Content aspect ratio of the 2D pool by hours, from res + bbox_game_area (metadata only).

    Hidden pillarboxing (a 4:3 game inside a 16:9 video without bbox_game_area) can't be seen here;
    stage 03 detects it from pixels, so the real non-16:9 share is >= what this shows.
    """
    parts = []
    for shard, sdf in d2.groupby("shard"):
        t = ds.dataset(meta_dir / "index" / f"SHARD_{int(shard):04d}.parquet").to_table(
            columns=["uuid", "bbox_game_area"], filter=pc.field("uuid").isin(sdf["uuid"].tolist()))
        parts.append(t.to_pandas())
    ga = pd.concat(parts).drop_duplicates("uuid").set_index("uuid")["bbox_game_area"]
    x = d2[["uuid", "video_id", "res_w", "res_h", "duration"]].copy()
    g = x["uuid"].map(ga).map(lambda v: json.loads(v) if isinstance(v, str) else None)
    lo_path = DATA_ROOT / "layout_overrides.parquet"  # after 02c: crop = metadata area ∩ detected area
    n_lo = 0
    if lo_path.exists():
        lo = pd.read_parquet(lo_path)
        lo = lo[lo["status"] == "ok"].set_index("video_id")["game_area"].dropna().map(json.loads)
        full = {"xtl": 0.0, "ytl": 0.0, "xbr": 1.0, "ybr": 1.0}

        def inter(a, b):
            return {"xtl": max(a["xtl"], b["xtl"]), "ytl": max(a["ytl"], b["ytl"]),
                    "xbr": min(a["xbr"], b["xbr"]), "ybr": min(a["ybr"], b["ybr"])}
        det = x["video_id"].map(lo)
        has = det.notna()
        n_lo = int(has.sum())
        g = pd.Series([inter(gg if isinstance(gg, dict) else full, dd) if isinstance(dd, dict) else gg
                       for gg, dd in zip(g, det)], index=g.index)
    fw = g.map(lambda d: (d["xbr"] - d["xtl"]) if isinstance(d, dict) else 1.0)
    fh = g.map(lambda d: (d["ybr"] - d["ytl"]) if isinstance(d, dict) else 1.0)
    x["aspect"] = (x["res_w"] * fw) / (x["res_h"] * fh).replace(0, np.nan)
    if lo_path.exists():  # 02c's final per-video rectangle (incl. pixel trims) is the most exact
        lo2 = pd.read_parquet(lo_path)
        if "content_rel" in lo2:
            cr = lo2[lo2["status"] == "ok"].set_index("video_id")["content_rel"].dropna().map(json.loads)
            m_ = x["video_id"].isin(cr.index)
            rel = x.loc[m_, "video_id"].map(cr)
            x.loc[m_, "aspect"] = [(r[2] * w) / max(r[3] * h, 1e-6) for r, w, h in zip(rel, x.loc[m_, "res_w"], x.loc[m_, "res_h"])]
    x["has_game_area"] = g.map(lambda d: isinstance(d, dict))

    def name(a):
        if not np.isfinite(a):
            return "unknown"
        n, r = min(ASPECTS, key=lambda t: abs(np.log(a / t[1])))
        return n if abs(a / r - 1) < 0.04 else f"other({a:.2f})"
    x["aspect_name"] = x["aspect"].map(name)
    t = x.groupby("aspect_name").agg(hours=("duration", "sum"), chunks=("uuid", "size"),
                                     with_game_area=("has_game_area", "mean"))
    t["hours"] /= 3600
    t["share"] = t["hours"] / t["hours"].sum()
    t = t.sort_values("hours", ascending=False)
    t.to_csv(reports / "aspect_distribution.csv")
    src = f"metadata ∩ 02c layout boxes for {n_lo:,} chunks" if n_lo else "metadata only; run 02c for post-crop ratios"
    print(f"\n Content aspect ratio of the 2D pool ({src}; stage 03 also trims hidden bars):")
    for k, r in t.iterrows():
        print(f"   {k:<14} {r.hours:>9.1f} h  {r.share:>6.1%}   (bbox_game_area on {r.with_game_area:.0%})")


def print_plan(d2: pd.DataFrame, targets: list[float], args, reports: Path, log) -> None:
    stats = d2.groupby("game_key").agg(game=("game_name", "first"), genre=("genre", "first"),
                                       videos=("video_id", "nunique"), n=("uuid", "size"), mh=("duration", "mean"))
    n, mh = stats["n"].to_numpy(), stats["mh"].to_numpy() / 3600
    avail = float((n * mh).sum())
    targets = sorted({t for t in targets if t < avail} | {avail})
    rows, per_game = [], stats[["game", "genre", "videos"]].assign(avail_h=n * mh)
    for t in targets:
        a = solve_alpha(n, mh, t, log, quiet=True)
        c = solve_cap(n, mh, t)
        for strat, h, param in (("power", quota_hours(n, mh, a), a), ("cap", cap_quota(n, mh, c) * mh, c)):
            m = allocation_metrics(h)
            rows.append({"strategy": strat, "target_h": t, "param": param, **m,
                         "disk_gb": m["hours"] * args.est_disk_gb_per_hour,
                         "download_tb": m["hours"] * args.est_download_gb_per_hour / 1000})
            per_game[f"{strat}_h@{t:.0f}"] = h
    plan = pd.DataFrame(rows)
    plan.to_csv(reports / "allocation_plan.csv", index=False)
    per_game.sort_values("avail_h", ascending=False).to_csv(reports / "allocation_per_game.csv")
    print("\n" + "=" * 118)
    print(" ALLOCATION PLAN   power: quota_g = floor(n_g ** alpha) (param = alpha; keeps size ordering, sublinear)")
    print("                   cap:   hours_g = min(avail_g, cap)      (param = cap in hours; most balanced for a total)")
    print(" eff_games = exp(entropy of per-game hour shares): how many equally-sized games the mix is 'worth'")
    print("=" * 118)
    print(f" {'strategy':<8} {'target_h':>9} {'param':>7} {'hours':>9} {'eff_games':>9} {'top1':>6} {'top5':>6} {'top15':>6} "
          f"{'med_game_h':>10} {'max_game_h':>10} {'≥10h':>5} {'≥100h':>6} {'disk_GB':>8} {'dl_TB':>6}")
    for r in rows:
        if r["strategy"] == "power" and r is not rows[0]:
            print()
        print(f" {r['strategy']:<8} {r['target_h']:>9.0f} {r['param']:>7.3f} {r['hours']:>9.0f} {r['eff_games']:>9.1f} "
              f"{r['top1_share']:>6.1%} {r['top5_share']:>6.1%} {r['top15_share']:>6.1%} {r['median_game_h']:>10.2f} "
              f"{r['max_game_h']:>10.1f} {r['games_ge_10h']:>5} {r['games_ge_100h']:>6} {r['disk_gb']:>8.0f} "
              f"{r['download_tb']:>6.1f}")
    print("=" * 118)
    print(f" disk = processed 256px clips @ {args.est_disk_gb_per_hour} GB/h (x2 while webdataset shards coexist); "
          f"download = source video @ {args.est_download_gb_per_hour} GB/h (480p30 default). Rough estimates.")
    print(f" Per-game hours for every target: {reports / 'allocation_per_game.csv'}")
    print(" Next: re-run with --target-hours <H> [--allocation power|cap] (or --alpha <a>) to write the manifest.")


def largest_remainder(weights: np.ndarray, total: int, caps: np.ndarray) -> np.ndarray:
    """Integer allocation of `total` proportional to weights, respecting per-item caps."""
    alloc = np.zeros(len(weights), int)
    remaining, active = total, np.ones(len(weights), bool)
    while remaining > 0 and active.any():
        w = weights * active
        share = w / w.sum() * remaining
        add = np.minimum(np.floor(share).astype(int), caps - alloc)
        if add.sum() == 0:  # hand out by largest fractional part
            order = np.argsort(-(share - np.floor(share)) * active)
            for i in order:
                if remaining == 0:
                    break
                if active[i] and alloc[i] < caps[i]:
                    alloc[i] += 1
                    remaining -= 1
        else:
            alloc += add
            remaining -= add.sum()
        active = alloc < caps
    return alloc


def _spread_order(n: int) -> np.ndarray:
    """Order 0..n-1 so every prefix is spread evenly over the range (bit-reversal / van der Corput)."""
    if n <= 1:
        return np.arange(n)
    bits = int(np.ceil(np.log2(n)))
    rev = np.array([int(format(i, f"0{bits}b")[::-1], 2) for i in range(1 << bits)])
    return rev[rev < n]


def download_priority(sel: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Priority (hours) such that processing chunks in ascending order makes EVERY prefix a water-filling
    ('cap') allocation: each game's chunks are ranked (round-robin over its videos, spread along each video's
    timeline) and priority = that game's cumulative hours up to and including the chunk."""
    prio = pd.Series(np.nan, index=sel.index)
    for _, g in sel.groupby("game_key"):
        per_vid = []
        for _, v in g.groupby("video_id"):
            idx = v.sort_values("chunk_id").index.to_numpy()
            n = len(idx)
            # start mid-video and spread outwards; the first/last ~5% of a stream (setup, menus, title
            # screens, outros) goes to the very end of the queue
            order = [(i + n // 2) % n for i in _spread_order(n)]
            edge = max(1, int(round(0.05 * n))) if n >= 6 else 0
            is_edge = [o < edge or o >= n - edge for o in order]
            order = [o for o, e in zip(order, is_edge) if not e] + [o for o, e in zip(order, is_edge) if e]
            per_vid.append(list(idx[order]))
        rng.shuffle(per_vid)
        per_vid.sort(key=len, reverse=True)
        order = [lst[i] for i in range(max(map(len, per_vid))) for lst in per_vid if i < len(lst)]
        prio.loc[order] = np.cumsum(sel.loc[order, "duration"].to_numpy()) / 3600
    return prio + rng.uniform(0, 1e-6, len(prio))  # random tie-break across games


def select_within_game(gdf: pd.DataFrame, quota: int, rng: np.random.Generator) -> np.ndarray:
    """Spread `quota` chunks across videos (∝ size) and evenly along each video's timeline."""
    vids = gdf.groupby("video_id").indices
    keys = list(vids)
    sizes = np.array([len(vids[k]) for k in keys])
    per_vid = largest_remainder(sizes.astype(float), quota, sizes)
    picked = []
    for k, q in zip(keys, per_vid):
        if q == 0:
            continue
        idx = vids[k][np.argsort(gdf["chunk_id"].to_numpy()[vids[k]])]
        if q >= len(idx):
            picked.extend(idx)
            continue
        edges = np.linspace(0, len(idx), q + 1)
        for a, b in zip(edges[:-1], edges[1:]):  # one random chunk per equal-width time stratum
            lo, hi = int(math.floor(a)), max(int(math.floor(a)) + 1, int(math.floor(b)))
            picked.append(idx[rng.integers(lo, hi)])
    return gdf.index.to_numpy()[np.array(picked, int)]


def recompute_activity(df: pd.DataFrame, meta_dir: Path, thresh: float, include_dpad: bool, log,
                       cache_path: Path | None = None) -> pd.Series:
    """frac(||j_left|| > thresh [OR any d-pad]) per chunk, recomputed from the stored actions.

    Cached per (uuid, thresh, include_dpad) so re-runs of stage 02 are instant.
    """
    tag = f"move_{thresh:.2f}_{'dpad' if include_dpad else 'stick'}"
    cache = pd.Series(dtype=np.float32)
    if cache_path is not None and cache_path.exists():
        c = pd.read_parquet(cache_path)
        if tag in c.columns:
            cache = c.set_index("uuid")[tag].dropna()
    need = df[~df["uuid"].isin(cache.index)]
    vals: dict[str, float] = {}
    if len(need):
        log.info("Computing %s for %s chunks from stored actions (cached afterwards) ...", tag, f"{len(need):,}")
        for shard, sdf in tqdm(need.groupby("shard"), desc=f"recompute {tag}"):
            t = ds.dataset(meta_dir / "actions" / f"SHARD_{int(shard):04d}.parquet").to_table(
                columns=["uuid", "actions"], filter=pc.field("uuid").isin(sdf["uuid"].tolist()))
            for u, b in zip(t.column("uuid").to_pylist(), t.column("actions").to_pylist()):
                a = np.frombuffer(b, np.float32).reshape(-1, ACTION_DIM)
                m = np.linalg.norm(a[:, :2], axis=1) > thresh
                if include_dpad:
                    m |= a[:, 4:8].max(1) > 0
                vals[u] = float(m.mean())
        if cache_path is not None:
            new = pd.Series(vals, dtype=np.float32)
            allv = pd.concat([cache, new])
            old = pd.read_parquet(cache_path).set_index("uuid") if cache_path.exists() else pd.DataFrame()
            old = old.reindex(old.index.union(allv.index))
            old[tag] = allv.reindex(old.index)
            atomic_write_parquet(old.rename_axis("uuid").reset_index(), cache_path)
        cache = pd.concat([cache, pd.Series(vals, dtype=np.float32)])
    return df["uuid"].map(cache).astype(np.float32)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_root_arg(ap)
    ap.add_argument("--meta-dir", default=str(DATA_ROOT / "nitrogen_meta"))
    ap.add_argument("--genres", default="./config/genres_2d.yaml")
    ap.add_argument("--labels-csv", default="./config/game_labels.csv",
                    help="optional manual overrides: columns game,is_2d[,genre]")
    ap.add_argument("--out", default=str(DATA_ROOT / "manifest_2d.parquet"))
    ap.add_argument("--reports-dir", default=str(DATA_ROOT / "reports"))
    # YouTube enrichment
    ap.add_argument("--enrich-youtube", action="store_true",
                    help="look up YouTube title/tags for game=='other' / unknown-slug videos")
    ap.add_argument("--check-availability", action="store_true",
                    help="also probe every 2D candidate video and drop private/removed ones")
    ap.add_argument("--llm-labels", default=str(DATA_ROOT / "llm_game_labels.parquet"),
                    help="per-video LLM game labels from 02b_llm_label_games.py (used if the file exists)")
    ap.add_argument("--llm-min-confidence", type=float, default=0.6)
    ap.add_argument("--llm-allow-multi", action="store_true", help="also use labels for multi-game videos")
    ap.add_argument("--llm-accept-perspectives", nargs="+", default=["side-view-2d", "top-down-2d"],
                    help="LLM perspectives counted as 2D for games not in the curated lists")
    ap.add_argument("--keyword-fallback", action="store_true",
                    help="accept unidentified videos whose YouTube tags/title vote 2D (each becomes its own "
                         "'unk:<video_id>' game, always kept in train)")
    ap.add_argument("--yt-cache", default=str(DATA_ROOT / "yt_meta_cache.jsonl"))
    ap.add_argument("--yt-workers", type=int, default=4)
    ap.add_argument("--yt-min-interval", type=float, default=0.5, help="seconds between YouTube requests")
    ap.add_argument("--yt-max-videos", type=int, default=0, help="cap lookups (0 = no cap; for testing)")
    ap.add_argument("--cookies", default=None)
    ap.add_argument("--cookies-from-browser", default=None)
    ap.add_argument("--yt-backend", choices=["auto", "api", "ytdlp"], default="auto",
                    help="metadata source for YouTube videos: 'api' = YouTube Data API v3 (no bot checks; "
                         "needs --yt-api-key or env YOUTUBE_API_KEY), 'ytdlp' = scraping, 'auto' = api if a key is set")
    ap.add_argument("--yt-api-key", default=os.environ.get("YOUTUBE_API_KEY"))
    ap.add_argument("--proxy", default=None, help="proxy for yt-dlp metadata requests, e.g. socks5://host:port")
    # activity filter
    ap.add_argument("--move-thresh", type=float, default=0.2)
    ap.add_argument("--min-move-frac", type=float, default=0.15)
    ap.add_argument("--move-includes-dpad", action=argparse.BooleanOptionalAction, default=True,
                    help="count d-pad presses as movement (many 2D players never touch the stick)")
    ap.add_argument("--min-btn-frac", type=float, default=0.08)
    ap.add_argument("--min-chunks-per-game", type=int, default=3)
    ap.add_argument("--fps", type=float, nargs="*", default=[30.0, 60.0],
                    help="keep only these annotation fps (default 30 60: both reach 30/15 fps with an integer "
                         "stride; ~1%% of videos are 25/50 fps). Pass no values to keep all.")
    # sampling + split
    ap.add_argument("--plan", action="store_true",
                    help="measure only: print pool distribution, idle-filter sensitivity and an allocation table "
                         "for --plan-targets, write reports, and exit without a manifest")
    ap.add_argument("--plan-targets", type=float, nargs="+",
                    default=[250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000])
    ap.add_argument("--target-hours", type=float, default=None, help="total sampled hours (train + val)")
    ap.add_argument("--alpha", type=float, default=None, help="use this exponent directly instead of solving")
    ap.add_argument("--allocation", choices=["power", "cap"], default="power",
                    help="power: floor(n_g**alpha) solved for --target-hours; cap: per-game hour cap solved for it")
    ap.add_argument("--est-disk-gb-per-hour", type=float, default=0.10, help="processed clips @ 15 fps, CRF 18 (measured)")
    ap.add_argument("--est-download-gb-per-hour", type=float, default=0.15,
                    help="source download per hour of data (~0.15 GB measured at 360p VP9; ~1.8 GB at 720p60)")
    ap.add_argument("--val-frac", type=float, default=0.10, help="fraction of games held out")
    ap.add_argument("--val-max-hours-pct", type=float, default=90.0,
                    help="only games at or below this percentile of hours are eligible for holdout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-profile", action="store_true")
    args = ap.parse_args()

    if not args.plan and args.target_hours is None and args.alpha is None:
        ap.error("choose an allocation: run with --plan first, then pass --target-hours H or --alpha A")
    log = setup_logging("02_profile_and_sample")
    rng = np.random.default_rng(args.seed)
    meta_dir, reports = Path(args.meta_dir), Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    if not args.skip_profile:
        profile(meta_dir, log)
    df = load_index(meta_dir, log)
    if args.fps:
        df = df[np.isin(df["fps"].round(), np.round(args.fps))]
        log.info("fps filter %s -> %s chunks", args.fps, f"{len(df):,}")

    # ---------------------------------------------------------------- labeling
    lab = GameLabeler(Path(args.genres), Path(args.labels_csv) if args.labels_csv else None)
    junk = df["game"].map(lab.is_junk_slug)
    if junk.any():
        log.info("Junk/chapter slugs treated as 'other': %d slugs, %.0f h (e.g. %s)", df.loc[junk, "game"].nunique(),
                 df.loc[junk, "duration"].sum() / 3600, ", ".join(df.loc[junk, "game"].value_counts().index[:6]))
        df.loc[junk, "game"] = "other"
    slug_lab = {g: lab.label_slug(g) for g in df["game"].unique()}
    df["status"] = df["game"].map(lambda g: slug_lab[g][0])
    df["genre"] = df["game"].map(lambda g: slug_lab[g][1])
    df["label_source"] = df["game"].map(lambda g: slug_lab[g][2])
    ckey = {g: lab.canonical_key(g) for g in df["game"].unique()}
    df["game_key"] = df["game"].map(ckey)
    disp = {k: (lab.title_2d[k][0] if k in lab.title_2d else lab.title_3d.get(k)) for k in set(ckey.values())}
    df["game_name"] = [disp.get(k) or g for g, k in zip(df["game"], df["game_key"])]
    log.info("Slug labels: %s", df.groupby("status")["duration"].sum().div(3600).round(1).to_dict())

    yt = None
    if args.enrich_youtube or args.check_availability:
        backend = args.yt_backend if args.yt_backend != "auto" else ("api" if args.yt_api_key else "ytdlp")
        yt = YouTubeMeta(Path(args.yt_cache), args.yt_workers, args.yt_min_interval, args.cookies,
                         args.cookies_from_browser, log, backend=backend, api_key=args.yt_api_key, proxy=args.proxy)
        urls = video_urls(Path(args.meta_dir))
    if args.enrich_youtube:
        unk_vids = df.loc[df["status"] == "unknown", "video_id"].unique().tolist()
        # biggest videos first so a capped run covers the most hours
        order = df[df["video_id"].isin(unk_vids)].groupby("video_id")["duration"].sum().sort_values(ascending=False)
        unk_vids = order.index.tolist()[: args.yt_max_videos or None]
        meta = yt.get_many(unk_vids, urls)
        vid_update, unk_slug_votes, slug_hit_keys = {}, defaultdict(Counter), defaultdict(set)
        slug_of_vid = df.drop_duplicates("video_id").set_index("video_id")["game"].to_dict()
        for v in unk_vids:
            m = meta.get(v, {})
            if m.get("status") != "ok":
                if m.get("status") == "unavailable":
                    vid_update[v] = ("unavailable", "", "youtube", None, None)
                continue
            text = " | ".join([m.get("title") or "", " ".join(m.get("tags") or [])])
            # the TITLE is strong evidence; tags/description are weak (streamers stuff tags with popular game
            # names: e.g. Celeste-mod streams tagged "super mario bros"), so LLM labels may override those
            hit, src = lab.match_text(m.get("title") or ""), "youtube_title"
            if not hit:
                hit = lab.match_text(" ".join(m.get("tags") or [])) or lab.match_text((m.get("description") or "")[:300])
                src = "youtube_tags"
            slug = slug_of_vid[v]
            if slug != "other":  # named but unlisted slug: per-video hits win, the rest may be labeled by vote
                if hit:
                    unk_slug_votes[slug][hit[0]] += 1
                    slug_hit_keys[slug].add(hit[1])
                else:
                    unk_slug_votes[slug][{1: "kw2d", -1: "kw3d", 0: "none"}[lab.keyword_vote(text + " " + (m.get("description") or ""))]] += 1
                    continue
            if hit:
                status, key, _ = hit
                if status == "2d":
                    disp, genre = lab.title_2d[key]
                    vid_update[v] = ("2d", genre, src, key, disp)
                else:
                    vid_update[v] = ("3d", "", src, key, lab.title_3d[key])
            elif args.keyword_fallback and lab.keyword_vote(text + " " + (m.get("description") or "")) > 0:
                vid_update[v] = ("2d", "2D (keyword)", "youtube_keywords", f"unk:{v}", f"unk:{v}")
        n_multi = 0
        for slug, votes in unk_slug_votes.items():
            if len(slug_hit_keys[slug]) >= 2:  # titles name different games -> chapter/junk slug: no slug-level vote
                n_multi += 1
                continue
            n = sum(votes.values())
            rest = (df["game"] == slug) & ~df["video_id"].isin(vid_update)  # videos without their own title hit
            if votes["2d"] / n > 0.5 or (args.keyword_fallback and (votes["2d"] + votes["kw2d"]) / n > 0.6
                                         and votes["3d"] + votes["kw3d"] == 0):
                df.loc[rest, ["status", "genre", "label_source"]] = ["2d", "2D (youtube vote)", "youtube_vote"]
                if slug_hit_keys[slug]:  # consistent single game: use its canonical key/name
                    k = next(iter(slug_hit_keys[slug]))
                    df.loc[rest, ["game_key", "game_name"]] = [k, lab.title_2d.get(k, (k,))[0]]
                else:  # vote-only (often a level/area name): unknown game identity -> train-only 'unk:' key
                    df.loc[rest, ["game_key", "game_name"]] = [f"unk:{slug}", f"unk:{slug}"]
            elif (votes["3d"] + votes["kw3d"]) / n > 0.5:
                df.loc[rest, ["status", "label_source"]] = ["3d", "youtube_vote"]
        log.info("Slug votes: %d named slugs looked multi-game (left per-video)", n_multi)
        if vid_update:
            upd = pd.DataFrame.from_dict(vid_update, orient="index",
                                         columns=["u_status", "u_genre", "u_src", "u_key", "u_name"])
            m = df["video_id"].isin(upd.index)
            u = upd.loc[df.loc[m, "video_id"]].set_index(df.index[m])
            df.loc[m, "status"] = u["u_status"]
            df.loc[m, "genre"] = u["u_genre"]
            df.loc[m, "label_source"] = u["u_src"]
            has_key = m & df["video_id"].map(lambda v: vid_update.get(v, (None,) * 5)[3] is not None)
            df.loc[has_key, "game_key"] = df.loc[has_key, "video_id"].map(lambda v: vid_update[v][3])
            df.loc[has_key, "game_name"] = df.loc[has_key, "video_id"].map(lambda v: vid_update[v][4])
        log.info("After YouTube enrichment: %s", df.groupby("status")["duration"].sum().div(3600).round(1).to_dict())

    # ---------------------------------------------------------------- LLM labels from stage 02b
    llm_path = Path(args.llm_labels) if args.llm_labels else None
    if llm_path and llm_path.exists():
        L = pd.read_parquet(llm_path).drop_duplicates("video_id", keep="last").set_index("video_id")
        L = L[L["llm_parse_ok"] & L["llm_game"].notna() & (L["llm_confidence"] >= args.llm_min_confidence)]
        if not args.llm_allow_multi:
            L = L[~L["llm_multiple_games"].astype(bool)]
        # apply where we have no label yet, or only weak evidence (keyword / slug vote)
        weak = (df["status"] == "unknown") | df["label_source"].isin(["youtube_keywords", "youtube_vote", "youtube_tags"])
        cand = set(df.loc[weak, "video_id"].unique()) & set(L.index)
        # perspective per game: game-level LLM answer (02b second pass) > majority of video-level answers
        gp_path = llm_path.with_name("llm_game_perspectives.parquet")
        game_persp: dict[str, str] = {}
        vl = L[~L["llm_perspective"].isin(["unknown", "mixed"])]
        for k, sgrp in vl.groupby("llm_game_key"):
            game_persp[k] = sgrp["llm_perspective"].mode().iat[0]
        n_vote = len(game_persp)
        if gp_path.exists():
            gp = pd.read_parquet(gp_path)
            gp = gp[gp["game_parse_ok"].astype(bool) & (gp["game_confidence"] >= args.llm_min_confidence)
                    & ~gp["game_perspective"].isin(["unknown"])]
            game_persp.update(dict(zip(gp["llm_game_key"], gp["game_perspective"])))
            log.info("Game-level perspectives: %d from %s, %d from video majority", len(gp), gp_path.name, n_vote)
        else:
            log.info("No %s (re-run 02b for the per-game pass); using video-majority perspectives", gp_path.name)
        upd = {}
        for v in cand:
            r = L.loc[v]
            key, name = r["llm_game_key"], r["llm_game"]
            status, genre, src = lab.label_slug(name)  # curated / manual lists win on the LLM's title
            if status == "unknown":  # e.g. "The Binding of Isaac: Repentance" -> curated "The Binding of Isaac"
                hit = lab.match_text(name)
                if hit:
                    key = hit[1]
                    name = lab.title_2d[key][0] if hit[0] == "2d" else lab.title_3d[key]
                    status, genre, src = lab.label_slug(name)
            if status == "unknown":
                p_ = game_persp.get(r["llm_game_key"], r["llm_perspective"])
                status = "2d" if p_ in args.llm_accept_perspectives else ("unknown" if p_ in ("unknown", "mixed") else "3d")
                genre, src = (r["llm_genre"] or ""), "llm"
            else:
                src = f"llm+{src}"
            key = lab.group_of.get(key, key)  # editions / series -> one game for the holdout
            if key in lab.title_2d:  # show the canonical title (e.g. Celeste, not one of its mod packs)
                name = lab.title_2d[key][0]
            elif key in lab.title_3d:
                name = lab.title_3d[key]
            upd[v] = (status, genre, src, key, name)
        if upd:
            u = pd.DataFrame.from_dict(upd, orient="index", columns=["status", "genre", "label_source", "game_key", "game_name"])
            m = df["video_id"].isin(u.index)
            df.loc[m, u.columns] = u.loc[df.loc[m, "video_id"]].to_numpy()
        log.info("LLM labels (%s): applied to %d videos (conf>=%.2f) -> %s", llm_path, len(upd),
                 args.llm_min_confidence, df.groupby("status")["duration"].sum().div(3600).round(1).to_dict())
    elif llm_path:
        log.info("No LLM labels at %s (run 02b_llm_label_games.py to add them)", llm_path)

    # safety net: final label vs a confident per-video LLM label (catches e.g. a whole channel mislabeled by tags)
    if llm_path and llm_path.exists():
        Lr = pd.read_parquet(llm_path).drop_duplicates("video_id", keep="last").set_index("video_id")
        Lr = Lr[Lr["llm_parse_ok"] & Lr["llm_game"].notna() & (Lr["llm_confidence"] >= 0.8) & ~Lr["llm_multiple_games"].astype(bool)]
        pv = df[df["status"] == "2d"].groupby("video_id").agg(game_key=("game_key", "first"), game=("game_name", "first"),
                                                              src=("label_source", "first"), hours=("duration", "sum"))
        pv = pv[pv.index.isin(Lr.index)]
        pv["llm_game"] = Lr.loc[pv.index, "llm_game"]
        pv["llm_key"] = [lab.group_of.get(lab.canonical_key(g), lab.canonical_key(g)) for g in pv["llm_game"]]
        dis = pv[pv["llm_key"] != pv["game_key"]].copy()
        dis["hours"] /= 3600
        dis.sort_values("hours", ascending=False).to_csv(reports / "label_llm_disagreements.csv")
        if len(dis):
            top = dis.groupby(["game", "src"])["hours"].agg(["sum", "size"]).sort_values("sum", ascending=False).head(8)
            log.warning("2D videos whose label disagrees with a confident LLM label: %d videos, %.0f h "
                        "(reports/label_llm_disagreements.csv). Largest:\n%s", len(dis), dis["hours"].sum(),
                        top.round(1).to_string())

    # label report for manual curation
    rep = df.groupby(["game_name", "status", "label_source"]).agg(
        videos=("video_id", "nunique"), hours=("duration", "sum")).reset_index()
    rep["hours"] = (rep["hours"] / 3600).round(2)
    rep.sort_values("hours", ascending=False).to_csv(reports / "game_label_report.csv", index=False)
    unl = rep[(rep.status == "unknown") & (rep.game_name != "other")].sort_values("hours", ascending=False)
    if len(unl):
        log.info("Top unlabeled named games (label them in %s, then re-run):\n%s", args.labels_csv,
                 unl.head(25)[["game_name", "videos", "hours"]].to_string(index=False))

    d2 = df[df["status"] == "2d"].copy()
    log.info("2D pool before activity filter: %s chunks, %.1f h", f"{len(d2):,}", d2.duration.sum() / 3600)

    # ---------------------------------------------------------------- availability
    if args.check_availability:
        meta = yt.get_many(d2["video_id"].unique().tolist(), urls)
        bad = {v for v, m in meta.items() if m.get("status") == "unavailable"}
        log.info("Availability: %d / %d 2D videos unavailable -> dropped", len(bad), d2.video_id.nunique())
        d2 = d2[~d2["video_id"].isin(bad)]
    elif yt is not None:
        bad = {v for v, m in yt.cache.items() if m.get("status") == "unavailable"}
        d2 = d2[~d2["video_id"].isin(bad)]

    # ---------------------------------------------------------------- activity filter
    col = f"frac_{'move' if args.move_includes_dpad else 'lstick'}_{int(round(args.move_thresh * 10)):02d}"
    if args.move_thresh in STICK_THRESHOLDS and col in d2.columns:
        move = d2[col]
    else:
        move = recompute_activity(d2, meta_dir, args.move_thresh, args.move_includes_dpad, log,
                                  cache_path=meta_dir / "activity_cache.parquet")
    if args.move_includes_dpad and args.plan and "frac_lstick_02" in d2.columns:
        stick_only = d2[f"frac_lstick_{int(round(args.move_thresh * 10)):02d}"] >= args.min_move_frac
        with_dpad = move >= args.min_move_frac
        log.info("Movement filter: stick-only keeps %.1f h, stick-or-d-pad keeps %.1f h (+%.1f h rescued)",
                 d2.loc[stick_only, "duration"].sum() / 3600, d2.loc[with_dpad, "duration"].sum() / 3600,
                 d2.loc[with_dpad & ~stick_only, "duration"].sum() / 3600)
    if args.plan:
        activity_sensitivity(d2, move, reports / "idle_filter_sensitivity.csv")
    keep = (move >= args.min_move_frac) & (d2["frac_btn_gameplay"] >= args.min_btn_frac)
    log.info("Activity filter (move>%.2f in >=%.0f%% frames AND buttons in >=%.0f%% frames): kept %.1f%% "
             "(%s -> %s chunks)", args.move_thresh, 100 * args.min_move_frac, 100 * args.min_btn_frac,
             100 * keep.mean() if len(keep) else 0, f"{len(d2):,}", f"{keep.sum():,}")
    d2 = d2[keep]
    counts = d2.groupby("game_key")["uuid"].transform("size")
    d2 = d2[counts >= args.min_chunks_per_game]
    if d2.empty:
        raise SystemExit("No 2D chunks left after filtering. Try --enrich-youtube, label more games, "
                         "or relax thresholds.")
    gtab = print_distribution("EMPIRICAL DISTRIBUTION — 2D, non-idle pool", d2)
    gtab.to_csv(reports / "distribution_pool.csv")

    # ---------------------------------------------------------------- alpha sampling
    if args.plan:
        aspect_report(d2, meta_dir, reports)
        print_plan(d2, args.plan_targets, args, reports, log)
        return
    stats = d2.groupby("game_key").agg(n=("uuid", "size"), mh=("duration", "mean"))
    if args.alpha is not None:
        alpha = float(np.clip(args.alpha, 0.0, 1.0))
        log.info("Using --alpha %.4f -> %.1f h", alpha,
                 quota_hours(stats["n"].to_numpy(), stats["mh"].to_numpy() / 3600, alpha).sum())
    elif args.allocation == "power":
        alpha = solve_alpha(stats["n"].to_numpy(), stats["mh"].to_numpy() / 3600, args.target_hours, log)
    if args.allocation == "cap" and args.alpha is None:
        cap_h = solve_cap(stats["n"].to_numpy(), stats["mh"].to_numpy() / 3600, args.target_hours)
        stats["quota"] = cap_quota(stats["n"].to_numpy(), stats["mh"].to_numpy() / 3600, cap_h)
        alpha = float("nan")
        log.info("Allocation 'cap': per-game cap %.1f h -> %.1f h", cap_h,
                 (stats["quota"] * stats["mh"] / 3600).sum())
    else:
        stats["quota"] = np.minimum(stats["n"], np.floor(stats["n"].astype(float) ** alpha)).astype(int)
    picked = []
    for key, gdf in tqdm(d2.groupby("game_key"), desc="sampling games"):
        picked.append(select_within_game(gdf, int(stats.at[key, "quota"]), rng))
    sel = d2.loc[np.concatenate(picked)].copy()
    sel["download_priority"] = download_priority(sel, rng)

    # ---------------------------------------------------------------- holdout split (by game)
    sg = per_game_table(sel)
    eligible = sg[~sg.index.str.startswith("unk:")]
    cap = np.percentile(eligible["hours"], args.val_max_hours_pct) if len(eligible) else 0
    eligible = eligible[eligible["hours"] <= cap].index.to_numpy()
    n_val = int(round(args.val_frac * len(sg[~sg.index.str.startswith("unk:")])))
    n_val = min(n_val, max(len(eligible) - 0, 0))
    val_games = set(rng.choice(eligible, size=n_val, replace=False)) if n_val > 0 else set()
    sel["split"] = np.where(sel["game_key"].isin(val_games), "val_unseen_games", "train")
    sel["sampling_alpha"] = alpha

    print_distribution(f"SAMPLED SUBSET — alpha={alpha:.4f}, target={args.target_hours or 0:.0f} h", sel)
    for sp, sdf in sel.groupby("split"):
        log.info("split %-17s games=%4d videos=%5d chunks=%7s hours=%7.1f", sp, sdf.game_key.nunique(),
                 sdf.video_id.nunique(), f"{len(sdf):,}", sdf.duration.sum() / 3600)
    assert not (set(sel.loc[sel.split == "train", "game_key"]) & set(sel.loc[sel.split != "train", "game_key"]))

    # ---------------------------------------------------------------- join heavy columns + write
    heavy = []
    for shard, sdf in sel.groupby("shard"):
        f = meta_dir / "index" / f"SHARD_{int(shard):04d}.parquet"
        heavy.append(ds.dataset(f).to_table(columns=HEAVY_COLS, filter=pc.field("uuid").isin(sdf["uuid"].tolist())))
    heavy_df = pa.concat_tables(heavy).to_pandas().drop_duplicates("uuid")
    man = sel.merge(heavy_df, on="uuid", how="left").drop(columns=["status"])
    man = man.sort_values(["split", "game_key", "video_id", "chunk_id"]).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(man, args.out)
    per_game_table(sel).assign(split=lambda x: np.where(x.index.isin(val_games), "val_unseen_games", "train")) \
        .to_csv(reports / "distribution_sampled.csv")
    json.dump({"alpha": alpha, "target_hours": args.target_hours, "val_games": sorted(val_games),
               "args": vars(args)}, open(reports / "sampling_info.json", "w"), indent=2)
    log.info("Wrote %s (%s chunks, %.1f h). Reports in %s", args.out, f"{len(man):,}",
             man.duration.sum() / 3600, reports)


if __name__ == "__main__":
    try:
        main()
    except BotBlockedError as e:
        raise SystemExit(f"\nSTOPPED: {e}")
