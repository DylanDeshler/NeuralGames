#!/usr/bin/env python
"""Diagnose processed clips: what 02c decided for the video, what 03 applied, what 04 returned.

  python inspect_clips.py                      # all processed clips (fine for smoke tests)
  python inspect_clips.py --video q4RmgW4qIQI  # one video
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ng_common import DATA_ROOT


def short(x, n=70):
    s = json.dumps(x) if not isinstance(x, str) else x
    return s if len(s) <= n else s[: n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=str(DATA_ROOT / "processed"))
    ap.add_argument("--manifest", default=str(DATA_ROOT / "manifest_2d.parquet"))
    ap.add_argument("--layout", default=str(DATA_ROOT / "layout_overrides.parquet"))
    ap.add_argument("--video", default=None)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    metas = sorted(Path(args.processed_dir).glob("*/*/meta.json"))
    if args.video:
        metas = [m for m in metas if m.parent.parent.name == args.video]
    man = pd.read_parquet(args.manifest, columns=["uuid", "bbox_game_area", "bbox_others"]).set_index("uuid")
    lo = pd.read_parquet(args.layout).set_index("video_id") if Path(args.layout).exists() else None
    print(f"{len(metas)} processed clips | layout overrides file: {'yes (%d videos)' % len(lo) if lo is not None else 'MISSING'}\n")
    for mp in metas[: args.limit]:
        m = json.loads(mp.read_text())
        v, u = m["video_id"], m["uuid"]
        meta_ga = man.at[u, "bbox_game_area"] if u in man.index else "?"
        print(f"== {u}  [{m.get('game')}]  split={m.get('split')}")
        print(f"   metadata bbox_game_area: {short(meta_ga)}")
        if lo is not None and v in lo.index:
            r = lo.loc[v]
            print(f"   02c: status={r.get('status')} fetch={r.get('fetch_method')} game_area={short(r.get('game_area'))} "
                  f"overlays={short(r.get('overlays'), 60)} frac_gameplay={r.get('frac_gameplay')} "
                  f"stream_layout={r.get('stream_layout')} n_frames={r.get('n_frames')} spread={r.get('box_spread')}")
        else:
            print("   02c: NOT CHECKED (video not in layout_overrides)")
        print(f"   03:  layout_override={m.get('layout_override')} src={m.get('src_wh')} crop_src={m.get('crop_src')} "
              f"borders={m.get('border_crop_ltrb')} bucket={m.get('bucket')} inpaint_boxes={short(m.get('inpaint_boxes'), 60)} "
              f"frac_black={m.get('frac_black')}")
        print(f"   04:  screens={m.get('vlm_screens')} frac_gameplay={m.get('vlm_frac_gameplay')} "
              f"title={m.get('vlm_game_title')!r} qc_frames={m.get('vlm_qc_frames')} "
              f"parse_failed={m.get('caption_raw') is not None}")
        if m.get("caption_raw"):
            print(f"        raw: {short(m['caption_raw'], 200)}")
        print()


if __name__ == "__main__":
    main()
