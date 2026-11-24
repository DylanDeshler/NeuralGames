#!/usr/bin/env python
"""Stage 06 — push packed WebDataset shards to a (private) Hugging Face dataset repo, then free local disk.

Designed for iterative builds: run after every `05_pack_webdataset.py --append` cycle (run_pipeline.py bulk
--push-to-hub does this). Each run:
  1. finds local shards not yet pushed (tracked in <root>/_pushed.txt),
  2. uploads them plus the current index.json / game_ids.json / name_mismatch_report.csv in ONE commit,
  3. verifies every uploaded shard's size on the Hub,
  4. deletes the verified local shards (--delete-local), so disk only ever holds the current cycle.
Nothing is deleted unless the commit succeeded and the remote sizes match. Re-running is safe.

The repo is created PRIVATE by default: the shards contain frames from other people's videos and games, and
NitroGen's annotations are CC BY-NC 4.0. Use --public only if you have the rights to publish.

Auth: `huggingface-cli login` or env HF_TOKEN.

Training-side streaming (same shapes per batch = one bucket's shards):
  import webdataset as wds
  url = "https://huggingface.co/datasets/<repo>/resolve/main/train/21x12/train-21x12-{000000..000123}.tar"
  ds = wds.WebDataset(f"pipe:curl -s -L -H 'Authorization:Bearer $HF_TOKEN' {url}").decode()
or `datasets.load_dataset("webdataset", data_files={"train": "hf://datasets/<repo>/train/21x12/*.tar"}, streaming=True)`.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ng_common import add_data_root_arg, DATA_ROOT, setup_logging

CARD = """---
license: other
license_name: research-only
pretty_name: NitroGen 2D world-model shards (private)
tags: [webdataset, world-model, video, gameplay]
---
# NitroGen 2D — action-conditioned world-model shards

Private research dataset built with this repo's pipeline from `nvidia/NitroGen` action annotations
(CC BY-NC 4.0) and the public gameplay videos they reference. Video content belongs to its creators and
the games' publishers. **Do not redistribute.**

Layout: `<split>/<bucket>/<split>-<bucket>-NNNNNN.tar` (one shard series per aspect bucket, e.g. `21x12` =
336×192 at patch 16), `<split>/index.json` (per-bucket samples/hours/shards), `game_ids.json`.
Each sample: `<uuid>.mp4` (15 fps, H.264 yuv444p), `.actions.npy` (T×21 float32), `.txt` (caption),
`.cls` (game id), `.json` (metadata incl. `inpaint_boxes`, `pad`, `content_wh` for loss masks).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_root_arg(ap)
    ap.add_argument("--repo-id", required=True, help="e.g. <your-username>/nitrogen-2d-wm")
    ap.add_argument("--root", default=str(DATA_ROOT / "webdataset"))
    ap.add_argument("--public", action="store_true", help="create the repo PUBLIC (default: private)")
    ap.add_argument("--delete-local", action="store_true", help="delete shards after a verified upload")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--max-repo-gb", type=float, default=950.0,
                    help="refuse to push past this total repo size (private storage quota; 0 = no limit)")
    ap.add_argument("--tag", default="", help="machine tag (multi-machine builds): index.json is uploaded as index_<tag>.json")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi

    log = setup_logging("06_push_to_hub")
    root = Path(args.root)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    info = api.repo_info(args.repo_id, repo_type="dataset")
    if getattr(info, "private", None) is False and not args.public:
        raise SystemExit(f"{args.repo_id} already exists and is PUBLIC. Refusing to upload; make it private or pass --public.")

    pushed_path = root / "_pushed.txt"
    pushed = set(pushed_path.read_text().split()) if pushed_path.exists() else set()
    shards = sorted(p for p in root.glob("*/*/*.tar"))
    new = [p for p in shards if p.relative_to(root).as_posix() not in pushed]
    meta_files = [p for p in [root / "game_ids.json", root / "name_mismatch_report.csv", *root.glob("*/index.json"),
                              *root.glob("*/_packed_uuids.txt")]
                  if p.exists()]
    existing = set(api.list_repo_files(args.repo_id, repo_type="dataset", revision=args.revision))
    def remote_path(p: Path) -> str:  # each build machine's index goes to <split>/index_<tag>.json
        rel = p.relative_to(root).as_posix()
        if rel.endswith("/_packed_uuids.txt"):  # what this machine packed: other machines' stage 03 skips these
            return f"meta/packed/{rel.split('/')[0]}_{args.tag or 'main'}.txt"
        return rel.replace("/index.json", f"/index_{args.tag}.json") if args.tag and rel.endswith("/index.json") else rel
    ops = [CommitOperationAdd(remote_path(p), str(p)) for p in new + meta_files]
    if "README.md" not in existing:
        ops.append(CommitOperationAdd("README.md", CARD.encode()))
    gb = sum(p.stat().st_size for p in new) / 1e9
    log.info("Repo %s (%s): %d new shards (%.1f GB) + %d metadata files", args.repo_id,
             "public" if args.public else "private", len(new), gb, len(meta_files))
    if not new and not meta_files:
        log.info("Nothing to push.")
        return
    if args.max_repo_gb > 0 and new:  # private storage quota: never start an upload that would overflow it
        used = sum(getattr(e, "size", 0) or 0 for e in api.list_repo_tree(args.repo_id, repo_type="dataset",
                                                                          recursive=True, revision=args.revision))
        log.info("Repo uses %.1f GB of the %.0f GB budget; this push adds %.1f GB", used / 1e9, args.max_repo_gb, gb)
        if used / 1e9 + gb > args.max_repo_gb:
            log.error("Pushing would exceed --max-repo-gb %.0f (%.1f + %.1f GB). Shards kept locally; stopping. Raise "
                      "the budget, or lower the encode size (e.g. CRF 20) for the rest of the build.",
                      args.max_repo_gb, used / 1e9, gb)
            raise SystemExit(4)
    api.create_commit(args.repo_id, repo_type="dataset", operations=ops, revision=args.revision,
                      commit_message=f"add {len(new)} shards ({gb:.1f} GB)")

    # verify sizes on the Hub before deleting anything
    paths = [p.relative_to(root).as_posix() for p in new]
    remote = {}
    for i in range(0, len(paths), 200):
        for e in api.get_paths_info(args.repo_id, paths[i:i + 200], repo_type="dataset", revision=args.revision):
            remote[e.path] = getattr(e, "size", None)
    ok = [p for p in new if remote.get(p.relative_to(root).as_posix()) == p.stat().st_size]
    bad = [p for p in new if p not in ok]
    with open(pushed_path, "a") as fh:
        for p in ok:
            fh.write(p.relative_to(root).as_posix() + "\n")
    if bad:
        log.error("%d shards failed size verification (kept locally, re-run to retry): %s", len(bad),
                  [p.name for p in bad[:5]])
    if args.delete_local:
        for p in ok:
            p.unlink()
        log.info("Deleted %d verified local shards (%.1f GB freed)", len(ok), sum(remote[p.relative_to(root).as_posix()] for p in ok) / 1e9)
    log.info("Pushed %d shards to https://huggingface.co/datasets/%s", len(ok), args.repo_id)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
