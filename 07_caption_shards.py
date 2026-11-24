#!/usr/bin/env python
"""Stage 07 (GPU) — caption + QC packed shards, separately from building them.

Split pipeline: CPU machines build and push the shards (`run_pipeline.py build`: 03 -> 05 -> 06, no VLM), and
this script, on a GPU box, captions what is already on the Hub (or in a local webdataset dir), shard by shard.
It runs the same per-clip VLM request as stage 04 (--qc-frames frames, caption of frame 0, per-frame screen
labels, 2D/3D, game title) and writes one parquet per shard:

    captions/<split>/<bucket>/<shard name>.parquet
        uuid, video_id, game_key, split, bucket, shard, caption, vlm_perspective, vlm_is_2d, vlm_screens,
        vlm_frac_gameplay, vlm_game_title, frame0_dark, vlm_qc_frames, parse_ok, caption_model

The shards themselves are never rewritten. At training time, join on uuid (sample["__key__"]) and filter,
e.g. keep vlm_frac_gameplay >= 0.75, vlm_screens[0] == "gameplay", vlm_is_2d is not False (see README
"Captions sidecar").

Resumable: shards whose caption parquet already exists (Hub or local) are skipped. With --follow it keeps
polling for new shards while the build machines push, and exits after --idle-exit-min without new shards.

  # GPU box (model server shared with 02c):
  vllm serve Qwen/Qwen3.8-27B-FP8 --limit-mm-per-prompt '{"image": 6}' &
  python 07_caption_shards.py --data-root /data/games --repo-id <user>/nitrogen-2d-wm --follow
  # local (smoke): python 07_caption_shards.py --data-root /data/games --local-root /data/games/smoke/webdataset
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import queue
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ng_common import add_data_root_arg, DATA_ROOT, DEFAULT_VLM, atomic_write_parquet, setup_logging

cap04 = importlib.import_module("04_caption_frames")  # prompt, parser, image prep and VLM backends


# ============================================================================ shard reading
def read_samples(tar_path: Path) -> list[dict]:
    """[{key, mp4: bytes, json: dict}] for every sample in a WebDataset tar."""
    groups: dict[str, dict] = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, _, ext = Path(m.name).name.partition(".")
            if ext not in ("mp4", "json"):
                continue
            g = groups.setdefault(key, {"key": key})
            data = tf.extractfile(m).read()
            g[ext] = json.loads(data) if ext == "json" else data
    return [g for g in groups.values() if "mp4" in g and "json" in g]


def clip_frames(sample: dict, image_size: int, n: int) -> list[Image.Image]:
    """n frames evenly spread over the clip (frame 0 first), letterbox removed, upscaled like stage 04."""
    import av
    meta = sample["json"]
    T = int(meta.get("n_frames", 0)) or 1
    want = sorted(set(np.linspace(0, T - 1, max(1, n)).round().astype(int).tolist()))
    got: dict[int, Image.Image] = {}
    with av.open(io.BytesIO(sample["mp4"])) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = f.to_image().convert("RGB")
            if i >= want[-1]:
                break
    return [cap04._prep(got[i], meta, image_size) for i in want if i in got]


# ============================================================================ sources
class HubSource:
    def __init__(self, args, log):
        from huggingface_hub import HfApi
        self.api, self.repo, self.args, self.log = HfApi(token=os.environ.get("HF_TOKEN")), args.repo_id, args, log

    def listing(self) -> tuple[list[str], set[str]]:
        files = self.api.list_repo_files(self.repo, repo_type="dataset")
        shards = sorted(f for f in files if f.endswith(".tar") and f.split("/")[0] in self.args.splits)
        done = {f for f in files if f.startswith("captions/") and f.endswith(".parquet")}
        return shards, done

    def fetch(self, rel: str, dest_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(self.repo, rel, repo_type="dataset", local_dir=dest_dir))

    def publish(self, files: list[tuple[str, Path]]) -> None:
        from huggingface_hub import CommitOperationAdd
        if not files:
            return
        ops = [CommitOperationAdd(rel, str(p)) for rel, p in files]
        for attempt in range(5):
            try:
                self.api.create_commit(self.repo, repo_type="dataset", operations=ops,
                                       commit_message=f"captions for {len(files)} shards")
                return
            except Exception as e:  # transient Hub errors / concurrent commits from the build machines
                self.log.warning("caption commit failed (%s), retrying in %ds", e, 30 * (attempt + 1))
                time.sleep(30 * (attempt + 1))
        raise RuntimeError("could not commit captions to the Hub")


class LocalSource:
    def __init__(self, args, log):
        self.root, self.args = Path(args.local_root), args

    def listing(self) -> tuple[list[str], set[str]]:
        shards = sorted(p.relative_to(self.root).as_posix() for s in self.args.splits
                        for p in (self.root / s).glob("*/*.tar"))
        done = {p.relative_to(self.root).as_posix() for p in (self.root / "captions").glob("**/*.parquet")}
        return shards, done

    def fetch(self, rel: str, dest_dir: Path) -> Path:
        return self.root / rel

    def publish(self, files: list[tuple[str, Path]]) -> None:
        pass  # written in place under <root>/captions


def caption_rel(shard_rel: str) -> str:
    return "captions/" + shard_rel[: -len(".tar")] + ".parquet"


class DummyBackend:
    """--backend dummy: fixed answer, for testing the plumbing without a GPU."""
    def __init__(self, args):
        pass

    def __call__(self, batch):
        return [json.dumps({"caption": "a test scene", "perspective": "side-scrolling", "is_2d": True,
                            "screens": ["gameplay"] * len(ims), "game_title": None}) for ims in batch]


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_root_arg(ap)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo-id", help="Hub dataset with the shards; captions are committed back to it")
    src.add_argument("--local-root", help="local webdataset dir instead of the Hub")
    ap.add_argument("--splits", nargs="+", default=["train", "val_unseen_games"])
    ap.add_argument("--backend", choices=["openai", "vllm", "transformers", "dummy"], default="openai")
    ap.add_argument("--model", default=DEFAULT_VLM)
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--concurrency", type=int, default=64, help="parallel requests to the model server")
    ap.add_argument("--qc-frames", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256, help="clips per backend call")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--max-pixels", type=int, default=320 * 28 * 28)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--decode-workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--work-dir", default=str(DATA_ROOT / "caption_work"))
    ap.add_argument("--commit-every", type=int, default=10, help="shards per Hub commit")
    ap.add_argument("--follow", action="store_true", help="keep polling for new shards (parallel to the build)")
    ap.add_argument("--poll-min", type=float, default=5.0)
    ap.add_argument("--idle-exit-min", type=float, default=120.0, help="--follow: stop after this long with nothing new")
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    log = setup_logging("07_caption_shards")
    source = HubSource(args, log) if args.repo_id else LocalSource(args, log)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    backend = (DummyBackend if args.backend == "dummy" else
               {"vllm": cap04.VLLMBackend, "transformers": cap04.HFBackend, "openai": cap04.OpenAIBackend}[args.backend])(args)
    pool = ThreadPoolExecutor(args.decode_workers)
    pending: list[tuple[str, Path]] = []
    seen: set[str] = set()  # shards handled in this run (the Hub listing lags behind our own commits)
    idle_since = time.time()
    n_shards = n_clips = 0

    def flush():
        source.publish(pending)
        for _, p in pending:
            if args.repo_id:
                p.unlink(missing_ok=True)
        pending.clear()

    while True:
        shards, done = source.listing()
        todo = [s for s in shards if caption_rel(s) not in done and s not in seen]
        if args.limit_shards:
            todo = todo[: max(0, args.limit_shards - n_shards)]
        log.info("%d shards listed, %d captioned, %d to caption", len(shards), len(done), len(todo))
        if todo:
            idle_since = time.time()
        # prefetch: download the next shards while the GPU works on the current one
        q: queue.Queue = queue.Queue(maxsize=2)

        def producer():
            for rel in todo:
                try:
                    q.put((rel, source.fetch(rel, work / "shards"), None))
                except Exception as e:
                    q.put((rel, None, e))
            q.put(None)
        threading.Thread(target=producer, daemon=True).start()

        while (item := q.get()) is not None:
            rel, path, err = item
            seen.add(rel)
            if err is not None:
                log.error("could not fetch %s: %s", rel, err)
                continue
            t0 = time.time()
            samples = read_samples(path)
            frames = list(pool.map(lambda s: clip_frames(s, args.image_size, args.qc_frames), samples))
            rows = []
            for i in range(0, len(samples), args.batch_size):
                sb, fb = samples[i:i + args.batch_size], frames[i:i + args.batch_size]
                outs = backend(fb)
                for s, ims, text in zip(sb, fb, outs):
                    r = cap04.parse_output(text, len(ims))
                    scr = r["screens"]
                    frac_gp = (sum(x == "gameplay" for x in scr) / len(scr)) if len(scr) >= max(1, (len(ims) + 1) // 2) else None
                    m = s["json"]
                    rows.append({"uuid": s["key"], "video_id": m.get("video_id"), "game_key": m.get("game_key"),
                                 "split": m.get("split"), "bucket": m.get("bucket"), "shard": rel,
                                 "caption": r["caption"], "vlm_perspective": r["perspective"], "vlm_is_2d": r["is_2d"],
                                 "vlm_screens": scr, "vlm_frac_gameplay": frac_gp, "vlm_game_title": r["game_title"],
                                 "frame0_dark": bool(np.asarray(ims[0]).mean() < 12) if ims else None,
                                 "vlm_qc_frames": len(ims), "parse_ok": r["parse_ok"], "caption_model": args.model})
            out_rel = caption_rel(rel)
            out_path = (work / out_rel) if args.repo_id else (Path(args.local_root) / out_rel)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_parquet(pd.DataFrame(rows), out_path)
            pending.append((out_rel, out_path))
            if args.repo_id:
                path.unlink(missing_ok=True)  # the downloaded shard is no longer needed
            n_shards += 1
            n_clips += len(rows)
            df = pd.DataFrame(rows)
            log.info("%s: %d clips in %.0fs (%.1f clips/s) | gameplay>=0.75 %.0f%% | 2D %.0f%% | unparsed %d",
                     rel, len(rows), time.time() - t0, len(rows) / max(1e-6, time.time() - t0),
                     100 * (df.vlm_frac_gameplay.fillna(0) >= 0.75).mean() if len(df) else 0,
                     100 * (df.vlm_is_2d != False).mean() if len(df) else 0,  # noqa: E712
                     int((~df.parse_ok).sum()) if len(df) else 0)
            if len(pending) >= args.commit_every:
                flush()
        flush()
        if args.limit_shards and n_shards >= args.limit_shards:
            break
        if not args.follow:
            break
        if n_shards and time.time() - idle_since > args.idle_exit_min * 60:  # never quit before the first shard
            log.info("No new shards for %.0f min: done.", args.idle_exit_min)
            break
        time.sleep(args.poll_min * 60)
    log.info("Captioned %d shards, %d clips.", n_shards, n_clips)


if __name__ == "__main__":
    main()
