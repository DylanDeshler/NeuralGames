#!/usr/bin/env python
"""Stage 04 — per-clip VLM pass: frame-0 caption + gameplay QC + game-title check.

One VLM request per clip sees --qc-frames frames spread over the clip (default 4: start, 1/3, 2/3, end)
and returns JSON:
  caption        1-2 sentence visual prompt for IMAGE 1 (art style, perspective, biome, main character;
                 no HUD / text / game name)  -> text-to-world initialization
  perspective    side-scrolling | top-down | isometric | 3d | other,  is_2d: bool
  screens        one label per image: gameplay | menu | title | loading | cutscene | map_or_inventory |
                 black | not_a_game  -> stage 05 drops clips that are mostly non-gameplay
  game_title     the game it recognizes (or null) -> stage 05 drops val_unseen_games clips whose frames look
                 like a different game that is in train (label-error leakage), and writes a mismatch report

Backends:
  vllm          (default) offline batched inference, Qwen/Qwen2.5-VL-7B-Instruct
  transformers  HF transformers generate() with left-padded batches
  openai        any OpenAI-compatible endpoint (e.g. `vllm serve Qwen/Qwen2.5-VL-7B-Instruct`)

Results go into each clip's meta.json (resumable: done clips are skipped; clips captioned by an older,
single-frame run are redone when --qc-frames > 1) and are merged into the manifest
(caption, vlm_perspective, vlm_is_2d, vlm_frac_gameplay, vlm_game_title, frame0_dark).

Multi-GPU data parallel: one process per GPU with --num-shards N --shard-index i (CUDA_VISIBLE_DEVICES=i),
then `python 04_caption_frames.py --merge-only`.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ng_common import DATA_ROOT, atomic_write_json, atomic_write_parquet, processed_dir_for, setup_logging

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
SCREENS = ["gameplay", "menu", "title", "loading", "cutscene", "map_or_inventory", "black", "not_a_game"]
PROMPT = """You are given {n} frames, in order, sampled from one 20-second gameplay clip (IMAGE 1 is the first frame).
They are used to train a generative video world model.

1. caption: a concise 1-2 sentence visual prompt describing ONLY IMAGE 1 (ignore images 2-{n} for the caption) that would
   let an artist recreate that exact scene. If IMAGE 1 is black, a menu or a title screen, describe exactly that. Cover the art
   style (pixel art, hand-drawn, painterly, cel-shaded, low-poly, realistic 3D...), camera perspective (side-scrolling,
   top-down, isometric, 3D), environment/biome, and the main player character's appearance and pose.
   NEVER mention HUD elements, health bars, score/coin counters, timers, minimaps, menus, text overlays, logos,
   watermarks, webcams, controller overlays, blurred/smudged patches or black bars. Do not name the game.
2. screens: for EACH of the {n} images, what is on screen: one of {screens}. "gameplay" means the player is
   controlling a character/vehicle in the game world (HUD visible is fine); pause/option/inventory screens are not.
3. game_title: the game's title if you recognize it with reasonable confidence, else null.

Respond with JSON only, no markdown:
{{"caption": "...", "perspective": "side-scrolling|top-down|isometric|3d|other", "is_2d": true|false,
 "screens": ["...", ...], "game_title": "..." or null}}"""


# ============================================================================ image prep
def _prep(img: Image.Image, meta: dict, image_size: int) -> Image.Image:
    px, py = meta.get("pad", [0, 0])  # drop pad bars (fit-mode clips) so the VLM sees only the game
    cw, ch = meta.get("content_wh", list(img.size))
    img = img.crop((px, py, px + cw, py + ch))
    s = image_size / max(img.size)
    if s > 1:
        img = img.resize((max(28, round(img.width * s)), max(28, round(img.height * s))), Image.Resampling.BICUBIC)
    return img


def load_frames(clip_dir: Path, meta: dict, image_size: int, n: int) -> list[Image.Image]:
    """n frames evenly spread over the clip (frame 0 from frame0.png when available)."""
    import av
    T = int(meta.get("n_frames", 0)) or 1
    want = sorted(set(np.linspace(0, T - 1, max(1, n)).round().astype(int).tolist()))
    got: dict[int, Image.Image] = {}
    p0 = clip_dir / "frame0.png"
    if p0.exists():
        got[0] = Image.open(p0).convert("RGB")
    if any(i not in got for i in want):
        with av.open(str(clip_dir / "frames.mp4")) as c:
            for i, f in enumerate(c.decode(video=0)):
                if i in want and i not in got:
                    got[i] = f.to_image().convert("RGB")
                if i >= want[-1]:
                    break
    return [_prep(got[i], meta, image_size) for i in want if i in got]


def to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def prompt_for(n: int) -> str:
    return PROMPT.format(n=n, screens="|".join(SCREENS))


def parse_output(text: str, n_images: int = 1) -> dict:
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            cap = " ".join(str(d.get("caption", "")).split())
            persp = str(d.get("perspective", "other")).lower().strip()
            is2d = d.get("is_2d")
            if isinstance(is2d, str):
                is2d = is2d.lower() in ("true", "yes", "1")
            screens = [str(x).lower().strip() for x in (d.get("screens") or [])][:n_images]
            screens = [x if x in SCREENS else "unknown" for x in screens]
            title = d.get("game_title")
            title = title.strip() if isinstance(title, str) and title.strip().lower() not in ("", "null", "none", "unknown") else None
            if cap:
                return {"caption": cap, "perspective": persp, "is_2d": bool(is2d) if is2d is not None else None,
                        "screens": screens, "game_title": title, "parse_ok": True}
        except json.JSONDecodeError:
            pass
    cap = re.sub(r"^```(json)?|```$", "", text).strip().strip('"')
    cap = re.sub(r'^\{?\s*"?caption"?\s*:\s*"?', "", cap).split('", "perspective')[0]
    return {"caption": " ".join(cap.split()), "perspective": "unknown", "is_2d": None, "screens": [],
            "game_title": None, "parse_ok": False}


# ============================================================================ backends (each call: list of image-lists)
class VLLMBackend:
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=args.model, max_model_len=args.max_model_len,
                       limit_mm_per_prompt={"image": max(1, args.qc_frames)},
                       tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_mem,
                       mm_processor_kwargs={"min_pixels": 64 * 28 * 28, "max_pixels": args.max_pixels},
                       dtype="bfloat16", seed=0, enforce_eager=args.enforce_eager)
        self.sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    def __call__(self, batch: list[list[Image.Image]]) -> list[str]:
        convs = [[{"role": "user", "content": [
            *({"type": "image_url", "image_url": {"url": to_data_uri(im)}} for im in ims),
            {"type": "text", "text": prompt_for(len(ims))}]}] for ims in batch]
        outs = self.llm.chat(convs, self.sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


class HFBackend:
    def __init__(self, args):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        except ImportError:
            from transformers import AutoModelForImageTextToText as M
        attn = "flash_attention_2"
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            attn = "sdpa"
        self.torch = torch
        self.model = M.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation=attn,
                                       device_map="auto").eval()
        self.proc = AutoProcessor.from_pretrained(args.model, min_pixels=64 * 28 * 28, max_pixels=args.max_pixels)
        self.proc.tokenizer.padding_side = "left"
        self.max_new = args.max_new_tokens

    def __call__(self, batch):
        texts, flat = [], []
        for ims in batch:
            msgs = [{"role": "user", "content": [*({"type": "image"} for _ in ims),
                                                 {"type": "text", "text": prompt_for(len(ims))}]}]
            texts.append(self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            flat.extend(ims)
        inputs = self.proc(text=texts, images=flat, padding=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class OpenAIBackend:
    def __init__(self, args):
        from openai import OpenAI
        self.client = OpenAI(base_url=args.api_base, api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.model, self.max_new, self.conc = args.model, args.max_new_tokens, args.concurrency

    def _one(self, ims):
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0.0, max_tokens=self.max_new,
                    messages=[{"role": "user", "content": [
                        *({"type": "image_url", "image_url": {"url": to_data_uri(im)}} for im in ims),
                        {"type": "text", "text": prompt_for(len(ims))}]}])
                return r.choices[0].message.content
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    def __call__(self, batch):
        with ThreadPoolExecutor(self.conc) as ex:
            return list(ex.map(self._one, batch))


# ============================================================================ manifest merge
def merge_into_manifest(manifest_path: str, processed_dir: str, log) -> None:
    man = pd.read_parquet(manifest_path)
    cols = {k: [] for k in ("caption", "vlm_perspective", "vlm_is_2d", "vlm_frac_gameplay", "vlm_game_title", "frame0_dark")}
    for v, u in zip(man.video_id, man.uuid):
        p = processed_dir_for(processed_dir, v, u) / "meta.json"
        m = json.loads(p.read_text()) if p.exists() else {}
        for k in cols:
            cols[k].append(m.get(k))
    for k, vals in cols.items():
        man[k] = vals
    atomic_write_parquet(man, manifest_path)
    c = man["caption"].notna()
    fg = pd.to_numeric(man.loc[c, "vlm_frac_gameplay"], errors="coerce")
    log.info("Manifest updated: %s / %s clips done; VLM non-2D: %d; mostly non-gameplay (<0.75): %d (%.1f%%)",
             f"{c.sum():,}", f"{len(man):,}", int((man["vlm_is_2d"] == False).sum()),  # noqa: E712
             int((fg < 0.75).sum()), 100 * float((fg < 0.75).mean()) if len(fg) else 0.0)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DATA_ROOT / "manifest_2d.parquet"))
    ap.add_argument("--processed-dir", default=str(DATA_ROOT / "processed"))
    ap.add_argument("--backend", choices=["vllm", "transformers", "openai"], default="vllm")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--qc-frames", type=int, default=4, help="frames per clip shown to the VLM (1 = caption only)")
    ap.add_argument("--batch-size", type=int, default=32, help="clips per batch")
    ap.add_argument("--image-size", type=int, default=448, help="upscale each frame's long side to this")
    ap.add_argument("--max-pixels", type=int, default=320 * 28 * 28, help="per image")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False,
                    help="skip torch.compile + CUDA graphs (use if vLLM engine startup fails)")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--no-merge", action="store_true", help="skip manifest merge (multi-process runs)")
    args = ap.parse_args()

    log = setup_logging(f"04_caption_frames_{args.shard_index}")
    if args.merge_only:
        merge_into_manifest(args.manifest, args.processed_dir, log)
        return

    man = pd.read_parquet(args.manifest, columns=["video_id", "uuid"])
    man = man.iloc[args.shard_index::args.num_shards]
    todo = []
    for v, u in zip(man.video_id, man.uuid):
        d = processed_dir_for(args.processed_dir, v, u)
        mp = d / "meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        done = "caption" in m and (args.qc_frames <= 1 or m.get("vlm_qc_frames", 1) >= args.qc_frames)
        if done and not args.overwrite:
            continue
        todo.append(d)
    if args.limit:
        todo = todo[: args.limit]
    log.info("%d clips to process (shard %d/%d) with %s:%s, %d frames/clip", len(todo), args.shard_index,
             args.num_shards, args.backend, args.model, args.qc_frames)
    if todo:
        backend = {"vllm": VLLMBackend, "transformers": HFBackend, "openai": OpenAIBackend}[args.backend](args)
        n_bad, n_nongp = 0, 0
        with open(Path(args.processed_dir) / f"_captions_{args.shard_index}.jsonl", "a") as fh:
            for i in tqdm(range(0, len(todo), args.batch_size), desc="clip batches"):
                dirs = todo[i:i + args.batch_size]
                metas = [json.loads((d / "meta.json").read_text()) for d in dirs]
                try:
                    frames = [load_frames(d, m, args.image_size, args.qc_frames) for d, m in zip(dirs, metas)]
                    outs = backend(frames)
                except Exception as e:
                    log.error("batch %d failed: %s", i, e)
                    continue
                for d, m, ims, text in zip(dirs, metas, frames, outs):
                    r = parse_output(text, len(ims))
                    n_bad += not r["parse_ok"]
                    scr = r["screens"]
                    frac_gp = (sum(s == "gameplay" for s in scr) / len(scr)) if scr else None
                    n_nongp += frac_gp is not None and frac_gp < 0.75
                    m.update(caption=r["caption"], vlm_perspective=r["perspective"], vlm_is_2d=r["is_2d"],
                             vlm_screens=scr, vlm_frac_gameplay=frac_gp, vlm_game_title=r["game_title"],
                             vlm_qc_frames=len(ims), caption_model=args.model,
                             caption_raw=text if not r["parse_ok"] else None,
                             frame0_dark=bool(np.asarray(ims[0]).mean() < 12) if ims else None)
                    atomic_write_json(d / "meta.json", m)
                    fh.write(json.dumps({"uuid": m["uuid"], "caption": r["caption"], "screens": scr,
                                         "game_title": r["game_title"]}) + "\n")
                if i == 0 and outs:
                    r0 = parse_output(outs[0], len(frames[0]))
                    log.info("Example: caption=%r screens=%s title=%r", r0["caption"], r0["screens"], r0["game_title"])
        log.info("Done (%d unparsable responses stored raw; %d clips mostly non-gameplay).", n_bad, n_nongp)
    if args.num_shards > 1:
        log.info("Multi-process run: merge with `python 04_caption_frames.py --merge-only` when all shards finish.")
    elif not args.no_merge:
        merge_into_manifest(args.manifest, args.processed_dir, log)


if __name__ == "__main__":
    main()
