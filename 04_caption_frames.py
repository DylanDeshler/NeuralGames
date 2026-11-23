#!/usr/bin/env python
"""Stage 04 — caption Frame 0 of each processed clip with a local VLM (text-to-world init prompts).

Backends:
  vllm          (default) offline batched inference, Qwen/Qwen2.5-VL-7B-Instruct
  transformers  HF transformers generate() with left-padded batches
  openai        any OpenAI-compatible endpoint (e.g. `vllm serve Qwen/Qwen2.5-VL-7B-Instruct`)

The VLM returns JSON {caption, perspective, is_2d}; captions describe art style, perspective,
environment/biome and the main character while ignoring HUD/health bars/score/text overlays.
`is_2d`/`perspective` give a per-chunk sanity check on the game-level 2D label (stage 05 can drop
chunks the VLM flags as 3D with --drop-vlm-3d).

Results are written into each clip's meta.json (resumable: captioned clips are skipped) and merged
into the manifest (columns: caption, vlm_perspective, vlm_is_2d, frame0_dark).

Multi-GPU data parallel: run one process per GPU with --num-shards N --shard-index i
(CUDA_VISIBLE_DEVICES=i), then `python 04_caption_frames.py --merge-only`.

Examples:
  python 04_caption_frames.py --backend vllm --batch-size 64
  python 04_caption_frames.py --backend openai --api-base http://localhost:8000/v1 --concurrency 32
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
PROMPT = """You are annotating the first frame of a gameplay clip for a generative video world model.
Write a concise 1-2 sentence visual prompt that would let an artist recreate this scene. Cover:
1. art style (e.g. pixel art, hand-drawn, painterly, cel-shaded, vector, low-poly, realistic 3D),
2. camera perspective (side-scrolling, top-down, isometric, or 3D third/first person),
3. environment / biome (e.g. mossy cavern, neon city rooftops, snowy forest, lava castle),
4. the main player character's appearance and pose.
STRICTLY IGNORE and never mention: HUD elements, health/mana bars, score or coin counters, timers,
minimaps, menus, buttons, subtitles, any text overlays, logos, watermarks, webcams, controller
overlays, blurred/smudged patches, or black letterbox bars. Do not name the game or franchise.
Respond with JSON only, no markdown:
{"caption": "<1-2 sentences>", "perspective": "side-scrolling|top-down|isometric|3d|other", "is_2d": true|false}"""


# ============================================================================ image prep
def load_frame0(clip_dir: Path, meta: dict, image_size: int) -> Image.Image:
    p = clip_dir / "frame0.png"
    if p.exists():
        img = Image.open(p).convert("RGB")
    else:  # fall back to decoding the mp4
        import av
        with av.open(str(clip_dir / "frames.mp4")) as c:
            img = next(c.decode(video=0)).to_image().convert("RGB")
    # drop letterbox bars so the VLM sees only the game area
    px, py = meta.get("pad", [0, 0])
    cw, ch = meta.get("content_wh", list(img.size))
    img = img.crop((px, py, px + cw, py + ch))
    s = image_size / max(img.size)
    if s > 1:
        img = img.resize((max(28, round(img.width * s)), max(28, round(img.height * s))), Image.Resampling.BICUBIC)
    return img


def to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_output(text: str) -> dict:
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
            if cap:
                return {"caption": cap, "perspective": persp, "is_2d": bool(is2d) if is2d is not None else None,
                        "parse_ok": True}
        except json.JSONDecodeError:
            pass
    cap = re.sub(r"^```(json)?|```$", "", text).strip().strip('"')
    cap = re.sub(r'^\{?\s*"?caption"?\s*:\s*"?', "", cap).split('", "perspective')[0]
    return {"caption": " ".join(cap.split()), "perspective": "unknown", "is_2d": None, "parse_ok": False}


# ============================================================================ backends
class VLLMBackend:
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=args.model, max_model_len=args.max_model_len, limit_mm_per_prompt={"image": 1},
                       tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_mem,
                       mm_processor_kwargs={"min_pixels": 64 * 28 * 28, "max_pixels": args.max_pixels},
                       dtype="bfloat16", seed=0, enforce_eager=args.enforce_eager)
        self.sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    def __call__(self, images: list[Image.Image]) -> list[str]:
        convs = [[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": to_data_uri(im)}},
            {"type": "text", "text": PROMPT}]}] for im in images]
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

    def __call__(self, images):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text] * len(images), images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.proc.batch_decode(out, skip_special_tokens=True)


class OpenAIBackend:
    def __init__(self, args):
        from openai import OpenAI
        self.client = OpenAI(base_url=args.api_base, api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.model, self.max_new, self.conc = args.model, args.max_new_tokens, args.concurrency

    def _one(self, im):
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0.0, max_tokens=self.max_new,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": to_data_uri(im)}},
                        {"type": "text", "text": PROMPT}]}])
                return r.choices[0].message.content
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    def __call__(self, images):
        with ThreadPoolExecutor(self.conc) as ex:
            return list(ex.map(self._one, images))


# ============================================================================ manifest merge
def merge_into_manifest(manifest_path: str, processed_dir: str, log) -> None:
    man = pd.read_parquet(manifest_path)
    caps, persp, is2d, dark = [], [], [], []
    for v, u in zip(man.video_id, man.uuid):
        p = processed_dir_for(processed_dir, v, u) / "meta.json"
        m = json.loads(p.read_text()) if p.exists() else {}
        caps.append(m.get("caption"))
        persp.append(m.get("vlm_perspective"))
        is2d.append(m.get("vlm_is_2d"))
        dark.append(m.get("frame0_dark"))
    man["caption"], man["vlm_perspective"], man["vlm_is_2d"], man["frame0_dark"] = caps, persp, is2d, dark
    atomic_write_parquet(man, manifest_path)
    c = man["caption"].notna()
    log.info("Manifest updated: %s / %s captioned; VLM perspective: %s; VLM says non-2D: %d",
             f"{c.sum():,}", f"{len(man):,}", man.loc[c, "vlm_perspective"].value_counts().head(6).to_dict(),
             int((man["vlm_is_2d"] == False).sum()))  # noqa: E712


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DATA_ROOT / "manifest_2d.parquet"))
    ap.add_argument("--processed-dir", default=str(DATA_ROOT / "processed"))
    ap.add_argument("--backend", choices=["vllm", "transformers", "openai"], default="vllm")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=448, help="upscale the (bucket-res) frame's long side to this")
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--max-model-len", type=int, default=4096)
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

    man = pd.read_parquet(args.manifest)
    man = man.iloc[args.shard_index::args.num_shards]
    todo = []
    for v, u in zip(man.video_id, man.uuid):
        d = processed_dir_for(args.processed_dir, v, u)
        mp = d / "meta.json"
        if not mp.exists():
            continue
        if not args.overwrite and "caption" in json.loads(mp.read_text()):
            continue
        todo.append(d)
    if args.limit:
        todo = todo[: args.limit]
    log.info("%d clips to caption (shard %d/%d) with %s:%s", len(todo), args.shard_index, args.num_shards,
             args.backend, args.model)
    if todo:
        backend = {"vllm": VLLMBackend, "transformers": HFBackend, "openai": OpenAIBackend}[args.backend](args)
        n_bad = 0
        with open(Path(args.processed_dir) / f"_captions_{args.shard_index}.jsonl", "a") as fh:
            for i in tqdm(range(0, len(todo), args.batch_size), desc="caption batches"):
                dirs = todo[i:i + args.batch_size]
                metas = [json.loads((d / "meta.json").read_text()) for d in dirs]
                imgs = [load_frame0(d, m, args.image_size) for d, m in zip(dirs, metas)]
                try:
                    outs = backend(imgs)
                except Exception as e:
                    log.error("batch %d failed: %s", i, e)
                    continue
                for d, m, im, text in zip(dirs, metas, imgs, outs):
                    r = parse_output(text)
                    n_bad += not r["parse_ok"]
                    m.update(caption=r["caption"], vlm_perspective=r["perspective"], vlm_is_2d=r["is_2d"],
                             caption_model=args.model, caption_raw=text if not r["parse_ok"] else None,
                             frame0_dark=bool(np.asarray(im).mean() < 12))
                    atomic_write_json(d / "meta.json", m)
                    fh.write(json.dumps({"uuid": m["uuid"], **{k: r[k] for k in ("caption", "perspective", "is_2d")}}) + "\n")
                if i == 0:
                    log.info("Example caption: %s", parse_output(outs[0])["caption"])
        log.info("Captioning done (%d unparsable JSON responses, stored as raw text).", n_bad)
    if args.num_shards > 1:
        log.info("Multi-process run: merge with `python 04_caption_frames.py --merge-only` when all shards finish.")
    elif not args.no_merge:
        merge_into_manifest(args.manifest, args.processed_dir, log)


if __name__ == "__main__":
    main()
