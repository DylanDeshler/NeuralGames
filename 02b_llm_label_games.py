#!/usr/bin/env python
"""Stage 02b (optional, GPU) — identify the game behind each video with an LLM, from its YouTube metadata.

Why: in the public NitroGen release `game` is "other" for ~2/3 of videos, and some named slugs are
level/chapter names (e.g. `profaned_capital`, `firelink_shrine`). A curated title list only catches
games it already knows. This stage asks an instruction-tuned LLM to read each video's YouTube title,
channel, tags and description and return:

    {"game": "<official title or null>", "confidence": 0-1, "perspective": "...", "genre": "...",
     "multiple_games": bool}

Results go to data/llm_game_labels.parquet (one row per video). Stage 02 picks them up automatically
(--llm-labels). There, the LLM's game name is first checked against the curated list and manual
labels; the LLM's own perspective is used only for games those don't cover.

Which videos are labeled (--scope):
  unknown  (default) videos whose NitroGen slug is "other" or not in genres_2d.yaml / game_labels.csv
  all      every video (also re-checks known slugs)

Resumable: labeled video_ids are cached in data/llm_game_labels.jsonl and skipped on re-runs.
YouTube metadata is shared with stage 02 via data/yt_meta_cache.jsonl.

Examples:
  python 02b_llm_label_games.py                                     # vLLM, Qwen2.5-14B-Instruct
  python 02b_llm_label_games.py --model Qwen/Qwen2.5-32B-Instruct --tensor-parallel-size 2
  python 02b_llm_label_games.py --backend openai --api-base http://localhost:8000/v1 --model <served-name>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from tqdm import tqdm

from ng_common import DATA_ROOT, atomic_write_parquet, norm_title, setup_logging
from ng_labels import BotBlockedError, GameLabeler, YouTubeMeta, video_urls

PERSPECTIVES = ["side-view-2d", "top-down-2d", "isometric", "3d", "mixed", "unknown"]
GENRES = ["Platformer", "Metroidvania", "Run and Gun", "Beat 'em up", "Arcade", "Shmup", "Fighting",
          "2D Action", "Roguelike", "Action-RPG", "Action-Adventure", "RPG", "Shooter", "Sports", "Racing",
          "Battle Royale", "Puzzle", "Simulation", "Strategy", "Other"]

SYSTEM = "You are an expert on video games. You identify games from YouTube video metadata and answer in strict JSON."
USER_TMPL = """Identify the video game played in this YouTube gameplay video.

NitroGen label (may be "other", a level/chapter/area name, or a game slug): {slug}
Title: {title}
Channel: {channel}
Tags: {tags}
Description (truncated): {desc}

Answer with JSON only:
{{"game": "<official game title in English, or null if you cannot tell>",
  "confidence": <0.0-1.0 that "game" is correct>,
  "perspective": "<one of: {persp}>",
  "genre": "<one of: {genres}>",
  "multiple_games": <true if the video clearly covers several different games>}}

Rules:
- Use the base game's official title (e.g. "Hollow Knight", not "Hollow Knight Steel Soul any%"). For mods,
  randomizers, or fan levels, give the base game (e.g. a Celeste mod -> "Celeste").
- A level/area/boss name in the NitroGen label is a hint about which game it is (e.g. "firelink_shrine" -> Dark Souls).
- perspective is the game's usual gameplay camera: "side-view-2d" includes 2.5D games played on a 2D plane
  (e.g. Ori, Trine, Little Nightmares); "top-down-2d" is overhead 2D (e.g. The Binding of Isaac);
  "isometric" is angled overhead (e.g. Hades, Diablo); "3d" is free 3D camera.
- If unsure, use null / "unknown" and a low confidence. Do not guess from the channel name alone."""


def build_prompt(slug: str, m: dict) -> list[dict]:
    tags = ", ".join((m.get("tags") or [])[:25])
    desc = " ".join((m.get("description") or "").split())[:600]
    user = USER_TMPL.format(slug=slug, title=m.get("title") or "", channel=m.get("channel") or "",
                            tags=tags or "(none)", desc=desc or "(none)", persp=", ".join(PERSPECTIVES),
                            genres=", ".join(GENRES))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    out = {"llm_game": None, "llm_confidence": 0.0, "llm_perspective": "unknown", "llm_genre": None,
           "llm_multiple_games": False, "llm_parse_ok": False}
    if not m:
        return out
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    g = d.get("game")
    g = g.strip() if isinstance(g, str) and g.strip() and g.strip().lower() not in ("null", "none", "unknown") else None
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    persp = str(d.get("perspective", "unknown")).strip().lower()
    multi = d.get("multiple_games", False)
    out.update(llm_game=g, llm_confidence=conf,
               llm_perspective=persp if persp in PERSPECTIVES else "unknown",
               llm_genre=d.get("genre") if d.get("genre") in GENRES else None,
               llm_multiple_games=(multi if isinstance(multi, bool) else str(multi).lower() == "true"),
               llm_parse_ok=True)
    return out


GAME_USER_TMPL = """Video game: "{name}"
Example YouTube titles of videos labeled with this game (for disambiguation only): {titles}

What is this game's camera perspective during normal moment-to-moment gameplay?
Answer with JSON only:
{{"perspective": "<one of: {persp}>", "genre": "<one of: {genres}>", "confidence": <0.0-1.0>}}

Definitions: "side-view-2d" = side-scrolling / 2D-plane gameplay, including 2.5D games rendered in 3D but played on a
2D plane (Ori, Trine, Little Nightmares, modern Prince of Persia: The Lost Crown); "top-down-2d" = overhead 2D view
(The Binding of Isaac, A Link to the Past, classic 2D Pokemon); "isometric" = angled overhead (Hades, Diablo);
"3d" = free 3D camera / 3D movement (Elden Ring, Pokemon Violet, Fortnite); "mixed" only if the game genuinely switches."""


def game_level_pass(df: pd.DataFrame, args, log, out_path: Path) -> pd.DataFrame:
    """Ask the LLM once per identified game (by name) for its perspective/genre; cached by game key."""
    cache = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["llm_game_key"])
    ok = df[df["llm_parse_ok"] & df["llm_game"].notna() & (df["llm_confidence"] >= 0.5)]
    games = (ok.groupby("llm_game_key")
             .agg(name=("llm_game", lambda s: s.mode().iat[0]), videos=("video_id", "size"),
                  titles=("yt_title", lambda s: [t for t in s.dropna().head(3)]))
             .reset_index())
    todo = games[~games["llm_game_key"].isin(cache["llm_game_key"])]
    log.info("Game-level perspective pass: %d games (%d cached, %d to ask)", len(games), len(games) - len(todo), len(todo))
    if len(todo):
        llm = get_llm(args)
        rows = []
        for i in tqdm(range(0, len(todo), args.batch_size), desc="game-level"):
            b = todo.iloc[i:i + args.batch_size]
            convs = [[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": GAME_USER_TMPL.format(
                          name=r.name, titles=" | ".join(f'"{t[:80]}"' for t in r.titles) or "(none)",
                          persp=", ".join(PERSPECTIVES), genres=", ".join(GENRES))}] for r in b.itertuples()]
            for r, text in zip(b.itertuples(), llm(convs)):
                p = parse(text)  # reuses the video parser: perspective / genre / confidence fields
                rows.append({"llm_game_key": r.llm_game_key, "game_name": r.name, "game_perspective": p["llm_perspective"],
                             "game_genre": p["llm_genre"], "game_confidence": p["llm_confidence"],
                             "game_parse_ok": p["llm_parse_ok"], "llm_model": args.model})
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("llm_game_key", keep="last")
        atomic_write_parquet(cache, out_path)
    return cache


_LLM = None


def get_llm(args):
    global _LLM
    if _LLM is None:
        _LLM = (VLLMText if args.backend == "vllm" else OpenAIText)(args)
    return _LLM


# ============================================================================ backends
class VLLMText:
    def __init__(self, args):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16",
                       max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem, seed=0,
                       enforce_eager=args.enforce_eager)
        self.sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        # Qwen3-style models: disable thinking for short structured answers
        self.kw = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in args.model.lower() else {}

    def __call__(self, convs):
        outs = self.llm.chat(convs, self.sp, use_tqdm=False, **self.kw)
        return [o.outputs[0].text for o in outs]


class OpenAIText:
    def __init__(self, args):
        from openai import OpenAI
        self.client = OpenAI(base_url=args.api_base, api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.model, self.max_new, self.conc = args.model, args.max_new_tokens, args.concurrency

    def _one(self, conv):
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=conv, temperature=0.0,
                                                        max_tokens=self.max_new)
                return r.choices[0].message.content
            except Exception:
                if attempt == 4:
                    return ""
                time.sleep(2 ** attempt)

    def __call__(self, convs):
        with ThreadPoolExecutor(self.conc) as ex:
            return list(ex.map(self._one, convs))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta-dir", default=str(DATA_ROOT / "nitrogen_meta"))
    ap.add_argument("--genres", default="./config/genres_2d.yaml")
    ap.add_argument("--labels-csv", default="./config/game_labels.csv")
    ap.add_argument("--out", default=str(DATA_ROOT / "llm_game_labels.parquet"))
    ap.add_argument("--scope", choices=["unknown", "all"], default="unknown")
    ap.add_argument("--min-video-hours", type=float, default=0.0, help="skip videos shorter than this")
    ap.add_argument("--limit", type=int, default=0, help="label at most N videos (largest first)")
    ap.add_argument("--backend", choices=["vllm", "openai"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct",
                    help="any instruct LLM; bigger models know more obscure games")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--skip-game-pass", action="store_true", help="skip the per-game perspective pass")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--max-model-len", type=int, default=2048, help="prompts are ~700 tokens")
    ap.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True,
                    help="skip torch.compile + CUDA graphs (robust startup; small speed cost for this short job). "
                         "--no-enforce-eager for max throughput once your vLLM install is known-good")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    # YouTube metadata (shared cache with stage 02)
    ap.add_argument("--yt-cache", default=str(DATA_ROOT / "yt_meta_cache.jsonl"))
    ap.add_argument("--yt-workers", type=int, default=4)
    ap.add_argument("--yt-min-interval", type=float, default=0.5)
    ap.add_argument("--cookies", default=None)
    ap.add_argument("--cookies-from-browser", default=None)
    ap.add_argument("--yt-backend", choices=["auto", "api", "ytdlp"], default="auto",
                    help="metadata source for YouTube videos: 'api' = YouTube Data API v3 (no bot checks; "
                         "needs --yt-api-key or env YOUTUBE_API_KEY), 'ytdlp' = scraping, 'auto' = api if a key is set")
    ap.add_argument("--yt-api-key", default=os.environ.get("YOUTUBE_API_KEY"))
    ap.add_argument("--proxy", default=None, help="proxy for yt-dlp metadata requests, e.g. socks5://host:port")
    args = ap.parse_args()

    log = setup_logging("02b_llm_label_games")
    out_path = Path(args.out)
    cache_path = out_path.with_suffix(".jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- per-video table
    idx = ds.dataset(sorted((Path(args.meta_dir) / "index").glob("*.parquet"))).to_table(
        columns=["video_id", "game", "duration"]).to_pandas()
    vids = idx.groupby("video_id").agg(game=("game", "first"), hours=("duration", "sum"))
    vids["hours"] /= 3600
    lab = GameLabeler(Path(args.genres), Path(args.labels_csv) if args.labels_csv else None)
    if args.scope == "unknown":
        vids = vids[vids["game"].map(lambda g: lab.label_slug(g)[0] == "unknown")]
    vids = vids[vids["hours"] >= args.min_video_hours].sort_values("hours", ascending=False)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done[r["video_id"]] = r
            except Exception:
                pass
    todo = vids[~vids.index.isin(list(done))]
    if args.limit:
        todo = todo.head(args.limit)
    log.info("Videos in scope: %s (%.0f h) | already labeled: %s | to label now: %s",
             f"{len(vids):,}", vids.hours.sum(), f"{len(done):,}", f"{len(todo):,}")

    if len(todo):
        # ---- YouTube metadata (cached; private/removed videos are skipped)
        backend = args.yt_backend if args.yt_backend != "auto" else ("api" if args.yt_api_key else "ytdlp")
        yt = YouTubeMeta(Path(args.yt_cache), args.yt_workers, args.yt_min_interval, args.cookies,
                         args.cookies_from_browser, log, backend=backend, api_key=args.yt_api_key, proxy=args.proxy)
        urls = video_urls(Path(args.meta_dir))
        meta = yt.get_many(todo.index.tolist(), urls)
        ok = [v for v in todo.index if meta.get(v, {}).get("status") == "ok"]
        log.info("YouTube metadata: %d ok, %d unavailable, %d errors (errors are retried next run)", len(ok),
                 sum(meta.get(v, {}).get("status") == "unavailable" for v in todo.index),
                 sum(meta.get(v, {}).get("status") not in ("ok", "unavailable") for v in todo.index))

        # ---- LLM
        backend = get_llm(args)
        with open(cache_path, "a") as fh:
            for i in tqdm(range(0, len(ok), args.batch_size), desc="llm batches"):
                batch = ok[i:i + args.batch_size]
                outs = backend([build_prompt(todo.at[v, "game"], meta[v]) for v in batch])
                for v, text in zip(batch, outs):
                    r = {"video_id": v, "nitrogen_game": todo.at[v, "game"], "yt_title": meta[v].get("title"),
                         **parse(text), "llm_model": args.model}
                    r["llm_game_key"] = norm_title(r["llm_game"]) if r["llm_game"] else None
                    if not r["llm_parse_ok"]:
                        r["llm_raw"] = (text or "")[:500]
                    done[v] = r
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                if i == 0 and batch:
                    for v in batch[:5]:
                        log.info("  %-60s -> %s (%.2f, %s)", (meta[v].get("title") or "")[:60], done[v]["llm_game"],
                                 done[v]["llm_confidence"], done[v]["llm_perspective"])

    # ---- write parquet + summary
    df = pd.DataFrame(list(done.values()))
    if df.empty:
        log.warning("No labels produced.")
        return
    atomic_write_parquet(df, out_path)
    if not args.skip_game_pass:
        gp = game_level_pass(df, args, log, out_path.with_name("llm_game_perspectives.parquet"))
        df = df.merge(gp[["llm_game_key", "game_perspective", "game_confidence"]], on="llm_game_key", how="left")
        df["final_perspective"] = df["game_perspective"].where(
            df["game_perspective"].notna() & (df["game_confidence"].fillna(0) >= 0.6) &
            ~df["game_perspective"].isin(["unknown"]), df["llm_perspective"])
    else:
        df["final_perspective"] = df["llm_perspective"]
    df = df.merge(idx.groupby("video_id")["duration"].sum().div(3600).rename("hours"), left_on="video_id",
                  right_index=True, how="left")
    print("\n" + "=" * 78)
    print(f" LLM game labels: {len(df):,} videos, {df.hours.sum():,.0f} h -> {out_path}")
    print(f" identified (conf>=0.6, single game): "
          f"{((df.llm_confidence >= 0.6) & ~df.llm_multiple_games & df.llm_game.notna()).mean():.1%} of videos")
    print(" hours by perspective (video-level): " + str(df.groupby("llm_perspective")["hours"].sum().round(0).to_dict()))
    print(" hours by perspective (game-level where confident): "
          + str(df.groupby("final_perspective")["hours"].sum().round(0).to_dict()))
    print(f" unparsable outputs: {(~df.llm_parse_ok).sum()}")
    top = (df[df.llm_game.notna()].groupby("llm_game_key")
           .agg(game=("llm_game", "first"), videos=("video_id", "size"), hours=("hours", "sum"),
                persp=("final_perspective", lambda s: s.mode().iat[0]))
           .sort_values("hours", ascending=False).head(25))
    print(f" {'game':<45} {'videos':>6} {'hours':>8}  perspective")
    for _, r in top.iterrows():
        print(f" {str(r.game)[:45]:<45} {r.videos:>6} {r.hours:>8.1f}  {r.persp}")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except BotBlockedError as e:
        raise SystemExit(f"\nSTOPPED: {e}")
