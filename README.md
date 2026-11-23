# NitroGen → 2D action-conditioned world-model dataset

A pipeline that turns the `nvidia/NitroGen` action annotations, plus the public YouTube videos they
refer to, into WebDataset shards of `(256×256 video, (T,21) actions, caption, game_id)`. Every stage
can be resumed: re-running a stage skips work that's already done.

```
01_fetch_metadata.py      HF tarballs -> compact per-shard index + action parquet       (CPU, ~25-60 GB)
02b_llm_label_games.py    (optional) LLM names the game per video from YouTube metadata    (GPU, ~minutes)
02_profile_and_sample.py  2D filter, idle filter, alpha-sampling, game holdout -> manifest (CPU, minutes)
03_download_and_process.py  20 s clip fetch, crop/letterbox 256, overlay inpaint, align   (network-bound)
04_caption_frames.py      Qwen2.5-VL caption of frame 0 (vLLM / transformers / OpenAI API) (GPU)
05_pack_webdataset.py     ~1 GB WebDataset shards: train/ and val_unseen_games/
verify_dataset.py         alignment/range assertions + visual debug grid & overlay videos
```

## Dataset facts (checked against the real shards, Nov 2026)

* The HF repo stores **100 × `actions/SHARD_XXXX.tar.gz`** (about 1.6 GB each, 164 GB total), not
  loose files. `snapshot_download(allow_patterns=["**/metadata.json", ...])` therefore matches nothing.
  Stage 01 downloads each tarball, streams through it once and keeps a compact form: about 12.5M
  chunks as loose files would take more than 450 GB.
* Parquet: 17 `int32` button columns, and `j_left`/`j_right` as `list<double>[2]` in [-1, 1].
  (-1,-1) is the top-left. Row count == `chunk_size`. Stage 01 detects the layout on the first chunk
  and also handles `j_left_x/j_left_y` or struct columns.
* fps is `chunk_size / duration`. In a sample of 190 videos spread over all 100 shards, it was 60 fps
  for ~90 %, 30 fps for ~9 %, and 25/50 fps for ~1 %.
* **No genre in the release.** The paper's Fig. 3b genre split (Action-RPG 34.9 %, Platformer
  18.4 %, Action-Adventure 9.2 %, Sports 5.8 %, Metroidvania 5.4 %, Roguelike 4.9 %, RPG 4.7 %,
  Battle Royale 4.0 %, Racing 3.3 %, Other 9.4 %) comes from internal labels. None of the ~17.7k
  sampled `metadata.json` files has a genre key, and the code and model repos ship no mapping.
* `game` is a **per-video** slug (constant across a video's chunks). It is `"other"` for **~66 % of
  videos and ~67 % of hours** in that sample, so the paper's 9.4 % "Other" is a *genre* bucket, not
  this field. Some named slugs are YouTube chapter or level names rather than titles
  (`profaned_capital`, `firelink_shrine`, `ten_desires`, `_untitled_chapter_1_`). Label those in
  `config/game_labels.csv`, or let the YouTube title match override them.
* Rough 2D budget: Platformer + Metroidvania + the 2D share of Roguelike/Other ≈ 25–30 % of 40k h,
  i.e. ~10k h before the idle filter. Measure the real number with `02_profile_and_sample.py --plan`.
* Besides `bbox_controller_overlay` (pixels, relative to `original_video.resolution` [H, W]), some
  chunks have `bbox_game_area` (the gameplay is a sub-rectangle, which stage 03 crops to) and
  `bbox_others` (webcams/banners, inpainted by default).

## Setup (on the GPU box)

```bash
sudo apt-get install -y ffmpeg               # or a static ffmpeg build
curl -fsSL https://deno.land/install.sh | sh # JS runtime yt-dlp needs for YouTube
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt               # drop vllm/transformers lines if captioning elsewhere
huggingface-cli login                         # optional (dataset is public; avoids rate limits)
export NG_DATA_ROOT=/data/games               # where ALL outputs go (default: ./data). Add to ~/.bashrc
```

Everything lands under `$NG_DATA_ROOT`. The Hugging Face cache (`HF_HOME`) also defaults to
`$NG_DATA_ROOT/hf_cache` unless you've already set it, so the Qwen weights stay off your home disk.

```
$NG_DATA_ROOT/
  nitrogen_meta/{index,actions,_done,_tars}/   stage 01 (~60 GB; _tars is scratch, ~2 GB per worker)
  yt_meta_cache.jsonl  llm_game_labels.parquet  stage 02b / 02
  manifest_2d.parquet  reports/                 stage 02
  processed/<video_id>/<uuid>/  tmp_03/         stage 03 (~0.3 GB per hour of data @ 30 fps)
  webdataset/{train,val_unseen_games}/          stage 05 (same size again; delete processed/ after verifying)
  hf_cache/                                     Qwen2.5-14B (~30 GB) + Qwen2.5-VL-7B (~16 GB)
  logs/  debug/
```
Every path can still be overridden per script (`--meta-dir`, `--processed-dir`, `--out`, ...).

## Step by step

### 1. Metadata sync
```bash
python 01_fetch_metadata.py --shards 0-1          # try 2 shards first (~3.5 GB download, a few min)
python 01_fetch_metadata.py --shards all --workers 4
python 01_fetch_metadata.py --summary-only        # re-print the summary
```
Writes `data/nitrogen_meta/{index,actions}/SHARD_XXXX.parquet`. Tarballs are deleted after
processing unless you pass `--keep-tar`. Check the log for the detected parquet schema and an example
`metadata.json`. At peak, each worker needs about 2 GB of scratch space for its tarball. Use
`--extract-loose` if you also want the original files on disk (only sensible for a shard or two).

### 2a. (Recommended) LLM game identification
```bash
python 02b_llm_label_games.py                       # vLLM + Qwen/Qwen2.5-14B-Instruct, one GPU
python 02b_llm_label_games.py --model Qwen/Qwen2.5-32B-Instruct --tensor-parallel-size 2   # knows more obscure games
```
For every video whose slug is `"other"` or unrecognised (including chapter names like
`profaned_capital`), this fetches the YouTube title, channel, tags and description, reusing the cache
shared with stage 02. An LLM then returns `{game, confidence, perspective, genre, multiple_games}`.
Output: `data/llm_game_labels.parquet`. Re-runs skip videos already labelled. The YouTube lookups are
the slow part: about 25k videos at 2 requests/s is ~3–4 h. Run it with `--limit 500` first and read
the summary table it prints.

Stage 02 uses the file automatically. The LLM's title is resolved against the curated list, aliases
and manual labels first, and the LLM's `perspective` decides only for games those don't cover
(`--llm-accept-perspectives side-view-2d top-down-2d`). Labels below `--llm-min-confidence 0.6`
are ignored, and so are multi-game videos unless you pass `--llm-allow-multi`. The LLM's title
becomes the game key, so `profaned_capital` and `firelink_shrine` both become "Dark Souls III" and
can't straddle the train/val split.

### 2. Profile, filter, sample
```bash
# (a) measure: full 2D pool, idle-filter sensitivity, allocation options. Writes reports, no manifest.
python 02_profile_and_sample.py --enrich-youtube --check-availability --plan
# (b) decide: pick a total and a strategy from the plan table, then write the manifest
python 02_profile_and_sample.py --target-hours 5000 --allocation power      # or --allocation cap, or --alpha 0.8
```
The plan prints, for each target in `--plan-targets` (default 250 h to 20k h, plus "everything"):

| column | meaning |
|---|---|
| `param` | α for `power`; per-game hour cap for `cap` |
| `eff_games` | exp(entropy of per-game hour shares): how many equal-sized games the mix is worth (**breadth**) |
| `top1/top5/top15` | share of hours taken by the biggest titles (**dominance**) |
| `≥10h`, `≥100h`, `med_game_h` | how many games get enough data to learn their physics (**depth**) |
| `disk_GB`, `dl_TB` | rough processed-disk and source-download volume |

Two strategies:
* **`power`**: `quota_g = floor(n_g^α)`. Big games keep more than small ones, but sublinearly.
  Gives depth on the major titles while still covering the tail.
* **`cap`**: `hours_g = min(available_g, cap)`. The most balanced mix possible for a given total
  (maximum `eff_games`), but at small totals no game gets much depth.

At "everything" the two are the same. The script also writes `reports/allocation_per_game.csv`
(hours per game for every target and strategy) and `reports/idle_filter_sensitivity.csv`.
`--target-hours` is required outside `--plan`, so nothing is sampled until you've chosen.
* **2D labeling** runs in this precedence order: `config/game_labels.csv` → `exclude_3d` →
  `games_2d` (`config/genres_2d.yaml`) → YouTube title/tags/description match for `"other"` and
  unknown slugs (`--enrich-youtube`, cached in `data/yt_meta_cache.jsonl`) → LLM labels from
  stage 02b → optional `--keyword-fallback`.
  **Iterate:** open `data/reports/game_label_report.csv`, label the unknown games with the most hours
  in `config/game_labels.csv`, and re-run. YouTube lookups are cached, so re-runs are fast.
* **Idle filter:** keeps a chunk only if `||j_left|| > 0.2` in ≥15 % of frames **and** a gameplay
  button is pressed in ≥8 % of frames. start/back/guide don't count. Flags: `--move-thresh`,
  `--min-move-frac`, `--min-btn-frac`.
* **Sampling:** prints the empirical distribution (games, hours, P50/P90/P99, top 15). Then brentq
  solves for α in `quota_g = floor(n_g^α)` so the total reaches `--target-hours`. Within a game,
  chunks are spread across videos in proportion to size, and evenly along each video's timeline
  (biome coverage, no near-duplicate neighbouring chunks).
* **Holdout:** ~10 % of games (`--val-frac`), drawn from games at or below P90 hours, go entirely to
  `val_unseen_games`. Keyword-only `unk:<video>` games always stay in train.

Outputs: `data/manifest_2d.parquet` plus `data/reports/*.csv`, `sampling_info.json`.

### 3. Download + process clips
```bash
python 03_download_and_process.py --limit 20 --workers 4      # smoke test, then inspect data/processed
python 03_download_and_process.py --workers 16 --max-starts-per-min 40
python 03_download_and_process.py --status
```
* `--method auto` (default): yt-dlp resolves the stream URL, then ffmpeg seeks with HTTP range
  requests and decodes only [start, end). If that fails, it falls back to
  `yt-dlp --download-sections "*start-end" --force-keyframes-at-cuts`.
* `crop(bbox_game_area) → fps=annotation fps → area-downscale → letterbox pad to 256×256 (black) →
  cv2.inpaint(TELEA)` over the controller overlay and `bbox_others`. Output has exactly `T` frames,
  where T is the length of the actions (tail deficits ≤6 frames are padded by repeating the last
  frame and recorded in `meta.json`).
* Output: `data/processed/<video_id>/<uuid>/{frames.mp4, actions.npy, frame0.png, meta.json}`. The
  video is H.264 yuv444p at CRF 10. Use `--enc-args "-preset veryslow -crf 0 -pix_fmt yuv444p"` for
  lossless, or `--vcodec h264_nvenc --enc-args "-preset p7 -tune hq -rc constqp -qp 10 -pix_fmt yuv444p"`
  to encode on the GPU.
* `--target-fps 30` (default) picks an integer stride per chunk (60→30 uses 2, 30→30 uses 1). Actions
  are pooled over each window: sticks are averaged, buttons are OR'd, so a 2-frame tap survives.
  Pooling composes, so you can go 30→15 at load time with the same rule. Stage 02's default `--fps 30 60`
  drops the ~1 % of 25/50 fps videos, which can't reach 30 with an integer stride.
* Private/removed videos are recorded as permanent failures in `data/processed/_failed.jsonl` and
  skipped on re-runs. HTTP 429 / bot checks trigger a global back-off. If YouTube starts demanding
  sign-in, pass `--cookies cookies.txt`, and keep yt-dlp updated.
* Budget: 1,000 h at 30 fps is about 108M frames. At CRF 10 yuv444p that's roughly 0.5–1.5 MB per
  20 s clip, i.e. ~0.1–0.3 GB per hour of data. Download is ~1.8 GB per source hour at 720p60, so
  5k h means ~9 TB to transfer. Use `--workers` sized to your bandwidth.

### 4. Caption frame 0
```bash
python 04_caption_frames.py --backend vllm --batch-size 64                    # one GPU
# multi-GPU data parallel:
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python 04_caption_frames.py --num-shards 4 --shard-index $i & done; wait
python 04_caption_frames.py --merge-only
# or against a server:  vllm serve Qwen/Qwen2.5-VL-7B-Instruct
python 04_caption_frames.py --backend openai --api-base http://localhost:8000/v1 --concurrency 32
```
The letterbox is cropped off and the frame upscaled to 448 px before captioning. The VLM returns JSON
`{caption, perspective, is_2d}`, and the prompt explicitly forbids mentioning HUD elements, health
bars, score counters, text, overlays and the game's name. Results go into each `meta.json` and into
the manifest columns `caption`, `vlm_perspective`, `vlm_is_2d` and `frame0_dark`.

### 5. Pack + verify
```bash
python 05_pack_webdataset.py --drop-vlm-3d          # ~1 GB shards in data/webdataset/{train,val_unseen_games}
python verify_dataset.py --num-shards 3 --debug-videos 2
```
Each sample has `<uuid>.mp4`, `.actions.npy` (T,21 float32), `.txt`, `.cls` (game_id) and `.json`.
`verify_dataset.py` asserts `len(frames) == len(actions)`, checks value ranges, binary buttons, black
letterbox bars and train/val game disjointness. It writes `data/debug/grid_<split>.png` and overlay
videos. Watch the overlay videos to confirm that stick motion matches on-screen motion before training.

## Action vector (21-d, float32)
`[j_left_x, j_left_y, j_right_x, j_right_y, dpad_down, dpad_left, dpad_right, dpad_up, left_shoulder,
left_thumb, left_trigger, right_shoulder, right_thumb, right_trigger, south, west, east, north, back,
start, guide]`. Sticks are in [-1, 1] (y = −1 is up); buttons are {0, 1}. `actions_processed.parquet`
is used when present, otherwise `actions_raw.parquet` (recorded as `action_source`).

## Known limitations
* Labels are per video, but some streams contain several games (e.g. "SotN randomizer, then
  Crimzon Clover"). The VLM's `is_2d` check together with `--drop-vlm-3d` catches part of this.
* The YouTube enrichment can only identify games that are listed in `genres_2d.yaml`/`aliases`.
  Coverage grows as you add labels.
* License: NitroGen annotations are CC BY-NC 4.0, and the videos belong to their creators. Use for
  research only.
