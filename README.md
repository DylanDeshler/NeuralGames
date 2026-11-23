# NitroGen → 2D action-conditioned world-model dataset

A pipeline that turns the `nvidia/NitroGen` action annotations, plus the public YouTube videos they
refer to, into WebDataset shards of `(video in an aspect-ratio bucket, (T,21) actions, caption, game_id)`. Every stage
can be resumed: re-running a stage skips work that's already done.

```
01_fetch_metadata.py      HF tarballs -> compact per-shard index + action parquet       (CPU, ~25-60 GB)
02b_llm_label_games.py    (optional) LLM names the game per video from YouTube metadata    (GPU, ~minutes)
02_profile_and_sample.py  2D filter, idle filter, alpha-sampling, game holdout -> manifest (CPU, minutes)
02c_detect_layout.py      VLM finds the game screen inside stream layouts (chat/webcam), per video (GPU+net)
03_download_and_process.py  20 s clip fetch, aspect bucket (no letterbox), inpaint, align (network-bound)
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
  processed/<video_id>/<uuid>/  tmp_03/         stage 03 (~0.10 GB per hour of data @ 15 fps, CRF 18)
  webdataset/{train,val_unseen_games}/          stage 05 (same size again; delete processed/ after verifying)
  hf_cache/                                     Qwen2.5-14B (~30 GB) + Qwen2.5-VL-7B (~16 GB)
  logs/  debug/
```
Every path can still be overridden per script (`--meta-dir`, `--processed-dir`, `--out`, ...).

## YouTube access from cloud machines
YouTube often answers datacenter IPs with "Sign in to confirm you're not a bot".
* **Metadata (stages 02/02b): use the YouTube Data API v3.** No bot checks, 50 videos per request,
  and ~300 quota units for 15k videos (10k/day free). Create a key at console.cloud.google.com →
  APIs & Services → enable "YouTube Data API v3" → Credentials → API key. Then
  `export YOUTUBE_API_KEY=...` (`--yt-backend auto` picks it up). Non-YouTube sources in the index
  (e.g. Twitch VODs, ids like `v2170866956`) are fetched from their own URL with yt-dlp, and expired
  VODs are marked unavailable.
* **Video downloads (stage 03) can't use the API.** Options, best first: run stage 03 from a residential
  network; `--cookies cookies.txt` from a logged-in (throwaway) account plus the
  `bgutil-ytdlp-pot-provider` PO-token plugin for yt-dlp; or `--proxy http://user:pass@host:port`
  (applied to both yt-dlp and the ffmpeg fetch, since the stream URLs are tied to the IP that
  resolved them).
* Both stages stop early with a clear message after repeated bot checks (`--max-bot-streak`) instead
  of grinding. Progress is cached, so just re-run once access is fixed.

## One command: `run_pipeline.py`
After stages 01, 02b and 02 (metadata, LLM labels, manifest), the download/processing stages run together:
```bash
python run_pipeline.py smoke --proxy http://127.0.0.1:8899     # ~20 clips in $NG_DATA_ROOT/smoke/, ends with smoke_results.tgz
python run_pipeline.py bulk  --proxy http://127.0.0.1:8899 --workers 8 --min-free-gb 60
```
* **smoke** uses an isolated copy of the manifest and its own processed/shard dirs, so it never touches
  bulk outputs. It runs 02c (first 40 videos) → 03 (`--clips 20`) → 04 → 05 → verify → inspect, then
  bundles the debug grids, debug videos, `verify.txt`, `inspect.txt` and name mismatches into one tarball.
* **bulk** runs 02c for every video (resumable), then repeats `03 --limit --cycle-clips` (20k clips, ~110 h)
  → `04` → `05 --append --delete-processed` until the manifest is done, the disk floor is reached, or
  downloads stop. Stop it any time (Ctrl-C) and re-run the same command to resume.
* `--resample` re-runs stage 02 first, `--skip-layout` skips 02c, and `--dry-run` prints the commands.

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

**Balanced-first download order.** The manifest has a `download_priority` column (a game's cumulative
hours up to and including that chunk), and stage 03 processes chunks in that order. **Any prefix
of the download is therefore a per-game-capped ("water-filling") subset.** If you sample generously
(even `--target-hours 1e9` = everything) and stop stage 03 whenever bandwidth or YouTube access runs out,
what you have is already balanced. `03_download_and_process.py --status` reports the downloaded
subset's game count, `eff_games`, top-game share, and the per-game cap reached so far. Per-game
re-weighting can also be done at training time with sampling weights.
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

### 2c. Stream-layout detection (before downloading clips)
Many videos are stream captures whose frame also holds Twitch chat, a webcam, alerts or decorative
panels, and NitroGen has no `bbox_game_area` for them. In the first real smoke test this affected 4 of 16 clips.
`02c_detect_layout.py` grabs 3 frames per video (all videos by default, because NitroGen's own game areas
often still include stream bars) and asks Qwen2.5-VL for the game-screen rectangle plus overlay boxes.
Overlays that touch an edge and span most of the frame (chat columns, status bars, bezels) are **cropped
off**; smaller ones inside the picture (webcams) are inpainted. Stage 03 crops to metadata area ∩ 02c area.
```bash
python 02c_detect_layout.py --proxy http://127.0.0.1:8899 --limit 50   # check the summary line, then run without --limit
python 03_download_and_process.py --proxy http://127.0.0.1:8899 --require-layout-check ...
```
`--require-layout-check` makes stage 03 skip videos 02c hasn't checked yet, so no chat or webcam layouts
slip in. `--min-video-gameplay-frac 0.5` additionally skips videos whose probe frames were mostly menus.

### 3. Download + process clips
```bash
python 03_download_and_process.py --limit 20 --workers 4      # smoke test, then inspect data/processed
python 03_download_and_process.py --workers 16 --max-starts-per-min 40
python 03_download_and_process.py --status
```
* `--method auto` (default): yt-dlp resolves the stream URL, then ffmpeg seeks with HTTP range
  requests and decodes only [start, end). If that fails, it falls back to
  `yt-dlp --download-sections "*start-end" --force-keyframes-at-cuts`.
* **Aspect-ratio buckets, no letterboxing** (built for pixel-space patch transformers like JiT):
  `crop(bbox_game_area) → detect static black pillar/letterbox bars → pick the bucket with the closest
  aspect → resize to cover it, centre-crop the excess (≤ --max-crop, typically 1–2 %) → cv2.inpaint(TELEA)`
  over the controller overlay and `bbox_others`. There's no distortion, and every token is game content.
  `--buckets auto` (default) uses the canonical buckets below plus filler buckets (24 in total,
  216–272 tokens at `--patch 16`), so every aspect from 0.5 to 2.45 is within 5 % of a bucket and no
  clip needs padding. Canonical buckets:

  | bucket | px | aspect | typical source |
  |---|---|---|---|
  | 21x12 | 336×192 | 1.75 | 16:9 (most modern games, ~75 % of hours by metadata) |
  | 19x14 | 304×224 | 1.36 | 4:3 (retro, often pillarboxed inside 16:9 videos) |
  | 17x15 | 272×240 | 1.13 | 8:7 (SNES/NES) |
  | 18x12 | 288×192 | 1.50 | 3:2 (GBA) |
  | 19x12 | 304×192 | 1.58 | 16:10 |
  | 16x16 | 256×256 | 1.00 | square |
  | 24x10 | 384×160 | 2.40 | 21:9 |
  | 14x19, 12x21 | 224×304, 192×336 | 0.74, 0.57 | vertical / TATE |

  Pass an explicit list to restrict them (e.g. `--buckets 21x12,19x14,16x16`); with fewer buckets,
  clips beyond `--max-crop` are fit and padded (`fit_mode`, `pad`, `content_wh` in the JSON, already
  excluded by `ng_masks.pixel_valid_mask`). `--patch 32` doubles every side for 512-class models.
  `02_profile_and_sample.py --plan` prints the pool's aspect-ratio split (metadata only; hidden
  pillarboxing is detected later, from pixels, in stage 03). `--status` and the run summary report the bucket counts. Output has exactly `T` frames,
  where T is the length of the actions (tail deficits ≤6 frames are padded by repeating the last
  frame and recorded in `meta.json`).
* Output: `data/processed/<video_id>/<uuid>/{frames.mp4, actions.npy, frame0.png, meta.json}`. The
  video is H.264 yuv444p at CRF 18. That's visually near-lossless on top of an already-lossy YouTube source
  (41.6 dB PSNR vs CRF 10, measured), at ~2.5× less disk. Use `--enc-args "-preset medium -crf 22 -pix_fmt yuv444p"`
  to save another ~35 %, `-crf 0` for lossless, or `--vcodec h264_nvenc --enc-args "-preset p7 -tune hq -rc constqp -qp 10 -pix_fmt yuv444p"`
  to encode on the GPU.
* Source format: the selector prefers **~480p**, otherwise the next tier up to `--max-src-height` (720 by
  default), and only falls back to lower resolutions if nothing else exists. YouTube's 360p is only
  ~0.4–0.5 Mbit/s and visibly smears pixel art, and many videos have 360p and 720p60 but no 480p. Source
  download is ~0.5 GB/h at 480p and ~1 GB/h at 720p60, through the proxy if you use one. `--max-src-height 480`
  falls back to 360p when there's no 480p, which saves bandwidth at a clear cost in quality.
* `--target-fps 15` (default; the training rate) picks an integer stride per chunk (60→15 uses 4,
  30→15 uses 2). Actions are pooled over each window: sticks are averaged, buttons are OR'd, so a 1-frame
  tap at 60 fps survives. The keyframe interval defaults to 2 s at the output fps. Stage 02's default
  `--fps 30 60` drops the ~1 % of 25/50 fps videos, which can't reach 15 with an integer stride. The
  chunk fps (n_frames/duration) is snapped to the nominal rate, and clip timing uses the metadata `start_time`.
* Private/removed videos are recorded as permanent failures in `data/processed/_failed.jsonl` and
  skipped on re-runs. HTTP 429 / bot checks trigger a global back-off. If YouTube starts demanding
  sign-in, pass `--cookies cookies.txt`, and keep yt-dlp updated.
* Budget (measured on a real 2D clip; busier games cost more): **~0.10 GB per hour of data** at 15 fps,
  CRF 18 (0.066 at CRF 22). Downloads stream to a temp dir that's deleted per clip: ~0.15 GB per source hour
  at 360–480p, never kept on disk. With `05 --delete-processed` only one copy exists at a time, so 350 GB
  of free space holds ~3,000+ h. Stage 03 stops at `--min-free-gb`, and because of the balanced-first
  order whatever fits is a per-game-capped subset.
* **Routing downloads through a home connection** (when YouTube bot-blocks the cloud IP):
  ```bash
  # Mac / home machine (keep it awake: caffeinate -i)
  pip install proxy.py && proxy --hostname 127.0.0.1 --port 8899
  ssh -N -o ServerAliveInterval=30 -R 8899:127.0.0.1:8899 user@cloud-box
  # cloud box
  python 03_download_and_process.py --proxy http://127.0.0.1:8899 --workers 8
  ```
  An HTTP proxy works for both yt-dlp and ffmpeg's stream fetch. Stream URLs are tied to the IP that
  resolved them, so both must go through the same proxy. Throughput is bounded by the home connection.

### 4. VLM pass per clip: caption, gameplay QC, title check
```bash
python 04_caption_frames.py --backend vllm                 # 4 frames per clip by default (--qc-frames)
# multi-GPU: CUDA_VISIBLE_DEVICES=i python 04_caption_frames.py --num-shards N --shard-index i, then --merge-only
```
One request per clip shows Qwen2.5-VL 4 frames (start, 1/3, 2/3, end) and returns:
* `caption` for frame 0 (art style, perspective, biome, main character; no HUD, text or game name);
* `screens`: a per-frame label (gameplay, menu, title, loading, cutscene, map_or_inventory, black,
  not_a_game) -> `vlm_frac_gameplay`. In the first real smoke test, menus, title screens and an emulator
  ROM browser all passed the idle filter because menu navigation produces inputs.
* `game_title` it recognizes, used only for the train/val leakage check below.

Results go into `meta.json` and the manifest (`caption`, `vlm_frac_gameplay`, `vlm_game_title`, ...). The
pass is resumable. Throughput is roughly 10–20 clips/s on one 48 GB GPU with vLLM (~900 tokens per clip),
and it can run while stage 03 is still downloading.

### 5. Pack + verify
```bash
python 05_pack_webdataset.py --drop-vlm-3d          # ~1 GB shards in data/webdataset/{train,val_unseen_games}
python verify_dataset.py --num-shards 3 --debug-videos 2
```
Stage 05 also drops clips with `vlm_frac_gameplay < --min-gameplay-frac` (0.75), clips with more than
25 % near-black frames (`--max-black-frac`) and, with `--val-leak-check` (default), **val_unseen_games
videos whose frames the VLM recognizes as a different game that's in train**. Some labels are wrong
(e.g. a "Street Fighters 2" video that's actually *Sunset Riders*), and this stops them leaking a training
game into the unseen-game split. All label/frames disagreements are listed in
`webdataset/name_mismatch_report.csv` for review; fix recurring ones in `config/game_labels.csv`.

Shards are written per bucket (`train/21x12/train-21x12-000000.tar`, ...), and `index.json` lists
each bucket's resolution, samples and hours. To train, sample a bucket in proportion to its hours,
then stream same-shape batches from its shards. Each sample has `<uuid>.mp4`, `.actions.npy`
(T,21 float32), `.txt`, `.cls` (game_id) and `.json` (incl. `width`, `height`, `bucket`, `inpaint_boxes`).
`verify_dataset.py` asserts `len(frames) == len(actions)`, checks value ranges, binary buttons, black
letterbox bars and train/val game disjointness. It writes `data/debug/grid_<split>.png` and overlay
videos. Watch the overlay videos to confirm that stick motion matches on-screen motion before training.

## Overlay masking at training time
Loss masking alone isn't enough. The overlay *is* the action label in pixel form, so if it's
visible in the context frames, the model can read actions from pixels and ignore the action input.
The pipeline therefore:
1. **removes it from the input**: stage 03 inpaints `bbox_controller_overlay` and `bbox_others`
   before anything is encoded;
2. **records the regions**: each sample's JSON has `inpaint_boxes` ([x, y, w, h] in output-pixel
   coordinates, already grown by `--mask-dilate`), plus `pad`/`content_wh` for the letterbox.

On the training side, use `ng_masks.py`:
```python
from ng_masks import pixel_valid_mask, grid_valid_mask, augment_fake_boxes
valid = pixel_valid_mask(info)                    # (H, W): 1 = supervise; 0 = inpainted overlay (or pad, fit-mode only)
# pixel-space models (JiT, x-prediction): mask the per-pixel loss directly
loss = (per_pixel_loss * valid[None, None]).sum() / (valid.sum() * C).clip(min=1)
# per-token view if needed (e.g. drop fully masked tokens): grid_valid_mask(valid, stride=patch)
# optional mask-channel conditioning: concat 1 - augment_fake_boxes(valid, rng) as a 4th input channel
# to the patch embedding, blank the same pixels in context frames, and pass all-ones at inference
```
With a temporally compressing VAE, mask a latent if the region is masked in any frame it covers. The
boxes are fixed per clip, so this is just broadcasting. Check controllability after training with an
action-shuffle test (prediction error should rise noticeably with shuffled actions).

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
