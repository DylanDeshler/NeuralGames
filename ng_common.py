"""Shared helpers for the NitroGen 2D world-model data pipeline.

Schema facts (verified against nvidia/NitroGen shards, 2026-11):
  * HF repo hosts `actions/SHARD_XXXX.tar.gz` (100 shards, ~1.6 GB each).
  * Inside: SHARD_XXXX/<video_id>/<video_id>_chunk_NNNN/{metadata.json, actions_raw.parquet,
    [actions_processed.parquet]}
  * Parquet: 17 int32 button columns + `j_left`/`j_right` as list<double>[2]; rows == chunk_size.
  * metadata.json: `game` is a snake_case slug (often "other"); fps = chunk_size / duration
    (30 or 60); `bbox_controller_overlay` = [x, y, w, h] in pixels of `original_video.resolution`
    ([H, W]); optional `bbox_game_area` / `bbox_others` in relative [0, 1] coords.

The action-column detection below is deliberately defensive so the pipeline keeps working if
NVIDIA changes the layout (e.g. `j_left_x`/`j_left_y` columns or struct columns).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

# Canonical button order (from the dataset README). Output action vector layout:
#   [j_left_x, j_left_y, j_right_x, j_right_y, *BUTTONS]  -> 21 dims, float32
BUTTONS = [
    "dpad_down", "dpad_left", "dpad_right", "dpad_up",
    "left_shoulder", "left_thumb", "left_trigger",
    "right_shoulder", "right_thumb", "right_trigger",
    "south", "west", "east", "north",
    "back", "start", "guide",
]
ACTION_NAMES = ["j_left_x", "j_left_y", "j_right_x", "j_right_y"] + BUTTONS
ACTION_DIM = len(ACTION_NAMES)  # 21
# Menu/system buttons are excluded from the "gameplay button activity" statistic.
NON_GAMEPLAY_BUTTONS = {"back", "start", "guide"}
GAMEPLAY_BUTTON_IDX = [4 + i for i, b in enumerate(BUTTONS) if b not in NON_GAMEPLAY_BUTTONS]
STICK_THRESHOLDS = (0.1, 0.2, 0.3)

# Root for ALL pipeline outputs (metadata, manifests, clips, shards, caches, logs).
# Set once:  export NG_DATA_ROOT=/data/games
DATA_ROOT = Path(os.environ.get("NG_DATA_ROOT", "./data")).expanduser()
# Keep HF model/dataset caches (Qwen weights ~15-30 GB each) on the same disk unless HF_HOME is already set.
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf_cache"))


# ----------------------------------------------------------------------------- logging
def setup_logging(name: str, log_dir: str | Path | None = None, level: str = "INFO") -> logging.Logger:
    log_dir = log_dir or DATA_ROOT / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ----------------------------------------------------------------------------- schema
def detect_action_schema(column_names: list[str], arrow_types: dict[str, str] | None = None) -> dict:
    """Figure out how sticks/buttons are stored. Returns a dict usable by `actions_to_array`."""
    cols = set(column_names)
    arrow_types = arrow_types or {}
    sticks = {}
    for stick in ("j_left", "j_right"):
        if stick in cols:
            t = arrow_types.get(stick, "")
            sticks[stick] = {"mode": "struct" if t.startswith("struct") else "list", "cols": [stick]}
        elif f"{stick}_x" in cols and f"{stick}_y" in cols:
            sticks[stick] = {"mode": "xy", "cols": [f"{stick}_x", f"{stick}_y"]}
        else:
            sticks[stick] = {"mode": "missing", "cols": []}
    present = [b for b in BUTTONS if b in cols]
    missing = [b for b in BUTTONS if b not in cols]
    extra = sorted(cols - set(BUTTONS) - {c for s in sticks.values() for c in s["cols"]})
    return {"sticks": sticks, "buttons_present": present, "buttons_missing": missing, "extra_columns": extra}


def _stick_from_column(col) -> np.ndarray:
    """Convert a pyarrow list/fixed-size-list/struct column to an (T, 2) float array."""
    import pyarrow as pa

    col = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    t = col.type
    if pa.types.is_struct(t):
        names = [t.field(i).name for i in range(t.num_fields)]
        xs = col.field(names[0]).to_numpy(zero_copy_only=False)
        ys = col.field(names[1]).to_numpy(zero_copy_only=False)
        return np.stack([xs, ys], 1).astype(np.float32)
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        if flat.size == 2 * len(col):
            return flat.reshape(-1, 2)
        # ragged / null rows: slow path
        out = np.zeros((len(col), 2), np.float32)
        for i, v in enumerate(col.to_pylist()):
            if v is not None and len(v) >= 2:
                out[i] = v[:2]
        return out
    raise TypeError(f"Unsupported stick column type: {t}")


def actions_to_array(table, schema: dict | None = None) -> np.ndarray:
    """pyarrow.Table -> (T, 21) float32 [j_left(2), j_right(2), buttons(17)]."""
    if schema is None:
        schema = detect_action_schema(table.column_names, {f.name: str(f.type) for f in table.schema})
    T = table.num_rows
    out = np.zeros((T, ACTION_DIM), np.float32)
    for k, stick in enumerate(("j_left", "j_right")):
        s = schema["sticks"][stick]
        if s["mode"] in ("list", "struct"):
            out[:, 2 * k:2 * k + 2] = _stick_from_column(table.column(s["cols"][0]))
        elif s["mode"] == "xy":
            out[:, 2 * k] = table.column(s["cols"][0]).to_numpy(zero_copy_only=False)
            out[:, 2 * k + 1] = table.column(s["cols"][1]).to_numpy(zero_copy_only=False)
    for i, b in enumerate(BUTTONS):
        if b in schema["buttons_present"]:
            v = table.column(b).to_numpy(zero_copy_only=False)
            out[:, 4 + i] = (np.nan_to_num(v.astype(np.float32)) > 0.5)
    np.nan_to_num(out, copy=False)
    np.clip(out[:, :4], -1.0, 1.0, out=out[:, :4])
    return out


def activity_stats(a: np.ndarray) -> dict[str, float]:
    """Per-chunk activity statistics used by the idle filter in stage 02."""
    if len(a) == 0:
        return {f"frac_lstick_{int(t * 10):02d}": 0.0 for t in STICK_THRESHOLDS} | {
            f"frac_move_{int(t * 10):02d}": 0.0 for t in STICK_THRESHOLDS} | {
            "frac_rstick_02": 0.0, "frac_btn_gameplay": 0.0, "frac_btn_any": 0.0,
            "mean_btn_per_frame": 0.0, "frac_any_input": 0.0}
    ln = np.linalg.norm(a[:, 0:2], axis=1)
    rn = np.linalg.norm(a[:, 2:4], axis=1)
    btn = a[:, 4:]
    gp = a[:, GAMEPLAY_BUTTON_IDX].max(1) > 0
    dpad = a[:, 4:8].max(1) > 0  # many 2D players move with the d-pad, not the stick
    stats = {f"frac_lstick_{int(round(t * 10)):02d}": float((ln > t).mean()) for t in STICK_THRESHOLDS}
    stats.update({f"frac_move_{int(round(t * 10)):02d}": float(((ln > t) | dpad).mean()) for t in STICK_THRESHOLDS})
    stats.update(
        frac_rstick_02=float((rn > 0.2).mean()),
        frac_btn_gameplay=float(gp.mean()),
        frac_btn_any=float((btn.max(1) > 0).mean()),
        mean_btn_per_frame=float(btn.sum(1).mean()),
        frac_any_input=float(((ln > 0.2) | (rn > 0.2) | gp).mean()),
    )
    return stats


# ----------------------------------------------------------------------------- misc
_ROMAN = {"ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9"}


def norm_words(s: str) -> list[str]:
    """Lowercase words with roman numerals II-IX as digits ('I' and 'X' are left alone: Mega Man X != 10)."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()  # Pokémon -> Pokemon
    words = re.sub(r"[^a-z0-9]+", " ", s.lower().replace("&", "and")).split()
    single = [len(w) == 1 for w in words]
    out = []
    for i, w in enumerate(words):
        # don't convert inside letter runs like "E.V.O." -> e v o
        near_letters = (i > 0 and single[i - 1]) or (i + 1 < len(words) and single[i + 1])
        out.append(w if near_letters else _ROMAN.get(w, w))
    return out


def norm_title(s: str) -> str:
    """'Hollow Knight: Silksong' / 'hollow_knight__silksong' -> 'hollowknightsilksong'; 'Blasphemous II' -> 'blasphemous2'."""
    return "".join(norm_words(s))


def parse_shard_spec(spec: str, n_total: int = 100) -> list[int]:
    """'all' | '0-9,15,20-22' -> sorted list of shard indices."""
    if spec.strip().lower() == "all":
        return list(range(n_total))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def dir_size_bytes(path: str | Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def free_gb(path: str | Path) -> float:
    Path(path).mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def atomic_write_parquet(df, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def processed_dir_for(processed_root: str | Path, video_id: str, uuid: str) -> Path:
    return Path(processed_root) / video_id / uuid


# ----------------------------------------------------------------------------- VLM / LLM helpers
DEFAULT_VLM = "Qwen/Qwen3.8-27B-FP8"   # native image+video model; fits a 48 GB GPU in FP8


def vllm_chat_kwargs(model: str) -> dict:
    """Qwen3.x models think by default: disable it for short structured answers."""
    return {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3" in model.lower() else {}


def vllm_mm_kwargs(model: str, max_pixels: int) -> dict:
    """min/max_pixels are Qwen2.5-VL processor options; newer Qwen processors use their own defaults."""
    if "qwen2.5-vl" in model.lower() or "qwen2_5" in model.lower():
        return {"mm_processor_kwargs": {"min_pixels": 64 * 28 * 28, "max_pixels": max_pixels}}
    return {}


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)


def openai_chat_many(client, model: str, convs: list[list[dict]], max_tokens: int, concurrency: int = 32) -> list[str]:
    """Send many chat requests to an OpenAI-compatible server (e.g. `vllm serve`) concurrently; returns texts.
    Thinking is disabled for Qwen3.x via chat_template_kwargs (vLLM honours it in extra_body)."""
    import time as _t
    from concurrent.futures import ThreadPoolExecutor as _TPE
    extra = vllm_chat_kwargs(model) or None

    def one(msgs):
        for attempt in range(5):
            try:
                r = client.chat.completions.create(model=model, messages=msgs, temperature=0.0, max_tokens=max_tokens,
                                                   extra_body=extra)
                return r.choices[0].message.content or ""
            except Exception:
                if attempt == 4:
                    return ""
                _t.sleep(2 ** attempt)
    with _TPE(concurrency) as ex:
        return list(ex.map(one, convs))
