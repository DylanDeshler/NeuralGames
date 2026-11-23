"""Game identification helpers shared by stage 02 and 02b.

GameLabeler  curated 2D whitelist / 3D blacklist / aliases (config/genres_2d.yaml) + manual CSV labels
YouTubeMeta  cached, rate-limited yt-dlp metadata lookups (title, tags, description; no media)
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from ng_common import norm_title

UNAVAILABLE_PATTERNS = re.compile(
    r"(private video|video unavailable|has been removed|no longer available|account associated|"
    r"copyright|members-only|join this channel|confirm your age|is not available|terminated)", re.I)




# ============================================================================ game labeling
class GameLabeler:
    def __init__(self, cfg_path: Path, labels_csv: Path | None):
        cfg = yaml.safe_load(cfg_path.read_text())
        self.title_2d: dict[str, tuple[str, str]] = {}  # norm -> (display, genre)
        for genre, titles in (cfg.get("games_2d") or {}).items():
            for t in titles or []:
                self.title_2d[norm_title(t)] = (t, genre)
        self.title_3d = {norm_title(t): t for t in (cfg.get("exclude_3d") or [])}
        self.kw2d = [k.lower() for k in cfg.get("keywords_2d") or []]
        self.kw3d = [k.lower() for k in cfg.get("keywords_3d") or []]
        self.manual: dict[str, tuple[bool, str]] = {}
        if labels_csv and labels_csv.exists():
            m = pd.read_csv(labels_csv, comment="#")
            for _, r in m.iterrows():
                val = str(r.get("is_2d", "")).strip().lower()
                if val in ("1", "true", "yes", "y", "2d"):
                    self.manual[norm_title(r["game"])] = (True, str(r.get("genre", "") or "manual"))
                elif val in ("0", "false", "no", "n", "3d"):
                    self.manual[norm_title(r["game"])] = (False, "")
        # word-level phrase patterns for title matching, longest first
        def phrases(d):
            items = []
            for key, val in d.items():
                disp = val[0] if isinstance(val, tuple) else val
                words = re.sub(r"[^a-z0-9]+", " ", disp.lower().replace("&", "and")).split()
                if words:
                    items.append((" " + " ".join(words) + " ", key))
            return sorted(items, key=lambda x: -len(x[0]))
        self.phr_2d, self.phr_3d = phrases(self.title_2d), phrases(self.title_3d)
        # aliases (exempt from the min-length rule)
        self.alias_phr: list[tuple[str, str, str]] = []  # (phrase, status, key)
        for canon, alist in (cfg.get("aliases") or {}).items():
            k = norm_title(canon)
            status = "3d" if k in self.title_3d else ("2d" if k in self.title_2d else None)
            if status is None:
                continue
            for a in alist or []:
                words = re.sub(r"[^a-z0-9]+", " ", a.lower().replace("&", "and")).split()
                if words:
                    self.alias_phr.append((" " + " ".join(words) + " ", status, k))
        self.alias_phr.sort(key=lambda x: -len(x[0]))
        self.ambiguous = {norm_title(t) for t in cfg.get("ambiguous_titles") or []}

    def label_slug(self, game: str):
        """-> (status in {'2d','3d','unknown'}, genre, source)"""
        k = norm_title(game)
        if k in self.manual:
            is2d, genre = self.manual[k]
            return ("2d" if is2d else "3d"), genre, "manual"
        if k in self.title_3d:
            return "3d", "", "exclude_3d"
        if k in self.title_2d:
            return "2d", self.title_2d[k][1], "whitelist"
        return "unknown", "", ""

    def match_text(self, text: str):
        """Find the longest known title (2D or 3D) mentioned in a YouTube title/tags string."""
        t = " " + " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower().replace("&", "and")).split()) + " "
        best = None
        for phr, status, key in self.alias_phr:
            if phr in t:
                best = (status, key, len(phr) + 100)  # explicit alias beats generic title hits
                break
        if best is not None:
            return best
        for phr, key in self.phr_2d:
            if key in self.ambiguous and not t.startswith(phr):
                continue
            if len(phr) > 6 and phr in t:  # skip super-short titles ("fez") to avoid false hits
                best = ("2d", key, len(phr))
                break
        for phr, key in self.phr_3d:
            if len(phr) > 6 and phr in t:
                if best is None or len(phr) >= best[2]:
                    best = ("3d", key, len(phr))
                break
        return best

    def keyword_vote(self, text: str) -> int:
        t = " " + " ".join(re.sub(r"[^a-z0-9']+", " ", (text or "").lower()).split()) + " "
        s2 = sum(f" {k} " in t for k in self.kw2d)
        s3 = sum(f" {k} " in t for k in self.kw3d)
        return (s2 > 0 and s3 == 0) - (s3 > 0 and s2 == 0)


# ============================================================================ YouTube metadata
class YouTubeMeta:
    """Cached, rate-limited yt-dlp metadata lookups (no media download)."""

    def __init__(self, cache_path: Path, workers: int, min_interval: float, cookies: str | None,
                 cookies_from_browser: str | None, log):
        self.cache_path, self.workers, self.min_interval, self.log = cache_path, workers, min_interval, log
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.cache[r["video_id"]] = r
                except Exception:
                    pass
        self.lock, self.rate_lock, self.last = threading.Lock(), threading.Lock(), 0.0
        self.opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True,
                     "socket_timeout": 30}
        if cookies:
            self.opts["cookiefile"] = cookies
        if cookies_from_browser:
            self.opts["cookiesfrombrowser"] = (cookies_from_browser,)

    def _throttle(self):
        with self.rate_lock:
            wait = self.last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()

    def _fetch(self, vid: str) -> dict:
        import yt_dlp
        for attempt in range(3):
            self._throttle()
            try:
                with yt_dlp.YoutubeDL(self.opts) as ydl:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False, process=False)
                return {"video_id": vid, "status": "ok", "title": info.get("title"),
                        "tags": (info.get("tags") or [])[:40], "categories": info.get("categories"),
                        "channel": info.get("channel"), "description": (info.get("description") or "")[:600],
                        "duration": info.get("duration"), "availability": info.get("availability")}
            except Exception as e:
                msg = str(e)
                if UNAVAILABLE_PATTERNS.search(msg) and "not a bot" not in msg:
                    return {"video_id": vid, "status": "unavailable", "error": msg[:300]}
                if "not a bot" in msg or "429" in msg:
                    time.sleep(30 * (attempt + 1))
                else:
                    time.sleep(3 * (attempt + 1))
                last = msg
        return {"video_id": vid, "status": "error", "error": last[:300]}

    def get_many(self, vids: list[str]) -> dict[str, dict]:
        todo = [v for v in dict.fromkeys(vids) if self.cache.get(v, {}).get("status") not in ("ok", "unavailable")]
        if todo:
            self.log.info("Fetching YouTube metadata for %d videos (%d cached) ...", len(todo), len(vids) - len(todo))
            with ThreadPoolExecutor(self.workers) as ex, open(self.cache_path, "a") as fh:
                futs = [ex.submit(self._fetch, v) for v in todo]
                for f in tqdm(as_completed(futs), total=len(futs), desc="yt-meta"):
                    r = f.result()
                    with self.lock:
                        self.cache[r["video_id"]] = r
                        fh.write(json.dumps(r) + "\n")
                        fh.flush()
        return {v: self.cache.get(v, {"status": "missing"}) for v in vids}
