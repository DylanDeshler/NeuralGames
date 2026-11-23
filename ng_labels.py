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

from ng_common import norm_title, norm_words

UNAVAILABLE_PATTERNS = re.compile(
    r"(private video|video unavailable|video is unavailable|is unavailable|has been removed|no longer available|"
    r"account associated|copyright|members-only|join this channel|confirm your age|is not available|terminated|"
    r"does not exist|deleted|404)", re.I)
BOT_PATTERNS = re.compile(r"(not a bot|sign in to confirm|HTTP Error 429|too many requests)", re.I)


class BotBlockedError(RuntimeError):
    pass


class _NullLogger:
    """Silences yt-dlp's own console output; errors are captured from the exceptions instead."""
    def debug(self, msg):
        pass

    warning = info = error = debug


def video_urls(meta_dir: Path) -> dict[str, str]:
    """video_id -> source URL from the stage-01 index (YouTube, Twitch, ...)."""
    import pyarrow.dataset as ds
    out: dict[str, str] = {}
    for f in sorted((Path(meta_dir) / "index").glob("*.parquet")):
        t = ds.dataset(f).to_table(columns=["video_id", "url", "source"]).to_pandas().drop_duplicates("video_id")
        for v, u, src in zip(t.video_id, t.url, t.source):
            if v not in out:
                out[v] = u if isinstance(u, str) and u else (
                    f"https://www.youtube.com/watch?v={v}" if (src or "youtube") == "youtube" else "")
    return out


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url




# ============================================================================ game labeling
class GameLabeler:
    def __init__(self, cfg_path: Path, labels_csv: Path | None):
        cfg = yaml.safe_load(cfg_path.read_text())
        bad = [t for v in (cfg.get("games_2d") or {}).values() for t in (v or []) if not isinstance(t, str)]
        bad += [t for t in (cfg.get("exclude_3d") or []) if not isinstance(t, str)]
        if bad:
            raise ValueError(f"{cfg_path}: non-string titles (quote titles containing ':'): {bad}")
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
                words = norm_words(disp)
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
                words = norm_words(a)
                if words:
                    self.alias_phr.append((" " + " ".join(words) + " ", status, k))
        self.alias_phr.sort(key=lambda x: -len(x[0]))
        self.ambiguous = {norm_title(t) for t in cfg.get("ambiguous_titles") or []}
        self.junk_re = re.compile("|".join(cfg.get("junk_slugs") or ["$^"]), re.I)
        # titles that should count as ONE game for grouping / unseen-game holdout
        self.group_of = {norm_title(t): norm_title(canon) for canon, ts in (cfg.get("game_groups") or {}).items()
                         for t in [canon, *(ts or [])]}

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
        # slug contains a known title/alias as a phrase: 'the_binding_of_isaac__repentance' -> Isaac,
        # 'dark_souls__remastered' -> Dark Souls, 'super_mario_world_2__yoshi_s_island' -> SMW
        if game and game != "other":
            hit = self.match_text(game.replace("_", " "))
            if hit:
                if hit[0] == "2d":
                    return "2d", self.title_2d[hit[1]][1], "slug_match"
                return "3d", "", "slug_match"
        return "unknown", "", ""

    def is_junk_slug(self, game: str) -> bool:
        """NitroGen 'game' values that are stream/chapter markers, not games (treated like 'other')."""
        return bool(game) and game != "other" and bool(self.junk_re.search(game))

    def canonical_key(self, game: str) -> str:
        """Game key used for grouping/holdout: the matched curated title if the slug contains one,
        then collapsed through `game_groups` (e.g. Isaac Rebirth/Repentance -> one game)."""
        k = norm_title(game)
        if not (k in self.manual or k in self.title_2d or k in self.title_3d or not game or game == "other"):
            hit = self.match_text(game.replace("_", " "))
            k = hit[1] if hit else k
        return self.group_of.get(k, k)

    def match_text(self, text: str):
        """Find the longest known title (2D or 3D) mentioned in a YouTube title/tags string."""
        t = " " + " ".join(norm_words(text)) + " "
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
    """Cached video metadata (title, tags, description, channel; no media).

    backend="api"    YouTube Data API v3 (needs YOUTUBE_API_KEY / --yt-api-key). 50 ids per request,
                     ~1 quota unit each (10k/day free). No bot checks. Missing ids => unavailable.
    backend="ytdlp"  yt-dlp extraction (rate-limited; cloud IPs often get "confirm you're not a bot":
                     pass --cookies or use the API backend).
    Non-YouTube sources (e.g. Twitch VODs) always go through yt-dlp using their real URL.
    """

    def __init__(self, cache_path: Path, workers: int, min_interval: float, cookies: str | None,
                 cookies_from_browser: str | None, log, backend: str = "ytdlp", api_key: str | None = None,
                 proxy: str | None = None, max_bot_streak: int = 25):
        self.cache_path, self.workers, self.min_interval, self.log = cache_path, workers, min_interval, log
        self.backend, self.api_key, self.max_bot_streak = backend, api_key, max_bot_streak
        if backend == "api" and not api_key:
            raise SystemExit("--yt-backend api needs an API key (--yt-api-key or env YOUTUBE_API_KEY)")
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                    self.cache[r["video_id"]] = r
                except Exception:
                    pass
        self.lock, self.rate_lock, self.last = threading.Lock(), threading.Lock(), 0.0
        self.bot_streak, self.n_ok = 0, 0
        self.sample_errors: list[str] = []
        self.opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True,
                     "socket_timeout": 30, "logger": _NullLogger()}
        if cookies:
            self.opts["cookiefile"] = cookies
        if cookies_from_browser:
            self.opts["cookiesfrombrowser"] = (cookies_from_browser,)
        if proxy:
            self.opts["proxy"] = proxy

    # ------------------------------------------------------------------ yt-dlp path
    def _throttle(self):
        with self.rate_lock:
            wait = self.last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()

    def _fetch_ytdlp(self, vid: str, url: str) -> dict:
        import yt_dlp
        last = ""
        for attempt in range(3):
            if self.bot_streak >= self.max_bot_streak:
                return {"video_id": vid, "status": "error", "error": "aborted: bot-blocked"}
            self._throttle()
            try:
                with yt_dlp.YoutubeDL(self.opts) as ydl:
                    info = ydl.extract_info(url, download=False, process=False)
                with self.lock:
                    self.bot_streak, self.n_ok = 0, self.n_ok + 1
                return {"video_id": vid, "status": "ok", "source_url": url, "title": info.get("title"),
                        "tags": (info.get("tags") or [])[:40], "categories": info.get("categories"),
                        "channel": info.get("channel") or info.get("uploader"),
                        "description": (info.get("description") or "")[:600], "duration": info.get("duration"),
                        "availability": info.get("availability")}
            except Exception as e:
                last = str(e)
                with self.lock:
                    if len(self.sample_errors) < 5 and last[:120] not in self.sample_errors:
                        self.sample_errors.append(last[:120])
                if BOT_PATTERNS.search(last) and is_youtube(url):  # only YouTube bot checks count toward a block
                    with self.lock:
                        self.bot_streak += 1
                    time.sleep(10 * (attempt + 1))
                elif BOT_PATTERNS.search(last):
                    time.sleep(10 * (attempt + 1))
                elif UNAVAILABLE_PATTERNS.search(last):
                    return {"video_id": vid, "status": "unavailable", "source_url": url, "error": last[:300]}
                else:
                    time.sleep(3 * (attempt + 1))
        return {"video_id": vid, "status": "error", "source_url": url, "error": last[:300]}

    # ------------------------------------------------------------------ YouTube Data API path
    def _fetch_api_batch(self, vids: list[str]) -> list[dict]:
        import urllib.error
        import urllib.parse
        import urllib.request
        q = urllib.parse.urlencode({"part": "snippet,status,contentDetails", "id": ",".join(vids),
                                    "key": self.api_key, "maxResults": 50})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"https://www.googleapis.com/youtube/v3/videos?{q}", timeout=30) as r:
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                if e.code == 403 and "quota" in body.lower():
                    raise RuntimeError(f"YouTube API quota exhausted: {body}") from e
                if e.code in (400, 401, 403):
                    raise RuntimeError(f"YouTube API error {e.code}: {body}") from e
                time.sleep(2 ** attempt)
            except Exception:
                time.sleep(2 ** attempt)
        else:
            return [{"video_id": v, "status": "error", "error": "api request failed"} for v in vids]
        got = {}
        for it in data.get("items", []):
            sn, st = it.get("snippet", {}), it.get("status", {})
            got[it["id"]] = {"video_id": it["id"], "status": "ok" if st.get("privacyStatus") != "private" else "unavailable",
                             "source_url": f"https://www.youtube.com/watch?v={it['id']}", "title": sn.get("title"),
                             "tags": (sn.get("tags") or [])[:40], "categories": [sn.get("categoryId")],
                             "channel": sn.get("channelTitle"), "description": (sn.get("description") or "")[:600],
                             "duration": it.get("contentDetails", {}).get("duration"),
                             "availability": st.get("privacyStatus"), "via": "api"}
        return [got.get(v, {"video_id": v, "status": "unavailable", "error": "not returned by API (removed/private)",
                            "via": "api"}) for v in vids]

    # ------------------------------------------------------------------ public
    def get_many(self, vids: list[str], urls: dict[str, str] | None = None, strict: bool = False) -> dict[str, dict]:
        """Fetch missing metadata. On a bot block: raise if strict, else warn and return what's cached."""
        urls = urls or {}
        todo = [v for v in dict.fromkeys(vids) if self.cache.get(v, {}).get("status") not in ("ok", "unavailable")]
        if todo:
            url_of = {v: urls.get(v) or f"https://www.youtube.com/watch?v={v}" for v in todo}
            yt = [v for v in todo if is_youtube(url_of[v])]
            other = [v for v in todo if not is_youtube(url_of[v])]
            self.log.info("Fetching metadata for %d videos (%d cached): %d YouTube via %s, %d other via yt-dlp",
                          len(todo), len(set(vids)) - len(todo), len(yt), self.backend, len(other))
            with open(self.cache_path, "a") as fh:
                def save(r):
                    with self.lock:
                        self.cache[r["video_id"]] = r
                        fh.write(json.dumps(r) + "\n")
                        fh.flush()
                if self.backend == "api" and yt:
                    for i in tqdm(range(0, len(yt), 50), desc="yt-api (50/req)"):
                        for r in self._fetch_api_batch(yt[i:i + 50]):
                            save(r)
                ytdlp_list = other + (yt if self.backend != "api" else [])
                if ytdlp_list:
                    with ThreadPoolExecutor(self.workers) as ex:
                        futs = [ex.submit(self._fetch_ytdlp, v, url_of[v]) for v in ytdlp_list]
                        for f in tqdm(as_completed(futs), total=len(futs), desc="yt-dlp meta"):
                            r = f.result()
                            if r.get("error") != "aborted: bot-blocked":
                                save(r)
                            if self.bot_streak >= self.max_bot_streak:
                                ex.shutdown(wait=False, cancel_futures=True)
                                err = BotBlockedError(
                                    f"YouTube is bot-blocking this IP ({self.bot_streak} consecutive 'confirm you're "
                                    f"not a bot' errors, {self.n_ok} successes). Progress is cached. Fix one of:\n"
                                    "  * --yt-backend api --yt-api-key KEY   (YouTube Data API v3; recommended for metadata)\n"
                                    "  * --cookies cookies.txt               (exported from a logged-in browser)\n"
                                    "  * run this step from a residential IP and copy yt_meta_cache.jsonl over")
                                if strict:
                                    raise err
                                self.log.warning("%s\nContinuing with the metadata already cached.", err)
                                break
                if self.sample_errors:
                    self.log.info("Sample yt-dlp errors: %s", " || ".join(self.sample_errors))
                n_fail = sum(self.cache.get(v, {}).get("status") not in ("ok", "unavailable") for v in todo)
                if n_fail:
                    self.log.warning("%d videos still without metadata (retried on the next run)", n_fail)
        return {v: self.cache.get(v, {"status": "missing"}) for v in vids}
