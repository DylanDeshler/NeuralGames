#!/usr/bin/env bash
# Set up direct YouTube access on the cloud box (no home proxy): JS runtime + PO-token provider + cookies check.
#
#   bash tools/setup_youtube_access.sh [cookies.txt] [test_video_url]
#
# Needs: cookies exported from a THROWAWAY YouTube account; default = the repo's www.youtube.com_cookies.txt.
# Idempotent: safe to re-run. The provider server listens on 127.0.0.1:4416 only.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
COOKIES="${1:-$REPO/www.youtube.com_cookies.txt}"
# default: yt-dlp's own long-lived test video (10 s); pass a long gameplay URL from your pool for a better test
TEST_VIDEO="${2:-https://www.youtube.com/watch?v=BaW_jenozKc}"

echo "== 1/5 deno (JavaScript runtime yt-dlp needs to avoid throttled stream URLs)"
if ! command -v deno >/dev/null 2>&1; then
  curl -fsSL https://deno.land/install.sh | sh
  export PATH="$HOME/.deno/bin:$PATH"
  grep -q '.deno/bin' ~/.bashrc || echo 'export PATH="$HOME/.deno/bin:$PATH"' >> ~/.bashrc
fi
deno --version | head -1

echo "== 2/5 yt-dlp (with challenge-solver scripts) + PO-token plugin"
python -m pip install -q -U "yt-dlp[default]" bgutil-ytdlp-pot-provider
yt-dlp --version

echo "== 3/5 PO-token provider server on 127.0.0.1:4416"
if curl -s -o /dev/null "http://127.0.0.1:4416" 2>/dev/null; then
  echo "already running"
elif command -v docker >/dev/null 2>&1; then
  docker rm -f bgutil-provider >/dev/null 2>&1 || true
  docker run --name bgutil-provider -d --init --restart unless-stopped \
    -p 127.0.0.1:4416:4416 brainicism/bgutil-ytdlp-pot-provider
else
  echo "no docker: building the Node server (needs node + npm; conda install -c conda-forge nodejs if missing)"
  command -v npm >/dev/null 2>&1 || conda install -y -q -c conda-forge nodejs
  DIR="$HOME/bgutil-ytdlp-pot-provider"
  [ -d "$DIR" ] || git clone --single-branch --branch 2.0.2 https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "$DIR"
  (cd "$DIR/server" && npm ci --silent && npx tsc)
  nohup node "$DIR/server/build/main.js" > "$HOME/bgutil-provider.log" 2>&1 &
  echo "started (log: ~/bgutil-provider.log); waiting for it..."
fi
for i in $(seq 1 30); do curl -s -o /dev/null "http://127.0.0.1:4416" && break; sleep 2; done

echo "== 4/5 cookies"
if [ ! -s "$COOKIES" ]; then
  echo "!! $COOKIES not found. Export cookies from the throwaway account (README), copy them here, re-run."; exit 1
fi
chmod 600 "$COOKIES"
head -1 "$COOKIES" | grep -q "Netscape" || echo "warning: $COOKIES doesn't look like a Netscape cookies.txt"

echo "== 5/5 verify: PO token + unthrottled download from THIS machine (no proxy)"
LOG=$(mktemp)
URL=$(yt-dlp -v --cookies "$COOKIES" -f "bv*[height<=720]" -g "$TEST_VIDEO" 2>"$LOG" | tail -1 || true)
grep -E "PO Token Providers|Generating a .* PO Token|JavaScript runtime|not a bot|ERROR" "$LOG" | sed 's/^/   /' | head -8
if [ -z "$URL" ]; then echo "!! yt-dlp could not get a stream URL (see above). Full log: $LOG"; exit 1; fi
SECS=$(python - "$URL" <<'PY'
import subprocess, sys, time
t = time.time()
subprocess.run(["ffmpeg", "-loglevel", "error", "-t", "10", "-i", sys.argv[1], "-c", "copy", "-f", "null", "-"])
print(f"{time.time() - t:.1f}")
PY
)
echo "   10 s of video fetched in ${SECS}s (throttled ≈ 10 s; healthy ≈ 1-2 s)"
echo "OK. Run the pipeline WITHOUT --proxy (repo cookies are picked up automatically)."
