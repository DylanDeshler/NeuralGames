#!/usr/bin/env bash
# Run a pipeline mode as a systemd service: starts at boot, restarts after a crash, survives daily reboots.
# Everything resumes from its outputs, so a restart only costs the model-server startup (~5 min on the GPU box).
#
#   bash tools/install_service.sh <name> <run_pipeline.py arguments...>      (run as your normal user, NOT sudo)
#
#   bash tools/install_service.sh gpu   gpu   --data-root /data/games --start-vlm-server --push-to-hub ddeshler/nitrogen-2d-wm --workers 16
#   bash tools/install_service.sh build build --data-root /data/games --push-to-hub ddeshler/nitrogen-2d-wm --workers 44
#
# Watch:   tail -f /data/games/logs/run_pipeline.log        (or: journalctl -u ng-<name> -f)
# Status:  systemctl status ng-<name>
# Stop:    sudo systemctl disable --now ng-<name>             (stops it and keeps it from starting at boot)
set -euo pipefail
[ "$(id -u)" -ne 0 ] || { echo "run as your normal user (the script calls sudo itself)"; exit 1; }
[ $# -ge 2 ] || { sed -n 2,13p "$0"; exit 1; }
NAME="ng-$1"; shift
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${NG_PYTHON:-$(command -v python || command -v python3)}"   # the env you normally run the pipeline with
PY="$(readlink -f "$PY")"
case " $* " in *" --data-root "*|*" --data "*) ;; *) echo "pass --data-root explicitly"; exit 1;; esac
ARGS=""
for x in "$@"; do ARGS="$ARGS $(printf '%q' "$x")"; done

UNIT="/etc/systemd/system/$NAME.service"
sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=NeuralGames pipeline ($NAME)
After=network-online.target docker.service
Wants=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$REPO
ExecStart=$PY $REPO/run_pipeline.py$ARGS
Environment=PYTHONUNBUFFERED=1
Environment=PATH=$(dirname "$PY"):$HOME/.deno/bin:/usr/local/bin:/usr/bin:/bin
Restart=on-failure
RestartSec=60
KillMode=control-group
TimeoutStopSec=90

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now "$NAME"
echo "installed $UNIT"
echo "  python : $PY"
echo "  command: run_pipeline.py$ARGS"
sleep 3; systemctl --no-pager status "$NAME" | head -5
echo "watch: tail -f <data-root>/logs/run_pipeline.log   |   stop: sudo systemctl disable --now $NAME"
