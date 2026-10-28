#!/usr/bin/env bash
# Start ComfyUI (:8188), the v2 API proxy (:8189) and the DollyGrip app (:8000).
# Logs go to $WORK/logs. Expose port 8000 on the pod for the public link.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/workspace}
COMFY=$WORK/ComfyUI
LOGS=$WORK/logs
mkdir -p "$LOGS"

wait_for() { for _ in $(seq 1 180); do curl -sf "$1" >/dev/null && return 0; sleep 1; done; echo "timeout: $1"; exit 1; }

if ! curl -sf http://127.0.0.1:8188/system_stats >/dev/null; then
  (cd "$COMFY" && nohup python main.py --listen 127.0.0.1 --port 8188 > "$LOGS/comfyui.log" 2>&1 &)
  wait_for http://127.0.0.1:8188/system_stats
fi
echo "comfyui up"

if ! curl -s -o /dev/null http://127.0.0.1:8189/; then
  nohup comfy-api-proxy --comfyui http://127.0.0.1:8188 --port 8189 > "$LOGS/proxy.log" 2>&1 &
  sleep 3
fi
echo "proxy up"

python "$REPO/scripts/build_workflow.py" --server http://127.0.0.1:8188 || {
  echo "workflow validation failed; see PROBLEMS above"; exit 1; }

cd "$REPO"
COMFY_BASE_URL=http://127.0.0.1:8189 nohup uvicorn app.server:app --host 0.0.0.0 --port 8000 > "$LOGS/app.log" 2>&1 &
wait_for http://127.0.0.1:8000/api/health
echo "app up on :8000"
