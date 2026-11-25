#!/usr/bin/env bash
# Poll GET <base>/v1/models until the vLLM server answers with a model list, or time out.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [--url URL | --port N] [--timeout SEC] [--interval SEC] [--pid PID] [--model ID]

  --url URL       base URL           (default: http://127.0.0.1:\$PORT, PORT default 8080)
  --port N        port on 127.0.0.1  (env PORT)
  --timeout SEC   give up after SEC  (default: 1800; first start compiles and captures CUDA graphs)
  --interval SEC  poll period        (default: 10)
  --pid PID       stop early if this process (the server) exits
  --model ID      also require this id in the /v1/models list
Exit codes: 0 ready, 1 timeout, 2 server process died, 3 bad args.
EOF
}

PORT="${PORT:-8080}"; URL=""; TIMEOUT=1800; INTERVAL=10; PID=""; MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --pid) PID="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 3 ;;
  esac
done
URL="${URL:-http://127.0.0.1:$PORT}"
URL="${URL%/}"

start=$(date +%s)
while true; do
  if body=$(curl -fsS --max-time 5 "$URL/v1/models" 2>/dev/null) && [[ "$body" == *'"data"'* ]]; then
    if [[ -z "$MODEL" || "$body" == *"\"$MODEL\""* ]]; then
      echo "[wait_healthy] ready after $(( $(date +%s) - start ))s: $URL"
      exit 0
    fi
  fi
  if [[ -n "$PID" ]] && ! kill -0 "$PID" 2>/dev/null; then
    echo "[wait_healthy] server pid $PID exited before becoming ready" >&2; exit 2
  fi
  if (( $(date +%s) - start >= TIMEOUT )); then
    echo "[wait_healthy] timeout after ${TIMEOUT}s: $URL/v1/models" >&2; exit 1
  fi
  sleep "$INTERVAL"
done
