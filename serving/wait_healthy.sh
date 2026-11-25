#!/usr/bin/env bash
# Poll GET <base>/v1/models until the server we started lists the expected model, or time out.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --model ID [--url URL | --port N] [--pid PID] [--timeout SEC] [--interval SEC]

Ready means: /v1/models returns JSON whose data[].id contains exactly ID.
With --pid (local URLs), each poll first checks that PID is alive. Ready is reported only once /proc
confirms that PID or one of its child processes owns the listening socket, so an old server on the
same port is never mistaken for the new one. For a remote --url (host other than localhost/127.x/::1)
the PID/port check is skipped, and a log line says so.

  --model ID      required: served model id, e.g. nvidia/Cosmos3-Nano
  --url URL       base URL           (default: http://127.0.0.1:\$PORT)
  --port N        port on 127.0.0.1  (env PORT, default 8080)
  --pid PID       the server process (recommended: \$! of the serve script)
  --timeout SEC   give up after SEC  (default: 1800; first start compiles and captures CUDA graphs)
  --interval SEC  poll period        (default: 10)
Exit codes: 0 ready, 1 timeout, 2 server process died / port owned by another process, 3 bad args.
EOF
}

bad() { echo "error: $*" >&2; usage >&2; exit 3; }
need_value() { [[ $# -ge 2 && "$2" != --* ]] || bad "$1 needs a value"; }

PORT="${PORT:-8080}"; URL=""; TIMEOUT=1800; INTERVAL=10; PID=""; MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) need_value "$@"; URL="$2"; shift 2 ;;
    --port) need_value "$@"; PORT="$2"; shift 2 ;;
    --timeout) need_value "$@"; TIMEOUT="$2"; shift 2 ;;
    --interval) need_value "$@"; INTERVAL="$2"; shift 2 ;;
    --pid) need_value "$@"; PID="$2"; shift 2 ;;
    --model) need_value "$@"; MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) bad "unknown arg: $1" ;;
  esac
done
[[ -n "$MODEL" ]] || bad "--model is required"
[[ "$PORT" =~ ^[1-9][0-9]*$ ]] && (( PORT <= 65535 )) || bad "bad --port '$PORT'"
[[ "$TIMEOUT" =~ ^[1-9][0-9]*$ ]] || bad "bad --timeout '$TIMEOUT'"
[[ "$INTERVAL" =~ ^[1-9][0-9]*$ ]] || bad "bad --interval '$INTERVAL'"
[[ -z "$PID" || "$PID" =~ ^[1-9][0-9]*$ ]] || bad "bad --pid '$PID'"
URL="${URL:-http://127.0.0.1:$PORT}"
URL="${URL%/}"
# Split the URL: scheme://host[:port][/path]. A missing port means the scheme default.
[[ "$URL" =~ ^(https?)://(\[[^]]+\]|[^/:]+)(:([0-9]+))?(/.*)?$ ]] || bad "bad --url '$URL' (want http[s]://host[:port])"
URL_SCHEME="${BASH_REMATCH[1]}"; URL_HOST="${BASH_REMATCH[2]}"; URL_PORT="${BASH_REMATCH[4]}"
if [[ -n "$URL_PORT" ]]; then
  PORT=$((10#$URL_PORT)); (( PORT >= 1 && PORT <= 65535 )) || bad "bad port in --url '$URL'"
elif [[ "$URL_SCHEME" == https ]]; then PORT=443; else PORT=80; fi
case "$URL_HOST" in
  localhost|127.*|0.0.0.0|"[::1]") LOCAL=1 ;;
  *) LOCAL=0 ;;
esac

PORTCHECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_portcheck.py"
# 0 = PID (or a child) listens on PORT, 1 = another process does, 2 = nothing listens yet / cannot tell.
port_owner_check() { python3 -I "$PORTCHECK" "$PORT" "$PID"; }

# Prints "ok" if the /v1/models JSON body (stdin) lists exactly MODEL in data[].id.
has_model() {
  python3 -I -c '
import json, sys
try:
    ids = [m.get("id") for m in json.load(sys.stdin).get("data", []) if isinstance(m, dict)]
except Exception:
    sys.exit(1)
sys.exit(0 if sys.argv[1] in ids else 1)
' "$MODEL"
}

start=$(date +%s)
deadline=$(( start + TIMEOUT ))
if [[ $LOCAL == 0 ]]; then
  echo "[wait_healthy] remote URL ($URL_HOST): skipping the PID/port ownership check${PID:+ (--pid $PID ignored)}" >&2
  PID=""
elif [[ -z "$PID" ]]; then
  echo "[wait_healthy] no --pid: cannot tell our server from an old one on port $PORT" >&2
fi
warned_unknown=0
while true; do
  if [[ -n "$PID" ]] && ! kill -0 "$PID" 2>/dev/null; then
    echo "[wait_healthy] server pid $PID exited before becoming ready" >&2; exit 2
  fi
  left=$(( deadline - $(date +%s) ))
  (( left < 1 )) && left=1
  ctimeout=$(( left < 5 ? left : 5 ))
  if body=$(curl -fsS --max-time "$ctimeout" "$URL/v1/models" 2>/dev/null) && has_model <<<"$body"; then
    if [[ -z "$PID" ]]; then
      echo "[wait_healthy] ready after $(( $(date +%s) - start ))s: $URL serves $MODEL"
      exit 0
    fi
    rc=0; port_owner_check || rc=$?
    case $rc in
      0) echo "[wait_healthy] ready after $(( $(date +%s) - start ))s: $URL serves $MODEL (pid $PID owns port $PORT)"
         exit 0 ;;
      1) echo "[wait_healthy] $URL answers, but port $PORT belongs to another process, not pid $PID" >&2; exit 2 ;;
      *) # Ownership unknown: do not report ready; keep polling until confirmed or timeout.
         if [[ $warned_unknown == 0 ]]; then
           echo "[wait_healthy] $URL answers, but cannot confirm pid $PID owns port $PORT yet; still waiting" >&2
           warned_unknown=1
         fi ;;
    esac
  fi
  now=$(date +%s)
  if (( now >= deadline )); then
    echo "[wait_healthy] timeout after ${TIMEOUT}s: $URL/v1/models (want $MODEL)" >&2; exit 1
  fi
  left=$(( deadline - now ))
  sleep $(( INTERVAL < left ? INTERVAL : left ))
done
