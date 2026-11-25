# shellcheck shell=bash
# Shared launcher for the native (no-docker) vLLM serve scripts. Source it; do not execute it.
# Each serve_*.sh sets MODEL_DEFAULT / TP_DEFAULT and a default_max_model_len() function,
# then calls `serve_main "$@"`.

SERVING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SERVING_DIR/.." && pwd)"
# The venv lives in the main checkout; git worktrees under .claude/worktrees/ reuse it.
DEFAULT_VLLM_VENV="$REPO_ROOT/.venv-vllm"
if [[ ! -d "$DEFAULT_VLLM_VENV" && "$REPO_ROOT" == */.claude/worktrees/* ]]; then
  DEFAULT_VLLM_VENV="${REPO_ROOT%%/.claude/worktrees/*}/.venv-vllm"
fi

serve_usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra vllm serve args]

Starts an OpenAI-compatible vLLM server in the foreground (native venv, no docker).
${SCRIPT_NOTE:-}
Options (each has an env var; the flag wins):
  --model ID            MODEL               (default: $MODEL_DEFAULT)
  --revision REV        REVISION            HF revision / commit sha (default: main)
  --port N              PORT                (default: 8080)
  --host H              HOST                (default: 127.0.0.1)
  --tp N                TP                  tensor-parallel size (default: $TP_DEFAULT)
  --gpus LIST           CUDA_VISIBLE_DEVICES (default: first TP GPUs, 0..TP-1)
  --max-model-len N     MAX_MODEL_LEN       (default depends on model and TP, see serving/README.md)
  --gpu-mem-util F      GPU_MEM_UTIL        (default: 0.9)
  --max-num-seqs N      MAX_NUM_SEQS        (default: vLLM default)
  --video-frames N      VIDEO_NUM_FRAMES    frames vLLM samples per local video (default: -1 = all, as in the reference)
  --video-fps F         VIDEO_FPS           (default: -1 = native fps, as in the reference)
  --mm-processor-kwargs JSON  MM_PROCESSOR_KWARGS  resolution caps; "none" = checkpoint defaults
                        (default: none, except 8B-class models at TP=1 get the TP=1 caps, see README)
  --fp8                 FP8=1               --quantization fp8 --kv-cache-dtype fp8 (changes numerics; off by default)
  --allow-download      ALLOW_DOWNLOAD=1    let vLLM fetch missing weights (default: HF_HUB_OFFLINE=1)
  --venv DIR            VLLM_VENV           (default: $DEFAULT_VLLM_VENV)
  --dry-run             print the command and exit
  -h, --help            show this help

Fixed flags (same as the reference docker recipe): --async-scheduling --enable-chunked-prefill
  --allowed-local-media-path / --mm-processor-cache-gb 0 --mm-encoder-tp-mode data,
  env VLLM_USE_FLASHINFER_SAMPLER=0. With TP>1 the script adds --disable-custom-all-reduce
  (the 4090 has no P2P).
Env passthrough: HF_HOME (default /cache/huggingface), EXTRA_ARGS (appended verbatim).
EOF
}

serve_main() {
  MODEL="${MODEL:-$MODEL_DEFAULT}"
  REVISION="${REVISION:-}"
  PORT="${PORT:-8080}"
  HOST="${HOST:-127.0.0.1}"
  TP="${TP:-$TP_DEFAULT}"
  GPUS="${CUDA_VISIBLE_DEVICES:-}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
  VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:--1}"
  VIDEO_FPS="${VIDEO_FPS:--1}"
  MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-}"
  FP8="${FP8:-0}"
  ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"
  VLLM_VENV="${VLLM_VENV:-$DEFAULT_VLLM_VENV}"
  DRY_RUN=0
  local passthrough=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model) MODEL="$2"; shift 2 ;;
      --revision) REVISION="$2"; shift 2 ;;
      --port) PORT="$2"; shift 2 ;;
      --host) HOST="$2"; shift 2 ;;
      --tp) TP="$2"; shift 2 ;;
      --gpus) GPUS="$2"; shift 2 ;;
      --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
      --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
      --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
      --video-frames) VIDEO_NUM_FRAMES="$2"; shift 2 ;;
      --video-fps) VIDEO_FPS="$2"; shift 2 ;;
      --mm-processor-kwargs) MM_PROCESSOR_KWARGS="$2"; shift 2 ;;
      --fp8) FP8=1; shift ;;
      --allow-download) ALLOW_DOWNLOAD=1; shift ;;
      --venv) VLLM_VENV="$2"; shift 2 ;;
      --dry-run) DRY_RUN=1; shift ;;
      -h|--help) serve_usage; exit 0 ;;
      --) shift; passthrough=("$@"); break ;;
      *) echo "unknown arg: $1 (put vllm args after --)" >&2; serve_usage >&2; exit 2 ;;
    esac
  done

  [[ "$TP" =~ ^[1-9][0-9]*$ ]] || { echo "--tp must be a positive integer" >&2; exit 2; }
  if [[ -z "$GPUS" ]]; then GPUS="$(seq -s, 0 $((TP - 1)))"; fi
  local ngpu; ngpu=$(awk -F, '{print NF}' <<<"$GPUS")
  if [[ "$ngpu" -lt "$TP" ]]; then
    echo "CUDA_VISIBLE_DEVICES=$GPUS has $ngpu GPU(s), TP=$TP needs $TP" >&2; exit 2
  fi
  [[ -n "$MAX_MODEL_LEN" ]] || MAX_MODEL_LEN="$(default_max_model_len "$MODEL" "$TP")"

  local cmd=(
    vllm serve "$MODEL"
    --host "$HOST" --port "$PORT"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --async-scheduling
    --enable-chunked-prefill
    --allowed-local-media-path /
    --mm-processor-cache-gb 0
    --mm-encoder-tp-mode data
    --media-io-kwargs "{\"video\": {\"num_frames\": $VIDEO_NUM_FRAMES, \"fps\": $VIDEO_FPS}}"
  )
  [[ -n "$REVISION" ]] && cmd+=(--revision "$REVISION")
  [[ -n "$MAX_NUM_SEQS" ]] && cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
  if [[ -z "$MM_PROCESSOR_KWARGS" ]] && declare -F default_mm_processor_kwargs >/dev/null; then
    MM_PROCESSOR_KWARGS="$(default_mm_processor_kwargs "$MODEL" "$TP")"
  fi
  if [[ -n "$MM_PROCESSOR_KWARGS" && "$MM_PROCESSOR_KWARGS" != none ]]; then
    cmd+=(--mm-processor-kwargs "$MM_PROCESSOR_KWARGS")
  fi
  [[ "$TP" -gt 1 ]] && cmd+=(--disable-custom-all-reduce)
  [[ "$FP8" == 1 ]] && cmd+=(--quantization fp8 --kv-cache-dtype fp8)
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_ARGS:-}" ]] && cmd+=($EXTRA_ARGS)
  cmd+=("${passthrough[@]}")

  export CUDA_VISIBLE_DEVICES="$GPUS"
  export HF_HOME="${HF_HOME:-/cache/huggingface}"
  export VLLM_USE_FLASHINFER_SAMPLER=0
  # Never pull 35-133 GB of weights by accident; fetch them on purpose with download_weights.sh.
  if [[ "$ALLOW_DOWNLOAD" != 1 ]]; then export HF_HUB_OFFLINE=1; fi

  echo "[serve] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES HF_HOME=$HF_HOME HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} venv=$VLLM_VENV" >&2
  echo "[serve] $(printf '%q ' "${cmd[@]}")" >&2
  [[ $DRY_RUN == 1 ]] && exit 0

  [[ -x "$VLLM_VENV/bin/vllm" ]] || { echo "no vllm in $VLLM_VENV; run serving/setup_vllm_env.sh" >&2; exit 1; }
  export PATH="$VLLM_VENV/bin:$PATH"
  exec "${cmd[@]}"
}

# Defaults for Qwen3-VL-8B-sized text towers (Cosmos3-Nano, Qwen3-VL-8B): 36 layers x 8 KV heads x 128 dim.
# KV cache = 144 KiB/token in bf16. See serving/README.md for the memory budget.
max_len_8b() {
  case "$1" in
    1) echo 16384 ;;    # ~17.6 GB weights on 24 GB leaves ~3 GiB KV ≈ 20k tokens
    *) echo 65536 ;;    # TP=2: ~9 GB weights/GPU, ~11 GiB KV/GPU ≈ 160k tokens pool
  esac
}

# Qwen3-VL-32B-sized text towers (Cosmos3-Super, Qwen3-VL-32B): 64 layers x 8 KV heads x 128 dim = 256 KiB/token.
max_len_32b() {
  case "$1" in
    1|2) echo "TP=$1 cannot hold ~66 GB of weights on 24 GB GPUs" >&2; echo 4096 ;;
    4) echo 32768 ;;    # ~16.5 GB weights/GPU, ~4 GiB KV/GPU at 64 KiB/token/GPU ≈ 65k tokens pool
    *) echo 128000 ;;   # TP=8: ~8.5 GB weights/GPU, ~11 GiB KV/GPU at 32 KiB/token/GPU ≈ 350k tokens pool
  esac
}

# At TP=1 a 24 GB card holds only a ~16k context. The checkpoint's own budgets let one image reach
# 16,384 tokens (longest_edge 16,777,216 px / 32^2) and one video 12,288 tokens (25,165,824 px over
# t*h*w / (32^2*2)), which would fill the whole context. Halve both so one image or video plus the
# prompt fits in 16k.
TP1_MM_CAPS='{"images_kwargs": {"max_pixels": 8388608}, "videos_kwargs": {"max_pixels": 12582912}}'
mm_caps_8b() { if [[ "$2" == 1 ]]; then echo "$TP1_MM_CAPS"; else echo none; fi; }
