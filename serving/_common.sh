# shellcheck shell=bash
# Shared launcher for the native (no-docker) vLLM serve scripts. Source it; do not execute it.
# Each serve_*.sh sets MODEL_DEFAULT, TP_DEFAULT, TP_ALLOWED and a default_max_model_len() function
# (optionally default_mm_processor_kwargs()), then calls `serve_main "$@"`.

SERVING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SERVING_DIR/.." && pwd)"
# The venv lives in the main checkout; git worktrees under .claude/worktrees/ reuse it.
DEFAULT_VLLM_VENV="${REPO_ROOT%%/.claude/worktrees/*}/.venv-vllm"

die() { echo "error: $*" >&2; exit 2; }

# need_value <flag> <remaining args...>: the flag must be followed by a value that is not another option.
need_value() {
  [[ $# -ge 2 ]] || die "$1 needs a value"
  [[ "$2" != --* ]] || die "$1 needs a value (got option $2)"
}

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_pos_int() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_pos_num() { [[ "$1" =~ ^([0-9]+\.?[0-9]*|\.[0-9]+)$ ]] && awk -v x="$1" 'BEGIN{exit !(x>0)}'; }

serve_usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra vllm serve args]

Starts an OpenAI-compatible vLLM server in the foreground (native venv, no docker).
${SCRIPT_NOTE:-}
Options (each has an env var; the flag wins):
  --model ID            MODEL               (default: $MODEL_DEFAULT)
  --revision REV        REVISION            HF revision / commit sha (default: main)
  --port N              PORT                (default: 8080; must be free)
  --host H              HOST                (default: 127.0.0.1)
  --tp N                TP                  tensor-parallel size, one of: ${TP_ALLOWED} (default: $TP_DEFAULT)
  --gpus LIST           CUDA_VISIBLE_DEVICES exactly TP distinct GPU indices, e.g. 0,1
                        (default when unset: 0..TP-1; a set-but-empty mask is an error)
  --max-model-len N     MAX_MODEL_LEN       (default depends on model and TP, see serving/README.md)
  --gpu-mem-util F      GPU_MEM_UTIL        fraction in (0, 1] (default: 0.9)
  --max-num-seqs N      MAX_NUM_SEQS        (default: vLLM default)
  --video-frames N      VIDEO_NUM_FRAMES    frames vLLM samples per local video, -1 = all (default: -1, as in the reference)
  --video-fps F         VIDEO_FPS           -1 = native fps (default: -1, as in the reference)
  --mm-processor-kwargs JSON  MM_PROCESSOR_KWARGS  resolution caps; "none" = checkpoint defaults
                        (default: none, except 8B-class models at TP=1 get the TP=1 caps, see README)
  --fp8                 FP8=1               --quantization fp8 --kv-cache-dtype fp8 (changes numerics; off by default)
  --allow-download      ALLOW_DOWNLOAD=1    let vLLM fetch missing weights (default: HF_HUB_OFFLINE=1)
  --venv DIR            VLLM_VENV           (default: $DEFAULT_VLLM_VENV)
  --dry-run             validate, print the command and exit (no GPU / port checks)
  -h, --help            show this help

Fixed flags (same as the reference docker recipe): --async-scheduling --enable-chunked-prefill
  --allowed-local-media-path / --mm-processor-cache-gb 0 --mm-encoder-tp-mode data,
  env VLLM_USE_FLASHINFER_SAMPLER=0. With TP>1 the script adds --disable-custom-all-reduce
  (the 4090 has no P2P).
Extra vllm args go after "--". Parallelism flags (--tensor-parallel-size/-tp, --pipeline-parallel-size/-pp,
  --data-parallel-size/-dp) and flags the script manages are refused there; use the options above.
Env: HF_HOME (default /cache/huggingface).
EOF
}

# validate_gpu_mask <mask> <tp>: non-empty, comma-separated distinct indices, exactly TP of them.
validate_gpu_mask() {
  local mask="$1" tp="$2"
  [[ -n "$mask" ]] || die "empty GPU mask (CUDA_VISIBLE_DEVICES is set but empty, or --gpus ''); TP=$tp needs $tp GPU(s)"
  [[ "$mask" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "bad GPU mask '$mask': want comma-separated indices like 0,1"
  local -a ids; IFS=, read -r -a ids <<<"$mask"
  local uniq; uniq=$(printf '%s\n' "${ids[@]}" | sort -n | uniq | wc -l)
  [[ "$uniq" -eq "${#ids[@]}" ]] || die "GPU mask '$mask' repeats a GPU"
  [[ "${#ids[@]}" -eq "$tp" ]] || die "GPU mask '$mask' has ${#ids[@]} GPU(s) but TP=$tp; give exactly $tp"
}

# reject_managed_passthrough <args...>: refuse vllm args that would override what the script validates.
reject_managed_passthrough() {
  local a
  for a in "$@"; do
    case "$a" in
      -tp|-tp=*|--tensor-parallel-size|--tensor-parallel-size=*|\
      -pp|-pp=*|--pipeline-parallel-size|--pipeline-parallel-size=*|\
      -dp|-dp=*|--data-parallel-size|--data-parallel-size=*)
        die "'$a' after -- is not allowed; use --tp / --gpus so the GPU mask is checked" ;;
      --port|--port=*|--host|--host=*|--max-model-len|--max-model-len=*|\
      --mm-processor-kwargs|--mm-processor-kwargs=*|--media-io-kwargs|--media-io-kwargs=*|\
      --revision|--revision=*|--gpu-memory-utilization|--gpu-memory-utilization=*)
        die "'$a' after -- is managed by this script; use the matching option (see --help)" ;;
    esac
  done
}

serve_main() {
  MODEL="${MODEL:-$MODEL_DEFAULT}"
  REVISION="${REVISION:-}"
  PORT="${PORT:-8080}"
  HOST="${HOST:-127.0.0.1}"
  TP="${TP:-$TP_DEFAULT}"
  # Unset mask -> derive 0..TP-1. Set-but-empty mask -> error (an empty mask means "no GPUs").
  local gpus_set=0 GPUS=""
  if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then gpus_set=1; GPUS="$CUDA_VISIBLE_DEVICES"; fi
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
  VIDEO_NUM_FRAMES="${VIDEO_NUM_FRAMES:--1}"
  VIDEO_FPS="${VIDEO_FPS:--1}"
  MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-}"
  FP8="${FP8:-0}"
  ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"
  VLLM_VENV="${VLLM_VENV:-$DEFAULT_VLLM_VENV}"
  local DRY_RUN=0
  local passthrough=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model) need_value "$@"; MODEL="$2"; shift 2 ;;
      --revision) need_value "$@"; REVISION="$2"; shift 2 ;;
      --port) need_value "$@"; PORT="$2"; shift 2 ;;
      --host) need_value "$@"; HOST="$2"; shift 2 ;;
      --tp) need_value "$@"; TP="$2"; shift 2 ;;
      --gpus) [[ $# -ge 2 ]] || die "--gpus needs a value"; gpus_set=1; GPUS="$2"; shift 2 ;;
      --max-model-len) need_value "$@"; MAX_MODEL_LEN="$2"; shift 2 ;;
      --gpu-mem-util) need_value "$@"; GPU_MEM_UTIL="$2"; shift 2 ;;
      --max-num-seqs) need_value "$@"; MAX_NUM_SEQS="$2"; shift 2 ;;
      --video-frames) need_value "$@"; VIDEO_NUM_FRAMES="$2"; shift 2 ;;
      --video-fps) need_value "$@"; VIDEO_FPS="$2"; shift 2 ;;
      --mm-processor-kwargs) need_value "$@"; MM_PROCESSOR_KWARGS="$2"; shift 2 ;;
      --fp8) FP8=1; shift ;;
      --allow-download) ALLOW_DOWNLOAD=1; shift ;;
      --venv) need_value "$@"; VLLM_VENV="$2"; shift 2 ;;
      --dry-run) DRY_RUN=1; shift ;;
      -h|--help) serve_usage; exit 0 ;;
      --) shift; passthrough=("$@"); break ;;
      *) echo "unknown arg: $1 (put vllm args after --)" >&2; serve_usage >&2; exit 2 ;;
    esac
  done

  # ---- validation: everything that ends up in the command line or in JSON is checked here ----
  [[ -n "$MODEL" && "$MODEL" != *[[:space:]]* ]] || die "bad --model '$MODEL'"
  [[ -z "$REVISION" || "$REVISION" =~ ^[A-Za-z0-9._/-]+$ ]] || die "bad --revision '$REVISION'"
  [[ "$HOST" =~ ^[A-Za-z0-9.:-]+$ ]] || die "bad --host '$HOST'"
  is_pos_int "$PORT" && (( PORT <= 65535 )) || die "--port must be 1-65535 (got '$PORT')"
  is_pos_int "$TP" || die "--tp must be a positive integer (got '$TP')"
  [[ " $TP_ALLOWED " == *" $TP "* ]] || die "TP=$TP not allowed for $MODEL; allowed: $TP_ALLOWED"
  if [[ $gpus_set == 0 ]]; then GPUS="$(seq -s, 0 $((TP - 1)))"; fi
  validate_gpu_mask "$GPUS" "$TP"
  [[ -n "$MAX_MODEL_LEN" ]] || MAX_MODEL_LEN="$(default_max_model_len "$MODEL" "$TP")"
  is_pos_int "$MAX_MODEL_LEN" || die "--max-model-len must be a positive integer (got '$MAX_MODEL_LEN')"
  is_pos_num "$GPU_MEM_UTIL" && awk -v x="$GPU_MEM_UTIL" 'BEGIN{exit !(x<=1)}' \
    || die "--gpu-mem-util must be in (0, 1] (got '$GPU_MEM_UTIL')"
  [[ -z "$MAX_NUM_SEQS" ]] || is_pos_int "$MAX_NUM_SEQS" || die "--max-num-seqs must be a positive integer"
  [[ "$VIDEO_NUM_FRAMES" == -1 ]] || is_pos_int "$VIDEO_NUM_FRAMES" \
    || die "--video-frames must be -1 or a positive integer (got '$VIDEO_NUM_FRAMES')"
  [[ "$VIDEO_FPS" == -1 ]] || is_pos_num "$VIDEO_FPS" || die "--video-fps must be -1 or a positive number (got '$VIDEO_FPS')"
  [[ "$FP8" == 0 || "$FP8" == 1 ]] || die "FP8 must be 0 or 1"
  [[ "$ALLOW_DOWNLOAD" == 0 || "$ALLOW_DOWNLOAD" == 1 ]] || die "ALLOW_DOWNLOAD must be 0 or 1"
  reject_managed_passthrough "${passthrough[@]}"

  if [[ -z "$MM_PROCESSOR_KWARGS" ]] && declare -F default_mm_processor_kwargs >/dev/null; then
    MM_PROCESSOR_KWARGS="$(default_mm_processor_kwargs "$MODEL" "$TP")"
  fi
  if [[ -n "$MM_PROCESSOR_KWARGS" && "$MM_PROCESSOR_KWARGS" != none ]]; then
    python3 -I -c 'import json,sys; d=json.loads(sys.argv[1]); assert isinstance(d, dict)' "$MM_PROCESSOR_KWARGS" 2>/dev/null \
      || die "--mm-processor-kwargs must be a JSON object (got '$MM_PROCESSOR_KWARGS')"
  fi

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
  if [[ -n "$REVISION" ]]; then cmd+=(--revision "$REVISION"); fi
  if [[ -n "$MAX_NUM_SEQS" ]]; then cmd+=(--max-num-seqs "$MAX_NUM_SEQS"); fi
  if [[ -n "$MM_PROCESSOR_KWARGS" && "$MM_PROCESSOR_KWARGS" != none ]]; then
    cmd+=(--mm-processor-kwargs "$MM_PROCESSOR_KWARGS")
  fi
  if (( TP > 1 )); then cmd+=(--disable-custom-all-reduce); fi
  if [[ "$FP8" == 1 ]]; then cmd+=(--quantization fp8 --kv-cache-dtype fp8); fi
  cmd+=("${passthrough[@]}")

  export CUDA_VISIBLE_DEVICES="$GPUS"
  export HF_HOME="${HF_HOME:-/cache/huggingface}"
  export VLLM_USE_FLASHINFER_SAMPLER=0
  # Never pull 35-133 GB of weights by accident; fetch them on purpose with download_weights.sh.
  if [[ "$ALLOW_DOWNLOAD" == 1 ]]; then
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  else
    export HF_HUB_OFFLINE=1
  fi

  echo "[serve] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES HF_HOME=$HF_HOME HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>} venv=$VLLM_VENV" >&2
  echo "[serve] $(printf '%q ' "${cmd[@]}")" >&2
  if [[ $DRY_RUN == 1 ]]; then exit 0; fi

  [[ -x "$VLLM_VENV/bin/vllm" ]] || { echo "no vllm in $VLLM_VENV; run serving/setup_vllm_env.sh" >&2; exit 1; }
  # GPU indices must exist on this host.
  if command -v nvidia-smi >/dev/null; then
    local ngpu id; ngpu=$(env -u CUDA_VISIBLE_DEVICES nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
    IFS=, read -r -a _ids <<<"$GPUS"
    for id in "${_ids[@]}"; do (( id < ngpu )) || die "GPU $id does not exist (host has $ngpu)"; done
  fi
  # Refuse to start if something already listens on the port (an old server would answer health checks).
  if python3 -I "$SERVING_DIR/_portcheck.py" "$PORT"; then
    die "port $PORT is already in use; stop the old server or pick --port"
  fi
  export PATH="$VLLM_VENV/bin:$PATH"
  exec "${cmd[@]}"
}

# Defaults for Qwen3-VL-8B-sized text towers (Cosmos3-Nano, Qwen3-VL-8B): 36 layers x 8 KV heads x 128 dim.
# KV cache = 144 KiB/token in bf16. See serving/README.md for the memory budget.
TP_ALLOWED_8B="1 2 4 8"
max_len_8b() {
  case "$1" in
    1) echo 16384 ;;    # ~17.6 GB weights on 24 GB leaves ~3.5 GiB KV ≈ 25k tokens
    *) echo 65536 ;;    # TP=2: ~9 GB weights/GPU, ~11 GiB KV/GPU ≈ 160k tokens pool
  esac
}

# Qwen3-VL-32B-sized text towers (Cosmos3-Super, Qwen3-VL-32B): 64 layers x 8 KV heads x 128 dim = 256 KiB/token.
# ~66 GB of weights: TP=1/2 cannot hold them on 24 GB GPUs.
TP_ALLOWED_32B="4 8"
max_len_32b() {
  case "$1" in
    4) echo 32768 ;;    # ~16.5 GB weights/GPU, ~4 GiB KV/GPU at 64 KiB/token/GPU ≈ 60k tokens pool
    8) echo 128000 ;;   # ~8.7 GB weights/GPU, ~11 GiB KV/GPU at 32 KiB/token/GPU ≈ 370k tokens pool
    *) die "no max-model-len default for a 32B-class model at TP=$1" ;;
  esac
}

# At TP=1 a 24 GB card holds only a ~16k context. The checkpoint's own budgets let one image reach
# 16,384 tokens (longest_edge 16,777,216 px / 32^2) and one video 12,288 tokens (25,165,824 px over
# t*h*w / (32^2*2)), which would fill the whole context. Halve both so one image or video plus the
# prompt fits in 16k. (Checked on CPU against vLLM 0.32's Qwen3-VL processing: an 8192x8192 image goes
# 16384 -> 8100 tokens, a 600-frame 4K video 9600 -> 5400 tokens.)
TP1_MM_CAPS='{"images_kwargs": {"max_pixels": 8388608}, "videos_kwargs": {"max_pixels": 12582912}}'
mm_caps_8b() { if [[ "$2" == 1 ]]; then echo "$TP1_MM_CAPS"; else echo none; fi; }
