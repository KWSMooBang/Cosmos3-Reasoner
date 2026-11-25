#!/usr/bin/env bash
# Download only the files vLLM reads for the Cosmos3 reasoner into $HF_HOME.
# vLLM's Cosmos3 model reads "[tv]*er/*.safetensors" (transformer/ + vision_encoder/), plus the root
# configs and tokenizer, so vae/ (1.4 GB), sound_tokenizer/ (2.0 GB) and assets/ are skipped.
# transformer/ still holds both towers (vLLM drops the generation tower at load time), so the
# savings are small: Nano 35.0 -> ~31.5 GB, Super 132.7 -> ~129.3 GB.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --model ID [--revision REV] [--dry-run]
  Downloads into HF_HOME (default /cache/huggingface). Needs explicit approval for the large models.
  --model ID      nvidia/Cosmos3-Nano | nvidia/Cosmos3-Super | Qwen/Qwen3-VL-8B-Instruct | ...
  --revision REV  commit sha to pin (recommended)
  --dry-run       print the hf command only
EOF
}

MODEL=""; REVISION=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$MODEL" ]] || { usage >&2; exit 2; }
export HF_HOME="${HF_HOME:-/cache/huggingface}"
VENV="${VLLM_VENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv-vllm}"
HF="$VENV/bin/hf"; [[ -x "$HF" ]] || HF=hf

cmd=("$HF" download "$MODEL")
[[ -n "$REVISION" ]] && cmd+=(--revision "$REVISION")
if [[ "$MODEL" == nvidia/Cosmos3-* ]]; then
  cmd+=(--include "*.json" "*.txt" "transformer/*" "vision_encoder/*" "text_tokenizer/*")
fi
printf '%q ' "${cmd[@]}"; echo
[[ $DRY == 1 ]] && exit 0
exec "${cmd[@]}"
