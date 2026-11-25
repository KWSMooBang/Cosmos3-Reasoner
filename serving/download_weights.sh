#!/usr/bin/env bash
# Download only the files vLLM reads for the Cosmos3 reasoner into $HF_HOME.
# vLLM's Cosmos3 model reads "[tv]*er/*.safetensors" (transformer/ + vision_encoder/), and the config,
# processor and tokenizer come from root files (checked on CPU with transformers 5.19: AutoConfig +
# AutoProcessor load from exactly the root files below). vae/ (1.4 GB), sound_tokenizer/ (2.0 GB),
# text_tokenizer/, assets/, images/ and scheduler/ are skipped. transformer/ still holds both towers
# (vLLM drops the generation tower at load time), so the savings are small:
# Nano 35.0 -> ~31.5 GB, Super 132.7 -> ~129.2 GB.
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
bad() { echo "error: $*" >&2; usage >&2; exit 2; }
need_value() { [[ $# -ge 2 && "$2" != --* ]] || bad "$1 needs a value"; }

# Root files vLLM / transformers need for Cosmos3 (config, generation defaults, processor, tokenizer).
COSMOS3_ROOT_FILES=(
  config.json generation_config.json chat_template.json
  preprocessor_config.json video_preprocessor_config.json
  tokenizer.json tokenizer_config.json vocab.json merges.txt
)
COSMOS3_DIRS=("transformer/*" "vision_encoder/*")

MODEL=""; REVISION=""; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) need_value "$@"; MODEL="$2"; shift 2 ;;
    --revision) need_value "$@"; REVISION="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) bad "unknown arg: $1" ;;
  esac
done
[[ -n "$MODEL" ]] || bad "--model is required"
export HF_HOME="${HF_HOME:-/cache/huggingface}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VLLM_VENV:-${REPO_ROOT%%/.claude/worktrees/*}/.venv-vllm}"
HF="$VENV/bin/hf"; [[ -x "$HF" ]] || HF=hf

cmd=("$HF" download "$MODEL")
if [[ -n "$REVISION" ]]; then cmd+=(--revision "$REVISION"); fi
if [[ "$MODEL" == nvidia/Cosmos3-* ]]; then
  # hf takes ONE pattern per --include; extra words would become literal FILENAMES. Repeat the flag.
  for pat in "${COSMOS3_ROOT_FILES[@]}" "${COSMOS3_DIRS[@]}"; do cmd+=(--include "$pat"); done
fi
echo "HF_HOME=$HF_HOME $(printf '%q ' "${cmd[@]}")"
if [[ $DRY == 1 ]]; then exit 0; fi
exec "${cmd[@]}"
