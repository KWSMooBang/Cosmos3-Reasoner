#!/usr/bin/env bash
# Serve the Qwen3-VL baselines with native vLLM, using the same flags as the Cosmos3 scripts.
#   Qwen/Qwen3-VL-8B-Instruct  : same text tower as Cosmos3-Nano -> TP=1 (16384, pixel caps) or TP=2 (65536)
#   Qwen/Qwen3-VL-32B-Instruct : same text tower as Cosmos3-Super -> TP=4 (32768) or TP=8 (128000)
# Use the same TP/max-model-len/caps as the Cosmos3 model you compare against.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL_DEFAULT="Qwen/Qwen3-VL-8B-Instruct"
is_32b() { [[ "$1" == *32B* ]]; }
# TP default depends on --model, so resolve it before serve_main parses the flags.
_model="${MODEL:-$MODEL_DEFAULT}"
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--model" ]]; then j=$((i + 1)); _model="${!j:-}"; fi
  [[ "${!i}" == "--" ]] && break
done
if is_32b "$_model"; then TP_DEFAULT=4; else TP_DEFAULT=1; fi
SCRIPT_NOTE="Qwen3-VL: --model Qwen/Qwen3-VL-8B-Instruct (TP=1 default) or Qwen/Qwen3-VL-32B-Instruct (TP=4 default)."
default_max_model_len() { if is_32b "$1"; then max_len_32b "$2"; else max_len_8b "$2"; fi; }
default_mm_processor_kwargs() { if is_32b "$1"; then echo none; else mm_caps_8b "$1" "$2"; fi; }

serve_main "$@"
