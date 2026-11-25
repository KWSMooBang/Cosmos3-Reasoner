#!/usr/bin/env bash
# Serve nvidia/Cosmos3-Super (reasoner/understanding tower only) with native vLLM.
# The understanding tower is Qwen3-VL-32B-sized (~33B params, ~66 GB bf16), so on 24 GB GPUs
# it needs TP=4 (default, max-model-len 32768) or TP=8 (max-model-len 128000).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL_DEFAULT="nvidia/Cosmos3-Super"
TP_DEFAULT=4
TP_ALLOWED="$TP_ALLOWED_32B"
SCRIPT_NOTE="Cosmos3-Super: TP must be 4 or 8 (8 KV heads, ~66 GB weights). TP=4 -> 32768, TP=8 -> 128000."
default_max_model_len() { max_len_32b "$2"; }

serve_main "$@"
