#!/usr/bin/env bash
# Serve nvidia/Cosmos3-Nano (reasoner/understanding tower only) with native vLLM.
# vLLM's Cosmos3ForConditionalGeneration (a Qwen3-VL-8B subclass) loads only the understanding
# weights (~8.8B params, ~17.6 GB bf16), so the default is TP=1 on one 4090.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL_DEFAULT="nvidia/Cosmos3-Nano"
TP_DEFAULT=1
TP_ALLOWED="$TP_ALLOWED_8B"
SCRIPT_NOTE="Cosmos3-Nano: TP=1 -> max-model-len 16384 + image/video pixel caps; --tp 2 -> 65536, no caps."
default_max_model_len() { max_len_8b "$2"; }
default_mm_processor_kwargs() { mm_caps_8b "$1" "$2"; }

serve_main "$@"
